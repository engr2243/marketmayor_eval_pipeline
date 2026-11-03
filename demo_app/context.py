"""
Turning a contact's message history into something the model can cite.

Builds a transcript that fits a budget without losing the parts that matter, and
returns the label-to-id map that lets verification check every quote against the
message it claims to come from.

Messages are labelled `[m1]`, `[m2]` rather than by uuid: a uuid costs about
fifteen tokens each and invites a mistype, while an invented short label simply
fails to resolve and is caught.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

# Kinds that carry words worth reading. A call or voicemail arrives as a
# timestamp and a direction only, so it counts as activity but contributes no
# text to read.
TEXTUAL_KINDS = frozenset({"note", "sms", "email", "chat", "social", "other"})

# Per-message cap. Long enough for a real message, short enough that one
# forwarded email thread cannot consume a whole contact's budget.
MAX_BODY_CHARS = 900
# Total transcript budget, roughly six thousand tokens.
DEFAULT_CHAR_BUDGET = 24_000
# When trimming, the share of the budget reserved for the oldest messages.
# Commitments usually live in an old one ("in March they said call me in
# September"), which a keep-the-newest window would throw away.
OLDEST_SHARE = 0.3


@dataclass
class Message:
    """One message, note or logged touch."""

    id: str
    occurred_at: datetime
    direction: str
    kind: str
    body: str = ""
    subject: str | None = None


@dataclass
class Contact:
    """The contact fields the prompt is given."""

    id: str
    first_name: str
    last_name: str | None = None
    stage: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class ContactContext:
    """One contact's transcript, and everything needed to verify a citation."""

    contact_id: str
    transcript: str
    # label -> message id, for resolving citations back to real rows.
    labels: dict[str, str] = field(default_factory=dict)
    # label -> normalised searchable text, for verifying quotes.
    texts: dict[str, str] = field(default_factory=dict)
    # label -> the message's own date, so a fabricated occurred_on is correctable.
    dates: dict[str, str] = field(default_factory=dict)
    # label -> the kind, so a consumer can record whether a quote came from a
    # note or a conversation.
    kinds: dict[str, str] = field(default_factory=dict)
    message_count: int = 0
    truncated: bool = False

    @property
    def has_content(self) -> bool:
        return self.message_count > 0


def normalise(text: str | None) -> str:
    """
    Collapse whitespace and case, so matching a quote against its source is not
    defeated by a line wrap or a capitalisation difference.

    Deliberately does NOT touch punctuation. That omission is the subject of one
    of the pipeline's own measurements: a model that substitutes a typographic
    apostrophe for a straight one loses a real fact here, and the near-miss
    detector exists to quantify how often.
    """
    return " ".join((text or "").split()).lower()


def _clip(body: str, limit: int = MAX_BODY_CHARS) -> str:
    body = " ".join(body.split())
    if len(body) <= limit:
        return body
    return body[:limit].rsplit(" ", 1)[0] + " …"


def _render(rows: list[Message]) -> tuple[str, dict, dict, dict, dict]:
    lines: list[str] = []
    labels: dict[str, str] = {}
    texts: dict[str, str] = {}
    dates: dict[str, str] = {}
    kinds: dict[str, str] = {}

    for index, row in enumerate(rows, start=1):
        label = f"m{index}"
        occurred = row.occurred_at
        if occurred.tzinfo is None:
            occurred = occurred.replace(tzinfo=UTC)
        day = occurred.date().isoformat()

        who = {"inbound": "contact", "outbound": "us"}.get(row.direction or "", "unknown")
        parts = [p for p in (row.subject, row.body) if p]
        body = _clip(" — ".join(parts)) if parts else "(no text captured)"

        lines.append(f"[{label}] {day} · {who} · {row.kind}: {body}")
        labels[label] = row.id
        texts[label] = normalise(" ".join(parts))
        dates[label] = day
        kinds[label] = row.kind

    return "\n".join(lines), labels, texts, dates, kinds


def _tag_lines(contact: Contact) -> tuple[list[str], dict, dict, dict]:
    """
    The tags on the record, as quotable lines.

    Tags used to sit in the prompt as a header line, which made them visible but
    useless: verification checks every quote against `texts`, which holds only
    message bodies, so a fact citing the header was rejected as ungrounded. A tag
    like "past client" or "wants to sell 2027" carries real information and had
    no way to become any.

    One line per tag rather than one blob, so a fact cites the tag it used. No
    `dates` entry: a tag has no date and must not borrow the contact's. The
    caller deliberately does not count these towards `message_count` — see
    `build()`.
    """
    lines: list[str] = []
    labels: dict[str, str] = {}
    texts: dict[str, str] = {}
    kinds: dict[str, str] = {}

    for index, tag in enumerate(contact.tags or [], start=1):
        text = str(tag or "").strip()
        if not text:
            continue
        label = f"t{index}"
        lines.append(f"[{label}] tag: {text}")
        # No message behind a tag, so the label resolves to nothing and a
        # consumer records the source as a tag instead of a message id.
        labels[label] = ""
        texts[label] = normalise(text)
        kinds[label] = "tag"
    return lines, labels, texts, kinds


def build(
    contact: Contact,
    messages: list[Message],
    *,
    char_budget: int = DEFAULT_CHAR_BUDGET,
) -> ContactContext:
    """
    Assemble one contact's transcript, oldest first.

    Takes the messages rather than fetching them, so this is a pure function and
    testable without a database. In a real deployment the caller does the query.
    """
    rows = sorted(
        (
            m
            for m in messages
            if m.kind in TEXTUAL_KINDS and (m.subject or m.body)
        ),
        key=lambda m: (m.occurred_at, m.id),
    )

    if not rows:
        return ContactContext(contact_id=contact.id, transcript="", message_count=0)

    transcript, labels, texts, dates, kinds = _render(rows)
    truncated = False

    if len(transcript) > char_budget:
        truncated = True
        # Keep a head and a tail, drop the middle, and say so: the head holds
        # old commitments, the tail holds current state.
        head_budget = int(char_budget * OLDEST_SHARE)
        head: list[Message] = []
        used = 0
        for row in rows:
            entry = len(_clip(" — ".join(p for p in (row.subject, row.body) if p))) + 40
            if used + entry > head_budget:
                break
            head.append(row)
            used += entry

        tail: list[Message] = []
        used = 0
        for row in reversed(rows[len(head):]):
            entry = len(_clip(" — ".join(p for p in (row.subject, row.body) if p))) + 40
            if used + entry > char_budget - head_budget:
                break
            tail.append(row)
            used += entry
        tail.reverse()

        kept = head + tail
        dropped = len(rows) - len(kept)
        transcript, labels, texts, dates, kinds = _render(kept)
        if dropped > 0:
            marker = (
                f"\n… [{dropped} message(s) omitted from the middle of this history] …\n"
            )
            split = len(head)
            head_text = "\n".join(transcript.splitlines()[:split])
            tail_text = "\n".join(transcript.splitlines()[split:])
            transcript = head_text + marker + tail_text
        rows = kept

    # Tags ride along at the end, after truncation has settled so they are never
    # the thing that gets dropped. They are excluded from message_count on
    # purpose: `has_content` gates whether a contact is worth a model call at
    # all, and a bare tag list is not worth paying for. So tags enrich the
    # contacts already being read and never widen the spend across a book.
    lines, tag_labels, tag_texts, tag_kinds = _tag_lines(contact)
    if lines:
        transcript = "\n".join([transcript, *lines])
        labels = {**labels, **tag_labels}
        texts = {**texts, **tag_texts}
        kinds = {**kinds, **tag_kinds}

    return ContactContext(
        contact_id=contact.id,
        transcript=transcript,
        labels=labels,
        texts=texts,
        dates=dates,
        kinds=kinds,
        message_count=len(rows),
        truncated=truncated,
    )
