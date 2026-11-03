"""
What a labelled case looks like.

The application's own enums are imported rather than restated. `FactKind`,
`ConversationState`, `DetailField` and the rest are the vocabulary the
application actually uses, and a parallel copy here would drift the first time
a kind is added — a silently unscored fact kind is worse than a loud import.

The gold shape is deliberately *not* the production `Fact`. It asserts less (no
`summary` prose, no `confidence`, which is a judgement nobody can label
consistently) and one thing more: `quotable`, which is the distinction the
headline metric rests on. A fact can be genuinely present and not
verbatim-quotable — implied across two messages, or paraphrased in the only line
that states it. Those facts *should* be discarded by the verifier, and counting
them as wrongful loss would flatter the number.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from demo_app.schemas import (
    ConversationState,
    DetailField,
    EngagementKind,
    FactKind,
    IntentState,
    RelationshipKind,
)

Strict = ConfigDict(extra="forbid")

Stratum = Literal["typical", "edge", "adversarial"]

# How a label came to exist. Reported alongside every metric derived from it,
# because "a model made these" and "a human made these blind" support very
# different claims.
LabelMethod = Literal["blind_human", "model_assisted_human", "model_only"]


class MessageSeed(BaseModel):
    """One message, note or logged touch in a case's history."""

    model_config = Strict

    day: date
    direction: Literal["inbound", "outbound"]
    # Mirrors demo_app.context.TEXTUAL_KINDS. A kind outside it contributes no
    # text, which is a legitimate case shape (it tests the empty path).
    kind: Literal["note", "sms", "email", "chat", "social", "other", "call", "voicemail"]
    body: str
    subject: str | None = None


class ContactSeed(BaseModel):
    """The contact fields the extraction prompt is given."""

    model_config = Strict

    first_name: str
    last_name: str | None = None
    stage: str | None = None
    tags: list[str] = Field(default_factory=list)


class GoldFact(BaseModel):
    """One fact a careful reader would take from the transcript."""

    model_config = Strict

    kind: FactKind
    # Which message states it, by the label the transcript renders ("m3", "t1").
    source_label: str
    # The span that supports it, as it appears in the source.
    quote: str

    # Asserted only where the gold has a view. None means "not asserted", which
    # is not the same as "asserted absent" — an unasserted field is skipped in
    # scoring rather than counted as a miss.
    due_on: date | None = None
    event_on: date | None = None
    person_name: str | None = None
    relationship_kind: RelationshipKind | None = None
    engagement_kind: EngagementKind | None = None
    detail_field: DetailField | None = None
    detail_value: str | None = None
    intent: IntentState | None = None

    # Whether a verbatim quote for this fact exists in one single message. False
    # means the verifier is right to drop it, and it is excluded from wrongful
    # loss. See the module docstring.
    quotable: bool = True

    # Whether losing this fact would cost the agent something real. Reported
    # separately; never used to excuse a miss.
    essential: bool = True


class LabelProvenance(BaseModel):
    """Who decided this case's labels, when, and under which guide."""

    model_config = Strict

    labeller: str
    labelled_at: datetime
    guide_version: str
    method: LabelMethod


class GoldCase(BaseModel):
    """One extraction test case: an input, and what should come out of it."""

    model_config = Strict

    case_id: str
    stratum: Stratum
    # Frozen per case. The prompt resolves relative dates against each message's
    # own date rather than this, but it is in the prompt and so has to be fixed
    # for a run to be reproducible.
    today: date

    contact: ContactSeed
    messages: list[MessageSeed]

    gold_facts: list[GoldFact] = Field(default_factory=list)
    gold_state: ConversationState
    gold_state_quotable: bool = True

    # What this case is for, in one line. Read by the labeller and quoted in
    # error-analysis tables.
    intent: str = ""
    provenance: LabelProvenance


class ToolCallSpec(BaseModel):
    """
    An expected tool call.

    Argument predicates rather than exact values: a correct call that phrased
    its arguments differently is still correct, and exact-matching a whole
    argument dict fails it. Each predicate is a named check registered in
    `scorers/tools.py`, so the dataset stays declarative.
    """

    model_config = Strict

    name: str
    arg_predicates: dict[str, str] = Field(default_factory=dict)


class ChatSeed(BaseModel):
    """
    Database state for one chat case: this tenant's rows, and the other's.

    The other tenant exists only to be absent from every answer.
    """

    model_config = Strict

    tenant_a: list[ContactSeed] = Field(default_factory=list)
    tenant_b: list[ContactSeed] = Field(default_factory=list)
    # Messages keyed by the first tenant's contact first name, so a case reads
    # as a scenario rather than as a foreign-key graph.
    messages: dict[str, list[MessageSeed]] = Field(default_factory=dict)


class ChatCase(BaseModel):
    """One chat test case."""

    model_config = Strict

    case_id: str
    stratum: Stratum
    today: date

    seed: ChatSeed
    prior_turns: list[dict] = Field(default_factory=list)
    message: str

    must_call: list[ToolCallSpec] = Field(default_factory=list)
    must_not_call: list[str] = Field(default_factory=list)
    # Tenant B's names and ids. Any appearance in a reply or a tool result fails
    # the whole run, not just the case.
    forbidden_strings: list[str] = Field(default_factory=list)

    expect_clarification: bool = False
    expect_decline: bool = False
    grounding: Literal["strict", "names_only", "skip"] = "strict"
    # Only where no deterministic check works. Left None, no judge runs.
    rubric: str | None = None

    intent: str = ""
    provenance: LabelProvenance
