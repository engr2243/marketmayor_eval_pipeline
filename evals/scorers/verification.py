"""
The headline measurement: what the verifier throws away, and whether it was right to.

The application already guarantees that no surviving fact quotes something the
source does not contain — `validate()` drops any fact whose quote cannot be
found in the message it cites, and 28 unit tests cover that rule. Precision
against invented facts is therefore high by construction, and re-proving it
would be theatre.

The number nobody has is the other side of the same guard: **how many good
facts it destroys.** That is the direct analogue of a false positive in a
compliance check — a rule firing on something clean — and it splits into two
quantities that must never be added together:

- **Wrongful loss.** The model found a fact the gold says is present and
  quotable, and the verifier discarded it anyway.
- **Correct loss.** The model invented or misquoted something and the verifier
  caught it. The guard working.

Two further quantities need no labels at all, which matters because labelling
time is the scarcest input here:

- **Near-miss rejections.** A quote rejected as absent which is in fact present
  once smart quotes, dashes and ellipses are folded. The words were copied; the
  typography was not. This is wrongful loss provable from the strings alone.
- **Cap loss.** Facts that passed every verification step and were then dropped
  by the twelve-fact ceiling. The verifier certified them as grounded and the
  cap threw them away regardless.

Nothing in this module reimplements any part of the verifier. Reasons are
attributed by replaying each fact through the real `validate()` on its own,
which leaves exactly one entry in the rejection histogram.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from demo_app.verify import MAX_FACTS_PER_CONTACT, validate
from evals.metrics import Ratio
from evals.scorers.alignment import Alignment, align_facts
from evals.scorers.fields import QuoteCheck, QuotePresence, check_quote

# Rejection reasons that discard a whole fact, as against those that strip one
# field and keep the fact. Taken from the verifier's own behaviour: a reason in
# this set means the fact never reaches the user.
FATAL_REASONS = frozenset(
    {
        "source_not_in_transcript",
        "quote_not_in_source",
        "relation_without_name",
        "detail_incomplete",
        "detail_date_invalid",
    }
)

# Reasons that cost a field and keep the claim. `due_date_invalid` additionally
# demotes a commitment to a timeline, which is a real downgrade in what the
# product does with it, so it is reported rather than ignored.
FIELD_ONLY_REASONS = frozenset(
    {"due_date_invalid", "event_date_invalid", "transaction_off_target"}
)


@dataclass
class FactOutcome:
    """What happened to one fact the model returned."""

    index: int
    kind: str
    confidence: str
    source_label: str

    passed_verification: bool
    """Would survive `validate()` on its own, before the twelve-fact cap."""

    reason: str | None
    """The single reason it was discarded, or None if it survived."""

    quote: QuoteCheck
    """Where its quote actually sits in the message it cited."""

    gold_index: int | None = None
    gold_quotable: bool | None = None

    @property
    def fatal(self) -> bool:
        return self.reason is not None and self.reason in FATAL_REASONS

    @property
    def wrongful(self) -> bool:
        """
        Discarded, and the gold says it was a real, quotable fact.

        Requires a gold pairing: without one there is no basis for calling the
        loss wrongful, and guessing would be inventing a label.
        """
        return (
            not self.passed_verification
            and self.gold_index is not None
            and self.gold_quotable is True
        )

    @property
    def correct(self) -> bool:
        """Discarded, and either unpaired or paired to a fact gold calls unquotable."""
        return not self.passed_verification and not self.wrongful


@dataclass
class CapLoss:
    """Facts the verifier certified and the ceiling discarded anyway."""

    dropped: int = 0
    certified: int = 0
    """How many passed verification before the cap was applied."""
    by_kind_confidence: Counter[tuple[str, str]] = field(default_factory=Counter)

    @property
    def rate(self) -> Ratio:
        return Ratio(self.dropped, self.certified)


@dataclass
class VerificationReport:
    """One case's verification outcome."""

    outcomes: list[FactOutcome] = field(default_factory=list)
    kept: int = 0
    raw_count: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    cap: CapLoss = field(default_factory=CapLoss)
    alignment: Alignment = field(default_factory=Alignment)
    note_dropped: bool = False
    state_evidence_cleared: bool = False

    # ---------------- the headline pair ----------------
    @property
    def wrongful_loss(self) -> Ratio:
        """
        Of the facts the model found that gold says should have survived, how
        many the verifier destroyed.

        Denominator is deliberately narrow: facts the model actually produced
        *and* gold marks quotable. A fact the model never found is a recall
        problem, not a verifier problem, and folding the two together would
        blame the guard for the model's silence.
        """
        wrongful = sum(1 for o in self.outcomes if o.wrongful)
        eligible = sum(
            1 for o in self.outcomes if o.gold_index is not None and o.gold_quotable
        )
        return Ratio(wrongful, eligible)

    @property
    def correct_loss(self) -> Ratio:
        """Of everything discarded, how much deserved it."""
        discarded = [o for o in self.outcomes if not o.passed_verification]
        return Ratio(sum(1 for o in discarded if o.correct), len(discarded))

    # ---------------- the label-free pair ----------------
    @property
    def near_miss(self) -> Ratio:
        """
        Of the quotes rejected as not found, how many are present after folding
        punctuation or at high ordered overlap.

        Needs no gold set. This is a lower bound on wrongful loss provable from
        the strings alone.
        """
        rejected_quotes = [
            o for o in self.outcomes if o.reason == "quote_not_in_source"
        ]
        return Ratio(
            sum(1 for o in rejected_quotes if o.quote.is_near_miss),
            len(rejected_quotes),
        )

    @property
    def punctuation_only_loss(self) -> Ratio:
        """
        The sharpest version of the same number: rejected, but present verbatim
        once smart quotes and dashes are folded. Not a judgement at all.
        """
        rejected_quotes = [
            o for o in self.outcomes if o.reason == "quote_not_in_source"
        ]
        return Ratio(
            sum(
                1
                for o in rejected_quotes
                if o.quote.presence is QuotePresence.PUNCTUATION
            ),
            len(rejected_quotes),
        )

    @property
    def unsupported_surviving(self) -> Ratio:
        """
        Surviving facts whose quote is not in the message they cited.

        Expected to be zero by construction. Reported as a regression tripwire,
        never presented as an achievement.
        """
        survivors = [o for o in self.outcomes if o.passed_verification]
        return Ratio(
            sum(1 for o in survivors if not o.quote.survives_production),
            len(survivors),
        )


def _sole_reason(rejected: dict[str, int]) -> str | None:
    """
    The one reason a single-fact replay produced.

    More than one means the replay was not isolated — a bug here rather than in
    the application — so it is surfaced joined rather than silently truncated.
    """
    keys = [k for k in rejected if k != "note_ungrounded"]
    if not keys:
        return None
    if len(keys) == 1:
        return keys[0]
    return "+".join(sorted(keys))


def verify(
    analysis: Any,
    ctx: Any,
    gold: Sequence[Any] = (),
) -> VerificationReport:
    """
    Run the real verifier, then account for everything it discarded.

    `analysis` is a production `ContactAnalysis`, `ctx` a `ContactContext`, and
    `gold` a sequence of `GoldFact`. Passing `gold=()` still yields every
    label-free metric, which is what makes the headline cheap.
    """
    raw_facts = list(analysis.facts)
    whole = validate(analysis, ctx)

    report = VerificationReport(
        kept=len(whole.facts),
        raw_count=len(raw_facts),
        rejected=dict(whole.rejected),
        note_dropped="note_ungrounded" in whole.rejected,
        state_evidence_cleared=whole.state_evidence is None
        and bool(analysis.state_evidence),
    )

    report.alignment = align_facts(raw_facts, gold)

    certified: list[Any] = []

    for index, fact in enumerate(raw_facts):
        # Replay this fact alone. `notes` is cleared so the
        # whole-analysis-only `note_ungrounded` rule cannot contaminate the
        # reason, and the facts list holds exactly one entry so the twelve-fact
        # cap cannot fire. What comes back is this fact's own verdict.
        solo = validate(
            analysis.model_copy(update={"facts": [fact], "notes": None}),
            ctx,
        )
        reason = _sole_reason(solo.rejected)
        passed = bool(solo.facts)

        label = str(getattr(fact, "source_message_id", "") or "").strip().strip("[]")
        # An unknown label has no message behind it, so there is nothing to
        # check the quote against — that is itself the finding.
        source_text = ctx.texts.get(label, "")
        quote = check_quote(str(getattr(fact, "evidence", "") or ""), source_text)

        gold_index = report.alignment.gold_for(index)
        gold_quotable: bool | None = None
        if gold_index is not None:
            gold_quotable = bool(getattr(gold[gold_index], "quotable", True))

        report.outcomes.append(
            FactOutcome(
                index=index,
                kind=str(getattr(fact, "kind", "")),
                confidence=str(getattr(fact, "confidence", "")),
                source_label=label,
                passed_verification=passed,
                reason=reason,
                quote=quote,
                gold_index=gold_index,
                gold_quotable=gold_quotable,
            )
        )

        if passed:
            certified.append(solo.facts[0])

    report.cap = _cap_loss(certified, whole.facts)
    return report


def _cap_loss(certified: Sequence[Any], kept: Sequence[Any]) -> CapLoss:
    """
    What the twelve-fact ceiling discarded, by kind and confidence.

    Counted as a multiset difference rather than by matching objects: the
    verifier re-sorts when it trims and returns copies, so object identity is
    gone by the time we see the result. Comparing `(kind, confidence)` counts
    between the facts that passed verification individually and the facts that
    actually survived gives the exact number and the exact distribution without
    needing to know which specific rows were cut.

    Both sides are post-verification facts, so a commitment already demoted to a
    timeline is counted as a timeline on both sides and cannot create a phantom
    difference.
    """
    loss = CapLoss(certified=len(certified))
    if len(certified) <= len(kept):
        return loss

    def key(fact: Any) -> tuple[str, str]:
        return (str(getattr(fact, "kind", "")), str(getattr(fact, "confidence", "")))

    difference = Counter(key(f) for f in certified) - Counter(key(f) for f in kept)
    loss.by_kind_confidence = difference
    loss.dropped = sum(difference.values())
    return loss


def could_hit_cap(fact_count: int) -> bool:
    """Whether the ceiling is even in play for this many facts."""
    return fact_count > MAX_FACTS_PER_CONTACT
