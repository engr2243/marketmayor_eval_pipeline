"""
The headline metric, against the real verifier.

These tests call the application's own `validate()` through `verify()`. That is
the point: the thing being measured is production behaviour, and a test using a
stand-in verifier would measure the pipeline agreeing with itself.

What has to hold:

- Per-fact reason attribution works via single-fact replay, for every reason.
- Wrongful loss counts only facts the gold says were present AND quotable.
- A fact gold marks unquotable is correct loss, not wrongful loss.
- Near-miss and cap loss are computed with no gold set at all.
"""

from __future__ import annotations

import pytest

from demo_app.context import ContactContext, normalise
from demo_app.schemas import ContactAnalysis, Fact
from demo_app.verify import MAX_FACTS_PER_CONTACT
from evals.models import GoldFact
from evals.scorers.fields import QuotePresence
from evals.scorers.verification import FATAL_REASONS, verify

M1 = "Thanks! Call me in three months about the rental."
M2 = "Her husband Mark is starting at UVA in the autumn."
M3 = "She said she’d definitely refer me to her sister."


@pytest.fixture
def ctx() -> ContactContext:
    """A context built directly — these tests need no transcript rendering."""
    return ContactContext(
        contact_id="c1",
        transcript="",
        labels={"m1": "i1", "m2": "i2", "m3": "i3", "t1": ""},
        texts={
            "m1": normalise(M1),
            "m2": normalise(M2),
            "m3": normalise(M3),
            "t1": normalise("past client"),
        },
        dates={"m1": "2026-03-04", "m2": "2026-04-02", "m3": "2026-05-10"},
        kinds={"m1": "sms", "m2": "note", "m3": "note", "t1": "tag"},
        message_count=3,
    )


def fact(**kw: object) -> Fact:
    base: dict[str, object] = {
        "kind": "preference",
        "summary": "s",
        "evidence": M1,
        "source_message_id": "m1",
        "occurred_on": "2026-03-04",
        "actor": "contact",
        "confidence": "high",
    }
    base.update(kw)
    return Fact.model_validate(base)


def analysis(*facts: Fact, notes: str | None = None) -> ContactAnalysis:
    return ContactAnalysis(
        conversation_state="dormant",
        state_evidence=M1,
        state_source_message_id="m1",
        facts=list(facts),
        notes=notes,
    )


def gold(**kw: object) -> GoldFact:
    base: dict[str, object] = {
        "kind": "preference",
        "source_label": "m1",
        "quote": M1,
        "quotable": True,
    }
    base.update(kw)
    return GoldFact.model_validate(base)


# --------------------------------------------------------------------------
# Reason attribution
# --------------------------------------------------------------------------
def test_a_verbatim_fact_survives_with_no_reason(ctx: ContactContext) -> None:
    report = verify(analysis(fact()), ctx)
    assert report.kept == 1
    assert report.outcomes[0].passed_verification
    assert report.outcomes[0].reason is None


def test_a_fabricated_quote_is_attributed_to_quote_not_in_source(ctx: ContactContext) -> None:
    report = verify(analysis(fact(evidence="I will email the paperwork on Friday.")), ctx)
    assert report.outcomes[0].reason == "quote_not_in_source"
    assert report.outcomes[0].fatal


def test_an_unknown_label_is_attributed_to_source_not_in_transcript(ctx: ContactContext) -> None:
    report = verify(analysis(fact(source_message_id="m9")), ctx)
    assert report.outcomes[0].reason == "source_not_in_transcript"


def test_a_relation_without_name_is_attributed_to_relation_without_name(ctx: ContactContext) -> None:
    report = verify(
        analysis(fact(kind="relationship", evidence=M2, source_message_id="m2")),
        ctx,
    )
    assert report.outcomes[0].reason == "relation_without_name"


def test_an_detail_incomplete_is_attributed_to_detail_incomplete(ctx: ContactContext) -> None:
    report = verify(analysis(fact(kind="contact_detail")), ctx)
    assert report.outcomes[0].reason == "detail_incomplete"


def test_a_due_date_before_its_message_costs_the_date_not_the_fact(ctx: ContactContext) -> None:
    # A field-only reason: the claim survives, demoted from commitment to
    # timeline. The fact still passes verification, which is why this reason is
    # not in FATAL_REASONS.
    report = verify(analysis(fact(kind="commitment", due_on="2026-01-01")), ctx)
    assert report.outcomes[0].reason == "due_date_invalid"
    assert report.outcomes[0].passed_verification
    assert "due_date_invalid" not in FATAL_REASONS


def test_every_reason_the_replay_reports_is_classified() -> None:
    # A reason the pipeline does not know about would be silently uncounted.
    assert FATAL_REASONS == {
        "source_not_in_transcript",
        "quote_not_in_source",
        "relation_without_name",
        "detail_incomplete",
        "detail_date_invalid",
    }


def test_the_replay_does_not_contaminate_a_reason_with_the_note_rule(
    ctx: ContactContext,
) -> None:
    # `note_ungrounded` is a whole-analysis rule. If the replay left the note in
    # place, every rejected fact would come back with two reasons.
    report = verify(
        analysis(fact(evidence="not in any message"), notes="Some prose."), ctx
    )
    assert report.outcomes[0].reason == "quote_not_in_source"
    assert report.note_dropped


# --------------------------------------------------------------------------
# Wrongful versus correct loss
# --------------------------------------------------------------------------
def test_a_discarded_quotable_gold_fact_is_wrongful_loss(ctx: ContactContext) -> None:
    # The model found the right fact and quoted it with a curly apostrophe the
    # source does not have. Gold says it is real and quotable, so the verifier
    # destroyed something good.
    predicted = fact(
        kind="endorsement",
        evidence="She said she'd definitely refer me to her sister.",
        source_message_id="m3",
    )
    g = gold(kind="endorsement", source_label="m3", quote=M3, quotable=True)
    report = verify(analysis(predicted), ctx, [g])

    outcome = report.outcomes[0]
    assert not outcome.passed_verification
    assert outcome.reason == "quote_not_in_source"
    assert outcome.gold_index == 0
    assert outcome.wrongful
    assert report.wrongful_loss.numerator == 1
    assert report.wrongful_loss.denominator == 1


def test_an_invented_fact_is_correct_loss_not_wrongful(ctx: ContactContext) -> None:
    report = verify(analysis(fact(evidence="I will email the paperwork on Friday.")), ctx, [])
    assert report.outcomes[0].correct
    assert not report.outcomes[0].wrongful
    assert report.correct_loss.numerator == 1


def test_a_fact_gold_calls_unquotable_is_correct_loss(ctx: ContactContext) -> None:
    # Genuinely present, but implied across two messages rather than stated in
    # one. The verifier is right to drop it and counting it as wrongful loss
    # would flatter the number.
    predicted = fact(kind="intent", evidence="she is planning to sell", source_message_id="m1")
    g = gold(kind="intent", source_label="m1", quote="she is planning to sell", quotable=False)
    report = verify(analysis(predicted), ctx, [g])

    outcome = report.outcomes[0]
    assert outcome.gold_index == 0
    assert not outcome.wrongful
    assert outcome.correct
    # Excluded from the denominator too: it was never eligible to survive.
    assert report.wrongful_loss.denominator == 0
    assert report.wrongful_loss.value is None


def test_wrongful_loss_ignores_facts_the_model_never_found(ctx: ContactContext) -> None:
    # Two gold facts; the model produced only the first. The second is a recall
    # problem, not a verifier problem, and folding the two together would blame
    # the guard for the model's silence. So the denominator is one, not two.
    found = gold(quote=M1, quotable=True)
    missed = gold(kind="commitment", source_label="m2", quote=M2, quotable=True)
    report = verify(analysis(fact()), ctx, [found, missed])

    assert report.outcomes[0].gold_index == 0
    assert report.alignment.unmatched_gold == [1]
    assert report.wrongful_loss.denominator == 1
    assert report.wrongful_loss.numerator == 0


# --------------------------------------------------------------------------
# The label-free metrics
# --------------------------------------------------------------------------
def test_near_miss_needs_no_gold_set(ctx: ContactContext) -> None:
    predicted = fact(
        kind="endorsement",
        evidence="She said she'd definitely refer me to her sister.",
        source_message_id="m3",
    )
    report = verify(analysis(predicted), ctx)  # no gold at all
    assert report.near_miss.numerator == 1
    assert report.punctuation_only_loss.numerator == 1
    assert report.outcomes[0].quote.presence is QuotePresence.PUNCTUATION


def test_a_genuine_fabrication_is_not_counted_as_a_near_miss(ctx: ContactContext) -> None:
    report = verify(analysis(fact(evidence="I will email the paperwork on Friday.")), ctx)
    assert report.near_miss.numerator == 0
    assert report.near_miss.denominator == 1


def test_unsupported_surviving_is_zero_by_construction(ctx: ContactContext) -> None:
    report = verify(analysis(fact(), fact(evidence=M2, source_message_id="m2")), ctx)
    assert report.unsupported_surviving.numerator == 0
    assert report.unsupported_surviving.denominator == 2


# --------------------------------------------------------------------------
# Cap loss
# --------------------------------------------------------------------------
def test_nothing_is_capped_below_the_ceiling(ctx: ContactContext) -> None:
    report = verify(analysis(*[fact() for _ in range(MAX_FACTS_PER_CONTACT)]), ctx)
    assert report.cap.dropped == 0
    assert report.cap.certified == MAX_FACTS_PER_CONTACT


def test_facts_over_the_ceiling_are_certified_then_discarded(ctx: ContactContext) -> None:
    # Every one of these passes verification on its own. The cap throws the
    # excess away regardless, which is recall loss with no model error behind it.
    over = MAX_FACTS_PER_CONTACT + 3
    report = verify(analysis(*[fact() for _ in range(over)]), ctx)
    assert report.cap.certified == over
    assert report.cap.dropped == 3
    assert report.kept == MAX_FACTS_PER_CONTACT
    assert "fact_cap" in report.rejected


def test_cap_loss_reports_what_kind_and_confidence_was_dropped(ctx: ContactContext) -> None:
    # The ceiling sorts by confidence then recency, so a low-confidence fact
    # loses to a high-confidence one. That is the finding, so the distribution
    # has to come back and not just a count.
    facts = [fact(confidence="high") for _ in range(MAX_FACTS_PER_CONTACT)]
    facts += [fact(kind="commitment", confidence="low", due_on="2026-06-04")]
    report = verify(analysis(*facts), ctx)
    assert report.cap.dropped == 1
    assert report.cap.by_kind_confidence[("commitment", "low")] == 1


def test_an_invented_fact_never_reaches_the_cap_accounting(ctx: ContactContext) -> None:
    facts = [fact() for _ in range(MAX_FACTS_PER_CONTACT)]
    facts += [fact(evidence="I will email the paperwork on Friday.")]
    report = verify(analysis(*facts), ctx)
    assert report.cap.certified == MAX_FACTS_PER_CONTACT
    assert report.cap.dropped == 0


# --------------------------------------------------------------------------
# Whole-analysis outcomes
# --------------------------------------------------------------------------
def test_an_unverifiable_state_quote_is_recorded_as_cleared(ctx: ContactContext) -> None:
    bad_state = ContactAnalysis(
        conversation_state="dormant",
        state_evidence="nothing anybody wrote",
        state_source_message_id="m1",
        facts=[fact()],
    )
    assert verify(bad_state, ctx).state_evidence_cleared


def test_an_empty_analysis_reports_zeroes_rather_than_dividing_by_them(
    ctx: ContactContext,
) -> None:
    report = verify(analysis(), ctx)
    assert report.raw_count == 0
    assert report.kept == 0
    assert report.wrongful_loss.value is None
    assert report.correct_loss.value is None
    assert report.near_miss.value is None
    assert report.cap.rate.value is None
