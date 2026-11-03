"""
Label agreement, and the annotator's own self-checks.

The agreement figure is what makes model-written labels defensible, so the cases
here are the ones that would make it lie: a `quotable` disagreement that must be
singled out rather than averaged away, cases present in only one file that must
be skipped rather than scored as total disagreement, and an annotator claiming a
quote it could not actually copy.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from evals.agreement import compare_labels, merge, render_markdown
from evals.annotate import AnnotatedFact, _to_gold
from evals.models import (
    ContactSeed,
    ConversationState,
    GoldCase,
    GoldFact,
    LabelMethod,
    LabelProvenance,
    MessageSeed,
)

QUOTE = "call me in three months about the rental"
OTHER = "her husband Mark is starting at UVA"


def prov(
    method: LabelMethod = "model_only", labeller: str = "annotator"
) -> LabelProvenance:
    return LabelProvenance(
        labeller=labeller,
        labelled_at=datetime(2026, 11, 2, tzinfo=UTC),
        guide_version="1.1",
        method=method,
    )


def case(
    case_id: str = "c1",
    facts: list[GoldFact] | None = None,
    state: ConversationState = "dormant",
    method: LabelMethod = "model_only",
) -> GoldCase:
    return GoldCase(
        case_id=case_id,
        stratum="typical",
        today=date(2026, 11, 2),
        contact=ContactSeed(first_name="Nadia", last_name="Okafor"),
        messages=[
            MessageSeed(day=date(2026, 3, 4), direction="inbound", kind="sms", body=QUOTE)
        ],
        gold_facts=facts if facts is not None else [],
        gold_state=state,
        intent="agreement fixture",
        provenance=prov(method),
    )


def other_party(cases: list[GoldCase]) -> list[GoldCase]:
    """
    The same cases, labelled by somebody else.

    `compare_labels` skips any pair whose two sides share a labeller, because
    that is a file compared against itself. Every comparison fixture therefore
    has to say who the second party is — which is the rule working, not an
    inconvenience.
    """
    return [
        c.model_copy(update={"provenance": prov(c.provenance.method, "human")})
        for c in cases
    ]


def fact(**kw: object) -> GoldFact:
    base: dict[str, object] = {
        "kind": "contact_request",
        "source_label": "m1",
        "quote": QUOTE,
    }
    base.update(kw)
    return GoldFact.model_validate(base)


# --------------------------------------------------------------------------
# Agreement
# --------------------------------------------------------------------------
def test_identical_labels_agree_completely() -> None:
    a = [case("c1", [fact()])]
    b = [case("c1", [fact()])]
    report = compare_labels(a, other_party(b))
    assert report.fact_agreement.value == 1.0
    assert report.quotable_agreement.value == 1.0
    assert not report.disagreements


def test_a_quotable_disagreement_is_caught_and_singled_out() -> None:
    # The headline metric is defined in terms of this flag, so disagreement here
    # must surface as its own number rather than average into the others.
    a = [case("c1", [fact(quotable=True)])]
    b = [case("c1", [fact(quotable=False)])]
    report = compare_labels(a, other_party(b))
    assert report.quotable_agreement.value == 0.0
    assert report.quotable_agreement.n == 1
    assert report.disagreements
    assert any("quotable" in c for c in report.disagreements[0].field_conflicts)


def test_quotable_is_compared_on_every_pair_even_when_both_agree() -> None:
    # It must never be skipped as "neither side has a view": both sides always
    # have a view, because the field has a default.
    report = compare_labels([case("c1", [fact()])], other_party([case("c1", [fact()])]))
    assert report.quotable_agreement.n == 1


def test_a_fact_only_one_side_found_is_counted_not_paired() -> None:
    a = [case("c1", [fact(), fact(kind="relationship", quote=QUOTE, person_name="Mark")])]
    b = [case("c1", [fact()])]
    report = compare_labels(a, other_party(b))
    assert report.matched_facts == 1
    assert report.only_in_a == 1
    assert report.only_in_b == 0
    assert report.fact_agreement.value is not None
    assert report.fact_agreement.value < 1.0


def test_cases_in_only_one_file_are_skipped_not_scored() -> None:
    # The blind subset is deliberately a slice. Scoring the other cases as
    # disagreements would make the figure meaningless.
    a = [case("c1", [fact()]), case("c2", [fact()]), case("c3", [fact()])]
    b = [case("c2", [fact()])]
    report = compare_labels(a, other_party(b))
    assert report.cases_compared == 1
    assert report.fact_agreement.value == 1.0


def test_a_case_labelled_by_the_same_party_on_both_sides_is_skipped() -> None:
    # `annotate --limit` writes the whole dataset with only the first N
    # annotated, so a naive comparison pairs most cases with themselves and
    # reads near 1.0 while measuring nothing. Independence is the thing being
    # measured, so it is enforced rather than left to the reader.
    a = [
        case("c1", [fact()], method="model_only"),
        case("c2", [fact()], method="model_only"),
    ]
    b = [
        case("c1", [fact(quotable=False)], method="model_only"),
        case("c2", [fact()], method="model_only"),
    ]
    # Same labeller on both sides of c2; different on c1.
    b[0] = b[0].model_copy(update={"provenance": prov("model_only", "annotator:sonnet")})

    report = compare_labels(a, b)
    assert report.cases_compared == 1
    assert report.cases_same_labeller == 1
    assert report.quotable_agreement.value == 0.0


def test_nothing_independently_labelled_yields_no_figure_at_all() -> None:
    a = [case("c1", [fact()])]
    b = [case("c1", [fact(quotable=False)])]  # same labeller as a
    report = compare_labels(a, b)
    assert report.cases_compared == 0
    assert report.cases_same_labeller == 1
    assert report.fact_agreement.value is None
    assert "Nothing independently labelled" in render_markdown(report)


def test_no_shared_cases_yields_no_figure_rather_than_a_perfect_one() -> None:
    report = compare_labels([case("c1", [fact()])], other_party([case("zzz", [fact()])]))
    assert report.cases_compared == 0
    assert report.fact_agreement.value is None
    assert "No cases in common" in render_markdown(report)


def test_a_state_disagreement_is_reported() -> None:
    a = [case("c1", [fact()], state="dormant")]
    b = [case("c1", [fact()], state="awaiting_my_reply")]
    report = compare_labels(a, other_party(b))
    assert report.state.value == 0.0
    assert report.disagreements[0].state == ("dormant", "awaiting_my_reply")


def test_a_date_disagreement_reports_its_offset() -> None:
    a = [case("c1", [fact(due_on=date(2026, 6, 4))])]
    b = [case("c1", [fact(due_on=date(2026, 6, 7))])]
    report = compare_labels(a, other_party(b))
    assert report.fields["due_on"].rate.value == 0.0
    assert any("+3d" in c for c in report.disagreements[0].field_conflicts)


def test_a_partial_name_is_credited_as_agreement() -> None:
    a = [case("c1", [fact(kind="relationship", person_name="Mark Chen")])]
    b = [case("c1", [fact(kind="relationship", person_name="Mark")])]
    report = compare_labels(a, other_party(b))
    assert report.fields["person_name"].rate.value == 1.0


def test_empty_label_sets_on_both_sides_agree_vacuously() -> None:
    report = compare_labels([case("c1", [])], other_party([case("c1", [])]))
    assert report.matched_facts == 0
    # Nothing to divide, so no figure rather than a perfect one.
    assert report.fact_agreement.value is None
    assert report.state.value == 1.0


def test_the_markdown_names_the_two_sides() -> None:
    report = compare_labels(
        [case("c1", [fact()])],
        other_party([case("c1", [fact()])]),
        label_a="annotated",
        label_b="blind",
    )
    rendered = render_markdown(report)
    assert "annotated" in rendered
    assert "blind" in rendered
    assert "quotable" in rendered


# --------------------------------------------------------------------------
# Merge
# --------------------------------------------------------------------------
def test_the_override_wins_where_it_has_a_case() -> None:
    primary = [case("c1", [fact()], method="model_only"), case("c2", [fact()])]
    override = [case("c1", [], method="blind_human")]
    merged = merge(primary, override)
    assert len(merged) == 2
    assert merged[0].provenance.method == "blind_human"
    assert merged[0].gold_facts == []


def test_a_case_the_override_lacks_keeps_its_own_label() -> None:
    primary = [case("c1", [fact()]), case("c2", [fact()])]
    merged = merge(primary, [case("c1", [], method="blind_human")])
    assert merged[1].case_id == "c2"
    assert merged[1].provenance.method == "model_only"
    assert len(merged[1].gold_facts) == 1


def test_merging_preserves_order_and_count() -> None:
    primary = [case(f"c{i}", [fact()]) for i in range(5)]
    merged = merge(primary, [case("c3", [], method="blind_human")])
    assert [c.case_id for c in merged] == [f"c{i}" for i in range(5)]


def test_an_override_case_not_in_primary_is_not_added() -> None:
    # Merge folds a subset in; it never grows the dataset, because a case the
    # reference does not contain has nothing to be scored against.
    merged = merge([case("c1", [fact()])], [case("zzz", [], method="blind_human")])
    assert [c.case_id for c in merged] == ["c1"]


# --------------------------------------------------------------------------
# The annotator's self-checks
# --------------------------------------------------------------------------
class _Ctx:
    """Just the `texts` map `_to_gold` reads."""

    def __init__(self, texts: dict[str, str]) -> None:
        self.texts = texts


def annotated(**kw: object) -> AnnotatedFact:
    base: dict[str, object] = {
        "kind": "contact_request",
        "source_label": "m1",
        "quote": QUOTE,
        "quotable": True,
        "essential": True,
    }
    base.update(kw)
    return AnnotatedFact.model_validate(base)


def test_a_verbatim_annotation_becomes_a_gold_fact() -> None:
    gold, unverbatim = _to_gold(annotated(), _Ctx({"m1": QUOTE}))
    assert gold is not None
    assert gold.quotable
    assert not unverbatim


def test_a_claimed_quote_the_annotator_could_not_copy_is_corrected() -> None:
    # It said quotable and the span is not in the message. Believe the strings,
    # not the claim — this is exactly the mistake that would corrupt the
    # headline metric.
    gold, unverbatim = _to_gold(
        annotated(quote="I will email the paperwork on Friday"), _Ctx({"m1": QUOTE})
    )
    assert gold is not None
    assert not gold.quotable
    assert unverbatim


def test_an_honestly_unquotable_annotation_is_not_counted_as_unverbatim() -> None:
    gold, unverbatim = _to_gold(
        annotated(quote="she is planning to move", quotable=False), _Ctx({"m1": QUOTE})
    )
    assert gold is not None
    assert not gold.quotable
    assert not unverbatim


def test_a_relation_without_name_is_dropped_rather_than_recorded() -> None:
    # The verifier would always discard it, so a gold label of that shape would
    # score as wrongful loss forever.
    gold, _ = _to_gold(
        annotated(kind="relationship", quote=OTHER), _Ctx({"m1": OTHER})
    )
    assert gold is None


def test_a_named_relation_survives() -> None:
    gold, _ = _to_gold(
        annotated(kind="relationship", quote=OTHER, person_name="Mark"),
        _Ctx({"m1": OTHER}),
    )
    assert gold is not None
    assert gold.person_name == "Mark"


def test_dates_are_parsed_and_a_bad_one_costs_only_the_date() -> None:
    gold, _ = _to_gold(annotated(due_on="2026-06-04"), _Ctx({"m1": QUOTE}))
    assert gold is not None and gold.due_on == date(2026, 6, 4)

    gold, _ = _to_gold(annotated(due_on="next spring"), _Ctx({"m1": QUOTE}))
    assert gold is not None
    assert gold.due_on is None


def test_a_bracketed_label_is_normalised() -> None:
    gold, _ = _to_gold(annotated(source_label="[m1]"), _Ctx({"m1": QUOTE}))
    assert gold is not None
    assert gold.source_label == "m1"


def test_a_blank_detail_value_becomes_none_rather_than_empty_string() -> None:
    gold, _ = _to_gold(
        annotated(kind="contact_detail", detail_field="city", detail_value="   "),
        _Ctx({"m1": QUOTE}),
    )
    assert gold is not None
    assert gold.detail_value is None


def test_a_quote_citing_an_unknown_label_cannot_be_quotable() -> None:
    gold, unverbatim = _to_gold(annotated(source_label="m9"), _Ctx({"m1": QUOTE}))
    assert gold is not None
    assert not gold.quotable
    assert unverbatim
