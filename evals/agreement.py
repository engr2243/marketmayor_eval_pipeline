"""
How far two sets of labels agree, and merging them into one reference.

This is the number that makes auto-labelling defensible. "A stronger model wrote
these labels and a human checked fifteen of them blind, agreeing on 0.91 of
fields (n=47)" is a real answer to "how were the labels created". "A model wrote
them" is not, and an interviewer will go straight at it.

So the comparison is symmetric and reported per field. It deliberately does not
treat either side as correct: a disagreement is a disagreement, and which side
is right is a judgement for whoever reads it. What it does single out is
**`quotable` agreement**, because the headline metric is defined in terms of
that flag — if the annotator and the human disagree about what is quotable, the
headline number is softer than it looks, whoever is right.

Merging is separate and opinionated in one direction only: a human label beats a
model label for the same case. Nothing else is inferred.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from evals.datasets import Dataset
from evals.metrics import Ratio
from evals.models import GoldCase, GoldFact
from evals.scorers.alignment import align_facts
from evals.scorers.fields import (
    match_date,
    match_enum,
    match_free_text,
    match_name,
    match_price,
    parse_iso_date,
)

# Fields compared on a matched pair, and how. Mirrors scorers/extraction.py so a
# field agreement figure and a model accuracy figure mean the same thing.
_DATE_DETAILS = frozenset({"birthday", "home_anniversary"})
_PRICE_DETAILS = frozenset({"price_range"})


@dataclass
class FieldAgreement:
    agreed: int = 0
    total: int = 0
    shapes: Counter[str] = field(default_factory=Counter)

    def record(self, agreed: bool, shape: str) -> None:
        self.total += 1
        self.agreed += int(agreed)
        self.shapes[shape] += 1

    @property
    def rate(self) -> Ratio:
        return Ratio(self.agreed, self.total)


@dataclass
class CaseDisagreement:
    """One case where the two label sets differ, for eyeballing."""

    case_id: str
    only_in_a: list[str] = field(default_factory=list)
    only_in_b: list[str] = field(default_factory=list)
    field_conflicts: list[str] = field(default_factory=list)
    state: tuple[str, str] | None = None


@dataclass
class AgreementReport:
    label_a: str = ""
    label_b: str = ""
    cases_compared: int = 0
    cases_same_labeller: int = 0
    """
    Shared case ids whose two labels came from the SAME labeller, and so were
    skipped. Comparing a label against itself inflates every figure towards 1.0
    and measures nothing.
    """

    matched_facts: int = 0
    only_in_a: int = 0
    only_in_b: int = 0

    fields: dict[str, FieldAgreement] = field(default_factory=dict)
    state: Ratio = field(default_factory=Ratio)
    disagreements: list[CaseDisagreement] = field(default_factory=list)

    @property
    def fact_agreement(self) -> Ratio:
        """
        Dice coefficient over facts: twice the matched, over both totals.

        Symmetric on purpose. Precision and recall would need one side to be
        the truth, and neither of these is.
        """
        total = (self.matched_facts + self.only_in_a) + (
            self.matched_facts + self.only_in_b
        )
        return Ratio(2 * self.matched_facts, total)

    @property
    def quotable_agreement(self) -> Ratio:
        """The one the headline metric depends on."""
        tally = self.fields.get("quotable")
        return tally.rate if tally else Ratio()

    @property
    def overall_field_agreement(self) -> Ratio:
        total = Ratio()
        for tally in self.fields.values():
            total = total + tally.rate
        return total


def _tally(report: AgreementReport, name: str) -> FieldAgreement:
    return report.fields.setdefault(name, FieldAgreement())


def _compare_pair(
    report: AgreementReport,
    case_id: str,
    a: GoldFact,
    b: GoldFact,
    disagreement: CaseDisagreement,
) -> None:
    """Field-by-field comparison of two labels for the same fact."""
    # `quotable` is compared on every pair, always, because the headline metric
    # is defined in terms of it. Everything else is compared only where at least
    # one side has a view.
    agreed = a.quotable == b.quotable
    _tally(report, "quotable").record(agreed, "same" if agreed else "differs")
    if not agreed:
        disagreement.field_conflicts.append(
            f"{a.kind}[{a.source_label}] quotable: {a.quotable} vs {b.quotable}"
        )

    agreed = a.essential == b.essential
    _tally(report, "essential").record(agreed, "same" if agreed else "differs")

    for name in ("relationship_kind", "engagement_kind", "intent", "detail_field"):
        va, vb = getattr(a, name), getattr(b, name)
        if va is None and vb is None:
            continue
        agreed = match_enum(va, vb)
        _tally(report, name).record(agreed, "same" if agreed else "differs")
        if not agreed:
            disagreement.field_conflicts.append(f"{a.kind} {name}: {va} vs {vb}")

    for name in ("due_on", "event_on"):
        va, vb = getattr(a, name), getattr(b, name)
        if va is None and vb is None:
            continue
        match = match_date(va, vb)
        _tally(report, name).record(match.correct, match.verdict.value)
        if not match.correct:
            disagreement.field_conflicts.append(
                f"{a.kind} {name}: {va} vs {vb}"
                + (f" ({match.offset_days:+d}d)" if match.offset_days else "")
            )

    if a.person_name is not None or b.person_name is not None:
        name_match = match_name(a.person_name, b.person_name)
        _tally(report, "person_name").record(name_match.correct, name_match.verdict.value)
        if not name_match.correct:
            disagreement.field_conflicts.append(
                f"{a.kind} person_name: {a.person_name!r} vs {b.person_name!r}"
            )

    if a.detail_value is not None or b.detail_value is not None:
        name = f"detail:{a.detail_field or b.detail_field or '?'}"
        if (a.detail_field or b.detail_field) in _DATE_DETAILS:
            match_result = match_date(parse_iso_date(a.detail_value), b.detail_value)
            agreed = match_result.correct
            shape = match_result.verdict.value
        elif (a.detail_field or b.detail_field) in _PRICE_DETAILS:
            agreed = match_price(a.detail_value, b.detail_value)
            shape = "digits_match" if agreed else "digits_differ"
        else:
            agreed = match_free_text(a.detail_value, b.detail_value)
            shape = "contained" if agreed else "differs"
        _tally(report, name).record(agreed, shape)
        if not agreed:
            disagreement.field_conflicts.append(
                f"{a.kind} detail_value: {a.detail_value!r} vs {b.detail_value!r}"
            )


def _describe(fact: GoldFact) -> str:
    quote = fact.quote if len(fact.quote) <= 48 else fact.quote[:45] + "..."
    return f"{fact.kind}[{fact.source_label}] {quote!r}"


def compare_labels(
    a: Sequence[GoldCase],
    b: Sequence[GoldCase],
    *,
    label_a: str = "A",
    label_b: str = "B",
) -> AgreementReport:
    """
    Compare two label sets over the case ids they share.

    Cases present in only one side are skipped rather than counted as total
    disagreement — the blind subset is deliberately a slice, and scoring the
    other fifty cases as disagreements would make the figure meaningless.

    Cases whose two labels came from the same labeller are also skipped, and
    that is not a nicety. `annotate --limit` writes the whole dataset with only
    the first N annotated, so comparing it against the authored labels pairs
    sixty-four cases with themselves. Agreement then reads 0.98 and means
    nothing. Independence is the thing being measured, so it is enforced here
    rather than left to whoever reads the number.
    """
    report = AgreementReport(label_a=label_a, label_b=label_b)
    by_id_a = {c.case_id: c for c in a}
    by_id_b = {c.case_id: c for c in b}
    shared = sorted(set(by_id_a) & set(by_id_b))
    report.cases_compared = len(shared)

    for case_id in shared:
        case_a, case_b = by_id_a[case_id], by_id_b[case_id]

        if case_a.provenance.labeller == case_b.provenance.labeller:
            report.cases_same_labeller += 1
            report.cases_compared -= 1
            continue

        disagreement = CaseDisagreement(case_id=case_id)

        alignment = align_facts(_as_predictions(case_a.gold_facts), case_b.gold_facts)
        report.matched_facts += len(alignment.pairs)
        report.only_in_a += len(alignment.unmatched_predicted)
        report.only_in_b += len(alignment.unmatched_gold)

        for pair in alignment.pairs:
            _compare_pair(
                report,
                case_id,
                case_a.gold_facts[pair.predicted_index],
                case_b.gold_facts[pair.gold_index],
                disagreement,
            )

        disagreement.only_in_a = [
            _describe(case_a.gold_facts[i]) for i in alignment.unmatched_predicted
        ]
        disagreement.only_in_b = [
            _describe(case_b.gold_facts[i]) for i in alignment.unmatched_gold
        ]

        states_agree = case_a.gold_state == case_b.gold_state
        report.state = report.state + Ratio(int(states_agree), 1)
        if not states_agree:
            disagreement.state = (case_a.gold_state, case_b.gold_state)

        if (
            disagreement.only_in_a
            or disagreement.only_in_b
            or disagreement.field_conflicts
            or disagreement.state
        ):
            report.disagreements.append(disagreement)

    return report


class _AsPrediction:
    """
    A `GoldFact` wearing the attribute names `align_facts` expects.

    The aligner reads `source_message_id` and `evidence` because that is
    what a production `Fact` calls them. Adapting here keeps the aligner a
    single implementation used by both the model scoring and this comparison —
    two label sets must be paired the same way predictions are, or an agreement
    figure and an accuracy figure would not be about the same thing.
    """

    __slots__ = ("evidence", "kind", "source_message_id")

    def __init__(self, fact: GoldFact) -> None:
        self.kind = fact.kind
        self.source_message_id = fact.source_label
        self.evidence = fact.quote


def _as_predictions(facts: Sequence[GoldFact]) -> list[_AsPrediction]:
    return [_AsPrediction(f) for f in facts]


def merge(primary: Sequence[GoldCase], override: Sequence[GoldCase]) -> list[GoldCase]:
    """
    One reference dataset: `override`'s labels win where it has them.

    Used to fold a human-labelled subset into a model-annotated dataset. The
    result carries mixed `provenance.method` values, which is what lets the
    report split the blind subset out and show it separately.

    Opinionated in exactly one direction and no further: a case absent from
    `override` keeps its primary label untouched, and nothing is inferred,
    averaged or reconciled.
    """
    by_id = {c.case_id: c for c in override}
    return [by_id.get(case.case_id, case) for case in primary]


def render_markdown(report: AgreementReport, *, max_cases: int = 15) -> str:
    """The agreement figure, as it will be quoted in the results doc."""
    lines: list[str] = []
    lines.append(f"# Label agreement: {report.label_a} vs {report.label_b}")
    lines.append("")

    if report.cases_compared <= 0:
        # Two different nothings, and the difference tells you what to fix.
        if report.cases_same_labeller:
            lines.append(
                f"**Nothing independently labelled.** All "
                f"{report.cases_same_labeller} shared case(s) carry the same "
                "labeller on both sides, so every pair would be a label compared "
                "against itself. No agreement figure can be computed from this."
            )
        else:
            lines.append(
                "**No cases in common.** Nothing to compare — the two files share "
                "no case ids, so no agreement figure can be computed."
            )
        return "\n".join(lines)

    lines.append(
        f"{report.cases_compared} independently labelled case(s) compared. Cases "
        "present in only one file are skipped, not counted as disagreement."
    )
    if report.cases_same_labeller:
        lines.append("")
        lines.append(
            f"**{report.cases_same_labeller} shared case(s) skipped** because both "
            "sides carry the same labeller, so comparing them would be a file "
            "against itself. That is the normal result of annotating with "
            "`--limit`, which writes the whole dataset with only the first N "
            "annotated."
        )
    lines.append("")

    lines.append("## Headline")
    lines.append("")
    lines.append("| Measure | Value |")
    lines.append("|---|---|")
    lines.append(f"| Facts agreed (Dice) | {report.fact_agreement.render()} |")
    lines.append(
        f"| **`quotable` agreement** | **{report.quotable_agreement.render()}** |"
    )
    lines.append(f"| All fields pooled | {report.overall_field_agreement.render()} |")
    lines.append(f"| Conversation state | {report.state.render()} |")
    lines.append(f"| Facts only in {report.label_a} | {report.only_in_a} |")
    lines.append(f"| Facts only in {report.label_b} | {report.only_in_b} |")
    lines.append("")
    lines.append(
        "`quotable` is singled out because the headline metric — how many "
        "quotable facts the verifier wrongly discards — is defined in terms of "
        "it. Disagreement here makes that number softer than it looks, "
        "whichever side turns out to be right."
    )
    lines.append("")

    if report.fields:
        lines.append("## Per field")
        lines.append("")
        lines.append("| Field | Agreement | Disagreement shapes |")
        lines.append("|---|---|---|")
        for name in sorted(report.fields):
            tally = report.fields[name]
            shapes = ", ".join(
                f"{k}×{v}"
                for k, v in sorted(tally.shapes.items(), key=lambda kv: -kv[1])
                if k not in ("same", "exact", "not_asserted")
            )
            lines.append(f"| `{name}` | {tally.rate.render()} | {shapes or '—'} |")
        lines.append("")

    if report.disagreements:
        lines.append("## Where they differ")
        lines.append("")
        for disagreement in report.disagreements[:max_cases]:
            lines.append(f"### `{disagreement.case_id}`")
            lines.append("")
            if disagreement.state:
                lines.append(
                    f"- state: `{disagreement.state[0]}` vs `{disagreement.state[1]}`"
                )
            for item in disagreement.only_in_a:
                lines.append(f"- only in {report.label_a}: {item}")
            for item in disagreement.only_in_b:
                lines.append(f"- only in {report.label_b}: {item}")
            for item in disagreement.field_conflicts:
                lines.append(f"- {item}")
            lines.append("")
        remaining = len(report.disagreements) - max_cases
        if remaining > 0:
            lines.append(f"...and {remaining} more case(s) with differences.")
            lines.append("")

    lines.append("## How to read this")
    lines.append("")
    lines.append(
        "This is not a score for either side. It is how much the reference "
        "labels can be trusted, and every metric derived from them is reported "
        "with this figure attached. If agreement is poor, the affected metrics "
        "are reported as unreliable rather than quietly used."
    )
    lines.append("")

    return "\n".join(lines)


def load_cases(dataset: Dataset) -> list[GoldCase]:
    return [c for c in dataset.cases if isinstance(c, GoldCase)]
