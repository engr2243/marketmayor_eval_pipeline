"""
Scoring one extraction case.

Two alignments happen per case and the difference between them is the point:

- **Against the validated facts** — what the agent actually sees. This is where
  precision, recall and per-field accuracy come from.
- **Against the raw facts** — what the model produced before the verifier ran.
  This is where verification loss comes from, and it lives in
  `scorers/verification.py`.

Scoring precision and recall on the raw output would flatter the product (facts
the user never sees counted as found); scoring verification loss on the
validated output would be circular (nothing discarded is present to examine).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from evals.metrics import PRF, Ratio
from evals.scorers.alignment import align_facts
from evals.scorers.fields import (
    DateMatch,
    DateVerdict,
    NameMatch,
    NameVerdict,
    match_date,
    match_enum,
    match_free_text,
    match_name,
    match_price,
    parse_iso_date,
)
from evals.scorers.verification import VerificationReport

# Facts whose loss costs the agent a dated action rather than a nice-to-know.
# Recall on these carries a higher floor; see thresholds.yaml.
COSTLY_KINDS = frozenset({"commitment", "contact_request", "engagement"})

# Detail fields whose value is a date and must be compared as one.
_DATE_DETAILS = frozenset({"birthday", "home_anniversary"})
# Detail fields whose value is money written however the speaker wrote it.
_PRICE_DETAILS = frozenset({"price_range"})


@dataclass
class FieldTally:
    """Per-field agreement, with the shape of each disagreement kept."""

    correct: int = 0
    total: int = 0
    verdicts: Counter[str] = field(default_factory=Counter)
    # Signed day offsets for date fields, so a systematic shift is visible as a
    # shift rather than as scattered wrongness.
    offsets: list[int] = field(default_factory=list)

    def record(self, ok: bool, verdict: str, offset: int | None = None) -> None:
        self.total += 1
        self.correct += int(ok)
        self.verdicts[verdict] += 1
        if offset is not None:
            self.offsets.append(offset)

    @property
    def accuracy(self) -> Ratio:
        return Ratio(self.correct, self.total)


@dataclass
class CaseScore:
    """Everything measured about one extraction case."""

    case_id: str
    stratum: str

    schema_valid: bool = True
    refused: bool = False
    call_failed: bool = False
    error: str | None = None

    per_kind: dict[str, PRF] = field(default_factory=dict)
    fields: dict[str, FieldTally] = field(default_factory=dict)

    state_correct: bool | None = None
    # Facts produced where the gold asserts none at all. The abstention failure.
    spurious_on_silence: int = 0
    gold_is_silent: bool = False
    note_produced_without_gold: bool = False

    verification: VerificationReport | None = None

    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def overall(self) -> PRF:
        total = PRF()
        for prf in self.per_kind.values():
            total = total + prf
        return total

    @property
    def costly(self) -> PRF:
        total = PRF()
        for kind, prf in self.per_kind.items():
            if kind in COSTLY_KINDS:
                total = total + prf
        return total


def _field_name(detail_field: str | None) -> str:
    return f"detail:{detail_field}" if detail_field else "detail:?"


def score_case(
    *,
    case_id: str,
    stratum: str,
    validated_facts: Sequence[Any],
    gold_facts: Sequence[Any],
    validated_state: str | None,
    gold_state: str | None,
    notes: str | None,
    verification: VerificationReport | None = None,
) -> CaseScore:
    """
    Compare what the agent will see against what the gold says is there.

    `validated_facts` are production `Fact` objects after `validate()`.
    """
    score = CaseScore(case_id=case_id, stratum=stratum, verification=verification)
    score.gold_is_silent = len(gold_facts) == 0

    alignment = align_facts(validated_facts, gold_facts)

    per_kind: dict[str, PRF] = defaultdict(PRF)

    for pair in alignment.pairs:
        pred = validated_facts[pair.predicted_index]
        g = gold_facts[pair.gold_index]
        kind = str(getattr(pred, "kind", ""))
        per_kind[kind] = per_kind[kind] + PRF(true_positives=1)
        _score_fields(score, pred, g)

    for index in alignment.unmatched_predicted:
        kind = str(getattr(validated_facts[index], "kind", ""))
        per_kind[kind] = per_kind[kind] + PRF(false_positives=1)

    for index in alignment.unmatched_gold:
        kind = str(getattr(gold_facts[index], "kind", ""))
        per_kind[kind] = per_kind[kind] + PRF(false_negatives=1)

    score.per_kind = dict(per_kind)

    # Abstention. A transcript that states nothing of substance should produce
    # nothing, and `notes` is prose that cannot be checked against a
    # quote, so producing it on silence is its own failure.
    if score.gold_is_silent:
        score.spurious_on_silence = len(validated_facts)
        score.note_produced_without_gold = bool(notes)

    if gold_state is not None:
        score.state_correct = match_enum(gold_state, validated_state)

    return score


def _tally(score: CaseScore, name: str) -> FieldTally:
    return score.fields.setdefault(name, FieldTally())


def _score_fields(score: CaseScore, pred: Any, gold: Any) -> None:
    """
    Field-by-field agreement for one matched pair.

    A field the gold does not assert is skipped rather than scored. Counting an
    unasserted field as a pass would inflate every accuracy figure with fields
    nobody had a view on.
    """
    # --- enums ---
    for name in ("relationship_kind", "engagement_kind", "intent", "detail_field"):
        g = getattr(gold, name, None)
        if g is None:
            continue
        ok = match_enum(g, getattr(pred, name, None))
        _tally(score, name).record(ok, "exact" if ok else "mismatch")

    # --- dates ---
    for name in ("due_on", "event_on"):
        g = getattr(gold, name, None)
        p = getattr(pred, name, None)
        if g is None and p is None:
            continue
        match: DateMatch = match_date(g, p)
        _tally(score, name).record(
            match.correct,
            match.verdict.value,
            match.offset_days if match.verdict is DateVerdict.OFF_BY else None,
        )

    # --- names ---
    g_name = getattr(gold, "person_name", None)
    p_name = getattr(pred, "person_name", None)
    if g_name is not None or p_name is not None:
        name_match: NameMatch = match_name(g_name, p_name)
        _tally(score, "person_name").record(
            name_match.correct,
            name_match.verdict.value,
        )
        if name_match.verdict is NameVerdict.SUBSET:
            # Credited, but tracked so a run passing entirely on partial names
            # is visible in the report rather than buried.
            _tally(score, "person_name:subset").record(True, "subset")

    # --- detail values, typed by which detail it is ---
    g_field = getattr(gold, "detail_field", None)
    g_value = getattr(gold, "detail_value", None)
    p_value = getattr(pred, "detail_value", None)
    if g_value is not None or p_value is not None:
        tally = _tally(score, _field_name(g_field))
        if g_field in _DATE_DETAILS:
            match = match_date(parse_iso_date(g_value), p_value)
            tally.record(
                match.correct,
                match.verdict.value,
                match.offset_days if match.verdict is DateVerdict.OFF_BY else None,
            )
        elif g_field in _PRICE_DETAILS:
            ok = match_price(g_value, p_value)
            tally.record(ok, "digits_match" if ok else "digits_differ")
        else:
            ok = match_free_text(g_value, p_value)
            tally.record(ok, "contained" if ok else "mismatch")
