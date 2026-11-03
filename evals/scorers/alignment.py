"""
Pairing what the model said with what the gold says.

Precision and recall need to know which predicted fact corresponds to which
gold fact, and the model does not number them. Getting this wrong is the
quietest way to produce confident nonsense: align too loosely and every near
miss counts as a hit, align too strictly and a correct fact phrased differently
counts as both a false positive and a miss, which is a double penalty for one
disagreement.

The rule: a prediction can only pair with a gold fact of the same kind, citing
the same message, whose quotes overlap. Within that, the best overlap wins, and
pairing is one to one. Everything unpaired is a real error on one side or the
other.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from evals.scorers.fields import normalise, ordered_overlap

# How much of the shorter quote must sit inside the longer one before two facts
# are talking about the same thing. Below this they are different claims that
# happen to cite the same message, which does occur — a single message can
# state a commitment and a contact detail.
MIN_QUOTE_OVERLAP = 0.60


@dataclass
class Pair:
    """One prediction matched to one gold fact."""

    predicted_index: int
    gold_index: int
    overlap: float


@dataclass
class Alignment:
    pairs: list[Pair] = field(default_factory=list)
    unmatched_predicted: list[int] = field(default_factory=list)
    """Predicted facts with no gold counterpart. False positives."""
    unmatched_gold: list[int] = field(default_factory=list)
    """Gold facts the model never produced. Misses."""

    def gold_for(self, predicted_index: int) -> int | None:
        for pair in self.pairs:
            if pair.predicted_index == predicted_index:
                return pair.gold_index
        return None

    def predicted_for(self, gold_index: int) -> int | None:
        for pair in self.pairs:
            if pair.gold_index == gold_index:
                return pair.predicted_index
        return None


def _quote_overlap(a: str, b: str) -> float:
    """
    Symmetric-ish overlap: the share of the shorter quote inside the longer.

    A model quoting one clause of a sentence the gold quotes whole is the same
    fact. Taking the shorter side as the needle means that reads as a strong
    match rather than a weak one.
    """
    na, nb = normalise(a), normalise(b)
    if not na or not nb:
        return 0.0
    needle, haystack = (na, nb) if len(na) <= len(nb) else (nb, na)
    return ordered_overlap(needle, haystack)


def align_facts(
    predicted: Sequence[Any],
    gold: Sequence[Any],
) -> Alignment:
    """
    Greedy one-to-one pairing, strongest overlap first.

    `predicted` holds production `Fact` objects (`kind`, `source_message_id`,
    `evidence`); `gold` holds `GoldFact` objects (`kind`, `source_label`,
    `quote`). Taking them structurally rather than by type keeps this module
    free of both schemas and therefore unit-testable without the application.

    Greedy rather than optimal: a true assignment problem would need a
    Hungarian solve, and with at most a dozen facts per contact and a hard
    same-kind, same-message constraint the two agree in practice. Worth
    revisiting only if a case appears where one prediction plausibly pairs with
    two different gold facts of the same kind on the same message.
    """
    # (overlap, length gap, predicted index, gold index)
    candidates: list[tuple[float, int, int, int]] = []

    for p_index, pred in enumerate(predicted):
        p_label = str(getattr(pred, "source_message_id", "") or "").strip().strip("[]")
        p_kind = getattr(pred, "kind", None)
        p_quote = str(getattr(pred, "evidence", "") or "")

        for g_index, g in enumerate(gold):
            if p_kind != getattr(g, "kind", None):
                continue
            g_label = str(getattr(g, "source_label", "") or "").strip().strip("[]")
            if p_label != g_label:
                continue
            g_quote = str(getattr(g, "quote", "") or "")
            overlap = _quote_overlap(p_quote, g_quote)
            if overlap >= MIN_QUOTE_OVERLAP:
                # Overlap is measured shorter-inside-longer, so a three-word
                # fragment of the gold quote scores the same 1.0 as the whole
                # sentence. When two predictions tie, the one closest in length
                # to the gold is the better pairing — otherwise a fragment can
                # win on nothing but its position in the list.
                length_gap = abs(len(normalise(p_quote)) - len(normalise(g_quote)))
                candidates.append((overlap, length_gap, p_index, g_index))

    # Strongest overlap first, then closest in length, then position so the
    # result is deterministic and a re-run of the same data pairs identically.
    candidates.sort(key=lambda c: (-c[0], c[1], c[2], c[3]))

    used_predicted: set[int] = set()
    used_gold: set[int] = set()
    pairs: list[Pair] = []

    for overlap, _gap, p_index, g_index in candidates:
        if p_index in used_predicted or g_index in used_gold:
            continue
        used_predicted.add(p_index)
        used_gold.add(g_index)
        pairs.append(Pair(p_index, g_index, overlap))

    pairs.sort(key=lambda p: p.predicted_index)
    return Alignment(
        pairs=pairs,
        unmatched_predicted=[i for i in range(len(predicted)) if i not in used_predicted],
        unmatched_gold=[i for i in range(len(gold)) if i not in used_gold],
    )
