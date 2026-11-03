"""
Pairing predictions with gold facts.

Getting this wrong is the quietest way to produce confident nonsense, so the
cases here are the ones that would hide a bug: a near miss that must not count
as a hit, a correct fact phrased differently that must not be penalised twice,
and two facts citing the same message that must stay apart.

Deliberately uses light stand-ins rather than the production `Fact` and the
the pipeline's own `GoldFact`: `align_facts` reads its inputs structurally, and keeping the
test free of both schemas is what proves that.
"""

from __future__ import annotations

from dataclasses import dataclass

from evals.scorers.alignment import MIN_QUOTE_OVERLAP, align_facts


@dataclass
class Pred:
    kind: str
    source_message_id: str
    evidence: str


@dataclass
class Gold:
    kind: str
    source_label: str
    quote: str


QUOTE = "call me in three months about the rental"
OTHER = "her birthday is 5 March"


def test_an_identical_fact_pairs() -> None:
    a = align_facts([Pred("contact_request", "m1", QUOTE)], [Gold("contact_request", "m1", QUOTE)])
    assert len(a.pairs) == 1
    assert a.pairs[0].overlap == 1.0
    assert not a.unmatched_predicted and not a.unmatched_gold


def test_a_clause_of_the_gold_quote_still_pairs() -> None:
    # The model quoting one clause of a sentence the gold quotes whole is the
    # same fact, and must not be scored as a false positive plus a miss.
    a = align_facts(
        [Pred("contact_request", "m1", "call me in three months")],
        [Gold("contact_request", "m1", QUOTE)],
    )
    assert len(a.pairs) == 1


def test_a_bracketed_label_resolves_the_same_as_a_bare_one() -> None:
    a = align_facts([Pred("commitment", "[m1]", QUOTE)], [Gold("commitment", "m1", QUOTE)])
    assert len(a.pairs) == 1


def test_a_different_kind_never_pairs() -> None:
    a = align_facts([Pred("commitment", "m1", QUOTE)], [Gold("contact_request", "m1", QUOTE)])
    assert not a.pairs
    assert a.unmatched_predicted == [0]
    assert a.unmatched_gold == [0]


def test_a_different_message_never_pairs() -> None:
    a = align_facts([Pred("commitment", "m2", QUOTE)], [Gold("commitment", "m1", QUOTE)])
    assert not a.pairs


def test_unrelated_quotes_on_the_same_message_stay_apart() -> None:
    # One message can legitimately state a commitment and a contact detail.
    # Same kind, same message, different claims — these are not the same fact.
    a = align_facts(
        [Pred("preference", "m1", OTHER)],
        [Gold("preference", "m1", QUOTE)],
    )
    assert not a.pairs
    assert a.unmatched_predicted == [0] and a.unmatched_gold == [0]


def test_pairing_is_one_to_one_and_takes_the_better_overlap() -> None:
    # Two predictions compete for one gold fact. The closer one wins; the other
    # is a false positive, not a second hit.
    a = align_facts(
        [
            Pred("commitment", "m1", "call me in three"),
            Pred("commitment", "m1", QUOTE),
        ],
        [Gold("commitment", "m1", QUOTE)],
    )
    assert len(a.pairs) == 1
    assert a.pairs[0].predicted_index == 1
    assert a.unmatched_predicted == [0]


def test_two_gold_facts_cannot_both_claim_one_prediction() -> None:
    a = align_facts(
        [Pred("commitment", "m1", QUOTE)],
        [Gold("commitment", "m1", QUOTE), Gold("commitment", "m1", QUOTE)],
    )
    assert len(a.pairs) == 1
    assert len(a.unmatched_gold) == 1


def test_lookups_resolve_in_both_directions() -> None:
    a = align_facts([Pred("commitment", "m1", QUOTE)], [Gold("commitment", "m1", QUOTE)])
    assert a.gold_for(0) == 0
    assert a.predicted_for(0) == 0
    assert a.gold_for(99) is None
    assert a.predicted_for(99) is None


def test_an_empty_prediction_list_is_all_misses() -> None:
    a = align_facts([], [Gold("commitment", "m1", QUOTE)])
    assert a.unmatched_gold == [0]
    assert not a.unmatched_predicted


def test_an_empty_gold_list_is_all_false_positives() -> None:
    a = align_facts([Pred("commitment", "m1", QUOTE)], [])
    assert a.unmatched_predicted == [0]
    assert not a.unmatched_gold


def test_an_empty_quote_cannot_pair_with_anything() -> None:
    a = align_facts([Pred("commitment", "m1", "")], [Gold("commitment", "m1", QUOTE)])
    assert not a.pairs


def test_the_overlap_floor_is_what_separates_a_match_from_a_coincidence() -> None:
    # Pinned so a future tweak to MIN_QUOTE_OVERLAP is a deliberate decision
    # rather than something that silently changes every recall figure.
    assert 0.0 < MIN_QUOTE_OVERLAP < 1.0
