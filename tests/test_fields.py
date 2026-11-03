"""
The matching rules, including the near-miss taxonomy the headline rests on.

The four `QuotePresence` levels are the most load-bearing thing in the pipeline:
`PUNCTUATION` is the claim that a rejection was typography and nothing else, and
that claim is what argues the verifier is wrong. So each level is pinned here by
example, including the cases that must *not* reach it.
"""

from __future__ import annotations

from datetime import date

from evals.scorers.fields import (
    DateVerdict,
    NameVerdict,
    QuotePresence,
    check_quote,
    digit_sequence,
    fold_punctuation,
    match_date,
    match_free_text,
    match_name,
    match_price,
    ordered_overlap,
    parse_iso_date,
)

SOURCE = "call me in three months about the rental, and say hi to mark."


# --------------------------------------------------------------------------
# Punctuation folding
# --------------------------------------------------------------------------
def test_curly_quotes_fold_to_straight() -> None:
    assert fold_punctuation("‘a’ “b”") == "'a' \"b\""


def test_dashes_fold_to_hyphen() -> None:
    assert fold_punctuation("a–b—c−d") == "a-b-c-d"


def test_non_breaking_space_folds_to_a_real_one() -> None:
    assert fold_punctuation("a b") == "a b"


def test_zero_width_characters_disappear() -> None:
    assert fold_punctuation("a​b﻿") == "ab"


# --------------------------------------------------------------------------
# Quote presence — the taxonomy
# --------------------------------------------------------------------------
def test_a_verbatim_quote_is_exact() -> None:
    check = check_quote("Call me in three months about the rental", SOURCE)
    assert check.presence is QuotePresence.EXACT
    assert check.survives_production
    assert not check.is_near_miss


def test_case_and_whitespace_are_already_handled_by_production() -> None:
    # `normalise` lowercases and collapses runs of whitespace, so these are
    # exact matches rather than near misses. Classifying them as near misses
    # would inflate the headline with differences the verifier never minded.
    assert check_quote("CALL ME   in Three Months", SOURCE).presence is QuotePresence.EXACT


def test_a_curly_apostrophe_is_punctuation_only() -> None:
    source = "she said she’d definitely refer me"
    check = check_quote("she said she'd definitely refer me", source)
    assert check.presence is QuotePresence.PUNCTUATION
    assert check.is_near_miss
    assert not check.survives_production


def test_an_em_dash_for_a_hyphen_is_punctuation_only() -> None:
    source = "the budget is 450—550k for a first move"
    assert check_quote("the budget is 450-550k", source).presence is QuotePresence.PUNCTUATION


def test_a_dropped_letter_is_a_near_miss_not_punctuation() -> None:
    # Every character of the quote appears in order, so the overlap is high,
    # but folding does not make it a substring: the model did not copy it.
    check = check_quote("call me in three month about the rental", SOURCE)
    assert check.presence is QuotePresence.HIGH_SIMILARITY
    assert check.is_near_miss


def test_an_added_ellipsis_is_a_near_miss_not_punctuation() -> None:
    # Folding turns the ellipsis into three dots, which ADDS characters the
    # source does not have. The model abbreviated rather than mistyped, so it
    # is reported as the judgement it is rather than as typography.
    check = check_quote("call me in three months about the rental…", SOURCE)
    assert check.presence is QuotePresence.HIGH_SIMILARITY


def test_a_fabricated_quote_is_absent() -> None:
    check = check_quote("I will email the paperwork on Friday", SOURCE)
    assert check.presence is QuotePresence.ABSENT
    assert not check.is_near_miss


def test_a_quote_from_another_message_is_absent() -> None:
    assert check_quote("her birthday is 5 March", SOURCE).presence is QuotePresence.ABSENT


def test_an_empty_quote_is_absent_and_not_a_near_miss() -> None:
    check = check_quote("", SOURCE)
    assert check.presence is QuotePresence.ABSENT
    assert check.similarity == 0.0


def test_an_source_not_in_transcript_has_nothing_to_match_against() -> None:
    # What a fact citing a label the transcript never had looks like.
    assert check_quote("anything at all", "").presence is QuotePresence.ABSENT


def test_ordered_overlap_is_asymmetric_in_the_needles_favour() -> None:
    # A short quote inside a long message scores 1.0; the reverse does not.
    # SequenceMatcher.ratio() would penalise both, which is wrong here.
    assert ordered_overlap("three months", SOURCE) == 1.0
    assert ordered_overlap(SOURCE, "three months") < 0.5


def test_ordered_overlap_of_nothing_is_zero_not_one() -> None:
    assert ordered_overlap("", SOURCE) == 0.0


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------
def test_an_exact_date_matches_with_a_zero_offset() -> None:
    match = match_date(date(2026, 6, 4), "2026-06-04")
    assert match.verdict is DateVerdict.EXACT
    assert match.offset_days == 0
    assert match.correct


def test_a_wrong_date_reports_its_signed_offset() -> None:
    # The sign is the diagnosis: anchoring on today instead of the message's
    # own date pushes dates one way, a misread month the other.
    assert match_date(date(2026, 6, 4), "2026-06-07").offset_days == 3
    assert match_date(date(2026, 6, 4), "2026-06-01").offset_days == -3


def test_a_date_with_a_timestamp_is_parsed_not_rejected() -> None:
    assert match_date(date(2026, 6, 4), "2026-06-04T09:00:00Z").verdict is DateVerdict.EXACT


def test_an_unparseable_date_is_not_an_offset() -> None:
    assert match_date(date(2026, 6, 4), "next spring").verdict is DateVerdict.UNPARSEABLE


def test_a_missing_date_differs_from_a_spurious_one() -> None:
    assert match_date(date(2026, 6, 4), None).verdict is DateVerdict.MISSING
    assert match_date(None, "2026-06-04").verdict is DateVerdict.SPURIOUS


def test_neither_side_asserting_a_date_is_not_scored_as_a_pass() -> None:
    match = match_date(None, None)
    assert match.verdict is DateVerdict.NOT_ASSERTED
    assert match.correct


def test_parse_iso_date_passes_a_date_through() -> None:
    assert parse_iso_date(date(2026, 6, 4)) == date(2026, 6, 4)
    assert parse_iso_date(None) is None
    assert parse_iso_date("rubbish") is None


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------
def test_an_identical_name_is_exact() -> None:
    assert match_name("Mark Chen", "Mark Chen").verdict is NameVerdict.EXACT


def test_case_and_punctuation_do_not_break_a_name() -> None:
    assert match_name("Mark O'Brien", "mark obrien").verdict is NameVerdict.EXACT


def test_a_first_name_against_a_full_name_is_a_subset() -> None:
    # Credited as correct, counted separately: a run passing entirely on
    # partial names should be visible rather than hidden in a recall figure.
    match = match_name("Mark Chen", "Mark")
    assert match.verdict is NameVerdict.SUBSET
    assert match.correct


def test_subset_works_in_both_directions() -> None:
    assert match_name("Mark", "Mark Chen").verdict is NameVerdict.SUBSET


def test_a_different_person_is_a_mismatch() -> None:
    match = match_name("Mark Chen", "Dan Whitfield")
    assert match.verdict is NameVerdict.MISMATCH
    assert not match.correct


def test_a_shared_surname_is_still_a_mismatch() -> None:
    # Two contacts with the same surname are two people. Nothing here may
    # treat them as interchangeable.
    assert match_name("Sam Whitaker", "Priya Whitaker").verdict is NameVerdict.MISMATCH


def test_a_missing_name_differs_from_a_spurious_one() -> None:
    assert match_name("Mark", None).verdict is NameVerdict.MISSING
    assert match_name(None, "Mark").verdict is NameVerdict.SPURIOUS
    assert match_name(None, None).verdict is NameVerdict.NOT_ASSERTED


def test_whitespace_only_counts_as_no_name() -> None:
    assert match_name("   ", None).verdict is NameVerdict.NOT_ASSERTED


# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------
def test_digit_sequences_ignore_currency_and_separators() -> None:
    assert digit_sequence("$450,000") == ["450", "000"]
    assert digit_sequence("around $450k") == ["450"]
    assert digit_sequence(None) == []


def test_shorthand_and_longhand_are_the_same_price() -> None:
    assert match_price("$450k", "450,000")
    assert match_price("around $450k", "$450k")


def test_a_different_number_is_a_different_price() -> None:
    assert not match_price("$450k", "$540k")


def test_a_price_with_no_digits_cannot_match_one_that_has_them() -> None:
    assert not match_price("$450k", "a bit more than they paid")


def test_two_absent_prices_agree() -> None:
    assert match_price(None, None)
    assert not match_price("$450k", None)


# --------------------------------------------------------------------------
# Free text
# --------------------------------------------------------------------------
def test_free_text_matches_on_containment_either_way() -> None:
    assert match_free_text("owns, three bedrooms", "Owns, three bedrooms, bought 2015")
    assert match_free_text("Owns, three bedrooms, bought 2015", "owns, three bedrooms")


def test_unrelated_free_text_does_not_match() -> None:
    assert not match_free_text("three bedrooms", "a garden flat")


def test_both_empty_free_text_agrees_but_one_sided_does_not() -> None:
    assert match_free_text(None, None)
    assert not match_free_text("three bedrooms", None)
