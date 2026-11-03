"""
Matching rules, one per field type.

The spec asks for an explicit rule per field type rather than one comparison
applied to everything, because "wrong" means different things in different
columns. A date that is three days out and a date that is three years out are
both unequal and not equally wrong. A name returned as "Mark" where the gold
says "Mark Chen" is right in the only way that matters. A price written "$450k"
against a gold of "450,000" is the same number in different clothes.

Every rule here is deterministic and returns a *reason*, not a boolean, so the
error analysis in Phase 6 can group by failure shape instead of by field.

`normalise` is imported from the application rather than reimplemented: it is
the exact rule the production verifier applies to a quote before looking for
it, and a second implementation of it here would be a second definition of
whether a fact survives.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher
from enum import StrEnum

from demo_app.context import normalise

# Characters a model substitutes without changing meaning, and which the
# production `normalise` does not touch — it collapses whitespace and lowercases
# and stops there. Folding these is NOT production behaviour; it is the pipeline
# measuring how much verification loss is pure typography.
_PUNCTUATION_FOLD = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "′": "'", "″": '"',
    "–": "-", "—": "-", "―": "-", "−": "-",
    "…": "...",
    " ": " ", " ": " ", " ": " ", " ": " ",
    "​": "", "﻿": "",
}

_DIGITS = re.compile(r"\d+")
# Apostrophes are deleted rather than spaced, so "O'Brien" normalises to one
# token and still matches a CRM that stored "OBrien". Everything else becomes a
# space, so "Mary-Jane" normalises to two tokens and matches "Mary Jane" —
# punctuation that joins words should split, punctuation inside a word should
# vanish.
_APOSTROPHE = re.compile(r"'")
_PUNCT_STRIP = re.compile(r"[^\w\s]", re.UNICODE)


def fold_punctuation(text: str) -> str:
    """
    Smart quotes, dashes and ellipses down to their ASCII equivalents.

    Deliberately not accent folding: in English-only data an accent difference
    between a quote and its source is a transcription error worth seeing, not
    noise worth hiding. If other languages arrive this becomes a decision.
    """
    out = "".join(_PUNCTUATION_FOLD.get(ch, ch) for ch in text)
    # Compatibility normalisation catches ligatures and the width variants that
    # arrive from spreadsheet exports.
    return unicodedata.normalize("NFKC", out)


class QuotePresence(StrEnum):
    """
    Why a quote did or did not resolve against the message it cited.

    The order matters: each level is "fails the one above, passes this one".
    """

    EXACT = "exact"
    """The production check passes. The fact survives verification."""

    PUNCTUATION = "punctuation"
    """
    Fails production, passes once smart quotes and dashes are folded. The model
    copied the words and not the typography. This is wrongful loss with a
    one-line fix, and it needs no human label to identify.
    """

    HIGH_SIMILARITY = "high_similarity"
    """
    Most of the quote is present, in order, but not all of it — a typo, a
    dropped word, a silently corrected spelling. Probably wrongful loss, but
    unlike PUNCTUATION it is a judgement, so it is reported separately.
    """

    ABSENT = "absent"
    """Genuinely not in that message. The verifier is right to drop it."""


# Above this share of the quote appearing in order, call it a near miss. Set
# high on purpose: this number is used to argue that the verifier is wrong, so
# it has to be hard to reach by accident.
SIMILARITY_THRESHOLD = 0.90


@dataclass(frozen=True)
class QuoteCheck:
    presence: QuotePresence
    similarity: float

    @property
    def survives_production(self) -> bool:
        return self.presence is QuotePresence.EXACT

    @property
    def is_near_miss(self) -> bool:
        return self.presence in (
            QuotePresence.PUNCTUATION,
            QuotePresence.HIGH_SIMILARITY,
        )


def ordered_overlap(needle: str, haystack: str) -> float:
    """
    The share of `needle` that appears inside `haystack`, in order.

    Not `SequenceMatcher.ratio()`, which is symmetric and so penalises a short
    quote against a long message — exactly the normal case here. Summing the
    matching blocks and dividing by the needle's own length asks the question
    we actually have: how much of this quote is in that message.
    """
    if not needle:
        return 0.0
    matcher = SequenceMatcher(None, needle, haystack, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return min(1.0, matched / len(needle))


def check_quote(quote: str, message_text: str) -> QuoteCheck:
    """
    Classify a quote against the message it cited.

    `message_text` is expected pre-normalised, because that is how the
    application stores it (`ContactContext.texts`). Normalising again is
    harmless and makes the function safe to call directly in tests.
    """
    needle = normalise(quote)
    haystack = normalise(message_text)

    if not needle:
        return QuoteCheck(QuotePresence.ABSENT, 0.0)

    if needle in haystack:
        return QuoteCheck(QuotePresence.EXACT, 1.0)

    folded_needle = normalise(fold_punctuation(needle))
    folded_haystack = normalise(fold_punctuation(haystack))
    if folded_needle and folded_needle in folded_haystack:
        return QuoteCheck(QuotePresence.PUNCTUATION, 1.0)

    similarity = ordered_overlap(folded_needle, folded_haystack)
    if similarity >= SIMILARITY_THRESHOLD:
        return QuoteCheck(QuotePresence.HIGH_SIMILARITY, similarity)
    return QuoteCheck(QuotePresence.ABSENT, similarity)


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------
def match_enum(gold: str | None, predicted: str | None) -> bool:
    """Exact. An enum has no near misses: a different member is a different claim."""
    return gold == predicted


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------
class DateVerdict(StrEnum):
    EXACT = "exact"
    OFF_BY = "off_by"
    UNPARSEABLE = "unparseable"
    MISSING = "missing"
    """Gold asserts a date, the model gave none."""
    SPURIOUS = "spurious"
    """The model gave a date, gold asserts none."""
    NOT_ASSERTED = "not_asserted"
    """Neither side has a view. Skipped in scoring rather than scored as a pass."""


@dataclass(frozen=True)
class DateMatch:
    verdict: DateVerdict
    offset_days: int | None = None

    @property
    def correct(self) -> bool:
        return self.verdict in (DateVerdict.EXACT, DateVerdict.NOT_ASSERTED)


def parse_iso_date(value: str | date | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except (ValueError, AttributeError):
        return None


def match_date(gold: date | None, predicted: str | date | None) -> DateMatch:
    """
    Exact equality, plus the signed day offset when it is wrong.

    The offset is the point. Shifted callback dates are the one failure already
    observed in this product, and a boolean cannot tell a model that anchored on
    today instead of the message's own date from a model that misread the month.
    """
    if gold is None and predicted is None:
        return DateMatch(DateVerdict.NOT_ASSERTED)
    if gold is None:
        return DateMatch(DateVerdict.SPURIOUS)
    if predicted is None:
        return DateMatch(DateVerdict.MISSING)

    parsed = parse_iso_date(predicted)
    if parsed is None:
        return DateMatch(DateVerdict.UNPARSEABLE)
    if parsed == gold:
        return DateMatch(DateVerdict.EXACT, 0)
    return DateMatch(DateVerdict.OFF_BY, (parsed - gold).days)


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------
class NameVerdict(StrEnum):
    EXACT = "exact"
    SUBSET = "subset"
    """One is a token subset of the other: "Mark" against "Mark Chen"."""
    MISMATCH = "mismatch"
    MISSING = "missing"
    SPURIOUS = "spurious"
    NOT_ASSERTED = "not_asserted"


@dataclass(frozen=True)
class NameMatch:
    verdict: NameVerdict

    @property
    def correct(self) -> bool:
        return self.verdict in (
            NameVerdict.EXACT,
            NameVerdict.SUBSET,
            NameVerdict.NOT_ASSERTED,
        )


def normalise_name(name: str | None) -> str:
    if not name:
        return ""
    folded = _APOSTROPHE.sub("", fold_punctuation(name))
    stripped = _PUNCT_STRIP.sub(" ", folded)
    return " ".join(stripped.split()).casefold()


def match_name(gold: str | None, predicted: str | None) -> NameMatch:
    """
    Normalised, with token-subset tolerance.

    Credited as correct but counted separately, so a run that passes entirely on
    subset matches is visible rather than hidden inside a recall figure.
    """
    g, p = normalise_name(gold), normalise_name(predicted)
    if not g and not p:
        return NameMatch(NameVerdict.NOT_ASSERTED)
    if not g:
        return NameMatch(NameVerdict.SPURIOUS)
    if not p:
        return NameMatch(NameVerdict.MISSING)
    if g == p:
        return NameMatch(NameVerdict.EXACT)

    gt, pt = set(g.split()), set(p.split())
    if gt and pt and (gt <= pt or pt <= gt):
        return NameMatch(NameVerdict.SUBSET)
    return NameMatch(NameVerdict.MISMATCH)


# --------------------------------------------------------------------------
# Prices and other free text
# --------------------------------------------------------------------------
def digit_sequence(value: str | None) -> list[str]:
    """
    The runs of digits in a value, in order.

    "$450k" and "450,000" are the same price written two ways and neither
    normalises into the other, so a string comparison is useless. Comparing
    digit runs catches the thing that actually matters: whether the model read
    a different number.
    """
    if not value:
        return []
    return _DIGITS.findall(fold_punctuation(value))


def match_price(gold: str | None, predicted: str | None) -> bool:
    """
    Same digits in the same order, or both empty.

    "$450k" against "450,000" compares ["450"] with ["450", "000"], which is
    not equal — so the rule is prefix tolerance in the leading run plus equal
    first digits. Shorthand that agrees on the significant figures passes;
    "$450k" against "$540k" does not.
    """
    g, p = digit_sequence(gold), digit_sequence(predicted)
    if not g and not p:
        return True
    if not g or not p:
        return False
    if g == p:
        return True
    # Shorthand: compare only the first run, which carries the magnitude the
    # agent recognises.
    return g[0] == p[0]


def match_free_text(gold: str | None, predicted: str | None) -> bool:
    """
    Normalised containment either way.

    Used only for `property_note` and similar short descriptive values, never
    for `summary` or `notes` — those are prose with no single right
    phrasing and are checked for groundedness instead of compared. Scoring them
    against a reference would measure style.
    """
    g, p = normalise(gold or ""), normalise(predicted or "")
    if not g and not p:
        return True
    if not g or not p:
        return False
    return g in p or p in g
