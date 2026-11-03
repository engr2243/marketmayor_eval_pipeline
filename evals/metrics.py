"""
Numbers that carry their own sample size.

Ground rule 6: every figure is reported as "0.82 (n=74)", never as a bare
percentage. That is easier to enforce with a type than with discipline, so a
ratio here is a numerator and a denominator and only becomes a float when
something asks. An undefined ratio — nothing in the denominator — stays
undefined rather than quietly becoming zero, because "no cases" and "all cases
failed" are opposite findings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean, stdev


@dataclass(frozen=True)
class Ratio:
    """A measured proportion and the sample it came from."""

    numerator: int = 0
    denominator: int = 0

    @property
    def value(self) -> float | None:
        """None when there is nothing to divide. Never 0.0 by default."""
        if self.denominator <= 0:
            return None
        return self.numerator / self.denominator

    @property
    def n(self) -> int:
        return self.denominator

    def __add__(self, other: Ratio) -> Ratio:
        return Ratio(
            self.numerator + other.numerator, self.denominator + other.denominator
        )

    def render(self, places: int = 3) -> str:
        v = self.value
        if v is None:
            return "n/a (n=0)"
        return f"{v:.{places}f} (n={self.denominator})"


@dataclass(frozen=True)
class PRF:
    """Precision, recall and F1, each with its own denominator."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> Ratio:
        return Ratio(self.true_positives, self.true_positives + self.false_positives)

    @property
    def recall(self) -> Ratio:
        return Ratio(self.true_positives, self.true_positives + self.false_negatives)

    @property
    def f1(self) -> float | None:
        """
        None only when precision or recall is itself unmeasurable.

        When both are measurable and both are zero — predictions were made,
        gold facts existed, none of them agreed — F1 is genuinely 0.0. Letting
        the zero-division guard return None there would report "not measured"
        for the worst possible result, which is the same mistake `Ratio` exists
        to prevent one level down.
        """
        p, r = self.precision.value, self.recall.value
        if p is None or r is None:
            return None
        if (p + r) == 0:
            return 0.0
        return 2 * p * r / (p + r)

    def __add__(self, other: PRF) -> PRF:
        return PRF(
            self.true_positives + other.true_positives,
            self.false_positives + other.false_positives,
            self.false_negatives + other.false_negatives,
        )

    def render(self) -> str:
        f1 = self.f1
        return (
            f"P {self.precision.render()} · "
            f"R {self.recall.render()} · "
            f"F1 {'n/a' if f1 is None else f'{f1:.3f}'}"
        )


@dataclass
class Spread:
    """
    A metric measured over several repeats.

    The reason `--repeats` exists: a model is not a function, and a change
    smaller than the run-to-run spread is not an improvement. The report says
    so in those words, and this is what it says it from.
    """

    samples: list[float] = field(default_factory=list)

    def add(self, value: float | None) -> None:
        if value is not None and not math.isnan(value):
            self.samples.append(value)

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def mean(self) -> float | None:
        return mean(self.samples) if self.samples else None

    @property
    def sd(self) -> float | None:
        """Sample standard deviation. None below two samples, not zero."""
        return stdev(self.samples) if len(self.samples) > 1 else None

    @property
    def lo(self) -> float | None:
        return min(self.samples) if self.samples else None

    @property
    def hi(self) -> float | None:
        return max(self.samples) if self.samples else None

    def render(self, places: int = 3) -> str:
        m = self.mean
        if m is None:
            return "n/a"
        if self.sd is None:
            return f"{m:.{places}f} (1 run)"
        return (
            f"{m:.{places}f} ±{self.sd:.{places}f} "
            f"[{self.lo:.{places}f}–{self.hi:.{places}f}, {self.n} runs]"
        )

    def exceeds_noise(self, other: Spread) -> bool | None:
        """
        Whether the gap between two measurements is bigger than their own noise.

        None when either side has too few repeats to have noise. Used by the
        report to label a delta as a real movement or as indistinguishable,
        rather than leaving the reader to eyeball two decimal places.
        """
        a, b = self.mean, other.mean
        if a is None or b is None:
            return None
        sa, sb = self.sd, other.sd
        if sa is None and sb is None:
            return None
        pooled = max(sa or 0.0, sb or 0.0)
        return abs(a - b) > pooled


def percentile(values: list[float], pct: float) -> float | None:
    """
    Nearest-rank percentile. Used for p50 and p95 latency.

    Deliberately not interpolated: with 60 cases an interpolated p95 invents a
    value between two real observations, and a latency figure should be a
    latency something actually took.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0, min(len(ordered) - 1, math.ceil(pct / 100 * len(ordered)) - 1))
    return ordered[rank]
