"""
The reporting primitives.

The one rule worth a test suite of its own: an undefined ratio stays undefined.
"No cases measured" and "every case failed" are opposite findings, and a `Ratio`
that quietly returns 0.0 on an empty denominator would report the second when
the truth is the first.
"""

from __future__ import annotations

from evals.metrics import PRF, Ratio, Spread, percentile


# --------------------------------------------------------------------------
# Ratio
# --------------------------------------------------------------------------
def test_a_ratio_divides_and_keeps_its_n() -> None:
    ratio = Ratio(41, 50)
    assert ratio.value == 0.82
    assert ratio.n == 50
    assert ratio.render() == "0.820 (n=50)"


def test_an_empty_ratio_is_undefined_not_zero() -> None:
    assert Ratio(0, 0).value is None
    assert Ratio().value is None


def test_an_undefined_ratio_renders_as_not_available() -> None:
    assert Ratio(0, 0).render() == "n/a (n=0)"


def test_a_zero_numerator_over_real_cases_is_genuinely_zero() -> None:
    # The distinction the type exists for.
    assert Ratio(0, 30).value == 0.0
    assert Ratio(0, 30).render() == "0.000 (n=30)"


def test_ratios_add_both_parts() -> None:
    assert Ratio(1, 2) + Ratio(3, 4) == Ratio(4, 6)


def test_a_negative_denominator_is_treated_as_no_data() -> None:
    assert Ratio(1, -1).value is None


# --------------------------------------------------------------------------
# PRF
# --------------------------------------------------------------------------
def test_precision_and_recall_have_different_denominators() -> None:
    prf = PRF(true_positives=8, false_positives=2, false_negatives=4)
    assert prf.precision == Ratio(8, 10)
    assert prf.recall == Ratio(8, 12)


def test_f1_is_the_harmonic_mean() -> None:
    prf = PRF(true_positives=1, false_positives=1, false_negatives=1)
    assert prf.f1 is not None
    assert abs(prf.f1 - 0.5) < 1e-9


def test_f1_is_undefined_when_there_is_nothing_to_score() -> None:
    assert PRF().f1 is None


def test_f1_is_undefined_rather_than_zero_when_precision_is_unmeasurable() -> None:
    # No predictions and no gold facts of this kind: nothing happened.
    assert PRF(0, 0, 0).f1 is None


def test_f1_is_zero_when_everything_was_wrong() -> None:
    prf = PRF(true_positives=0, false_positives=3, false_negatives=3)
    assert prf.f1 == 0.0


def test_prfs_add() -> None:
    total = PRF(1, 2, 3) + PRF(4, 5, 6)
    assert (total.true_positives, total.false_positives, total.false_negatives) == (5, 7, 9)


# --------------------------------------------------------------------------
# Spread
# --------------------------------------------------------------------------
def test_one_sample_has_a_mean_but_no_spread() -> None:
    spread = Spread()
    spread.add(0.8)
    assert spread.mean == 0.8
    assert spread.sd is None
    assert "1 run" in spread.render()


def test_several_samples_report_mean_and_range() -> None:
    spread = Spread()
    for value in (0.80, 0.84, 0.82):
        spread.add(value)
    assert spread.n == 3
    assert spread.lo == 0.80
    assert spread.hi == 0.84
    assert spread.sd is not None


def test_undefined_samples_are_not_counted_as_zero() -> None:
    spread = Spread()
    spread.add(None)
    spread.add(0.5)
    assert spread.n == 1
    assert spread.mean == 0.5


def test_an_empty_spread_renders_as_not_available() -> None:
    assert Spread().render() == "n/a"


def test_a_change_inside_the_spread_does_not_exceed_the_noise() -> None:
    # The claim the whole --repeats flag exists to support.
    before, after = Spread(), Spread()
    for value in (0.80, 0.84, 0.82):
        before.add(value)
    for value in (0.81, 0.85, 0.83):
        after.add(value)
    assert after.exceeds_noise(before) is False


def test_a_change_larger_than_the_spread_does_exceed_it() -> None:
    before, after = Spread(), Spread()
    for value in (0.80, 0.81, 0.80):
        before.add(value)
    for value in (0.92, 0.93, 0.92):
        after.add(value)
    assert after.exceeds_noise(before) is True


def test_noise_is_unknown_with_a_single_run_each() -> None:
    # Reported as unknown rather than as "no change": one run apiece cannot
    # tell an improvement from a coin toss.
    before, after = Spread(), Spread()
    before.add(0.80)
    after.add(0.90)
    assert after.exceeds_noise(before) is None


# --------------------------------------------------------------------------
# Percentiles
# --------------------------------------------------------------------------
def test_percentiles_are_real_observations_not_interpolations() -> None:
    values = [100.0, 200.0, 300.0, 400.0]
    assert percentile(values, 50) in values
    assert percentile(values, 95) in values


def test_p95_picks_the_slowest_of_a_small_sample() -> None:
    assert percentile([10.0, 20.0, 900.0], 95) == 900.0


def test_a_single_observation_is_every_percentile() -> None:
    assert percentile([42.0], 50) == 42.0
    assert percentile([42.0], 95) == 42.0


def test_no_observations_has_no_percentile() -> None:
    assert percentile([], 50) is None
