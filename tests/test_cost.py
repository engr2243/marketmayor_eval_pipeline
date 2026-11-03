"""
Pricing and the spend ceiling.

The ceiling is the only thing in the pipeline that can cost real money by being
wrong, so the cases here are the unsafe directions: an unknown model must
over-estimate rather than under-estimate, and the cap must stop a run rather
than let it finish quietly over budget.
"""

from __future__ import annotations

import pytest

from evals.cost import (
    FALLBACK,
    PRICES,
    SpendCapReached,
    SpendTracker,
    Usage,
    estimate_run_usd,
    price_for,
    usd,
)


# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------
def test_a_known_model_is_found() -> None:
    price, known = price_for("claude-haiku-4-5")
    assert known
    assert price.input == 1.00


def test_a_point_release_matches_its_prefix() -> None:
    _price, known = price_for("claude-sonnet-5-20260801")
    assert known


def test_an_unknown_model_over_estimates_rather_than_guessing_low() -> None:
    # An unknown model should trip the cap early, not overspend silently.
    price, known = price_for("some-other-vendor-model")
    assert not known
    assert price == FALLBACK
    assert price.input >= max(p.input for p in PRICES.values())


def test_the_extraction_batch_price_agrees_with_the_application() -> None:
    # The application hardcodes 0.50 / 2.50 per Mtok as the extraction model's
    # batch pricing for its own budget guard. Agreement with that is the only
    # external check available on this table, so it is pinned here.
    price, _ = price_for("claude-haiku-4-5")
    batch = price.batch()
    assert (batch.input, batch.output) == (0.50, 2.50)


def test_a_cache_read_is_a_tenth_of_fresh_input() -> None:
    price, _ = price_for("claude-sonnet-5")
    assert price.cache_read == pytest.approx(price.input * 0.10)


# --------------------------------------------------------------------------
# Costing a call
# --------------------------------------------------------------------------
def test_a_calls_cost_is_its_tokens_at_its_models_price() -> None:
    cost = usd(Usage(input_tokens=1_000_000, output_tokens=0), "claude-haiku-4-5")
    assert cost == pytest.approx(1.00)


def test_batching_halves_it() -> None:
    tokens = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    live = usd(tokens, "claude-haiku-4-5")
    batched = usd(tokens, "claude-haiku-4-5", batch=True)
    assert batched == pytest.approx(live / 2)


def test_cache_reads_are_added_not_subtracted() -> None:
    # The provider bills cache reads separately and does not fold them into
    # input_tokens, so treating them as a discount on input would undercount.
    without = usd(Usage(input_tokens=1_000_000), "claude-sonnet-5")
    with_cache = usd(
        Usage(input_tokens=1_000_000, cache_read_tokens=1_000_000), "claude-sonnet-5"
    )
    assert with_cache > without


def test_an_empty_call_costs_nothing() -> None:
    assert usd(Usage(), "claude-haiku-4-5") == 0.0


def test_usage_adds_every_field() -> None:
    total = Usage(1, 2, 3, 4) + Usage(10, 20, 30, 40)
    assert (
        total.input_tokens,
        total.output_tokens,
        total.cache_read_tokens,
        total.cache_creation_tokens,
    ) == (11, 22, 33, 44)


def test_usage_reads_the_applications_shape_by_attribute() -> None:
    class AppUsage:
        input_tokens = 5
        output_tokens = 6
        cache_read_tokens = 7
        cache_creation_tokens = 8

    assert Usage.from_app_usage(AppUsage()) == Usage(5, 6, 7, 8)


def test_usage_tolerates_a_missing_field() -> None:
    class Partial:
        input_tokens = 5

    assert Usage.from_app_usage(Partial()) == Usage(5, 0, 0, 0)


# --------------------------------------------------------------------------
# The ceiling
# --------------------------------------------------------------------------
def test_a_tracker_accumulates_spend_and_calls() -> None:
    tracker = SpendTracker(cap_usd=1.0)
    tracker.record(Usage(input_tokens=100_000), "claude-haiku-4-5")
    tracker.record(Usage(input_tokens=100_000), "claude-haiku-4-5")
    assert tracker.calls == 2
    assert tracker.spent_usd == pytest.approx(0.20)
    assert tracker.remaining_usd == pytest.approx(0.80)


def test_the_cap_raises_rather_than_returning_a_flag() -> None:
    # Raised so no call site can forget to check it.
    tracker = SpendTracker(cap_usd=0.05)
    tracker.record(Usage(input_tokens=100_000), "claude-haiku-4-5")
    with pytest.raises(SpendCapReached, match="cap of"):
        tracker.check()


def test_a_run_under_the_cap_is_not_stopped() -> None:
    tracker = SpendTracker(cap_usd=10.0)
    tracker.record(Usage(input_tokens=100_000), "claude-haiku-4-5")
    tracker.check()


def test_a_zero_cap_disables_the_guard() -> None:
    # Matches the application's own convention for its monthly budget.
    tracker = SpendTracker(cap_usd=0.0)
    tracker.record(Usage(input_tokens=100_000_000), "claude-opus-5")
    tracker.check()


def test_remaining_never_goes_negative() -> None:
    tracker = SpendTracker(cap_usd=0.01)
    tracker.record(Usage(input_tokens=1_000_000), "claude-opus-5")
    assert tracker.remaining_usd == 0.0


# --------------------------------------------------------------------------
# Pre-flight estimate
# --------------------------------------------------------------------------
def test_the_estimate_covers_every_repeat() -> None:
    _cost, calls = estimate_run_usd(
        cases=60,
        repeats=3,
        model="claude-haiku-4-5",
        input_tokens_per_case=2000,
        output_tokens_per_case=900,
    )
    assert calls == 180


def test_the_estimate_is_halved_by_batching() -> None:
    live, _ = estimate_run_usd(
        cases=60, repeats=3, model="claude-haiku-4-5",
        input_tokens_per_case=2000, output_tokens_per_case=900,
    )
    batched, _ = estimate_run_usd(
        cases=60, repeats=3, model="claude-haiku-4-5",
        input_tokens_per_case=2000, output_tokens_per_case=900, batch=True,
    )
    assert batched == pytest.approx(live / 2)


def test_zero_repeats_still_costs_one_pass() -> None:
    _cost, calls = estimate_run_usd(
        cases=10, repeats=0, model="claude-haiku-4-5",
        input_tokens_per_case=1000, output_tokens_per_case=500,
    )
    assert calls == 10
