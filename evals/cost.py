"""
What a run costs, and stopping it before it costs more than agreed.

Separate from the application's own spend accounting on purpose, and the
separation is a requirement rather than a tidiness preference: the application
has a per-user monthly AI budget, and an evaluation must not draw against it. A
labelling afternoon that quietly exhausted a real agent's allowance would switch
off their nightly read.

The prices below are a tripwire, not accounting. The provider's dashboard is the
authority. They are per million tokens, and the pair for the extraction model
agrees with the batch figures already hardcoded in the application, which is the
only external check available here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    input: float
    output: float

    @property
    def cache_read(self) -> float:
        """Cache reads bill at a tenth of the input rate."""
        return self.input * 0.10

    @property
    def cache_write(self) -> float:
        """A cache write costs a quarter more than the same tokens fresh."""
        return self.input * 1.25

    def batch(self) -> Price:
        """The Message Batches discount: half price, both directions."""
        return Price(self.input * 0.5, self.output * 0.5)


# Keyed by model id prefix so a point release does not silently fall back to the
# default. Verify against current published pricing before quoting a cost
# figure anywhere it matters.
PRICES: dict[str, Price] = {
    "claude-haiku-4-5": Price(1.00, 5.00),
    "claude-haiku-5-5": Price(1.00, 5.00),
    "claude-sonnet-5": Price(3.00, 15.00),
    "claude-sonnet-5-5": Price(3.00, 15.00),
    "claude-opus-5": Price(15.00, 75.00),
    "claude-opus-5-5": Price(15.00, 75.00),
    "claude-fable-5-1": Price(3.00, 15.00),
}

# Used when a model id is not in the table. Deliberately the most expensive
# entry: an unknown model should over-estimate and trip the cap early rather
# than under-estimate and overspend.
FALLBACK = Price(15.00, 75.00)


def price_for(model: str) -> tuple[Price, bool]:
    """
    The price for a model id, and whether it was actually found.

    The flag matters: a cost figure derived from the fallback is a guess, and
    the report says so rather than printing it as a measurement.
    """
    for prefix, price in PRICES.items():
        if model.startswith(prefix):
            return price, True
    return FALLBACK, False


@dataclass(frozen=True)
class Usage:
    """Token counts for one call. Mirrors the application's `Usage` shape."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_creation_tokens + other.cache_creation_tokens,
        )

    @classmethod
    def from_app_usage(cls, usage: object) -> Usage:
        """
        From the application's own `adapters.ai.Usage`.

        Read by attribute rather than by type so this module does not have to
        import the application, which keeps it unit-testable with the
        application absent from the path entirely.
        """
        return cls(
            int(getattr(usage, "input_tokens", 0) or 0),
            int(getattr(usage, "output_tokens", 0) or 0),
            int(getattr(usage, "cache_read_tokens", 0) or 0),
            int(getattr(usage, "cache_creation_tokens", 0) or 0),
        )


def usd(usage: Usage, model: str, *, batch: bool = False) -> float:
    """
    What one call's tokens cost.

    Cache reads are billed separately and are *not* included in
    `input_tokens` by the provider, so they are added rather than subtracted.
    """
    price, _known = price_for(model)
    if batch:
        price = price.batch()
    return (
        usage.input_tokens / 1_000_000 * price.input
        + usage.output_tokens / 1_000_000 * price.output
        + usage.cache_read_tokens / 1_000_000 * price.cache_read
        + usage.cache_creation_tokens / 1_000_000 * price.cache_write
    )


class SpendCapReached(RuntimeError):
    """
    The run has spent what it was allowed to.

    Raised rather than returned so no call site can forget to check it. The
    runner catches it, stops dispatching, and reports the run as capped with
    the cases that did complete — a truncated run with its n stated is useful,
    a silently shortened one is not.
    """


@dataclass
class SpendTracker:
    """
    Running spend against a ceiling, safe to share across concurrent tasks.

    Checked before dispatch *and* accumulated after each call, so the cap can
    only ever be overshot by the calls already in flight. With bounded
    concurrency that overshoot is small and known; pretending it is zero would
    need a reservation scheme that stalls the run for no real benefit.
    """

    cap_usd: float
    spent_usd: float = 0.0
    calls: int = 0
    usage: Usage = field(default_factory=Usage)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record(self, usage: Usage, model: str, *, batch: bool = False) -> float:
        cost = usd(usage, model, batch=batch)
        with self._lock:
            self.spent_usd += cost
            self.calls += 1
            self.usage = self.usage + usage
        return cost

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.cap_usd - self.spent_usd)

    def check(self) -> None:
        """Raise if the ceiling is reached. Called before each dispatch."""
        if self.cap_usd > 0 and self.spent_usd >= self.cap_usd:
            raise SpendCapReached(
                f"Eval spend cap of ${self.cap_usd:.2f} reached "
                f"after {self.calls} call(s) (${self.spent_usd:.4f})"
            )


def estimate_run_usd(
    *,
    cases: int,
    repeats: int,
    model: str,
    input_tokens_per_case: int,
    output_tokens_per_case: int,
    batch: bool = False,
) -> tuple[float, int]:
    """
    A pre-flight estimate, and the number of calls it covers.

    Shown to the operator before any run over fifty calls, per the brief. The
    token figures are per case and come from measuring the rendered prompt, not
    from a guess — see `targets/extraction.py`, which counts them before
    dispatching anything.
    """
    calls = cases * max(1, repeats)
    total = Usage(
        input_tokens=input_tokens_per_case * calls,
        output_tokens=output_tokens_per_case * calls,
    )
    return usd(total, model, batch=batch), calls
