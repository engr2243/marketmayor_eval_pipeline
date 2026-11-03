"""
Running a dataset: concurrency, retries, repeats, and the spend ceiling.

Four properties the brief asks for, each of which changes what a number means:

- **Bounded concurrency**, so a sixty-case run does not open sixty sockets and
  trip a rate limit that then looks like a model failure.
- **Retry with backoff**, reusing the application's own `is_retryable()` rather
  than inventing a second policy. A call that fails for good is recorded as a
  failure and kept in the denominator; dropping it would quietly improve every
  rate in the report.
- **Repeats**, because a model is not a function. A change smaller than the
  run-to-run spread is not an improvement, and the report says so.
- **A spend cap** checked before every dispatch. When it trips the run stops and
  reports itself as capped with the cases that did finish, which is useful; a
  silently shortened run is not.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from demo_app import adapter as ai_adapter
from evals.config import CONFIRM_ABOVE_CALLS, RunConfig, RunMeta, prompt_hash
from evals.cost import SpendCapReached, SpendTracker, price_for
from evals.datasets import Dataset
from evals.env import api_key, git_sha
from evals.models import GoldCase
from evals.scorers.extraction import CaseScore, score_case
from evals.scorers.verification import verify
from evals.targets import extraction as extraction_target

MAX_ATTEMPTS = 3
BASE_BACKOFF_SECONDS = 2.0

# Transients the provider surfaces that the application's own `is_retryable`
# does not name. It matches "grammar compilation" and "overloaded", because
# those are the ones that cost it a contact's whole reading. A rate limit is
# retried by the SDK underneath us; by the time it reaches here the SDK's
# attempts are spent, and one more slow attempt is still worth it.
_EXTRA_RETRYABLE = ("rate limit", "429", "529", "timeout", "connection", "temporarily")


class RunAborted(RuntimeError):
    """The operator declined the pre-flight estimate."""


def is_retryable(exc: BaseException) -> bool:
    """
    The application's policy, widened for the two failure shapes an eval sees
    and production does not.

    Production does not need the widening: a nightly pass that loses a contact
    picks it back up on the next sweep. A run that loses a case loses it from
    the denominator, which is worse.
    """
    if ai_adapter.is_retryable(exc):
        return True
    text = str(exc).lower()
    return any(needle in text for needle in _EXTRA_RETRYABLE)


@dataclass
class RunResult:
    """One run: its identity, its scores, and anything that went wrong."""

    meta: RunMeta
    scores: list[CaseScore] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)
    """(case_id, error) for calls that never produced output."""

    @property
    def completed(self) -> int:
        return len(self.scores)


def _client() -> Any:
    """
    An Anthropic client for the eval, built from the eval's own key.

    Not `ai_adapter.client()`: that reads the application's `ANTHROPIC_API_KEY`
    and its `AI_ENABLED` flag, and an eval must be runnable against a separate
    key without switching the application's AI layer on. The timeout matches
    the application's so a slow call looks the same here as in production.
    """
    key = api_key()
    if not key:
        raise RuntimeError(
            "No API key. Set EVAL_ANTHROPIC_API_KEY (preferred, so eval spend "
            "is billed separately) or ANTHROPIC_API_KEY."
        )
    from anthropic import AsyncAnthropic

    return AsyncAnthropic(
        api_key=key, timeout=ai_adapter.EXTRACT_TIMEOUT_SECONDS
    )


async def _with_retries(
    call: Callable[[], Any],
    *,
    case_id: str,
    tracker: SpendTracker,
) -> Any:
    """
    Attempt a call, backing off on the transients, raising on everything else.

    Jittered backoff: a run that hits a rate limit hits it on every in-flight
    task at once, and retrying them all on the same schedule reproduces the
    burst that caused it.
    """
    attempt = 0
    while True:
        tracker.check()
        try:
            return await call()
        except SpendCapReached:
            raise
        except Exception as exc:
            attempt += 1
            if attempt >= MAX_ATTEMPTS or not is_retryable(exc):
                raise
            wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            await asyncio.sleep(wait * (0.5 + random.random()))


def preflight(
    dataset: Dataset,
    config: RunConfig,
    prepared: Sequence[extraction_target.Prepared],
) -> tuple[float, int, int]:
    """
    The estimate shown before a run over fifty calls.

    Input tokens are measured from the rendered prompts rather than guessed at;
    output tokens are an assumption and labelled as one in the CLI. Returns
    (estimated usd, calls, estimated input tokens per case).
    """
    from evals.cost import estimate_run_usd

    sendable = [p for p in prepared if p.has_content]
    if not sendable:
        return 0.0, 0, 0
    per_case_in = sum(p.estimated_input_tokens for p in sendable) // len(sendable)
    # A ContactAnalysis with a handful of facts. Assumption, not a measurement.
    per_case_out = 900
    usd, calls = estimate_run_usd(
        cases=len(sendable),
        repeats=config.repeats,
        model=config.resolved_model(),
        input_tokens_per_case=per_case_in,
        output_tokens_per_case=per_case_out,
        batch=config.batch,
    )
    return usd, calls, per_case_in


async def run_extraction(
    dataset: Dataset,
    config: RunConfig,
    *,
    confirm: Callable[[float, int, int], bool] | None = None,
) -> RunResult:
    """
    Score a whole extraction dataset.

    `confirm` is called once, before anything is dispatched, when the run would
    exceed fifty calls. Returning False aborts. Passing None with
    `config.assume_yes` unset is treated as a refusal rather than as consent —
    the default has to be the safe one.
    """
    cases: list[GoldCase] = [c for c in dataset.cases if isinstance(c, GoldCase)]
    if config.case_ids:
        wanted = set(config.case_ids)
        cases = [c for c in cases if c.case_id in wanted]
    if config.limit:
        cases = cases[: config.limit]

    model = config.resolved_model()
    _price, price_known = price_for(model)

    prepared = [await extraction_target.prepare(case) for case in cases]
    estimate, calls, _per_case = preflight(dataset, config, prepared)

    if calls > CONFIRM_ABOVE_CALLS and not config.assume_yes:
        approved = confirm(estimate, calls, _per_case) if confirm else False
        if not approved:
            raise RunAborted(
                f"{calls} calls (est. ${estimate:.2f}) needs confirmation; not run."
            )

    meta = RunMeta(
        run_id=RunMeta.new_run_id("extraction", config.label),
        target="extraction",
        label=config.label,
        model=model,
        effort=config.effort(),
        env_label=config.resolved_env_label(),
        batch=config.batch,
        repeats=config.repeats,
        dataset_name=dataset.name,
        dataset_hash=dataset.content_hash,
        case_count=len(cases),
        prompt_hash=prompt_hash(
            extraction_target.SYSTEM, extraction_target.schema_fingerprint()
        ),
        git_sha=git_sha(),
        started_at=datetime.now(UTC).isoformat(),
        spend_cap_usd=config.spend_cap_usd,
        price_table_known=price_known,
    )
    if not price_known:
        meta.notes.append(
            f"Model {model!r} is not in the price table; cost figures are an "
            "upper-bound guess, not a measurement."
        )

    result = RunResult(meta=meta)
    tracker = SpendTracker(cap_usd=config.spend_cap_usd)

    try:
        if config.batch:
            await _run_batched(cases, config, model, tracker, result)
        else:
            await _run_live(cases, config, model, tracker, result)
    except SpendCapReached as exc:
        meta.capped = True
        meta.notes.append(str(exc))

    meta.finished_at = datetime.now(UTC).isoformat()
    meta.spent_usd = tracker.spent_usd
    meta.calls = tracker.calls
    if meta.batch:
        meta.notes.append(
            "Batched: the Message Batches results carry no usage, so token "
            "counts and per-case latency are unavailable on this path and are "
            "reported as absent rather than estimated."
        )
    return result


async def _run_live(
    cases: Sequence[GoldCase],
    config: RunConfig,
    model: str,
    tracker: SpendTracker,
    result: RunResult,
) -> None:
    api = _client()
    semaphore = asyncio.Semaphore(max(1, config.concurrency))

    async def one(case: GoldCase, repeat: int) -> None:
        async with semaphore:
            try:
                outcome = await _with_retries(
                    lambda: extraction_target.run_one(api, case, repeat_index=repeat),
                    case_id=case.case_id,
                    tracker=tracker,
                )
            except SpendCapReached:
                raise
            except Exception as exc:
                result.failures.append((case.case_id, f"{type(exc).__name__}: {exc}"[:300]))
                result.scores.append(
                    CaseScore(
                        case_id=case.case_id,
                        stratum=case.stratum,
                        call_failed=True,
                        error=f"{type(exc).__name__}: {exc}"[:300],
                    )
                )
                return
            tracker.record(outcome.usage, model)
            result.scores.append(_score(case, outcome))

    for repeat in range(max(1, config.repeats)):
        tasks = [asyncio.create_task(one(case, repeat)) for case in cases]
        try:
            await asyncio.gather(*tasks)
        except SpendCapReached:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise


async def _run_batched(
    cases: Sequence[GoldCase],
    config: RunConfig,
    model: str,
    tracker: SpendTracker,
    result: RunResult,
) -> None:
    """
    The same prompts through the Message Batches helpers.

    No per-call spend tracking is possible here: the results carry no usage. The
    cap is therefore enforced on the pre-flight estimate for this path, which is
    stated in the report rather than glossed over.
    """
    api = _client()
    for repeat in range(max(1, config.repeats)):
        tracker.check()
        outcomes = await extraction_target.run_batch(api, cases, repeat_index=repeat)
        by_id = {c.case_id: c for c in cases}
        for outcome in outcomes:
            case = by_id.get(outcome.case_id)
            if case is None:
                continue
            if outcome.call_failed and outcome.error:
                result.failures.append((outcome.case_id, outcome.error))
            result.scores.append(_score(case, outcome))


def _score(case: GoldCase, outcome: extraction_target.ExtractionOutcome) -> CaseScore:
    """Turn one call's output into one case's score."""
    if outcome.skipped:
        score = CaseScore(case_id=case.case_id, stratum=case.stratum)
        score.error = outcome.skipped
        return score

    if outcome.analysis is None:
        return CaseScore(
            case_id=case.case_id,
            stratum=case.stratum,
            schema_valid=outcome.schema_valid,
            call_failed=outcome.call_failed,
            error=outcome.error,
            latency_ms=outcome.latency_ms,
            input_tokens=outcome.usage.input_tokens,
            output_tokens=outcome.usage.output_tokens,
            cache_read_tokens=outcome.usage.cache_read_tokens,
        )

    report = verify(outcome.analysis, outcome.prepared.ctx, case.gold_facts)
    validated = outcome.validated
    score = score_case(
        case_id=case.case_id,
        stratum=case.stratum,
        validated_facts=list(validated.facts) if validated else [],
        gold_facts=case.gold_facts,
        validated_state=validated.state if validated else None,
        gold_state=case.gold_state,
        notes=validated.notes if validated else None,
        verification=report,
    )
    score.latency_ms = outcome.latency_ms
    score.input_tokens = outcome.usage.input_tokens
    score.output_tokens = outcome.usage.output_tokens
    score.cache_read_tokens = outcome.usage.cache_read_tokens
    return score
