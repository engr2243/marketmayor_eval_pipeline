"""
The extraction target: the real pipeline, with fixtures where the datastore was.

Every piece of the application is called rather than copied — `context.build()`
to render the transcript, `prompt.SYSTEM` as the instructions,
`prompt.build_user_text()` for the per-contact half, `ContactAnalysis` as the
schema, `adapter.extract_one()` for the call, and `verify.validate()` for
verification. A second implementation of any of them would mean the pipeline
could pass while the application failed.

The only substitution is where the rows come from: a case declares its messages
as fixtures instead of them being fetched.

**If you are reading this as a template.** The demo application's context
builder is a pure function that takes its messages as an argument. A real one
usually takes a database session and runs the query itself, in which case this
module needs a small stand-in that answers that one read-only query from the
case's fixtures. That is worth doing rather than avoiding: the transcript
rendering, the per-message clip, the head-and-tail truncation and the tag
channel are all real parts of the pipeline, and reimplementing them here to dodge
a database would mean evaluating a copy of the thing you care about.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from datetime import time as clock_time
from typing import Any

from demo_app import adapter as ai_adapter
from demo_app import context as ctxlib
from demo_app.context import Contact, Message
from demo_app.prompt import SYSTEM, build_user_text
from demo_app.schemas import ContactAnalysis
from demo_app.verify import validate
from evals.cost import Usage
from evals.models import GoldCase

# Rough characters per token for English prose. Used only for the pre-flight
# estimate shown before a large run; actual usage comes back from the provider
# and is what gets reported.
CHARS_PER_TOKEN = 3.8


def _as_utc(day: Any) -> datetime:
    """A case's date at midday UTC, so a timezone cannot shift it a day."""
    return datetime.combine(day, clock_time(12, 0), tzinfo=UTC)


def build_contact(case: GoldCase) -> Contact:
    """The contact fields the prompt is given."""
    return Contact(
        id=f"eval-{case.case_id}",
        first_name=case.contact.first_name,
        last_name=case.contact.last_name,
        stage=case.contact.stage,
        tags=list(case.contact.tags),
    )


def build_messages(case: GoldCase) -> list[Message]:
    """The case's messages. `build()` sorts them, as a real query's ORDER BY would."""
    return [
        Message(
            id=f"eval-{case.case_id}-m{index}",
            occurred_at=_as_utc(seed.day),
            direction=seed.direction,
            kind=seed.kind,
            subject=seed.subject,
            body=seed.body,
        )
        for index, seed in enumerate(case.messages, start=1)
    ]


@dataclass
class Prepared:
    """One case, rendered into exactly what the model will be sent."""

    case: GoldCase
    contact: Contact
    ctx: Any
    system: str
    user_text: str

    @property
    def has_content(self) -> bool:
        return bool(self.ctx.has_content)

    @property
    def estimated_input_tokens(self) -> int:
        return int((len(self.system) + len(self.user_text)) / CHARS_PER_TOKEN)


async def prepare(case: GoldCase) -> Prepared:
    """
    Render a case without calling anything.

    Separated from the call so the pre-flight estimate is measured from the real
    rendered prompt rather than guessed, and so a dataset can be checked for
    shape without spending a cent.
    """
    contact = build_contact(case)
    ctx = ctxlib.build(contact, build_messages(case))
    return Prepared(
        case=case,
        contact=contact,
        ctx=ctx,
        system=SYSTEM,
        user_text=build_user_text(contact, ctx, case.today.isoformat()),
    )


@dataclass
class ExtractionOutcome:
    """What one call produced, before any scoring."""

    case_id: str
    prepared: Prepared

    analysis: Any | None = None
    validated: Any | None = None

    schema_valid: bool = True
    refused: bool = False
    call_failed: bool = False
    error: str | None = None
    skipped: str | None = None
    """Set when the case was never sent — an empty transcript, or the cap."""

    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    repeat_index: int = 0


async def run_one(
    api: Any, case: GoldCase, *, repeat_index: int = 0
) -> ExtractionOutcome:
    """
    One live extraction through `extract_one()`, with the client injected.

    `extract_one` takes the client as its first argument, which is the whole
    reason this target needs no change to the application. It returns
    `(parsed, usage)` and never raises for content reasons, so a `None` here is
    a refusal or a schema failure rather than an exception.
    """
    prepared = await prepare(case)
    outcome = ExtractionOutcome(
        case_id=case.case_id, prepared=prepared, repeat_index=repeat_index
    )

    if not prepared.has_content:
        # The application skips these before building a request, so sending one
        # would measure a path it never takes.
        outcome.skipped = "no readable transcript"
        return outcome

    started = time.perf_counter()
    try:
        analysis, usage = await ai_adapter.extract_one(
            api,
            system=prepared.system,
            user_text=prepared.user_text,
            schema_model=ContactAnalysis,
        )
    except Exception as exc:
        outcome.latency_ms = (time.perf_counter() - started) * 1000
        outcome.call_failed = True
        outcome.error = f"{type(exc).__name__}: {exc}"[:500]
        return outcome

    outcome.latency_ms = (time.perf_counter() - started) * 1000
    outcome.usage = Usage.from_app_usage(usage)

    if analysis is None:
        # The adapter collapses a refusal and an unparseable response into the
        # same return. Both mean no usable output, and both are reported.
        outcome.schema_valid = False
        outcome.error = "no usable result (refusal or schema validation failure)"
        return outcome

    outcome.analysis = analysis
    outcome.validated = validate(analysis, prepared.ctx)
    return outcome


# --------------------------------------------------------------------------
# Batch path — half price, and the shape a scheduled pass would use
# --------------------------------------------------------------------------
BATCH_POLL_SECONDS = 15.0
BATCH_TIMEOUT_SECONDS = 60 * 60


async def run_batch(
    api: Any,
    cases: Iterable[GoldCase],
    *,
    repeat_index: int = 0,
    poll_seconds: float = BATCH_POLL_SECONDS,
    timeout_seconds: float = BATCH_TIMEOUT_SECONDS,
) -> list[ExtractionOutcome]:
    """
    The same prompts through the Message Batches helpers.

    Results arrive in arbitrary order, so they are keyed by `custom_id` and
    never by position — a positional read would silently score every case
    against another case's output.

    No usage comes back from `batch_results`, which yields only the id and the
    text. Token counts are therefore unavailable on this path and are reported
    as absent rather than estimated.
    """
    prepared_by_id: dict[str, Prepared] = {}
    requests: list[dict] = []
    outcomes: list[ExtractionOutcome] = []

    for case in cases:
        prepared = await prepare(case)
        if not prepared.has_content:
            outcomes.append(
                ExtractionOutcome(
                    case_id=case.case_id,
                    prepared=prepared,
                    skipped="no readable transcript",
                    repeat_index=repeat_index,
                )
            )
            continue
        custom_id = f"{case.case_id}__r{repeat_index}"
        prepared_by_id[custom_id] = prepared
        requests.append(
            ai_adapter.batch_request(
                custom_id,
                system=prepared.system,
                user_text=prepared.user_text,
                schema_model=ContactAnalysis,
            )
        )

    if not requests:
        return outcomes

    started = time.perf_counter()
    batch_id = await ai_adapter.submit_batch(api, requests)

    deadline = started + timeout_seconds
    while True:
        status = await ai_adapter.batch_status(api, batch_id)
        if status == "ended":
            break
        if time.perf_counter() > deadline:
            for prepared in prepared_by_id.values():
                outcomes.append(
                    ExtractionOutcome(
                        case_id=prepared.case.case_id,
                        prepared=prepared,
                        call_failed=True,
                        error=f"batch {batch_id} still {status} after "
                        f"{timeout_seconds:.0f}s",
                        repeat_index=repeat_index,
                    )
                )
            return outcomes
        await asyncio.sleep(poll_seconds)

    elapsed_ms = (time.perf_counter() - started) * 1000
    seen: set[str] = set()

    async for custom_id, text in ai_adapter.batch_results(api, batch_id):
        result_for = prepared_by_id.get(custom_id)
        if result_for is None:
            continue
        seen.add(custom_id)
        outcome = ExtractionOutcome(
            case_id=result_for.case.case_id,
            prepared=result_for,
            repeat_index=repeat_index,
            # One figure for the whole batch: a batch has no per-case latency
            # and inventing one would be a fabricated number.
            latency_ms=elapsed_ms,
        )
        if not text:
            outcome.schema_valid = False
            outcome.error = "no usable result in batch"
            outcomes.append(outcome)
            continue
        try:
            analysis = ContactAnalysis.model_validate_json(text)
        except Exception as exc:
            outcome.schema_valid = False
            outcome.error = f"schema validation failed: {exc}"[:500]
            outcomes.append(outcome)
            continue
        outcome.analysis = analysis
        outcome.validated = validate(analysis, result_for.ctx)
        outcomes.append(outcome)

    # Anything the provider never returned a row for, recorded as a failure
    # rather than quietly absent from the denominator.
    for custom_id, prepared in prepared_by_id.items():
        if custom_id in seen:
            continue
        outcomes.append(
            ExtractionOutcome(
                case_id=prepared.case.case_id,
                prepared=prepared,
                call_failed=True,
                error="no result returned for this request",
                repeat_index=repeat_index,
            )
        )

    return outcomes


def schema_fingerprint() -> dict[str, Any]:
    """The schema the model is constrained to, for the run's prompt hash."""
    return ContactAnalysis.model_json_schema()
