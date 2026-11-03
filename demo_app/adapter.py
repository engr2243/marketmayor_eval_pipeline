"""
The only place that knows which LLM vendor is in use.

Vendor specifics live here and callers deal in plain Python objects, so swapping
providers is this file plus the model constants.

The shape that matters to the pipeline: **the client is a parameter, not a
module-level singleton.** `extract_one(api, ...)` takes it, which is what lets an
evaluation drive the real call path with its own client, its own key and its own
spend ceiling without reconfiguring the application or monkeypatching this
module. A pipeline that has to patch the module under test is measuring itself.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)

# Extraction is high-volume and mechanical, so it gets the fastest model.
EXTRACT_MODEL = "claude-haiku-4-5"
EXTRACT_MAX_TOKENS = 4096

# One contact's transcript is a small request, so a call that has not answered in
# ninety seconds is not going to. The SDK default is ten minutes, and a reading
# pass works a book in series: one slow call there holds up every contact behind
# it.
EXTRACT_TIMEOUT_SECONDS = 90.0

# Server-side transients the SDK does not retry on its own, matched on the
# message because the status is a plain 400. "Grammar compilation timed out" is
# the structured-output schema failing to compile under load — the same request
# usually succeeds a moment later, and losing it costs the contact its whole
# reading until some later pass happens to pick them up.
#
# One retry, not two. A schema the API is struggling to compile will usually
# struggle again, and each attempt costs the timeout above.
_RETRYABLE_MESSAGES = ("grammar compilation", "overloaded")
EXTRACT_RETRIES = 1
EXTRACT_RETRY_BACKOFF_SECONDS = 2.0

# Off because the system prompt measures well under the minimum cacheable prefix
# on the extraction model, so marking it cacheable wrote and read nothing. If
# the prompt grows past the threshold, flip this and confirm with
# usage.cache_read_input_tokens on the second call rather than assuming.
CACHE_SYSTEM = False


@dataclass(frozen=True)
class Usage:
    """Token counts for one call."""

    input_tokens: int = 0
    output_tokens: int = 0
    # A non-zero cache_read is the only proof a cache is working.
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @classmethod
    def from_response(cls, usage: Any) -> Usage:
        return cls(
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            cache_creation_tokens=int(
                getattr(usage, "cache_creation_input_tokens", 0) or 0
            ),
        )


def is_retryable(exc: BaseException) -> bool:
    """
    Whether this failure is worth trying again.

    Deliberately a message match rather than a status check: these arrive as
    400s, which every sane retry policy treats as the caller's fault. This
    particular 400 is not.
    """
    text = str(exc).lower()
    return any(needle in text for needle in _RETRYABLE_MESSAGES)


def _schema_format(model: type[BaseModel]) -> dict:
    """Structured-output config from a Pydantic model."""
    return {"format": {"type": "json_schema", "schema": model.model_json_schema()}}


def extraction_params(
    *,
    system: str,
    user_text: str,
    schema_model: type[BaseModel],
    model: str = EXTRACT_MODEL,
    cacheable_system: bool = CACHE_SYSTEM,
) -> dict:
    """
    The request body for one extraction, shared by the live and batch paths so
    they cannot drift.

    No effort or adaptive-thinking setting: the extraction model rejects both,
    and the task is mechanical enough not to want them.
    """
    system_block: Any = system
    if cacheable_system:
        system_block = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
    return {
        "model": model,
        "max_tokens": EXTRACT_MAX_TOKENS,
        "system": system_block,
        "messages": [{"role": "user", "content": user_text}],
        "output_config": _schema_format(schema_model),
    }


async def extract_one(
    api: Any,
    *,
    system: str,
    user_text: str,
    schema_model: type[M],
    model: str = EXTRACT_MODEL,
) -> tuple[M | None, Usage]:
    """
    One live extraction. Returns (parsed, usage), or (None, usage) on a refusal
    or output that does not validate.

    Never raises for content reasons: the caller keeps whatever deterministic
    behaviour it had either way.
    """
    params = extraction_params(
        system=system, user_text=user_text, schema_model=schema_model, model=model
    )
    response = await _create_with_retries(api, params)
    usage = Usage.from_response(getattr(response, "usage", None))

    if getattr(response, "stop_reason", None) == "refusal":
        logger.warning("Extraction refused by the model")
        return None, usage

    text = next(
        (b.text for b in response.content if getattr(b, "type", None) == "text"), None
    )
    if not text:
        return None, usage

    try:
        return schema_model.model_validate_json(text), usage
    except Exception:
        logger.warning("Extraction output failed schema validation", exc_info=True)
        return None, usage


async def _create_with_retries(api: Any, params: dict) -> Any:
    """
    The call, with a short retry on the transients that are not our fault.

    Raises the last error once the attempts are spent, so the caller can record
    the contact as unread and let a later pass pick it back up.
    """
    attempt = 0
    while True:
        try:
            return await api.messages.create(**params)
        except Exception as exc:
            attempt += 1
            if attempt > EXTRACT_RETRIES or not is_retryable(exc):
                raise
            wait = EXTRACT_RETRY_BACKOFF_SECONDS * attempt
            logger.warning(
                "Extraction call failed transiently (%s); retrying in %.0fs "
                "[attempt %s of %s]",
                str(exc)[:120],
                wait,
                attempt,
                EXTRACT_RETRIES,
            )
            await asyncio.sleep(wait)


# --------------------------------------------------------------------------
# Batch path — half price, and the shape a nightly pass would use
# --------------------------------------------------------------------------
def batch_request(
    custom_id: str,
    *,
    system: str,
    user_text: str,
    schema_model: type[BaseModel],
    model: str = EXTRACT_MODEL,
) -> dict:
    """One entry for a Message Batches submission."""
    return {
        "custom_id": custom_id,
        "params": extraction_params(
            system=system, user_text=user_text, schema_model=schema_model, model=model
        ),
    }


async def submit_batch(api: Any, requests: Iterable[dict]) -> str:
    batch = await api.messages.batches.create(requests=list(requests))
    return batch.id


async def batch_status(api: Any, batch_id: str) -> str:
    """`in_progress` | `canceling` | `ended`."""
    batch = await api.messages.batches.retrieve(batch_id)
    return batch.processing_status


async def batch_results(
    api: Any, batch_id: str
) -> AsyncIterator[tuple[str, str | None]]:
    """
    Yields (custom_id, parsed_text_or_None).

    Results arrive in arbitrary order, so callers must key by custom_id and
    never by position. Note that no usage comes back on this path, which is why
    the pipeline reports token counts as absent for a batched run rather than
    estimating them.
    """
    async for result in await api.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            yield result.custom_id, None
            continue
        message = result.result.message
        text = next(
            (b.text for b in message.content if getattr(b, "type", None) == "text"),
            None,
        )
        yield result.custom_id, text
