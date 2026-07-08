
"""Async Anthropic client wrapper: bounded concurrency, timeout, retries on
transient errors, and structured-output parsing into Pydantic models.

Callers (graph/nodes.py) ask for a `output_model` and get back a validated
instance of it — they never see raw Anthropic response objects.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, TypeVar

import structlog
from anthropic import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncAnthropic,
    InternalServerError,
    RateLimitError,
)
from anthropic.types import MessageParam, ToolChoiceToolParam, ToolParam
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import settings

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

# Errors worth retrying: transient network/server-side conditions. Anything
# else (bad request, auth, invalid schema) is a bug and should surface
# immediately rather than being retried.
_TRANSIENT_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)


class LLMError(Exception):
    """Base for typed LLM failures — callers (the API layer) catch this
    once and turn it into a clean HTTP error instead of a raw traceback."""


class LLMOutputError(LLMError):
    """Raised when the model's output still fails schema validation after
    one repair attempt."""


class LLMRequestError(LLMError):
    """Raised when the Anthropic API itself rejects the request — auth,
    billing, invalid request, or a transient error that exhausted retries.
    Not a parsing problem, so no repair attempt applies."""


class AnthropicStructuredClient:
    """Thin wrapper around AsyncAnthropic that always returns a validated
    Pydantic model, or raises LLMOutputError.
    """

    def __init__(self, *, max_concurrency: int = 10) -> None:
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def call_structured(
        self,
        *,
        system: str,
        user: str,
        output_model: type[T],
        temperature: float,
        max_tokens: int = 1024,
    ) -> T:
        """Call the model, forcing it to respond via a tool call matching
        output_model's schema, and return a validated instance.

        On the first validation failure, retries once with a stricter
        instruction appended (the "repair" pass). If that also fails,
        raises LLMOutputError — callers should treat this as a typed,
        user-facing error rather than crashing the graph.
        """
        async with self._semaphore:
            try:
                raw = await self._call_with_retries(system, user, output_model, temperature, max_tokens)
            except APIError as exc:
                logger.error(
                    "llm.request_failed",
                    output_model=output_model.__name__,
                    error=str(exc),
                )
                raise LLMRequestError(f"Anthropic API request failed: {exc}") from exc

        try:
            return output_model.model_validate(raw)
        except ValidationError as exc:
            logger.warning(
                "llm.output_validation_failed",
                output_model=output_model.__name__,
                error=str(exc),
            )
            return await self._repair(system, user, output_model, temperature, max_tokens, exc)

    async def _call_with_retries(
        self,
        system: str,
        user: str,
        output_model: type[BaseModel],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        @retry(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(_TRANSIENT_ERRORS),
        )
        async def _do_call() -> dict[str, Any]:
            start = time.monotonic()
            try:
                response = await asyncio.wait_for(
                    self._client.messages.create(
                        model=settings.model_name,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=system,
                        messages=[MessageParam(role="user", content=user)],
                        tools=[_tool_schema(output_model)],
                        tool_choice=ToolChoiceToolParam(type="tool", name=output_model.__name__),
                    ),
                    timeout=settings.llm_timeout,
                )
            finally:
                duration_ms = (time.monotonic() - start) * 1000
                logger.info(
                    "llm.call",
                    model=settings.model_name,
                    output_model=output_model.__name__,
                    duration_ms=round(duration_ms, 1),
                )
            return _extract_tool_input(response, output_model.__name__)

        return await _do_call()

    async def _repair(
        self,
        system: str,
        user: str,
        output_model: type[T],
        temperature: float,
        max_tokens: int,
        exc: ValidationError,
    ) -> T:
        stricter_user = (
            f"{user}\n\n"
            f"Your previous response was invalid: {exc}\n"
            f"Call the `{output_model.__name__}` tool again, with arguments that "
            f"strictly match its schema."
        )
        try:
            raw = await self._call_with_retries(system, stricter_user, output_model, temperature, max_tokens)
        except APIError as exc:
            logger.error(
                "llm.request_failed",
                output_model=output_model.__name__,
                error=str(exc),
                phase="repair",
            )
            raise LLMRequestError(f"Anthropic API request failed during repair: {exc}") from exc

        try:
            return output_model.model_validate(raw)
        except ValidationError as exc2:
            logger.error(
                "llm.output_validation_failed_after_repair",
                output_model=output_model.__name__,
                error=str(exc2),
            )
            raise LLMOutputError(
                f"{output_model.__name__}: model output still invalid after repair attempt"
            ) from exc2


def _tool_schema(output_model: type[BaseModel]) -> ToolParam:
    return ToolParam(
        name=output_model.__name__,
        description=f"Return a {output_model.__name__}.",
        input_schema=output_model.model_json_schema(),
    )


def _extract_tool_input(response: Any, tool_name: str) -> dict[str, Any]:
    for block in response.content:
        if block.type == "tool_use" and block.name == tool_name:
            input_ = block.input
            assert isinstance(input_, dict)
            return input_
    raise LLMOutputError(f"model did not call the {tool_name} tool")
