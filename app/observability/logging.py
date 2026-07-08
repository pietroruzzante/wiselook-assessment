"""structlog configuration: JSON logs with per-request correlation ids.

Configured once at app startup (see app/main.py). Every log line picks up
whatever context is bound via structlog.contextvars for the current
request — request_id from the middleware below, plus session_id bound by
the route handlers — so a full assessment can be traced end-to-end by
grepping one id, without threading it through every function call.
"""

from __future__ import annotations

import contextvars
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import Any

import structlog
from fastapi import Request, Response
from structlog.types import EventDict

from app.config import settings

logger = structlog.get_logger(__name__)

# Matches the session_id path segment directly, since it's needed before
# routing runs (see request_logging_middleware's docstring for why).
_SESSION_PATH_RE = re.compile(r"^/v1/sessions/(?P<session_id>[0-9a-fA-F-]{36})")

# Set only while inside a capture_logs() block; None the rest of the time,
# in which case _capture_processor is a no-op and logging works as normal.
_capture_sink: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "_capture_sink", default=None
)


def _capture_processor(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Mirrors every log event into whatever list capture_logs() bound for
    the current context, on top of the normal stdout JSON output."""
    sink = _capture_sink.get()
    if sink is not None:
        sink.append(dict(event_dict))
    return event_dict


@contextmanager
def capture_logs() -> Iterator[list[dict[str, Any]]]:
    """Collects every log event emitted within this `with` block into a
    list, for the caller to persist however it likes.

    Used by evaluation/run_eval.py to save one JSON file of logs per
    golden-transcript run, alongside the normal stdout output — nothing
    about request-serving logging changes.
    """
    events: list[dict[str, Any]] = []
    token = _capture_sink.set(events)
    try:
        yield events
    finally:
        _capture_sink.reset(token)


def configure_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", level=level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _capture_processor,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


async def request_logging_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Binds a fresh request_id (and session_id, when the URL has one) to
    every log line emitted while handling this request.

    FastAPI's middleware runs the route handler in a separate context
    (BaseHTTPMiddleware), so anything bound *inside* the handler — e.g.
    for a freshly created session_id, which isn't in the URL yet — won't
    be visible back here after call_next returns. Binding it here from
    the URL instead makes sure it's on the summary log line below too.
    """
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=str(uuid.uuid4()))
    if match := _SESSION_PATH_RE.match(request.url.path):
        structlog.contextvars.bind_contextvars(session_id=match.group("session_id"))

    start = time.monotonic()
    response = await call_next(request)
    duration_ms = (time.monotonic() - start) * 1000

    logger.info(
        "http.request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration_ms, 1),
    )
    return response
