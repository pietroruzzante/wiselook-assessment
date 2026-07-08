"""Shared test fixtures.

The only thing worth faking is the LLM: everything else (graph, API,
checkpointer) is real, so tests exercise the actual routing and contract
logic instead of a reimplementation of it.
"""

from __future__ import annotations

import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from collections.abc import AsyncIterator
from pathlib import Path
from typing import TypeVar, cast

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel

import app.graph.nodes as nodes_module
import app.observability.logging as logging_module
from app.api.dependencies import get_graph
from app.domain.models import DimensionScore, SufficiencyVerdict
from app.graph.builder import build_graph
from app.main import app

T = TypeVar("T", bound=BaseModel)


@pytest.fixture(autouse=True)
def _isolate_session_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The request middleware writes real per-session log files as a side
    effect of serving a request (see app/observability/logging.py) — the
    API tests below go through that same middleware, so without this
    they'd litter the repo's logs/sessions/ with test-run files instead
    of real chat sessions."""
    monkeypatch.setattr(logging_module, "SESSION_LOG_DIR", tmp_path / "logs" / "sessions")


class FakeLLMClient:
    """Deterministic stand-in for AnthropicStructuredClient.

    `sufficiency_sequence` lets a test script the sufficiency gate across
    successive calls (e.g. insufficient once, then sufficient); the last
    value repeats once the sequence is exhausted.
    """

    def __init__(
        self,
        sufficiency_sequence: tuple[bool, ...] = (True,),
        score: int = 3,
        rationale: str = "The answer describes a concrete, specific reaction.",
    ) -> None:
        self._sufficiency_sequence = list(sufficiency_sequence)
        self.score = score
        self.rationale = rationale

    async def call_structured(
        self,
        *,
        system: str,
        user: str,
        output_model: type[T],
        temperature: float,
        max_tokens: int = 1024,
    ) -> T:
        if output_model is SufficiencyVerdict:
            sufficient = (
                self._sufficiency_sequence.pop(0)
                if len(self._sufficiency_sequence) > 1
                else self._sufficiency_sequence[0]
            )
            return cast(T, SufficiencyVerdict(sufficient=sufficient, reason="needs a concrete example"))
        if output_model is DimensionScore:
            return cast(T, DimensionScore(score=self.score, rationale=self.rationale))
        raise AssertionError(f"unexpected output_model: {output_model}")


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLMClient:
    client = FakeLLMClient()
    monkeypatch.setattr(nodes_module, "get_client", lambda: client)
    return client


@pytest.fixture
async def api_client(fake_llm: FakeLLMClient) -> AsyncIterator[AsyncClient]:
    # One graph instance per test, reused across requests within it — the
    # dependency override must return the *same* instance so its
    # MemorySaver keeps session state between turns.
    graph = build_graph()
    app.dependency_overrides[get_graph] = lambda: graph
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()
