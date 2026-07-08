"""End-to-end tests of the three /v1/sessions endpoints against the real
graph, with only the LLM faked (see conftest.py). These are the tests that
prove the API contract in CLAUDE.md section 6 actually holds.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from httpx import AsyncClient

from app.domain.constants import DIMENSION_ORDER


async def test_create_session_returns_first_question(api_client: AsyncClient) -> None:
    response = await api_client.post("/v1/sessions")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "in_progress"
    assert body["current_dimension"] == DIMENSION_ORDER[0].value
    assert body["turn"] == 1
    assert body["message"]


async def test_full_assessment_completes_with_valid_profile(api_client: AsyncClient) -> None:
    session = (await api_client.post("/v1/sessions")).json()
    session_id = session["session_id"]

    body = session
    for _ in DIMENSION_ORDER:
        response = await api_client.post(
            f"/v1/sessions/{session_id}/reply", json={"answer": "A specific, concrete answer."}
        )
        assert response.status_code == 200
        body = response.json()

    assert body["status"] == "completed"
    for dimension in DIMENSION_ORDER:
        score = body["profile"][dimension.value]
        assert 1 <= score["score"] <= 5
        assert score["rationale"]
    assert 0 <= body["confidence"] <= 1
    assert body["metadata"]["prompt_version"]
    assert body["metadata"]["model"]


async def test_reply_to_unknown_session_returns_404(api_client: AsyncClient) -> None:
    response = await api_client.post(f"/v1/sessions/{uuid4()}/reply", json={"answer": "hi"})
    assert response.status_code == 404


async def test_insufficient_answer_triggers_followup_before_scoring(
    monkeypatch: pytest.MonkeyPatch, api_client: AsyncClient
) -> None:
    import app.graph.nodes as nodes_module

    from tests.conftest import FakeLLMClient

    client = FakeLLMClient(sufficiency_sequence=(False, True))
    monkeypatch.setattr(nodes_module, "get_client", lambda: client)

    session = (await api_client.post("/v1/sessions")).json()
    session_id = session["session_id"]
    first_dimension = session["current_dimension"]

    followup = (
        await api_client.post(f"/v1/sessions/{session_id}/reply", json={"answer": "professionally"})
    ).json()
    assert followup["current_dimension"] == first_dimension
    assert followup["turn"] == 2

    scored = (
        await api_client.post(f"/v1/sessions/{session_id}/reply", json={"answer": "A concrete example."})
    ).json()
    assert scored["current_dimension"] != first_dimension
