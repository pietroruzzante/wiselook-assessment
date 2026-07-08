"""The three assessment endpoints (see CLAUDE.md section 6 for the contract)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.api.dependencies import get_graph, thread_config
from app.config import settings
from app.domain.constants import DIMENSION_ORDER
from app.domain.models import AssessmentMetadata, AssessmentResult, BigFiveProfile, DimensionScore
from app.domain.state import AssessmentState
from app.llm.prompts import PROMPT_VERSION

router = APIRouter(prefix="/v1")


class ReplyRequest(BaseModel):
    answer: str = Field(min_length=1)


class SessionTurnResponse(BaseModel):
    session_id: UUID
    status: str = "in_progress"
    current_dimension: str
    message: str
    turn: int
    partial_profile: BigFiveProfile | None = None


def _initial_state(session_id: UUID) -> AssessmentState:
    return {
        "session_id": str(session_id),
        "status": "in_progress",
        "current_dimension": DIMENSION_ORDER[0].value,
        "dimension_index": 0,
        "current_dimension_turns": [],
        "followups_used": 0,
        "total_followups_used": 0,
        "last_sufficiency": False,
        "last_sufficiency_reason": "",
        "profile": {},
        "pending_question": "",
        "turn": 0,
        "transcript": [],
        "confidence": 0.0,
    }


def _profile_from_state(raw_profile: dict[str, Any]) -> BigFiveProfile:
    return BigFiveProfile(**{k: DimensionScore(**v) for k, v in raw_profile.items()})


def _in_progress_response(session_id: UUID, state: AssessmentState) -> SessionTurnResponse:
    profile = _profile_from_state(state["profile"])
    return SessionTurnResponse(
        session_id=session_id,
        current_dimension=state["current_dimension"],
        message=state["pending_question"],
        turn=state["turn"],
        partial_profile=profile if state["profile"] else None,
    )


def _completed_response(session_id: UUID, state: AssessmentState) -> AssessmentResult:
    return AssessmentResult(
        session_id=session_id,
        profile=_profile_from_state(state["profile"]),
        confidence=state["confidence"],
        metadata=AssessmentMetadata(
            model=settings.model_name,
            prompt_version=PROMPT_VERSION,
            created_at=datetime.now(UTC),
        ),
    )


@router.post("/sessions", response_model=SessionTurnResponse, response_model_exclude_none=True)
async def create_session(graph: Any = Depends(get_graph)) -> SessionTurnResponse:
    session_id = uuid4()
    structlog.contextvars.bind_contextvars(session_id=str(session_id))
    state = await graph.ainvoke(_initial_state(session_id), config=thread_config(session_id))
    return _in_progress_response(session_id, state)


@router.post(
    "/sessions/{session_id}/reply",
    response_model=SessionTurnResponse | AssessmentResult,
    response_model_exclude_none=True,
)
async def reply(
    session_id: UUID, body: ReplyRequest, graph: Any = Depends(get_graph)
) -> SessionTurnResponse | AssessmentResult:
    # session_id is already bound to the log context by the request
    # middleware, which parses it straight from the URL (see
    # observability/logging.py for why that's done there, not here).
    config = thread_config(session_id)
    snapshot = await graph.aget_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="session not found")

    state = await graph.ainvoke(Command(resume=body.answer), config=config)
    if state["status"] == "completed":
        return _completed_response(session_id, state)
    return _in_progress_response(session_id, state)


@router.get(
    "/sessions/{session_id}",
    response_model=SessionTurnResponse | AssessmentResult,
    response_model_exclude_none=True,
)
async def get_session(
    session_id: UUID, graph: Any = Depends(get_graph)
) -> SessionTurnResponse | AssessmentResult:
    # session_id is already bound to the log context by the request
    # middleware (see the comment in reply(), above).
    config = thread_config(session_id)
    snapshot = await graph.aget_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="session not found")

    state = snapshot.values
    if state["status"] == "completed":
        return _completed_response(session_id, state)
    return _in_progress_response(session_id, state)
