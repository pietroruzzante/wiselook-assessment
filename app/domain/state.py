"""The LangGraph state shape.

This TypedDict is the single object threaded through every node in
app/graph/nodes.py and persisted by the checkpointer between HTTP turns.
Keep it a flat, serializable TypedDict — no Pydantic models with methods,
no non-JSON-serializable values, since the checkpointer may need to
(de)serialize it.
"""

from __future__ import annotations
from typing import Literal, TypedDict
from app.domain.constants import Dimension


class QATurn(TypedDict):
    """One question/answer exchange."""

    question: str
    answer: str


class ScoredDimension(TypedDict):
    """Plain, serializable mirror of app.domain.models.DimensionScore.

    State stays flat/plain data (no Pydantic instances) so the
    checkpointer can (de)serialize it freely; conversion to DimensionScore
    happens at the API boundary.
    """

    score: int
    rationale: str


class AssessmentState(TypedDict):
    session_id: str
    status: Literal["in_progress", "completed"]

    current_dimension: Dimension
    dimension_index: int

    # Q&A turns for the dimension currently being scored. Reset to []
    # each time we advance to the next dimension.
    current_dimension_turns: list[QATurn]
    followups_used: int

    # Scored so far, keyed by Dimension.value; only present once scored.
    profile: dict[str, ScoredDimension]

    # The question or follow-up just emitted, awaiting the user's reply
    # via wait_for_input's interrupt().
    pending_question: str

    # Monotonic count of questions/follow-ups emitted so far this session
    # (the API's `turn` field) — distinct from dimension_index, which only
    # advances once per dimension.
    turn: int

    # Full Q&A history across all dimensions, never reset. Used by
    # GET /v1/sessions/{id} to rehydrate the whole conversation for the UI.
    transcript: list[QATurn]
