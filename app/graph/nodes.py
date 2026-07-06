"""LangGraph nodes: the actions in the assessment conversation graph.

Node functions take AssessmentState and return a partial dict of updates
(LangGraph merges these into the running state). LLM calls go through a
lazily-constructed shared client (see get_client) so tests can monkeypatch
it with a fake, without needing a real ANTHROPIC_API_KEY or touching node
logic.
"""

from __future__ import annotations

from typing import Any

import structlog
from langgraph.types import interrupt

from app.config import settings
from app.domain.constants import DIMENSION_ORDER, PRIMARY_QUESTIONS, Dimension
from app.domain.models import DimensionScore, SufficiencyVerdict
from app.domain.state import AssessmentState, QATurn
from app.llm.client import AnthropicStructuredClient
from app.llm.prompts import (
    SCORE_SYSTEM_PROMPT,
    SUFFICIENCY_SYSTEM_PROMPT,
    build_score_user_prompt,
    build_sufficiency_user_prompt,
)

logger = structlog.get_logger(__name__)

_client: AnthropicStructuredClient | None = None


def get_client() -> AnthropicStructuredClient:
    global _client
    if _client is None:
        _client = AnthropicStructuredClient()
    return _client


async def ask_question(state: AssessmentState) -> dict[str, Any]:
    """Emit the primary question for the current dimension, resetting the
    per-dimension follow-up counter and turn accumulator."""
    dimension = Dimension(state["current_dimension"])
    logger.info("graph.ask_question", dimension=dimension.value)
    return {
        "pending_question": PRIMARY_QUESTIONS[dimension],
        "current_dimension_turns": [],
        "followups_used": 0,
        "turn": state["turn"] + 1,
    }


async def wait_for_input(state: AssessmentState) -> dict[str, Any]:
    """Suspend the graph until the caller resumes with Command(resume=answer)."""
    answer = interrupt({"question": state["pending_question"]})
    turn: QATurn = {"question": state["pending_question"], "answer": answer}
    return {
        "current_dimension_turns": [*state["current_dimension_turns"], turn],
        "transcript": [*state["transcript"], turn],
    }


async def assess_sufficiency(state: AssessmentState) -> dict[str, Any]:
    """LLM gate: do we have enough signal to score the current dimension?"""
    dimension = Dimension(state["current_dimension"])
    verdict = await get_client().call_structured(
        system=SUFFICIENCY_SYSTEM_PROMPT,
        user=build_sufficiency_user_prompt(dimension, state["current_dimension_turns"]),
        output_model=SufficiencyVerdict,
        temperature=0.0,
    )
    logger.info(
        "graph.assess_sufficiency",
        dimension=dimension.value,
        sufficient=verdict.sufficient,
        reason=verdict.reason,
    )
    return {
        "last_sufficiency": verdict.sufficient,
        "last_sufficiency_reason": verdict.reason,
    }


async def ask_followup(state: AssessmentState) -> dict[str, Any]:
    """Deterministic follow-up: probes the gap assess_sufficiency identified.

    Not a third LLM role (CLAUDE.md section 7 names exactly two) — this
    turns the sufficiency verdict's `reason` into a direct question.
    """
    reason = state["last_sufficiency_reason"]
    dimension = Dimension(state["current_dimension"])
    question = f"Could you say more? Specifically: {reason}"
    logger.info("graph.ask_followup", dimension=dimension.value)
    return {
        "pending_question": question,
        "followups_used": state["followups_used"] + 1,
        "total_followups_used": state["total_followups_used"] + 1,
        "turn": state["turn"] + 1,
    }


async def score_dimension(state: AssessmentState) -> dict[str, Any]:
    """LLM: score the current dimension from its accumulated turns, then
    advance to the next dimension (if any remain)."""
    dimension = Dimension(state["current_dimension"])
    score = await get_client().call_structured(
        system=SCORE_SYSTEM_PROMPT,
        user=build_score_user_prompt(dimension, state["current_dimension_turns"]),
        output_model=DimensionScore,
        temperature=0.0,
    )
    logger.info("graph.score_dimension", dimension=dimension.value, score=score.score)

    profile = {
        **state["profile"],
        dimension.value: {"score": score.score, "rationale": score.rationale},
    }
    updates: dict[str, Any] = {"profile": profile}

    next_index = state["dimension_index"] + 1
    if next_index < len(DIMENSION_ORDER):
        updates["dimension_index"] = next_index
        updates["current_dimension"] = DIMENSION_ORDER[next_index].value

    return updates


async def finalize(state: AssessmentState) -> dict[str, Any]:
    """Mark the session complete and compute the confidence heuristic.

    Heuristic, not a calibrated probability (documented in the README per
    CLAUDE.md section 14): fewer follow-ups needed overall implies clearer
    signal, so confidence scales down with total follow-ups used relative
    to the maximum possible across all five dimensions.
    """
    max_possible_followups = len(DIMENSION_ORDER) * settings.max_followups
    if max_possible_followups > 0:
        confidence = 1.0 - (state["total_followups_used"] / max_possible_followups)
    else:
        confidence = 1.0
    confidence = max(0.0, min(1.0, confidence))

    logger.info("graph.finalize", confidence=confidence)
    return {"status": "completed", "confidence": confidence}
