"""Tests for the node logic that doesn't involve an LLM call and isn't
otherwise exercised by the API happy path (test_api.py's full run never
uses a follow-up, so confidence there is always 1.0).
"""

from __future__ import annotations

from typing import cast

import pytest

from app.config import settings
from app.domain.state import AssessmentState
from app.graph.nodes import ask_followup, finalize


async def test_ask_followup_increments_both_followup_counters() -> None:
    state = cast(
        AssessmentState,
        {
            "last_sufficiency_reason": "too vague",
            "current_dimension": "openness",
            "followups_used": 0,
            "total_followups_used": 1,
            "turn": 2,
        },
    )
    updates = await ask_followup(state)
    assert updates["followups_used"] == 1
    assert updates["total_followups_used"] == 2
    assert updates["turn"] == 3


async def test_ask_followup_never_leaks_the_sufficiency_reason_to_the_user() -> None:
    """assess_sufficiency's `reason` names the trait and the model's read
    on the answer (e.g. "could reflect either genuine curiosity (high
    openness)...") — if that ever ends up in pending_question, the
    assessment stops being blind and a user could game later answers."""
    state = cast(
        AssessmentState,
        {
            "last_sufficiency_reason": "could reflect either genuine curiosity (high openness) or impulsive enthusiasm",
            "current_dimension": "openness",
            "followups_used": 0,
            "total_followups_used": 0,
            "turn": 1,
        },
    )
    updates = await ask_followup(state)
    assert "openness" not in updates["pending_question"].lower()
    assert "curiosity" not in updates["pending_question"].lower()


async def test_finalize_confidence_drops_with_more_followups_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_followups", 2)

    no_followups = await finalize(cast(AssessmentState, {"total_followups_used": 0}))
    some_followups = await finalize(cast(AssessmentState, {"total_followups_used": 5}))

    assert no_followups["confidence"] == 1.0
    assert 0.0 <= some_followups["confidence"] < no_followups["confidence"]
