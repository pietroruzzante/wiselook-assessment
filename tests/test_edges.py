"""Unit tests for the graph's one genuine decision point: the conditional
edges. Pure functions over AssessmentState, so no LLM or graph needed.
"""

from __future__ import annotations

from typing import cast

import pytest

from app.config import settings
from app.domain.state import AssessmentState
from app.graph.edges import route_after_score, route_after_sufficiency


@pytest.fixture(autouse=True)
def _fixed_followup_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_followups", 2)


def test_sufficient_routes_straight_to_scoring() -> None:
    state = cast(AssessmentState, {"last_sufficiency": True, "followups_used": 0})
    assert route_after_sufficiency(state) == "score_dimension"


def test_insufficient_with_budget_left_asks_followup() -> None:
    state = cast(AssessmentState, {"last_sufficiency": False, "followups_used": 1})
    assert route_after_sufficiency(state) == "ask_followup"


def test_insufficient_with_budget_exhausted_scores_anyway() -> None:
    state = cast(AssessmentState, {"last_sufficiency": False, "followups_used": 2})
    assert route_after_sufficiency(state) == "score_dimension"


def test_route_after_score_continues_when_dimensions_remain() -> None:
    state = cast(AssessmentState, {"profile": {"openness": {"score": 3, "rationale": "x"}}})
    assert route_after_score(state) == "ask_question"


def test_route_after_score_finalizes_once_all_five_scored() -> None:
    state = cast(
        AssessmentState,
        {
            "profile": {
                "openness": {"score": 3, "rationale": "x"},
                "conscientiousness": {"score": 3, "rationale": "x"},
                "extraversion": {"score": 3, "rationale": "x"},
                "agreeableness": {"score": 3, "rationale": "x"},
                "neuroticism": {"score": 3, "rationale": "x"},
            }
        },
    )
    assert route_after_score(state) == "finalize"
