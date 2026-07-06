"""Conditional edge functions: the graph's routing decisions.

Each function inspects AssessmentState and returns the name of the next
node — kept separate from nodes.py so the branching logic (the whole
reason we're using LangGraph over a plain loop) is easy to read and test
in isolation.
"""

from __future__ import annotations

from typing import Literal

from app.config import settings
from app.domain.constants import DIMENSION_ORDER
from app.domain.state import AssessmentState


def route_after_sufficiency(state: AssessmentState) -> Literal["ask_followup", "score_dimension"]:
    """sufficient -> score now. insufficient -> follow up, unless the
    per-dimension follow-up budget is already exhausted."""
    if state["last_sufficiency"]:
        return "score_dimension"
    if state["followups_used"] < settings.max_followups:
        return "ask_followup"
    return "score_dimension"


def route_after_score(state: AssessmentState) -> Literal["ask_question", "finalize"]:
    """All five dimensions scored -> finalize. Otherwise ask the next question."""
    if len(state["profile"]) >= len(DIMENSION_ORDER):
        return "finalize"
    return "ask_question"
