"""OCEAN dimension definitions and the fixed question bank.

This module is pure data: no logic, no I/O. It is the single source of
truth for "what dimensions exist, in what order, and what do we ask."
"""

from __future__ import annotations

from enum import StrEnum


class Dimension(StrEnum):
    """The Big Five / OCEAN dimensions, as string values (used directly in
    API responses and LLM prompts, so keep these lowercase and stable)."""

    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"


DIMENSION_ORDER: list[Dimension] = [
    Dimension.OPENNESS,
    Dimension.CONSCIENTIOUSNESS,
    Dimension.EXTRAVERSION,
    Dimension.AGREEABLENESS,
    Dimension.NEUROTICISM,
]

# Situational-judgment style: each question narrates a concrete, third-person
# workplace scenario (what a colleague/manager did or said, what happened as
# a result) and then asks how the person would react. The scenario itself
# supplies the specificity, so a generic reply ("I'd handle it professionally")
# stands out clearly as insufficient against a concrete prompt.
PRIMARY_QUESTIONS: dict[Dimension, str] = {
    Dimension.OPENNESS: (
        "One of your colleagues just came back from a conference and told the "
        "team about a completely different project-management methodology "
        "she wants to trial — it would mean dropping the workflow you've all "
        "used for two years, with no guarantee it's actually better. She's "
        "already posted in the team channel asking who wants to pilot it "
        "starting Monday. How would you react, and what would you do?"
    ),
    Dimension.CONSCIENTIOUSNESS: (
        "A teammate just went on unexpected leave for two weeks, right in the "
        "middle of a project with a hard client deadline next Friday. Their "
        "work — three tasks the rest of the project depends on — now has to "
        "be picked up by the remaining team, with no extra time added to the "
        "schedule. How would you handle the situation?"
    ),
    Dimension.EXTRAVERSION: (
        "Your manager just announced that starting next quarter, most "
        "cross-team coordination will move from written updates to daily "
        "stand-up calls and more frequent in-person syncs, on top of your "
        "existing meeting load. A few colleagues have already pushed back, "
        "saying it will eat into their focus time. How would you react to "
        "this change, and how would it affect the way you work?"
    ),
    Dimension.AGREEABLENESS: (
        "In a meeting about a shared project, a colleague pushes back hard on "
        "your proposed approach in front of the rest of the team, argues for a "
        "completely different direction, and dismisses your reasoning as "
        "overly cautious. A decision has to be made by the end of the "
        "meeting. How would you respond in the room, and afterward?"
    ),
    Dimension.NEUROTICISM: (
        "You just found out that a colleague's mistake caused a bug to reach "
        "production, and it's now affecting several customers. Your manager "
        "messages you and the colleague asking for an explanation and a fix, "
        "right in the middle of your own deadline crunch. How would you react "
        "in that moment, and how would you handle the hours that follow?"
    ),
}

MAX_FOLLOWUPS_DEFAULT: int = 3

# (high, low) trait anchors per CLAUDE.md's domain spec table — given, not
# invented. Used to give the LLM the same rubric a human rater would use.
DIMENSION_DESCRIPTORS: dict[Dimension, tuple[str, str]] = {
    Dimension.OPENNESS: ("creative, curious, open", "practical, conventional"),
    Dimension.CONSCIENTIOUSNESS: ("organized, disciplined", "flexible, spontaneous"),
    Dimension.EXTRAVERSION: ("sociable, energetic", "reserved, independent"),
    Dimension.AGREEABLENESS: ("cooperative, empathetic", "competitive, direct"),
    Dimension.NEUROTICISM: ("sensitive, reactive", "stable, calm"),
}
