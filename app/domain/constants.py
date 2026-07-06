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

# Grounded in day-to-day work scenarios so the answer draws on lived
# experience rather than abstract self-description.
PRIMARY_QUESTIONS: dict[Dimension, str] = {
    Dimension.OPENNESS: (
        "Tell me about a time your team was asked to try a new tool, process, "
        "or approach that replaced something familiar. How did you react, and "
        "what did you do?"
    ),
    Dimension.CONSCIENTIOUSNESS: (
        "Walk me through how you handled a project with a hard deadline and "
        "several moving parts. How did you plan the work and keep it on track?"
    ),
    Dimension.EXTRAVERSION: (
        "Describe a typical week at work in terms of meetings, team "
        "interactions, and heads-down solo work. Which parts energize you and "
        "which drain you?"
    ),
    Dimension.AGREEABLENESS: (
        "Tell me about a time you disagreed with a colleague or manager on how "
        "something should be done. How did you handle the disagreement?"
    ),
    Dimension.NEUROTICISM: (
        "Think of a time something went wrong at work unexpectedly, like a "
        "production incident, a missed deadline, or critical feedback. How did "
        "you react in the moment and afterward?"
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
