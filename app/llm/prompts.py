"""Versioned prompt templates for the two LLM roles (see CLAUDE.md section 7).

Prompts are treated as reviewable code: every change to the wording below
should bump PROMPT_VERSION, since it flows into AssessmentMetadata and
makes results auditable/reproducible.
"""

from __future__ import annotations

from app.domain.constants import DIMENSION_DESCRIPTORS, Dimension, PRIMARY_QUESTIONS
from app.domain.state import QATurn

PROMPT_VERSION = "v1"


def format_transcript(turns: list[QATurn]) -> str:
    """Render accumulated Q&A turns for a dimension as plain text for the prompt."""
    lines = []
    for i, turn in enumerate(turns, start=1):
        lines.append(f"Q{i}: {turn['question']}")
        lines.append(f"A{i}: {turn['answer']}")
    return "\n".join(lines)


def _rubric_line(dimension: Dimension) -> str:
    high, low = DIMENSION_DESCRIPTORS[dimension]
    return f"High {dimension.value} looks like: {high}. Low {dimension.value} looks like: {low}."


# --- assess_sufficiency -----------------------------------------------------

SUFFICIENCY_SYSTEM_PROMPT = """\
You are the sufficiency gate in a Big Five (OCEAN) personality assessment.

You are shown one OCEAN dimension, its rubric, and the transcript of the
question(s) asked and answer(s) given so far for that dimension only. Decide
whether the transcript already gives enough concrete signal to confidently
assign a 1-5 score for this dimension, or whether a follow-up question is
still needed.

Favor "sufficient" when the answer describes a specific situation and gives a
clear read on where the person falls on the high/low spectrum for this
dimension. Favor "insufficient" when the answer is vague, generic, off-topic,
too short to reveal anything, or could plausibly support either a high or a
low score.

This is a fast, cheap gate, not the final judgment — when genuinely unsure,
prefer "insufficient" so a follow-up can gather more signal, but do not ask
for follow-ups indefinitely if the person has already given a substantive,
on-topic answer.

Call the SufficiencyVerdict tool with:
- sufficient: your verdict
- reason: one sentence explaining what drove it (e.g. what's missing, or what
  makes the existing answer enough)
"""


def build_sufficiency_user_prompt(dimension: Dimension, turns: list[QATurn]) -> str:
    return (
        f"Dimension: {dimension.value}\n"
        f"Rubric: {_rubric_line(dimension)}\n"
        f"Primary question: {PRIMARY_QUESTIONS[dimension]}\n\n"
        f"Transcript so far:\n{format_transcript(turns)}"
    )


# --- score_dimension ---------------------------------------------------------

SCORE_SYSTEM_PROMPT = """\
You are the scorer in a Big Five (OCEAN) personality assessment.

You are shown one OCEAN dimension, its rubric, and the full transcript of
question(s) and answer(s) collected for that dimension. Assign an integer
score from 1 to 5, where 1 matches the "low" end of the rubric and 5 matches
the "high" end.

Your rationale must be directly grounded in what the person actually said —
quote or closely paraphrase specific details from their answer(s) — not a
generic restatement of the rubric. A rationale that could apply to any
transcript for this dimension is not acceptable. If the transcript only
weakly supports a score, say so in the rationale rather than overstating
confidence.

Call the DimensionScore tool with:
- score: integer 1-5
- rationale: 1-3 sentences, grounded in the transcript, justifying the score
"""


def build_score_user_prompt(dimension: Dimension, turns: list[QATurn]) -> str:
    return (
        f"Dimension: {dimension.value}\n"
        f"Rubric: {_rubric_line(dimension)}\n"
        f"Primary question: {PRIMARY_QUESTIONS[dimension]}\n\n"
        f"Transcript so far:\n{format_transcript(turns)}"
    )
