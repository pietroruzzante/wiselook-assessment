"""Pydantic schemas shared by the API, the graph nodes, and the LLM client.

These models serve double duty:
  - they are the *structured output* contracts the LLM calls are parsed
    into (see app/llm/client.py), and
  - they are the API response contracts (see app/api/routes.py).

Keep this module free of any LangGraph / FastAPI / Anthropic imports —
domain models are pure and everything else depends on them, not the
other way around.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from app.domain.constants import DIMENSION_ORDER, Dimension


class DimensionScore(BaseModel):
    """A single scored OCEAN dimension.

    This is also the structured-output contract for the `score_dimension`
    LLM call — the LLM is asked to return exactly
    this shape.
    """

    score: int = Field(ge=1, le=5)
    rationale: str = Field(min_length=1)

    @field_validator("rationale")
    @classmethod
    def _rationale_must_be_substantive(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 10:
            raise ValueError("rationale is too short to be a real justification")
        return value


class SufficiencyVerdict(BaseModel):
    """Structured output contract for the `assess_sufficiency` LLM call."""

    sufficient: bool
    reason: str = Field(min_length=1)


class BigFiveProfile(BaseModel):
    """The five scored dimensions. Fields are optional individually so this
    same model can represent a `partial_profile` (mid-assessment) or the
    final complete `profile`.
    """

    openness: DimensionScore | None = None
    conscientiousness: DimensionScore | None = None
    extraversion: DimensionScore | None = None
    agreeableness: DimensionScore | None = None
    neuroticism: DimensionScore | None = None

    def is_complete(self) -> bool:
        """True once all five dimensions are scored."""
        return all(
            getattr(self, dimension.value) is not None for dimension in DIMENSION_ORDER
        )

    def set_score(self, dimension: Dimension, score: DimensionScore) -> None:
        """Set the score for a given dimension by name."""
        setattr(self, dimension.value, score)


class AssessmentMetadata(BaseModel):
    """Traceability metadata attached to every completed result."""

    model: str
    prompt_version: str
    created_at: datetime


class AssessmentResult(BaseModel):
    """The final `POST /v1/sessions/{id}/reply` response body once the
    assessment is complete."""

    session_id: UUID
    status: Literal["completed"] = "completed"
    profile: BigFiveProfile
    confidence: float = Field(ge=0, le=1)
    metadata: AssessmentMetadata

    @model_validator(mode="after")
    def _profile_must_be_complete(self) -> AssessmentResult:
        if not self.profile.is_complete():
            raise ValueError("AssessmentResult requires a fully scored profile")
        return self
