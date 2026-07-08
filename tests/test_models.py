"""Validation edge cases for the Pydantic schemas.

These are also the structured-output contracts the LLM's responses are
parsed into, so rejecting a bad shape here is what protects the graph
from an unparseable or nonsensical model response.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.domain.models import (
    AssessmentMetadata,
    AssessmentResult,
    BigFiveProfile,
    DimensionScore,
)


def _metadata() -> AssessmentMetadata:
    return AssessmentMetadata(model="claude-sonnet-4-5", prompt_version="v3", created_at=datetime.now(UTC))


@pytest.mark.parametrize("score", [0, 6, -1])
def test_dimension_score_rejects_out_of_range_score(score: int) -> None:
    with pytest.raises(ValidationError):
        DimensionScore(score=score, rationale="A rationale long enough to pass.")


def test_dimension_score_rejects_too_short_rationale() -> None:
    with pytest.raises(ValidationError):
        DimensionScore(score=3, rationale="short")


def test_dimension_score_accepts_valid_input() -> None:
    score = DimensionScore(score=4, rationale="A specific, concrete rationale.")
    assert score.score == 4


def test_big_five_profile_is_complete_only_once_all_five_are_set() -> None:
    profile = BigFiveProfile()
    assert not profile.is_complete()

    for field in ("openness", "conscientiousness", "extraversion", "agreeableness"):
        setattr(profile, field, DimensionScore(score=3, rationale="Concrete evidence given."))
    assert not profile.is_complete()

    profile.neuroticism = DimensionScore(score=3, rationale="Concrete evidence given.")
    assert profile.is_complete()


def test_assessment_result_rejects_incomplete_profile() -> None:
    incomplete_profile = BigFiveProfile(openness=DimensionScore(score=3, rationale="Concrete evidence given."))
    with pytest.raises(ValidationError):
        AssessmentResult(
            session_id=uuid4(),
            profile=incomplete_profile,
            confidence=0.5,
            metadata=_metadata(),
        )


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_assessment_result_rejects_confidence_out_of_range(confidence: float) -> None:
    full_profile = BigFiveProfile(
        **{
            field: DimensionScore(score=3, rationale="Concrete evidence given.")
            for field in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
        }
    )
    with pytest.raises(ValidationError):
        AssessmentResult(
            session_id=uuid4(),
            profile=full_profile,
            confidence=confidence,
            metadata=_metadata(),
        )
