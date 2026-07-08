"""Deterministic pass/fail checks run against a completed AssessmentResult.

These are the cheap, schema-level checks (CLAUDE.md section 8, item 2) —
the LLM-as-judge in judge.py catches the subtler failure (plausible but
unfounded rationale) that a range check can't.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.domain.constants import DIMENSION_ORDER
from app.domain.models import AssessmentResult


class GoldenCase(BaseModel):
    """One fixed conversation: a scripted answer per dimension, and the
    score range a reasonable rater would assign for that answer."""

    name: str
    answers: dict[str, str]
    expected_ranges: dict[str, tuple[int, int]]


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


def run_checks(golden: GoldenCase, result: AssessmentResult) -> list[CheckResult]:
    missing = [d.value for d in DIMENSION_ORDER if getattr(result.profile, d.value) is None]
    checks = [
        CheckResult(
            name="all_dimensions_present",
            passed=not missing,
            detail=f"missing: {missing}" if missing else "",
        ),
        CheckResult(name="confidence_in_range", passed=0.0 <= result.confidence <= 1.0),
        CheckResult(
            name="metadata_complete",
            passed=bool(
                result.metadata.model and result.metadata.prompt_version and result.metadata.created_at
            ),
        ),
    ]

    for dimension, (low, high) in golden.expected_ranges.items():
        scored = getattr(result.profile, dimension)
        if scored is None:
            checks.append(CheckResult(name=f"{dimension}_in_expected_range", passed=False, detail="not scored"))
            continue
        checks.append(
            CheckResult(
                name=f"{dimension}_in_expected_range",
                passed=low <= scored.score <= high,
                detail=f"got {scored.score}, expected [{low}, {high}]",
            )
        )

    return checks
