"""CLI: runs the golden set through the real graph and the real LLM, then
prints a compact pass/fail report (CLAUDE.md section 8, item 4).

Unlike tests/, this hits the network — it needs a real ANTHROPIC_API_KEY
(see .env.example). Run with: python -m evaluation.run_eval
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from langgraph.types import Command

from app.api.dependencies import thread_config
from app.api.routes import _completed_response, _initial_state
from app.config import settings
from app.domain.constants import DIMENSION_ORDER, PRIMARY_QUESTIONS, Dimension
from app.domain.models import AssessmentResult
from app.graph.builder import build_graph
from app.llm.client import AnthropicStructuredClient
from evaluation.checks import GoldenCase, run_checks
from evaluation.judge import judge_rationale

GOLDEN_DIR = Path(__file__).parent / "golden"


async def run_case(golden: GoldenCase) -> AssessmentResult:
    """Drive one golden conversation through the real graph.

    Each dimension gets its scripted answer repeated if the sufficiency
    gate asks a follow-up (up to the configured budget) — golden answers
    are written to be sufficient on the first try, so this is a safety
    net, not the intended path.
    """
    graph: Any = build_graph()
    session_id = uuid4()
    config = thread_config(session_id)

    state = await graph.ainvoke(_initial_state(session_id), config=config)
    for dimension in DIMENSION_ORDER:
        answer = golden.answers[dimension.value]
        attempts = 0
        while state["current_dimension"] == dimension.value and attempts <= settings.max_followups:
            state = await graph.ainvoke(Command(resume=answer), config=config)
            attempts += 1

    return _completed_response(session_id, state)


async def main() -> int:
    golden_files = sorted(GOLDEN_DIR.glob("*.json"))
    if not golden_files:
        print(f"no golden transcripts found in {GOLDEN_DIR}")
        return 1

    judge_client = AnthropicStructuredClient()
    all_passed = True

    for path in golden_files:
        golden = GoldenCase.model_validate(json.loads(path.read_text()))
        print(f"\n=== {golden.name} ===")

        result = await run_case(golden)
        for check in run_checks(golden, result):
            status = "PASS" if check.passed else "FAIL"
            suffix = f" — {check.detail}" if check.detail else ""
            print(f"  [{status}] {check.name}{suffix}")
            all_passed = all_passed and check.passed

        for dimension in DIMENSION_ORDER:
            scored = getattr(result.profile, dimension.value)
            if scored is None:
                continue
            verdict = await judge_rationale(
                judge_client,
                question=PRIMARY_QUESTIONS[Dimension(dimension.value)],
                answer=golden.answers[dimension.value],
                score=scored.score,
                rationale=scored.rationale,
            )
            status = "PASS" if verdict.grounded else "FAIL"
            print(f"  [{status}] {dimension.value}_rationale_grounded — {verdict.explanation}")
            all_passed = all_passed and verdict.grounded

    print("\nRESULT:", "ALL PASSED" if all_passed else "SOME FAILED")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
