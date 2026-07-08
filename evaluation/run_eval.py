"""CLI: runs the golden set through the real graph and the real LLM, then
prints a compact pass/fail report (CLAUDE.md section 8, item 4).

Unlike tests/, this hits the network — it needs a real ANTHROPIC_API_KEY
(see .env.example). Run with: python -m evaluation.run_eval
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from langgraph.types import Command

from app.api.dependencies import thread_config
from app.api.routes import _completed_response, _initial_state
from app.config import settings
from app.domain.constants import DIMENSION_ORDER, PRIMARY_QUESTIONS, Dimension
from app.domain.models import AssessmentResult
from app.graph.builder import build_graph
from app.llm.client import AnthropicStructuredClient
from app.observability.logging import capture_logs, configure_logging
from evaluation.checks import GoldenCase, run_checks
from evaluation.judge import judge_rationale

GOLDEN_DIR = Path(__file__).parent / "golden"
LOG_DIR = Path(__file__).parent / "logs"


async def run_case(golden: GoldenCase) -> tuple[UUID, AssessmentResult]:
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

    return session_id, _completed_response(session_id, state)


def _save_session_log(golden_name: str, session_id: UUID, started_at: datetime, events: list[dict[str, Any]]) -> Path:
    """Writes every log event from one golden-case run to its own JSON
    file, so a run can be inspected after the fact without re-running it.

    Named with the run timestamp (sorts chronologically), the golden
    case's name (human-readable), and the session_id — the same
    correlation id used throughout the app's logs, so it doubles as the
    reference to find this run's entries if logs are also being
    collected elsewhere (e.g. shipped to a log aggregator in production).
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    path = LOG_DIR / f"{timestamp}_{golden_name}_{session_id}.json"
    path.write_text(
        json.dumps(
            {
                "golden_case": golden_name,
                "session_id": str(session_id),
                "started_at": started_at.isoformat(),
                "logs": events,
            },
            indent=2,
        )
    )
    return path


async def main() -> int:
    configure_logging()

    golden_files = sorted(GOLDEN_DIR.glob("*.json"))
    if not golden_files:
        print(f"no golden transcripts found in {GOLDEN_DIR}")
        return 1

    judge_client = AnthropicStructuredClient()
    all_passed = True

    for path in golden_files:
        golden = GoldenCase.model_validate(json.loads(path.read_text()))
        print(f"\n=== {golden.name} ===")
        started_at = datetime.now(UTC)

        with capture_logs() as events:
            session_id, result = await run_case(golden)
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

        log_path = _save_session_log(golden.name, session_id, started_at, events)
        print(f"  (logs saved to {log_path})")

    print("\nRESULT:", "ALL PASSED" if all_passed else "SOME FAILED")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
