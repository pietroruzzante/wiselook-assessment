"""LLM-as-judge (CLAUDE.md section 8, item 3): verdicts whether a
dimension's rationale is actually grounded in what the person said,
catching plausible-but-unfounded scoring that a range check can't.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.llm.client import AnthropicStructuredClient

JUDGE_SYSTEM_PROMPT = """\
You are auditing a Big Five personality scorer. You are shown the question
asked, the person's answer, the score assigned (1-5), and the rationale
given for that score. Decide whether the rationale is grounded in what the
person actually said — it should reference or closely paraphrase specific
details from the answer, not just assert a plausible-sounding conclusion.

Favor grounded=false if the rationale could apply to almost any answer, or
if it asserts something not actually present in the answer.
"""


class JudgeVerdict(BaseModel):
    grounded: bool
    explanation: str = Field(min_length=1)


def build_judge_user_prompt(question: str, answer: str, score: int, rationale: str) -> str:
    return (
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Score given: {score}\n"
        f"Rationale given: {rationale}\n\n"
        "Is this rationale grounded in the answer?"
    )


async def judge_rationale(
    client: AnthropicStructuredClient, *, question: str, answer: str, score: int, rationale: str
) -> JudgeVerdict:
    return await client.call_structured(
        system=JUDGE_SYSTEM_PROMPT,
        user=build_judge_user_prompt(question, answer, score, rationale),
        output_model=JudgeVerdict,
        temperature=0.0,
    )
