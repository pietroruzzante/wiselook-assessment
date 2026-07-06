# Wiselook Technical Test — Build Spec

This document is the single source of truth for building the Wiselook AI Engineer technical assessment. Read it fully before writing any code.

---

## 1. What we're building

A **conversational service that infers a Big Five (OCEAN) personality profile** from a user's answers, delivered as a FastAPI backend with a minimal chat UI. This is a technical test for the AI Engineer role at Wiselook, a startup building an LLM-powered conversational engine for competency and personality assessment in both text and voice.

The domain (Big Five) is deliberately simple and given — **we are not being evaluated on psychometric depth**. We are evaluated on **engineering judgment**: service/contract design, code structure, robustness, structured LLM outputs, testing/evaluation, async/concurrency, observability, and a design note on extending to voice.

We are targeting **Option C** (the highest tier), but built as a solid, well-scoped slice rather than a sprawling half-finished repo. The guiding principle from the brief: *"A small, solid slice with well-argued decisions beats a large, half-done repo."*

---

## 2. Domain spec (given — do not invent)

The Big Five / OCEAN model. Each dimension is scored **1–5** with a short rationale.

| Dimension | High (5) | Low (1) |
| --- | --- | --- |
| Openness | Creative, curious, open | Practical, conventional |
| Conscientiousness | Organized, disciplined | Flexible, spontaneous |
| Extraversion | Sociable, energetic | Reserved, independent |
| Agreeableness | Cooperative, empathetic | Competitive, direct |
| Neuroticism | Sensitive, reactive | Stable, calm |

The assessment walks through the five dimensions in order, asking one primary question per dimension plus up to one or two adaptive follow-ups when the answer isn't informative enough, then produces a final validated profile.

---

## 3. Architecture overview

The core is a **LangGraph state machine** that drives a multi-turn conversation, wrapped by a **FastAPI** service, with an **evaluation harness** and a static chat UI.

### Why LangGraph

The conversation flow has one genuine decision point: after each user answer, decide whether we have **enough signal to score the current dimension** or whether we need a **follow-up**. That conditional branching, plus human-in-the-loop multi-turn state, is exactly what LangGraph's conditional edges + `interrupt()` + checkpointer are designed for. We use LangGraph for that reason — not as decoration. Be ready to justify this in the interview: a plain loop would also work, but LangGraph gives us persisted per-session state, a clean separation between "decide" and "act" nodes, and a trivial path to Postgres-backed persistence in production.

### Graph shape

```
START
  → ask_question        (emit the primary question for current_dimension)
  → wait_for_input      (interrupt — suspend until user replies)
  → assess_sufficiency  (LLM: is this answer enough to score this dimension?)
       ├─ insufficient & followups_left → ask_followup → wait_for_input
       └─ sufficient (or followup budget exhausted) → score_dimension
  → score_dimension     (LLM: produce {score, rationale} for the dimension)
       ├─ more dimensions remain → ask_question (next)
       └─ all five scored        → finalize → END
```

`assess_sufficiency` and `score_dimension` are both **structured-output LLM calls** (Pydantic-validated). `wait_for_input` uses `langgraph.types.interrupt` so the graph suspends between HTTP turns.

### Multi-turn / state persistence

Each conversation is a LangGraph **thread** identified by a `session_id`. State is persisted by a **checkpointer** (`MemorySaver` for the test; note in the README that `AsyncPostgresSaver` is a one-line swap for production). Each user turn resumes the graph from the checkpoint via `Command(resume=...)`.

---

## 4. Repo structure

```
wiselook-assessment/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app, lifespan, static mount, exception handlers
│   ├── config.py               # Settings via pydantic-settings (env vars)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py           # POST /sessions, POST /sessions/{id}/reply, GET /sessions/{id}
│   │   └── dependencies.py     # graph/app singletons, request-id, etc.
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── models.py           # Pydantic: DimensionScore, BigFiveProfile, AssessmentResult
│   │   ├── state.py            # LangGraph AssessmentState (TypedDict)
│   │   └── constants.py        # OCEAN dimensions, question bank, followup budget
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── builder.py          # build + compile the StateGraph with checkpointer
│   │   ├── nodes.py            # ask_question, assess_sufficiency, score_dimension, finalize
│   │   └── edges.py            # conditional edge functions
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py           # async Anthropic client wrapper: retries, timeout, structured output
│   │   └── prompts.py          # versioned prompt templates (prompt_version constant)
│   └── observability/
│       ├── __init__.py
│       └── logging.py          # structlog config, request-id correlation, per-call latency
├── static/
│   └── index.html              # provided chat UI (already built)
├── evaluation/
│   ├── __init__.py
│   ├── golden/                 # golden transcripts (JSON): input turns + expected ranges
│   │   └── *.json
│   ├── checks.py               # schema/range/consistency checks
│   ├── judge.py                # LLM-as-judge (structured verdict)
│   └── run_eval.py             # CLI: run harness over golden set, print report
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # fixtures: test client, mocked LLM
│   ├── test_models.py          # Pydantic validation + parsing edge cases
│   ├── test_nodes.py           # node logic with mocked LLM
│   ├── test_edges.py           # conditional routing
│   └── test_api.py             # endpoint tests (mocked LLM, no network)
├── .env.example
├── .gitignore
├── pyproject.toml              # deps + tool config (ruff, pytest, mypy)
├── README.md
└── CLAUDE.md                   # this file
```

---

## 5. Tech stack

- **Python 3.11+**
- **FastAPI** + **uvicorn** — service layer
- **LangGraph** — conversation orchestration, checkpointer, interrupts
- **Anthropic Python SDK** (async) — the LLM (use `claude-sonnet-*`; make the model configurable via env)
- **Pydantic v2** + **pydantic-settings** — schemas, validation, config
- **structlog** — structured JSON logging
- **tenacity** — retries with exponential backoff (or hand-rolled; justify)
- **pytest** + **pytest-asyncio** + **httpx** — testing
- **ruff** (lint + format) and **mypy** (types) — quality gates

Keep dependencies lean. No database, no message queue, no frontend framework — in-memory state and a static HTML page are correct for this scope.

---

## 6. API contract

Design the contract carefully — it's explicitly evaluated. Version it (`/v1`).

### `POST /v1/sessions`
Creates a new assessment session and returns the first question.

Response:
```json
{
  "session_id": "uuid",
  "status": "in_progress",
  "current_dimension": "openness",
  "message": "First question text...",
  "turn": 1
}
```

### `POST /v1/sessions/{session_id}/reply`
Submits a user answer; resumes the graph; returns either the next question/follow-up or the final profile.

Request:
```json
{ "answer": "user's free-text answer" }
```

Response (still going):
```json
{
  "session_id": "uuid",
  "status": "in_progress",
  "current_dimension": "conscientiousness",
  "message": "Next question or follow-up...",
  "turn": 4,
  "partial_profile": { "openness": { "score": 4, "rationale": "..." } }
}
```

Response (complete) — this is the deliverable contract from the brief:
```json
{
  "session_id": "uuid",
  "status": "completed",
  "profile": {
    "openness":          { "score": 4, "rationale": "..." },
    "conscientiousness": { "score": 5, "rationale": "..." },
    "extraversion":      { "score": 3, "rationale": "..." },
    "agreeableness":     { "score": 4, "rationale": "..." },
    "neuroticism":       { "score": 2, "rationale": "..." }
  },
  "confidence": 0.78,
  "metadata": {
    "model": "claude-sonnet-...",
    "prompt_version": "v1",
    "created_at": "ISO-8601"
  }
}
```

### `GET /v1/sessions/{session_id}`
Returns the current state of a session (for the UI to rehydrate / for inspection).

Model the schema thoughtfully: score constrained to 1–5 (Pydantic `conint`/`Field(ge=1, le=5)`), rationale non-empty, confidence 0–1, metadata always present. Include `prompt_version` and `model` so results are reproducible and auditable — Wiselook explicitly cares about traceability and explainability.

---

## 7. LLM handling

- **All LLM outputs must be structured and validated.** Use Anthropic tool-use / structured output and parse into Pydantic models. Never trust raw text — validate ranges, required fields, and reject/repair malformed responses.
- **Prompt as code**: prompts live in `llm/prompts.py`, are versioned via a `PROMPT_VERSION` constant that flows into response metadata, and are treated as reviewable artifacts.
- **Robustness**: every LLM call has a **timeout**, **retries with exponential backoff** (on transient errors only), and a **graceful fallback** if the model returns something unparseable (e.g. one repair retry with a stricter instruction, then a typed error).
- **Two LLM roles**:
  1. `assess_sufficiency` → `{ "sufficient": bool, "reason": str }` — cheap, fast, low temperature.
  2. `score_dimension` → `{ "score": 1-5, "rationale": str }` — scores the current dimension from the accumulated turns for that dimension.
- Keep temperature low (0.0–0.3) for stability and reproducibility; note this trade-off in the README.

---

## 8. Evaluation layer (Option C)

Explain *what* you measure and *why*. Build:

1. **Golden transcripts** — a handful of fixed conversations (`evaluation/golden/*.json`), each with the user turns and an **expected score range** per dimension (a range, not an exact number — scoring is probabilistic).
2. **Deterministic checks** (`checks.py`): schema validity, scores within 1–5, scores fall inside expected ranges, all five dimensions present, confidence in 0–1, metadata complete.
3. **LLM-as-judge** (`judge.py`): given a transcript and the produced rationale, a judge model returns a structured verdict on whether the rationale is *consistent with and justified by* the user's answers — catches plausible-but-unfounded scoring.
4. **Runner** (`run_eval.py`): runs the harness over the golden set and prints a compact pass/fail report with per-dimension deltas.

In the README, state the limitation honestly: a golden set this small measures regression and gross errors, not true psychometric validity. That's the right scope for a test, and saying so is a plus.

---

## 9. Observability (Option C)

- **structlog** JSON logging throughout.
- A **request/correlation id** generated per session and per HTTP request, attached to every log line and every LLM call, so a full assessment can be traced end-to-end.
- **Per-LLM-call latency** logged (start/end, duration_ms, model, prompt_version, token usage if available).
- Log the graph node transitions (entered `assess_sufficiency`, decision=`insufficient`, etc.) — this doubles as a lightweight trace.

Don't over-build: no OpenTelemetry collector, no dashboards. Structured logs with correlation and latency are the right depth. Describe in the README how you'd extend to real distributed tracing (e.g. OTel + Langfuse/LangSmith) in production.

---

## 10. Concurrency (Option C)

The service must handle several assessments in parallel without degrading. Because every LLM call is **async** and state is isolated per `session_id` thread, concurrency mostly comes for free — but be deliberate:

- All I/O (LLM calls, graph invocations) is `async`.
- No shared mutable state between sessions; the checkpointer keys everything by thread id.
- Explain limits in the README: in-memory `MemorySaver` is single-process (fine for the test, doesn't scale horizontally); production would use `AsyncPostgresSaver` and stateless app instances behind a load balancer. Mention rate-limit / concurrency caps against the Anthropic API (e.g. a bounded semaphore) as the real bottleneck.

---

## 11. Voice design note (Option C — README, ½–1 page)

No implementation required. Write a crisp design note covering how you'd take this text engine to **real-time voice**:

- **Latency budget**: streaming ASR → LLM → streaming TTS; target sub-second perceived turn latency; where the current synchronous scoring would need to move off the critical path (score between turns, not during).
- **Turn-taking & interruptions**: VAD-driven endpointing, barge-in handling, partial-transcript speculation.
- **Imperfect transcription (ASR noise)**: the scoring prompts must be robust to disfluencies, filler words, and transcription errors; consider confidence-weighting and asking clarifying follow-ups when ASR confidence is low.
- **Text↔voice scoring parity**: the *same* scoring/inference layer must consume a normalized transcript regardless of channel, so a spoken and a typed version of the same answer produce the same profile. Propose a shared normalization step and a parity eval (same golden answers, fed via text and via synthesized-then-transcribed audio, assert score deltas stay within tolerance).
- Reference the tools Wiselook uses (LiveKit for the real-time transport, ElevenLabs for TTS) — I have hands-on ElevenLabs experience from a prior TTS project, so ground the note in that.

---

## 12. Conventions & quality bar

- **Typed everywhere**: full type hints, `mypy` clean.
- **Separation of concerns**: API knows nothing about LLM internals; graph knows nothing about HTTP; domain models are pure.
- **Small, meaningful commits** that show the build progression (schema → graph → API → eval → docs). The brief explicitly values commit history — do **not** squash into one giant commit.
- **Tests that matter**: parsing/validation edge cases, conditional-edge routing, and at least one endpoint test — all with the LLM **mocked** (no network in the test suite).
- **Config via env only**: `ANTHROPIC_API_KEY`, `MODEL_NAME`, `LLM_TIMEOUT`, `MAX_FOLLOWUPS`, `LOG_LEVEL`. Ship `.env.example`, never commit real keys.
- **README** must state: which option (C), setup/run instructions, design decisions and trade-offs, assumptions, and "what I'd do with more time."

---

## 13. Suggested build order

1. `domain/models.py` + `domain/state.py` + `domain/constants.py` — lock the schemas and question bank first.
2. `llm/client.py` + `llm/prompts.py` — async structured-output client with retries/timeout, mockable.
3. `graph/` — nodes, edges, builder; get the graph running end-to-end from a script before touching HTTP.
4. `api/` + `main.py` — wire the graph behind the three endpoints; mount `static/index.html`.
5. `tests/` — models, edges, one endpoint test (mocked LLM).
6. `evaluation/` — golden set, checks, judge, runner.
7. `observability/` — structlog + correlation ids + latency (can be woven in from step 2).
8. `README.md` — including the voice design note.

Start at step 1. Confirm the schemas and graph shape look right before building outward.

---

## 14. Assumptions (state these in the README)

- Fixed five-dimension walk, one primary question per dimension, up to `MAX_FOLLOWUPS` adaptive follow-ups each.
- In-memory session state (single process) — acceptable for a technical test.
- Confidence is a simple heuristic derived from answer informativeness / follow-ups used — documented as such, not presented as calibrated.
- English-language assessment for the test; note that multilingual is a straightforward prompt/eval extension.

