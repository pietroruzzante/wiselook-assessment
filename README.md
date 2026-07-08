# Wiselook — Conversational Big Five Assessment

A FastAPI service that walks a user through a five-question conversation and
infers a Big Five (OCEAN) personality profile, orchestrated by a LangGraph
state machine and scored by an LLM with structured, validated output.

This is the Wiselook AI Engineer technical assessment, built to **Option C**
(the highest tier requested in the brief): the core conversational engine,
plus an evaluation harness, observability, and a documented approach to
concurrency and voice. This README covers what actually got built, the
decisions behind it, and what I'd do next.

---

## What's here

- A **LangGraph** state machine that asks one question per OCEAN dimension,
  decides whether an answer is informative enough to score or needs a
  follow-up, scores it, and moves on — five dimensions, then done.
- A **FastAPI** service (`/v1/sessions`, `/v1/sessions/{id}/reply`,
  `/v1/sessions/{id}`) wrapping that graph, with a static chat UI (see the
  note below — this piece was provided by the brief, not something the
  assessment is evaluated on, though it got a bit more polish than strictly
  required).
- An **evaluation harness** (`evaluation/`) that runs fixed conversations
  through the real model and checks both the schema *and* whether the
  scoring rationale is actually justified by what the person said.
- **structlog** JSON logging with request/session correlation, so one id
  traces a full assessment across every LLM call and node transition.
- A test suite (`tests/`) that mocks only the LLM — everything else (graph,
  checkpointer, FastAPI app) runs for real.

---

## Setup and running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY

uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000/` for the chat UI, or drive the API
directly:

```bash
curl -X POST http://127.0.0.1:8000/v1/sessions
curl -X POST http://127.0.0.1:8000/v1/sessions/<id>/reply \
  -H "Content-Type: application/json" -d '{"answer": "..."}'
```

**Tests** (mocked LLM, no network, no API key needed beyond a dummy value):

```bash
pytest
```

**Evaluation** (hits the real Anthropic API — needs a real key in `.env`):

```bash
python -m evaluation.run_eval
```

**Lint / types**:

```bash
ruff check .
mypy .
```

---

## A note on the chat UI

`static/index.html` was supplied by the brief as the "already built" piece,
and the assessment isn't being evaluated on frontend work — the backend,
graph, evaluation, and observability are the point. That said, a few small
changes went in beyond
what was strictly necessary, because they affect whether the UI is actually
usable for testing the assessment, not because of frontend polish for its
own sake:

- A landing screen with a single "Start Assessment" button, since the
  chat previously auto-started a session (and burned an LLM call) the
  moment the page loaded, before a user had decided to begin.
- Removing a rationale leak in the follow-up question — it was echoing
  the sufficiency verdict's internal reasoning (which trait, and why the
  answer looked weak) straight into the chat, which breaks the "blind"
  part of a blind personality assessment. This one is a correctness fix
  more than UI polish; it happened to surface in the UI first.
- Trimming the completion banner's copy.

None of this changes the API contract or the graph; it's presentation-layer
only, called out here so it doesn't read as in-scope deliverable work.

---

## Why LangGraph, concretely

The conversation has exactly one real decision point: after an answer,
*is this enough to score, or do we need a follow-up?* That's conditional
branching over multi-turn, per-session state — which is what LangGraph's
conditional edges, `interrupt()`, and checkpointer are for. A plain
`while` loop with a dict would also work for a five-question flow this
size, and I want to be upfront about that rather than oversell the choice.
What LangGraph actually buys here:

- **Persisted, resumable state per session** via a checkpointer, instead of
  hand-rolling a session store.
- **A clean split between "decide" and "act"** (`edges.py` vs `nodes.py`),
  which made the routing logic trivial to unit-test in isolation
  (`tests/test_edges.py`) without spinning up the graph or mocking an LLM.
- **A one-line swap to Postgres** (`AsyncPostgresSaver` instead of
  `MemorySaver`) if this needed to survive a process restart or scale past
  one instance — the interface is identical, only the constructor changes.

The honest trade-off: LangGraph adds a dependency and a bit of ceremony
(`interrupt`/`Command(resume=...)`) for a graph this small. I judged that
worth it here because the brief explicitly asks for this to be evaluated as
a piece of production-shaped infrastructure, not a one-off script — and the
checkpointer/interrupt pattern is exactly what a longer, more adaptive
assessment would need anyway.

## The graph shape

```
START → ask_question → wait_for_input → assess_sufficiency
                                             ├─ insufficient & budget left → ask_followup → wait_for_input
                                             └─ sufficient (or budget exhausted) → score_dimension
                                                                                       ├─ more dimensions → ask_question
                                                                                       └─ all five scored → finalize → END
```

`wait_for_input` suspends the graph with `langgraph.types.interrupt()`
between HTTP turns; each reply resumes it with `Command(resume=answer)`.
`assess_sufficiency` and `score_dimension` are the two LLM roles, both
structured-output calls validated into Pydantic models
(`SufficiencyVerdict`, `DimensionScore`) — the graph never sees raw model
text.

---

## LLM handling

- **Structured output only.** Every LLM call is forced through a tool call
  matching a Pydantic schema (`app/llm/client.py`), then validated with
  `model_validate`. If validation fails, one repair attempt is made with a
  stricter instruction appended; if that also fails, a typed `LLMOutputError`
  is raised rather than letting malformed text leak into the graph.
- **Two typed failure modes**, not one. `LLMOutputError` (unparseable
  output, worth a repair attempt) is distinct from `LLMRequestError`
  (auth/billing/exhausted-retries — an API-level failure a repair prompt
  can't fix). The FastAPI layer catches the shared `LLMError` base and
  returns a 502, but the split matters for anyone debugging logs later.
- **Retries and timeouts**: `tenacity` retries transient errors
  (`APIConnectionError`, `APITimeoutError`, `RateLimitError`,
  `InternalServerError`) with exponential backoff, capped at 3 attempts;
  every call has a hard timeout (`LLM_TIMEOUT` env var). Non-transient
  errors (bad request, auth) are not retried — retrying a bug just delays
  the failure.
- **Low temperature (0.0)** on both roles, for reproducibility — scoring a
  personality trait shouldn't be a coin flip on rerun. The real trade-off is
  reduced diversity in rationale phrasing across runs, which is fine here
  since we're optimizing for consistency, not creative writing.
- **Prompts are versioned code** (`app/llm/prompts.py`, `PROMPT_VERSION`),
  and that version flows into every completed result's metadata — anyone
  looking at an old profile can tell exactly which prompt wording produced
  it.
- **Model used during development**: `claude-haiku-4-5`, set in the local
  `.env` — cheap enough to run the eval harness repeatedly while iterating
  on prompts (see the neuroticism fix below, which took three real eval
  runs to nail down). `.env.example` defaults to `claude-sonnet-4-5`, which
  is what I'd actually run in production for the better judgment call on
  ambiguous answers; the model is a plain env var either way.

---

## A real bug the eval harness caught

This is worth calling out specifically because it's the best evidence I
have that the evaluation layer earns its keep rather than being a checkbox.

Early runs of `evaluation/run_eval.py` against the real model had a golden
transcript where the answer was calm and methodical under pressure
("I'd stay level... give me ten minutes... loop in the colleague to fix it
together") but the neuroticism score came back **5** — the high/reactive
end — when it should have been 1–2. Two things were going on:

1. The rubric said high neuroticism is "sensitive, reactive." The model was
   reading "reacts quickly and decisively" as "reactive," when the Big Five
   sense is emotional volatility, not response speed. Fixed by rewording
   the rubric in `app/domain/constants.py` to spell out the distinction
   explicitly.
2. That alone didn't fix it. The score still came back 5, even though the
   model's own rationale now correctly argued for *low* neuroticism — a
   halo/valence bias: a calm, competent-sounding answer pulled the number
   upward by default. For the other four dimensions, "high" reads as the
   professionally desirable end, but neuroticism is inverted — calm (the
   good-sounding answer) is the *low* score. Fixed with an explicit
   instruction in `SCORE_SYSTEM_PROMPT` that score direction follows the
   rubric labels only, not how good the response sounds.

Both fixes are in `app/llm/prompts.py` and `app/domain/constants.py`
(prompt version bumped v3 → v5). I only caught this because the harness
runs real conversations through the real model and checks the *number*
against an expected range — a mocked test suite would have happily passed
throughout, since nothing about the code was broken, only the prompt's
framing.

---

## Evaluation: what I measure and why

Three layers, in `evaluation/`:

1. **Golden transcripts** (`evaluation/golden/*.json`) — two fixed
   conversations, one scripted to read as consistently high across most
   dimensions, one consistently low, each with an **expected score range**
   per dimension (a range, not an exact number, since scoring an LLM is
   probabilistic even at temperature 0).
2. **Deterministic checks** (`checks.py`) — all five dimensions present,
   confidence in [0, 1], metadata complete, and each score inside its
   golden range. Cheap, fast, no second LLM call needed.
3. **LLM-as-judge** (`judge.py`) — a second model call reads the question,
   answer, score, and rationale, and verdicts whether the rationale is
   actually grounded in what the person said. This is the check that caught
   nothing wrong on its own in the bug above (the judge agreed the
   rationale was well-grounded — it correctly described *low* neuroticism)
   but that mismatch between "grounded rationale" and "wrong number" is
   exactly what led me to the number itself, not the reasoning, being the
   bug.

`run_eval.py` runs both transcripts against the real graph and real model
and prints a compact pass/fail report per check.

**Honest limitation**: two golden transcripts measure regression and gross
scoring errors, not psychometric validity. They will catch "the model now
scores confident people as neurotic" but they cannot tell you whether the
Big Five model, this question set, or this scoring approach actually
predicts anything real about a person. That's out of scope for a technical
test, and I'd rather say so than imply otherwise.

---

## Observability

- **structlog, configured for JSON output** (`app/observability/logging.py`)
  — every log line is a JSON object with level, ISO timestamp, and whatever
  context is bound for that call.
- **Correlation ids**: a `request_id` is generated per HTTP request by
  `request_logging_middleware` and bound via `structlog.contextvars` so
  every log line during that request carries it automatically, without
  threading it through every function signature. `session_id` is bound the
  same way, parsed straight from the URL.
- One subtlety worth documenting because it cost me a debugging cycle:
  FastAPI's `app.middleware("http")` runs the route handler in a separate
  context (`BaseHTTPMiddleware`), so a value bound *inside* the handler
  doesn't propagate back to the middleware's own post-`call_next` log line.
  I first bound `session_id` inside each route handler and it silently
  vanished from the request summary log; binding it in the middleware
  itself (parsed from the URL path, since it's known before routing runs)
  fixed it. The in-handler bind is still needed for `create_session`, where
  the id doesn't exist until the handler generates it.
- **Per-LLM-call latency** (`app/llm/client.py`): every call logs
  `duration_ms`, model, and which structured output it was for.
- **Node transitions** (`app/graph/nodes.py`) log every `ask_question`,
  `assess_sufficiency` (with its verdict and reason), `ask_followup`, and
  `score_dimension` (with the score) — this doubles as a lightweight trace
  of the whole conversation.

Given one `session_id`, you can grep the logs and reconstruct the entire
assessment: every question asked, every sufficiency verdict and why, every
score, and every LLM call's latency.

**What I didn't build, on purpose**: no OpenTelemetry collector, no
dashboards, no metrics backend. For this scope, structured logs with
correlation and latency are the right depth — adding a tracing backend here
would be building infrastructure nobody's asked to operate yet. In
production I'd reach for OTel spans around each graph node and LLM call,
exported to something like Langfuse or LangSmith (both have first-class
LangGraph integrations), which would turn the current log-based trace into
an actual queryable, visualized trace with span timing.

---

## Concurrency

All I/O in the request path is `async` — the Anthropic calls, the graph
invocations, the checkpointer reads. Sessions are isolated by
`thread_id` (the session's `session_id`), so concurrent assessments don't
share mutable state; there's nothing to lock.

Two real limits, both documented rather than solved here:

- **`MemorySaver` is single-process.** It's the right choice for this
  test — zero setup, works out of the box — but it means state lives in
  one process's memory: no horizontal scaling, and a restart drops every
  in-progress session. `AsyncPostgresSaver` is a drop-in replacement
  (`langgraph.checkpoint.postgres.aio`) implementing the same
  `BaseCheckpointSaver` interface; swapping it in `graph/builder.py` is the
  only change needed to make this horizontally scalable and restart-safe.
- **The Anthropic API itself is the real bottleneck**, not this service.
  `AnthropicStructuredClient` already bounds concurrency with a semaphore
  (`max_concurrency=10`) so a burst of sessions can't blow through your
  account's rate limit and take down every in-flight request at once — it's
  a soft local cap, not a distributed one, so with multiple app instances
  you'd want a shared limiter (e.g. a token-bucket in Redis) or just size
  each instance's cap to your actual per-key rate limit divided by instance
  count.

---

## Voice design note

No implementation here — this is the design note the brief asks for on how
this text engine would extend to real-time voice, grounded in the tools
Wiselook actually uses (LiveKit for transport, ElevenLabs for TTS — I've
built a TTS integration with ElevenLabs before, so this isn't abstract to
me).

**Latency budget.** The current architecture already separates *asking*
from *scoring the previous answer* — `assess_sufficiency` and
`score_dimension` both run between turns, not on the critical path of
"stop talking, get a response." That's the right shape for voice: streaming
ASR feeds a growing transcript, VAD-driven endpointing decides when the
person's turn is over, and only *then* does the sufficiency/scoring call
run, in parallel with streaming TTS starting to speak the next question (or
a quick acknowledgment) as soon as its text is available. The one thing
that has to move: today the API waits for `assess_sufficiency` to complete
before returning the next question. In voice, you'd want to speak a filler
acknowledgment immediately, run sufficiency scoring in the background, and
only then speak the actual next question or follow-up — sub-second
perceived turn latency depends on never blocking speech on an LLM round
trip.

**Turn-taking and interruptions.** VAD-based endpointing decides when to
stop listening and start the sufficiency call; barge-in means the person
can start talking again before TTS finishes the question, which should cut
audio immediately and start a fresh ASR buffer rather than trying to merge
two utterances. Partial-transcript speculation (starting sufficiency
scoring on a partial ASR hypothesis before the person finishes) is tempting
for latency but risky here specifically, since `assess_sufficiency`'s job
is to judge whether an answer is *complete enough* — scoring a partial
transcript as sufficient would be exactly wrong. I'd keep sufficiency
judgment gated on a finalized transcript, and only speculate on cheaper
things like TTS pre-synthesis of the likely next question.

**ASR noise.** Disfluencies, filler words, and misrecognitions are
inevitable, and the sufficiency/scoring prompts as written already lean on
concrete behavioral detail over surface phrasing — that happens to be
somewhat robust to "um, I would, like, probably just, uh, tell them" coming
through as noisy text, since the prompts are already looking for the
underlying action, not clean prose. What I'd add: pass ASR confidence
through to `assess_sufficiency` as an explicit signal, and when confidence
is low, prefer a clarifying follow-up over a low-confidence score — the
same "insufficient" path already used for vague answers, just triggered by
transcription quality instead of content.

**Text/voice parity.** The scoring layer must consume a normalized
transcript regardless of channel, so a typed and a spoken version of the
same answer produce the same profile. Concretely: one normalization step
(strip filler words, fix obvious ASR homophone errors, but preserve
hedging and self-corrections, since those are real signal for some
dimensions) sits between "raw input" and `assess_sufficiency`/
`score_dimension`, and both channels feed through it. The parity eval this
implies is a direct extension of the existing golden-transcript harness:
take each golden answer, run it through text as today, then run a
synthesized-then-transcribed version of the same answer through the voice
path, and assert the score delta between the two stays within a small
tolerance (say, ±1 point). That's the same `checks.py` machinery, just
comparing two runs against each other instead of against a fixed expected
range.

---

## Assumptions

- Fixed five-dimension walk, one primary question per dimension, up to
  `MAX_FOLLOWUPS` adaptive follow-ups each (env-configurable, default 2 in
  `.env.example`).
- In-memory session state (`MemorySaver`), single process — correct for
  this test's scope, not for production (see Concurrency, above).
- `confidence` in the final result is a simple heuristic — it scales down
  with how many follow-ups were needed across the whole assessment,
  relative to the maximum possible. It is **not** a calibrated probability;
  treating it as one would overstate how well-understood that number is.
- English-language assessment. Multilingual support is a prompt and eval
  extension (translate the question bank, and either translate answers
  before scoring or verify the scoring prompt generalizes) rather than an
  architectural change.
- The follow-up question itself is deterministic, not a third LLM call — a
  single generic template asking for more concrete detail, the same for
  every dimension. Only two LLM roles exist (`assess_sufficiency` and
  `score_dimension`); the template deliberately doesn't repeat
  `assess_sufficiency`'s internal `reason` field back to the user — that
  field names the trait and the model's read on the answer, and surfacing
  it would break the blind assessment (a user could game later answers
  once they know what's being measured). The reason is still logged for
  traceability, just never shown in the chat.

---

## What I'd do with more time

- **More golden transcripts, and a couple with intentionally ambiguous
  answers** — the current two are written to read clearly high or low
  across the board, which is good for catching gross regressions but
  doesn't exercise the middle of the scale or genuine trait mixes.
- **Fix the confidence heuristic's blind spot**: it only reacts to
  follow-up count, so a confidently-wrong single answer scores the same
  confidence as a confidently-right one. A better heuristic would fold in
  something like the sufficiency verdict's own certainty, if the model
  exposed it.
- **A repair-loop metric**: right now a validation failure triggers one
  repair attempt silently; I'd add a counter/log specifically for how often
  repair is needed per prompt version, since a rising rate is an early
  signal that a prompt change made structured output harder to hit
  reliably.
- **Postgres checkpointer**, actually wired up behind an env flag, rather
  than just documented as a one-line swap — I stopped short of this since
  it adds a real dependency (a running Postgres instance) for a test whose
  scope explicitly says in-memory is fine.
- **The voice pipeline itself**, if this were a longer engagement — the
  design note above is deliberately not code, and turning it into a LiveKit
  agent with streaming ElevenLabs TTS is the natural next milestone.
