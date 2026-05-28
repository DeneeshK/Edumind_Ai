# EduMind Evaluation System — Implementation Prompt for Atigravity

---

## CONTEXT (Read This First Before Anything Else)

You are working on an existing production project called **EduMind** — an agentic adaptive learning platform for JEE/NEET physics students.

The project is already fully built and running. Your job is to **add an evaluation system to it without breaking anything that already works.**

Here is the current project structure you will be working with:

```
edumind/
├── app/
│   └── api.py                  ← FastAPI app, all HTTP endpoints, SSE stream
├── agents/
│   ├── base_agent.py           ← BaseAgent class all agents inherit from
│   ├── orchestrator.py         ← OrchestratorAgent — controls the session loop
│   ├── curriculum_architect.py ← CurriculumArchitectAgent — builds module plan
│   ├── tutor.py                ← TutorAgent — delivers lessons
│   ├── evaluator.py            ← EvaluatorAgent — tests the student
│   └── adaptation_engine.py   ← AdaptationEngine — decides what to do next
├── core/
│   ├── student_model.py        ← StudentState, CurriculumPlan, EvaluationReport
│   └── rag_pipeline.py         ← retrieve(), hyde() — full RAG pipeline
├── db/
│   ├── postgres.py             ← all PostgreSQL tables, init_db(), get_conn()
│   └── chromadb_client.py      ← insert(), search(), rerank()
├── clients/
│   ├── groq_client.py          ← tool_call_loop(), generate(), stream()
│   └── tavily_client.py        ← search()
├── config.py                   ← Settings (pydantic-settings, reads .env)
└── main.py                     ← uvicorn entrypoint
```

**Existing database tables** (already in `db/postgres.py` `_SCHEMA_SQL`):
- `students` — identity
- `concept_mastery` — mastery scores per concept per student
- `evaluation_history` — every EvaluationReport written mid-session
- `session_memory` — end-of-session summaries
- `decision_log` — every agent decision
- `doubt_log` — doubt events
- `curricula` — curriculum plans
- `module_embeddings` — ChromaDB IDs per module
- `dynamic_concept_summaries` — generated summaries per module

**Existing settings** (in `config.py`):
- `settings.reasoning_model` = `"llama-3.3-70b-versatile"` (used for tool-call loops)
- `settings.generation_model` = `"llama-3.1-8b-instant"` (used for plain text generation)
- `settings.groq_api_key`, `settings.tavily_api_key`, `settings.database_url`

**LLM call pattern** — all LLM calls go through `clients/groq_client.py`:
```python
from clients.groq_client import generate as groq_generate
response = await groq_generate(
    messages=[{"role": "user", "content": "..."}],
    model=settings.generation_model,
)
# returns a plain string
```

---

## YOUR TASK

Build a **standalone evaluation module** for EduMind. This module must:

1. Collect numeric scores for every major component of the system (RAG pipeline, each agent, learning outcomes)
2. Store every score in its own dedicated database table
3. Expose a clean API so scores can be read at any time
4. Run automatically at the right moments during a session (not in a separate process, not as a cron job — inline but non-blocking)
5. Never slow down or break an active teaching session
6. Support a manual trigger endpoint so the reviewer can run a full evaluation report on demand

---

## GROUND RULES (Follow These Exactly)

**Rule 1 — Do not modify any existing file's core logic.**
You may ADD new imports and ADD new function calls at the end of existing functions (hook points listed below), but never change existing logic, existing function signatures, or existing database writes.

**Rule 2 — All evaluation runs are fire-and-forget.**
Use `asyncio.create_task()` for every evaluation call that runs during a live session. The session must never `await` an evaluation function directly. Evaluation failures must be caught and logged; they must never propagate to the session.

**Rule 3 — Every metric must write to the database.**
No metric should only exist in memory or in logs. Every score must land in a database table so it can be read later via the API.

**Rule 4 — All new database tables go in `db/postgres.py` `_SCHEMA_SQL`.**
Append new `CREATE TABLE IF NOT EXISTS` statements to the existing `_SCHEMA_SQL` string. Do not create a separate schema file. `CREATE TABLE IF NOT EXISTS` is idempotent — it is safe to add to an existing schema.

**Rule 5 — All new settings go in `config.py`.**
Add new fields to the existing `Settings` class. Never hardcode thresholds or schedule values.

**Rule 6 — The evaluation module lives in its own directory.**
Create `evaluation/` at the project root. Never put evaluation code inside `agents/`, `core/`, or `db/`.

**Rule 7 — All LLM judge calls use `settings.generation_model`, not `settings.reasoning_model`.**
The cheap fast model is enough for scoring. Do not use the expensive 70b model for evaluation.

**Rule 8 — Never block the event loop.**
All SentenceTransformer `.encode()` calls and cross-encoder `.predict()` calls must be wrapped in `asyncio.to_thread(lambda: ...)`.

**Rule 9 — Schedule logic is config-driven, not hardcoded.**
The settings class controls when aggregation reports are generated. Start with safe defaults (weekly). The reviewer can change this without touching code.

---

## WHAT TO BUILD

### Step 1 — Create the evaluation directory

```
evaluation/
├── __init__.py
├── metrics/
│   ├── __init__.py
│   ├── rag_metrics.py          ← all RAG metrics
│   ├── agent_metrics.py        ← all agent metrics
│   └── outcome_metrics.py      ← learning outcome metrics
├── collector.py                ← writes metric rows to DB
├── runner.py                   ← orchestrates which metrics run when
├── scheduler.py                ← APScheduler-based aggregation jobs
└── api_router.py               ← FastAPI router mounted in app/api.py
```

---

### Step 2 — New database tables

Append these to the `_SCHEMA_SQL` string in `db/postgres.py`. Add them after the last existing table definition, before the closing `"""`:

```sql
-- 11. eval_metric_runs  ← one row per individual metric computation
CREATE TABLE IF NOT EXISTS eval_metric_runs (
    id              SERIAL PRIMARY KEY,
    student_id      TEXT REFERENCES students(student_id) ON DELETE SET NULL,
    session_id      TEXT,
    metric_name     TEXT NOT NULL,      -- e.g. "hyde_quality", "lesson_quality"
    component       TEXT NOT NULL,      -- "rag", "agent", "outcome", "system"
    score           FLOAT NOT NULL,     -- the numeric score 0.0-1.0
    details_json    JSONB NOT NULL DEFAULT '{}',  -- full breakdown dict
    trigger         TEXT NOT NULL DEFAULT 'auto', -- "auto" | "manual" | "scheduled"
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_eval_runs_student  ON eval_metric_runs(student_id);
CREATE INDEX IF NOT EXISTS idx_eval_runs_metric   ON eval_metric_runs(metric_name);
CREATE INDEX IF NOT EXISTS idx_eval_runs_created  ON eval_metric_runs(created_at);

-- 12. eval_session_reports  ← one row per session, aggregated scores
CREATE TABLE IF NOT EXISTS eval_session_reports (
    id              SERIAL PRIMARY KEY,
    session_id      TEXT NOT NULL UNIQUE,
    student_id      TEXT REFERENCES students(student_id) ON DELETE SET NULL,
    topic           TEXT NOT NULL DEFAULT '',
    rag_score       FLOAT,
    agent_score     FLOAT,
    outcome_score   FLOAT,
    system_score    FLOAT,
    full_report_json JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 13. eval_aggregated_reports  ← weekly/monthly rollup across all sessions
CREATE TABLE IF NOT EXISTS eval_aggregated_reports (
    id              SERIAL PRIMARY KEY,
    period_type     TEXT NOT NULL,     -- "weekly" | "monthly"
    period_start    TIMESTAMPTZ NOT NULL,
    period_end      TIMESTAMPTZ NOT NULL,
    avg_rag_score   FLOAT,
    avg_agent_score FLOAT,
    avg_system_score FLOAT,
    sessions_counted INT NOT NULL DEFAULT 0,
    students_counted INT NOT NULL DEFAULT 0,
    full_report_json JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

---

### Step 3 — New settings fields

Add these to the `Settings` class in `config.py`, after the existing fields:

```python
# Evaluation settings
eval_enabled: bool = True
eval_judge_model: str = "llama-3.1-8b-instant"   # cheap model for LLM-as-judge
eval_embed_model: str = "all-MiniLM-L6-v2"        # sentence-transformers model
eval_faithfulness_claim_limit: int = 15            # max claims to check per lesson
eval_precision_k: int = 10                         # K for ChromaDB precision@K
eval_schedule_weekly: bool = True                  # run weekly aggregation
eval_schedule_monthly: bool = True                 # run monthly aggregation
eval_schedule_timezone: str = "Asia/Kolkata"       # timezone for scheduler
```

---

### Step 4 — Build `evaluation/collector.py`

This is the single function that writes any metric score to the database:

```python
# evaluation/collector.py

from __future__ import annotations
import json
from loguru import logger
from db.postgres import get_conn


async def record_metric(
    metric_name: str,
    component: str,        # "rag" | "agent" | "outcome" | "system"
    score: float,
    details: dict,
    session_id: str | None = None,
    student_id: str | None = None,
    trigger: str = "auto",  # "auto" | "manual" | "scheduled"
) -> None:
    """
    Write one metric result to eval_metric_runs.
    All evaluation functions call this. Never raises — failures are logged only.
    """
    try:
        async with get_conn() as conn:
            await conn.execute(
                """
                INSERT INTO eval_metric_runs
                    (student_id, session_id, metric_name, component, score, details_json, trigger)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                student_id,
                session_id,
                metric_name,
                component,
                round(float(score), 6),
                json.dumps(details),
                trigger,
            )
    except Exception as exc:
        logger.warning("eval collector failed for '{}': {}", metric_name, exc)


async def write_session_report(
    session_id: str,
    student_id: str,
    topic: str,
    rag_score: float | None,
    agent_score: float | None,
    outcome_score: float | None,
    system_score: float | None,
    full_report: dict,
) -> None:
    """Write the aggregated session-level report."""
    try:
        async with get_conn() as conn:
            await conn.execute(
                """
                INSERT INTO eval_session_reports
                    (session_id, student_id, topic,
                     rag_score, agent_score, outcome_score, system_score, full_report_json)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (session_id) DO UPDATE SET
                    rag_score    = EXCLUDED.rag_score,
                    agent_score  = EXCLUDED.agent_score,
                    outcome_score = EXCLUDED.outcome_score,
                    system_score = EXCLUDED.system_score,
                    full_report_json = EXCLUDED.full_report_json
                """,
                session_id, student_id, topic,
                rag_score, agent_score, outcome_score, system_score,
                json.dumps(full_report),
            )
    except Exception as exc:
        logger.warning("eval session report write failed: {}", exc)
```

---

### Step 5 — Build `evaluation/metrics/rag_metrics.py`

Implement these five functions. Each one must:
- Accept only plain Python types (strings, lists, dicts, floats) — no ORM objects
- Return a dict with a `score` key (float 0-1) and a `details` sub-dict
- Never raise — wrap everything in try/except and return a zero score on failure
- Call `record_metric()` itself at the end

The reference implementations for all formulas are in the attached `edumind_evaluation_metrics.md` file. Implement them exactly as described, including the weight combinations and thresholds.

Functions to implement:

```python
async def hyde_quality_score(
    original_query: str,
    hypothetical_answer: str,
    concept_card_text: str,       # text from ChromaDB for this concept
    session_id: str,
    student_id: str,
) -> dict: ...

async def chromadb_precision_at_k(
    concept: str,
    retrieved_chunks: list[str],
    session_id: str,
    student_id: str,
) -> dict: ...

async def tavily_relevance_score(
    concept: str,
    tavily_results: list[dict],   # raw dicts from tavily_search()
    session_id: str,
    student_id: str,
) -> dict: ...

async def reranker_gain_score(
    query: str,
    chunks_before: list[str],
    chunks_after: list[str],
    session_id: str,
    student_id: str,
) -> dict: ...

async def rag_faithfulness_score(
    lesson_text: str,
    retrieved_chunks: list[str],
    session_id: str,
    student_id: str,
) -> dict: ...
```

**Important implementation notes for RAG metrics:**

All `SentenceTransformer` usage must:
1. Load the model once at module level using a global: `_embed_model = None`
2. Use a `_get_embed_model()` function with lazy initialization
3. Wrap `.encode()` calls in `await asyncio.to_thread(lambda: ...)`

For LLM-as-judge calls, use `groq_generate` with `settings.eval_judge_model`, not the reasoning model.

For `tavily_relevance_score`, compute both relevance (semantic similarity) and freshness (date decay), then combine as `0.7 * relevance + 0.3 * freshness` as described in the metrics document.

---

### Step 6 — Build `evaluation/metrics/agent_metrics.py`

Implement these six functions with the same pattern (return dict with `score` and `details`, call `record_metric()`, never raise):

```python
async def curriculum_coverage_score(
    topic: str,
    domain: str,
    generated_modules: list[dict],  # list of {"concept":..., "title":...}
    session_id: str,
    student_id: str,
) -> dict: ...

async def curriculum_ordering_score(
    modules: list[dict],   # list of {"concept":..., "prerequisites":[...]}
    session_id: str,
    student_id: str,
) -> dict: ...

async def lesson_quality_score(
    lesson_text: str,
    module: dict,          # {"title":..., "concept":..., "depth_level":..., "estimated_minutes":...}
    student_pace: str,
    session_id: str,
    student_id: str,
) -> dict: ...

async def question_quality_score(
    questions: list[dict],  # list of {"question":..., "question_type":..., "source_quote":...}
    lesson_text: str,
    session_id: str,
    student_id: str,
) -> dict: ...

async def scoring_consistency_score(
    qa_log: list[dict],   # list of {"question":..., "answer":..., "correctness_score":...}
    concept: str,
    session_id: str,
    student_id: str,
) -> dict: ...

def routing_accuracy_score(
    mastery_score: float,
    advance_threshold: float,
    actual_action: str,
    misconception_type: str | None,
    reteach_count: int,
    session_id: str,
    student_id: str,
) -> dict: ...
```

Note: `routing_accuracy_score` is a synchronous function — it uses no LLM and no embeddings, just pure arithmetic. It should still call `record_metric()` but via `asyncio.ensure_future()` or schedule it as a task from the async caller.

---

### Step 7 — Build `evaluation/metrics/outcome_metrics.py`

Implement these three functions:

```python
def mastery_progression_rate(
    mastery_history: list[float],
    concept: str,
    session_id: str,
    student_id: str,
) -> dict: ...

def calibration_quality_score(
    calibration_deltas: list[float],
    session_id: str,
    student_id: str,
) -> dict: ...

def session_efficiency_score(
    modules_attempted: int,
    modules_mastered: int,
    total_modules_in_curriculum: int,
    reteach_events: int,
    session_duration_minutes: float,
    pace: str,
    session_id: str,
    student_id: str,
) -> dict: ...
```

All three are synchronous (no LLM, no embeddings). Same pattern: return dict with `score` and `details`, call `record_metric()`.

---

### Step 8 — Build `evaluation/runner.py`

This is the orchestrator for the evaluation system. It collects all individual metric scores and assembles them into a session report.

```python
# evaluation/runner.py

class EvaluationRunner:
    """
    Collects evaluation data during a session and runs metrics
    at the appropriate hook points.
    All public methods are called via asyncio.create_task() from the session.
    They never propagate exceptions.
    """
    
    def __init__(self, session_id: str, student_id: str, topic: str, pace: str):
        self.session_id = session_id
        self.student_id = student_id
        self.topic = topic
        self.pace = pace
        
        # Data buffers — populated by hook calls during the session
        self._rag_scores: list[float] = []
        self._agent_scores: list[float] = []
        self._outcome_scores: list[float] = []
        self._reteach_count: int = 0
        self._modules_attempted: int = 0
        self._modules_mastered: int = 0
        self._session_start: datetime = datetime.now(timezone.utc)
    
    # ── RAG hook ─────────────────────────────────────────────────────────────
    async def on_rag_retrieve(
        self,
        query: str,
        hypothetical: str,
        concept_card: str,
        chroma_chunks_before_rerank: list[str],
        chroma_chunks_after_rerank: list[str],
        tavily_results: list[dict],
    ) -> None:
        """Called once per retrieve() call. Runs HyDE, precision, Tavily, reranker metrics."""
        ...
    
    # ── Lesson hook ───────────────────────────────────────────────────────────
    async def on_lesson_delivered(
        self,
        lesson_text: str,
        retrieved_chunks: list[str],
        module: dict,
    ) -> None:
        """Called after deliver_lesson tool executes. Runs faithfulness + lesson quality."""
        ...
    
    # ── Curriculum hook ───────────────────────────────────────────────────────
    async def on_curriculum_built(
        self,
        modules: list[dict],
    ) -> None:
        """Called after build_curriculum() returns. Runs coverage + ordering metrics."""
        ...
    
    # ── Evaluation hook ───────────────────────────────────────────────────────
    async def on_evaluation_complete(
        self,
        concept: str,
        qa_log: list[dict],
        questions_asked: list[dict],
        lesson_text: str,
        mastery_score: float,
        advance_threshold: float,
        actual_action: str,
        misconception_type: str | None,
        reteach_count: int,
        calibration_delta: float,
        mastery_history: list[float],
        modules_attempted: int,
        modules_mastered: int,
    ) -> None:
        """Called after each EvaluatorAgent.evaluate() returns. Runs all evaluator + outcome metrics."""
        ...
    
    # ── Session end ───────────────────────────────────────────────────────────
    async def on_session_end(
        self,
        modules_mastered: int,
        total_curriculum_modules: int,
        reteach_events: int,
        calibration_deltas: list[float],
    ) -> None:
        """Called at the end of _end_session(). Computes and writes the session report."""
        ...
    
    def _compute_system_score(
        self,
        rag_score: float | None,
        agent_score: float | None,
        outcome_score: float | None,
    ) -> float | None:
        """
        Weighted combination:
        system_score = 0.35 * rag + 0.40 * agent + 0.25 * outcome
        Returns None if no scores available.
        """
        ...
```

---

### Step 9 — Hook points in existing files

These are the **only changes** you make to existing files. Add these calls at the exact points listed. Each call is wrapped in `asyncio.create_task()` so it never blocks the session.

**`core/rag_pipeline.py` — inside `retrieve()`**

After `final = await rerank(...)` and before `return final`, add:

```python
# ── Evaluation hook ──────────────────────────────────────────────────────
if _eval_runner_ref is not None:
    asyncio.create_task(
        _eval_runner_ref.on_rag_retrieve(
            query=query,
            hypothetical=hypothetical,
            concept_card="",          # populated by runner from ChromaDB if available
            chroma_chunks_before_rerank=chunks,
            chroma_chunks_after_rerank=final,
            tavily_results=tavily_results,
        )
    )
```

`_eval_runner_ref` is a module-level variable in `core/rag_pipeline.py`:
```python
_eval_runner_ref: "EvaluationRunner | None" = None

def set_eval_runner(runner) -> None:
    global _eval_runner_ref
    _eval_runner_ref = runner

def clear_eval_runner() -> None:
    global _eval_runner_ref
    _eval_runner_ref = None
```

**`agents/tutor.py` — inside `_execute_tool("deliver_lesson", ...)`**

After `self.state.record_module_content(module.id, lesson_text)`, add:

```python
if hasattr(self, "_eval_runner") and self._eval_runner is not None:
    asyncio.create_task(
        self._eval_runner.on_lesson_delivered(
            lesson_text=lesson_text,
            retrieved_chunks=self._retrieved_chunks,
            module={"title": module.title, "concept": module.concept,
                    "depth_level": module.depth_level, "estimated_minutes": module.estimated_minutes},
        )
    )
```

`self._eval_runner` is set by the Orchestrator before calling `tutor.teach()`.

**`agents/curriculum_architect.py` — inside `build_curriculum()`**

After `self.state.curriculum = plan`, add:

```python
if hasattr(self, "_eval_runner") and self._eval_runner is not None:
    asyncio.create_task(
        self._eval_runner.on_curriculum_built(
            modules=[{"concept": m.concept, "title": m.title,
                       "prerequisites": m.prerequisites} for m in plan.modules],
        )
    )
```

**`agents/evaluator.py` — inside `evaluate()`**

After `self.state.evaluation_history.append(report)`, add:

```python
if hasattr(self, "_eval_runner") and self._eval_runner is not None:
    mastery_history = [
        r.mastery_score for r in self.state.evaluation_history
        if r.concept == concept
    ]
    asyncio.create_task(
        self._eval_runner.on_evaluation_complete(
            concept=concept,
            qa_log=self._qa_log,
            questions_asked=[{"question": q["question"], "source_quote": q.get("source_quote",""),
                               "question_type": q.get("type","")} for q in self._qa_log],
            lesson_text=self._module_grounding_text(concept),
            mastery_score=report.mastery_score,
            advance_threshold=self.state.advance_threshold,
            actual_action=report.recommended_action,
            misconception_type=report.misconception_type,
            reteach_count=self.state.metacognition.consecutive_reteach_count,
            calibration_delta=report.calibration_delta,
            mastery_history=mastery_history,
            modules_attempted=self.state.evaluation_cycle_count,
            modules_mastered=len([r for r in self.state.evaluation_history
                                   if r.mastery_score >= self.state.advance_threshold]),
        )
    )
```

**`agents/orchestrator.py` — inside `_run_session()` (or at the start of `_run_module_loop()`)**

After creating the `OrchestratorAgent`, create the runner and wire it:

```python
from evaluation.runner import EvaluationRunner
from core.rag_pipeline import set_eval_runner, clear_eval_runner

if settings.eval_enabled:
    eval_runner = EvaluationRunner(
        session_id=ctx.state.session_id,
        student_id=ctx.state.student_id,
        topic=ctx.state.curriculum.topic if ctx.state.curriculum else "",
        pace=ctx.state.pace,
    )
    set_eval_runner(eval_runner)
else:
    eval_runner = None
```

Pass `eval_runner` to every agent:
```python
architect._eval_runner = eval_runner
tutor._eval_runner = eval_runner
evaluator._eval_runner = eval_runner
```

At the very end of `_end_session()`, before returning:
```python
if eval_runner is not None:
    asyncio.create_task(
        eval_runner.on_session_end(
            modules_mastered=len(completed_modules),
            total_curriculum_modules=len(curriculum.modules),
            reteach_events=reteach_event_count,
            calibration_deltas=[r.calibration_delta for r in self.state.evaluation_history],
        )
    )
    clear_eval_runner()
```

---

### Step 10 — Build `evaluation/scheduler.py`

Use **APScheduler** (add `apscheduler` to requirements). The scheduler runs aggregation jobs on a configurable cadence.

```python
# evaluation/scheduler.py

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from config import settings
from loguru import logger

_scheduler: AsyncIOScheduler | None = None


async def _run_weekly_aggregation() -> None:
    """Read all session reports from last 7 days, compute averages, write to eval_aggregated_reports."""
    try:
        from evaluation.runner import build_aggregated_report
        await build_aggregated_report(period_type="weekly")
    except Exception as exc:
        logger.warning("Weekly eval aggregation failed: {}", exc)


async def _run_monthly_aggregation() -> None:
    """Read all session reports from last 30 days, compute averages, write to eval_aggregated_reports."""
    try:
        from evaluation.runner import build_aggregated_report
        await build_aggregated_report(period_type="monthly")
    except Exception as exc:
        logger.warning("Monthly eval aggregation failed: {}", exc)


def start_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        return

    _scheduler = AsyncIOScheduler(timezone=settings.eval_schedule_timezone)

    if settings.eval_schedule_weekly:
        _scheduler.add_job(
            _run_weekly_aggregation,
            CronTrigger(day_of_week="sun", hour=2, minute=0),  # every Sunday 2am
            id="weekly_eval",
            replace_existing=True,
            misfire_grace_time=3600,
        )

    if settings.eval_schedule_monthly:
        _scheduler.add_job(
            _run_monthly_aggregation,
            CronTrigger(day=1, hour=3, minute=0),  # 1st of every month 3am
            id="monthly_eval",
            replace_existing=True,
            misfire_grace_time=3600,
        )

    _scheduler.start()
    logger.info("Evaluation scheduler started (weekly={}, monthly={})",
                settings.eval_schedule_weekly, settings.eval_schedule_monthly)


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
```

Add `start_scheduler()` and `stop_scheduler()` to the FastAPI `lifespan` context manager in `app/api.py`:

```python
# Inside the existing lifespan() context manager, after init_db():
from evaluation.scheduler import start_scheduler, stop_scheduler
if settings.eval_enabled:
    start_scheduler()

yield

# After yield (shutdown section):
if settings.eval_enabled:
    stop_scheduler()
```

---

### Step 11 — Build `evaluation/api_router.py`

```python
# evaluation/api_router.py

from fastapi import APIRouter, Query
from db.postgres import get_conn
import json

router = APIRouter(prefix="/eval", tags=["evaluation"])


@router.get("/session/{session_id}")
async def get_session_eval(session_id: str):
    """Return the aggregated evaluation report for a session."""
    async with get_conn() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM eval_session_reports WHERE session_id=$1", session_id
        )
    if not row:
        return {"error": "No evaluation report found for this session."}
    return dict(row)


@router.get("/metrics")
async def list_metrics(
    student_id: str = Query(None),
    metric_name: str = Query(None),
    component: str = Query(None),
    limit: int = Query(50, le=200),
):
    """List recent metric runs, filterable by student, metric name, or component."""
    conditions = []
    params = []
    if student_id:
        conditions.append(f"student_id=${len(params)+1}")
        params.append(student_id)
    if metric_name:
        conditions.append(f"metric_name=${len(params)+1}")
        params.append(metric_name)
    if component:
        conditions.append(f"component=${len(params)+1}")
        params.append(component)
    
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(limit)
    
    async with get_conn() as conn:
        rows = await conn.fetch(
            f"SELECT * FROM eval_metric_runs {where} ORDER BY created_at DESC LIMIT ${len(params)}",
            *params
        )
    return [dict(r) for r in rows]


@router.get("/aggregated")
async def get_aggregated_reports(period_type: str = Query("weekly")):
    """Return recent aggregated reports (weekly or monthly)."""
    async with get_conn() as conn:
        rows = await conn.fetch(
            """SELECT * FROM eval_aggregated_reports
               WHERE period_type=$1
               ORDER BY created_at DESC LIMIT 10""",
            period_type
        )
    return [dict(r) for r in rows]


@router.post("/run-manual/{session_id}")
async def run_manual_eval(session_id: str):
    """
    Manually trigger a full evaluation report for a past session.
    Reads existing data from evaluation_history and session_memory tables.
    Does not re-run the session.
    """
    async with get_conn() as conn:
        evals = await conn.fetch(
            "SELECT * FROM evaluation_history WHERE session_id=$1", session_id
        )
        session = await conn.fetchrow(
            "SELECT * FROM session_memory WHERE session_id=$1", session_id
        )
    
    if not evals:
        return {"error": "No evaluation data found for this session."}
    
    # Re-compute outcome metrics from stored data
    from evaluation.metrics.outcome_metrics import calibration_quality_score, session_efficiency_score
    
    calibration_deltas = [float(r["calibration_delta"]) for r in evals]
    cal_result = calibration_quality_score(calibration_deltas, session_id, evals[0]["student_id"])
    
    return {
        "session_id": session_id,
        "student_id": evals[0]["student_id"] if evals else None,
        "evaluations_found": len(evals),
        "calibration_quality": cal_result,
        "note": "Full re-evaluation from stored data. RAG and lesson metrics require live session data.",
        "trigger": "manual",
    }
```

Mount this router in `app/api.py` by adding after the existing router setup:

```python
from evaluation.api_router import router as eval_router
app.include_router(eval_router)
```

---

### Step 12 — Add dependencies

Add these to your `requirements.txt` (or `pyproject.toml`). Check each one is not already listed:

```
sentence-transformers>=2.7.0
apscheduler>=3.10.4
```

`sentence-transformers` already brings `torch` and `transformers` as dependencies.

---

## WHAT NOT TO DO

- Do NOT add evaluation code inside agent `run()` methods or inside `tool_call_loop()`
- Do NOT use `asyncio.sleep()` inside any evaluation function
- Do NOT use background threads (`threading.Thread`) — use `asyncio.to_thread()` for CPU-bound work
- Do NOT read from the database inside a metric function to get data that should be passed in as arguments
- Do NOT create a separate FastAPI app for evaluation — mount the router on the existing app
- Do NOT add evaluation logic to `BaseAgent` — it must stay in the `evaluation/` directory
- Do NOT run curriculum coverage or ordering scores inside the live session loop — they are triggered by `on_curriculum_built()` which is a fire-and-forget task

---

## VERIFICATION CHECKLIST

After implementation, verify these work:

```
[ ] python main.py starts with no import errors
[ ] POST /session/start creates a session normally with no change in behavior
[ ] After a complete session, eval_session_reports has one new row
[ ] After a complete session, eval_metric_runs has multiple rows (one per metric fired)
[ ] GET /eval/session/{session_id} returns a JSON report with rag_score, agent_score, etc.
[ ] GET /eval/metrics?component=rag returns only RAG metric rows
[ ] POST /eval/run-manual/{session_id} works on a past completed session
[ ] Killing the eval task (asyncio cancel) during a session does not affect lesson delivery
[ ] settings.eval_enabled = False in .env disables all evaluation with zero code change
[ ] Weekly scheduler job appears in APScheduler job list at startup
[ ] No changes to existing table schemas (only new tables added)
[ ] All LLM-as-judge calls use settings.eval_judge_model (8b), not reasoning_model (70b)
[ ] SentenceTransformer is loaded lazily (not at import time) — startup time not affected
```

---

## ATTACHED REFERENCE

The file `edumind_evaluation_metrics.md` contains the exact formulas, weight combinations, and threshold values for every metric. Implement them exactly as written — do not invent alternative formulas. The document is the source of truth for all numeric computations.
