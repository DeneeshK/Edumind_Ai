"""
db/postgres.py
Connection pool (size 5) + init_db() + all EduMind tables.

Tables:
  1. students          — identity + course preference fields
  2. concept_mastery   — per-concept mastery scores
  3. metacognition     — MetacognitionProfile JSON
  4. curricula         — CurriculumPlan JSON + current_index
  5. evaluation_history— every EvaluationReport (written mid-session)
  6. session_memory    — end-of-session summaries
  7. decision_log      — every agent decision per session/course
  8. doubt_log         — doubt events with concept + type + count
  9. module_embeddings — chromadb ids for per-curriculum embeddings
 10. dynamic_concept_summaries — generated summaries linked to ChromaDB
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator
import uuid

import asyncpg
from loguru import logger

from config import settings

# ── Module-level pool (initialised once via init_db) ─────────────────────────
_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    """Return the module-level connection pool; raise if not initialised."""
    if _pool is None:
        raise RuntimeError("DB pool not initialised — call init_db() first.")
    return _pool


@asynccontextmanager
async def get_conn() -> AsyncGenerator[asyncpg.Connection, None]:
    """Async context manager that checks out a connection from the pool."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        yield conn


# ── Schema ────────────────────────────────────────────────────────────────────
_SCHEMA_SQL = """
-- 1. students
CREATE TABLE IF NOT EXISTS students (
    student_id      TEXT PRIMARY KEY,
    name            TEXT NOT NULL DEFAULT '',
    domain          TEXT NOT NULL DEFAULT '',
    goal            TEXT NOT NULL DEFAULT '',
    pace            TEXT NOT NULL DEFAULT 'medium',  -- fast | medium | deep
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. concept_mastery
CREATE TABLE IF NOT EXISTS concept_mastery (
    id              SERIAL PRIMARY KEY,
    student_id      TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    concept         TEXT NOT NULL,
    mastery_score   FLOAT NOT NULL DEFAULT 0.0,
    correctness     FLOAT NOT NULL DEFAULT 0.0,
    depth           FLOAT NOT NULL DEFAULT 0.0,
    sessions_seen   INT NOT NULL DEFAULT 0,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (student_id, concept)
);

-- 3. metacognition
CREATE TABLE IF NOT EXISTS metacognition (
    student_id          TEXT PRIMARY KEY REFERENCES students(student_id) ON DELETE CASCADE,
    profile_json        JSONB NOT NULL DEFAULT '{}',
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 4. curricula
CREATE TABLE IF NOT EXISTS curricula (
    id              SERIAL PRIMARY KEY,
    student_id      TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    topic           TEXT NOT NULL,
    plan_json       JSONB NOT NULL,          -- full CurriculumPlan
    current_index   INT NOT NULL DEFAULT 0,
    version         INT NOT NULL DEFAULT 1,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    pace            TEXT NOT NULL DEFAULT 'medium',  -- fast | medium | deep — stored per-curriculum so sync never uses stale student pace
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
-- Migration: add pace column to existing curricula tables that pre-date this change
DO $$ BEGIN
    ALTER TABLE curricula ADD COLUMN IF NOT EXISTS pace TEXT NOT NULL DEFAULT 'medium';
EXCEPTION WHEN others THEN NULL;
END $$;

-- 5. evaluation_history   ← written MID-SESSION immediately after each eval
CREATE TABLE IF NOT EXISTS evaluation_history (
    id                  SERIAL PRIMARY KEY,
    student_id          TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    session_id          TEXT NOT NULL,
    concept             TEXT NOT NULL,
    correctness_score   FLOAT NOT NULL,
    depth_score         FLOAT NOT NULL,
    mastery_score       FLOAT NOT NULL,
    misconception_type  TEXT,               -- conceptual | formula_misuse | application_error | NULL
    misconception_detail TEXT NOT NULL DEFAULT '',
    confidence_stated   INT NOT NULL,        -- 1-5
    calibration_delta   FLOAT NOT NULL,
    questions_asked     INT NOT NULL,
    recommended_action  TEXT NOT NULL,
    raw_json            JSONB NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 6. session_memory   ← written at session END
CREATE TABLE IF NOT EXISTS session_memory (
    id              SERIAL PRIMARY KEY,
    student_id      TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    session_id      TEXT NOT NULL UNIQUE,
    summary_text    TEXT NOT NULL DEFAULT '',
    modules_covered JSONB NOT NULL DEFAULT '[]',
    started_at      TIMESTAMPTZ,
    ended_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 7. decision_log   ← written at session END and after course creation
CREATE TABLE IF NOT EXISTS decision_log (
    id              SERIAL PRIMARY KEY,
    student_id      TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    session_id      TEXT NOT NULL,
    course_id       TEXT NOT NULL DEFAULT '',
    agent           TEXT NOT NULL,          -- orchestrator | evaluator | adaptation_engine etc.
    action          TEXT NOT NULL,
    rationale       TEXT NOT NULL DEFAULT '',
    payload_json    JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
ALTER TABLE decision_log
    ADD COLUMN IF NOT EXISTS course_id TEXT NOT NULL DEFAULT '';
CREATE INDEX IF NOT EXISTS idx_decision_log_course
    ON decision_log(course_id) WHERE course_id != '';

-- 8. doubt_log
CREATE TABLE IF NOT EXISTS doubt_log (
    id              SERIAL PRIMARY KEY,
    student_id      TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    session_id      TEXT NOT NULL,
    concept         TEXT NOT NULL,
    doubt_text      TEXT NOT NULL DEFAULT '',
    doubt_type      TEXT NOT NULL DEFAULT 'general',   -- general | prerequisite | application
    count           INT NOT NULL DEFAULT 1,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 9. module_embeddings
CREATE TABLE IF NOT EXISTS module_embeddings (
    id              SERIAL PRIMARY KEY,
    student_id      TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    curriculum_id   INT NOT NULL REFERENCES curricula(id) ON DELETE CASCADE,
    module_id       TEXT NOT NULL,
    chromadb_id     TEXT NOT NULL,
    domain          TEXT NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (student_id, module_id)
);

-- 10. dynamic_concept_summaries
CREATE TABLE IF NOT EXISTS dynamic_concept_summaries (
    id              SERIAL PRIMARY KEY,
    student_id      TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    curriculum_id   INT NOT NULL REFERENCES curricula(id) ON DELETE CASCADE,
    module_id       TEXT NOT NULL,
    concept         TEXT NOT NULL,
    domain          TEXT NOT NULL DEFAULT '',
    summary_text    TEXT NOT NULL DEFAULT '',
    chromadb_id     TEXT NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (student_id, module_id)
);

-- 11. eval_metric_runs  ← one row per individual metric computation
CREATE TABLE IF NOT EXISTS eval_metric_runs (
    id              SERIAL PRIMARY KEY,
    student_id      TEXT REFERENCES students(student_id) ON DELETE SET NULL,
    session_id      TEXT,
    metric_name     TEXT NOT NULL,
    component       TEXT NOT NULL,
    score           FLOAT NOT NULL,
    details_json    JSONB NOT NULL DEFAULT '{}',
    trigger         TEXT NOT NULL DEFAULT 'auto',
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
    period_type     TEXT NOT NULL,
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

-- 14. users  ← auth abstraction; dev-login now, Google OAuth later
CREATE TABLE IF NOT EXISTS users (
    id              TEXT PRIMARY KEY,
    google_sub      TEXT UNIQUE,
    email           TEXT UNIQUE NOT NULL,
    name            TEXT NOT NULL DEFAULT '',
    avatar_url      TEXT NOT NULL DEFAULT '',
    student_id      TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_users_student ON users(student_id);

-- 15. courses  ← frontend-facing course history
CREATE TABLE IF NOT EXISTS courses (
    id              TEXT PRIMARY KEY,
    student_id      TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    curriculum_id   INT REFERENCES curricula(id) ON DELETE SET NULL,
    topic           TEXT NOT NULL,
    goal            TEXT NOT NULL DEFAULT '',
    pace            TEXT NOT NULL DEFAULT 'medium',
    title           TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'active',
    prior_knowledge TEXT NOT NULL DEFAULT '',
    personalization_profile JSONB NOT NULL DEFAULT '{}',
    progress        FLOAT NOT NULL DEFAULT 0.0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (curriculum_id)
);
CREATE INDEX IF NOT EXISTS idx_courses_student ON courses(student_id);
ALTER TABLE courses
    ADD COLUMN IF NOT EXISTS personalization_profile JSONB NOT NULL DEFAULT '{}';
-- Per-course web-search toggle. When FALSE the agent is never offered the MCP
-- web-search tools, so no Tavily/embedding work runs for that course.
ALTER TABLE courses
    ADD COLUMN IF NOT EXISTS web_search_enabled BOOLEAN NOT NULL DEFAULT FALSE;

-- 16. course_modules  ← stable reopenable modules and saved lesson content
CREATE TABLE IF NOT EXISTS course_modules (
    id                  TEXT NOT NULL,
    course_id           TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    module_index        INT NOT NULL,
    title               TEXT NOT NULL,
    concept             TEXT NOT NULL,
    description         TEXT NOT NULL DEFAULT '',
    prerequisites       JSONB NOT NULL DEFAULT '[]',
    estimated_minutes   INT NOT NULL DEFAULT 10,
    depth_level         TEXT NOT NULL DEFAULT 'standard',
    difficulty          TEXT NOT NULL DEFAULT 'introductory',
    roadmap_step_id     TEXT NOT NULL DEFAULT '',
    module_metadata     JSONB NOT NULL DEFAULT '{}',
    status              TEXT NOT NULL DEFAULT 'not_started',
    content_markdown    TEXT NOT NULL DEFAULT '',
    content_generated_at TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (course_id, id)
);
CREATE INDEX IF NOT EXISTS idx_course_modules_course ON course_modules(course_id);
CREATE INDEX IF NOT EXISTS idx_course_modules_status ON course_modules(status);
ALTER TABLE course_modules
    ADD COLUMN IF NOT EXISTS module_metadata JSONB NOT NULL DEFAULT '{}';
ALTER TABLE course_modules
    ADD COLUMN IF NOT EXISTS roadmap_step_id TEXT NOT NULL DEFAULT '';

-- 17. module_chat_messages
CREATE TABLE IF NOT EXISTS module_chat_messages (
    id                              TEXT PRIMARY KEY,
    student_id                      TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    course_id                       TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    module_id                       TEXT NOT NULL,
    role                            TEXT NOT NULL,
    message                         TEXT NOT NULL,
    doubt_type                      TEXT,
    related_concepts                JSONB NOT NULL DEFAULT '[]',
    possible_missing_prerequisites  JSONB NOT NULL DEFAULT '[]',
    created_at                      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_module_chat_course_module ON module_chat_messages(course_id, module_id);
CREATE INDEX IF NOT EXISTS idx_module_chat_student ON module_chat_messages(student_id);

-- 18. module_questions
CREATE TABLE IF NOT EXISTS module_questions (
    id              TEXT PRIMARY KEY,
    course_id       TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    module_id       TEXT NOT NULL,
    order_index     INT NOT NULL DEFAULT 0,
    question_text   TEXT NOT NULL,
    expected_answer TEXT NOT NULL DEFAULT '',
    source_quote    TEXT NOT NULL DEFAULT '',
    concepts_tested JSONB NOT NULL DEFAULT '[]',
    source_section  TEXT NOT NULL DEFAULT '',
    is_answerable_from_lesson BOOLEAN NOT NULL DEFAULT TRUE,
    difficulty      TEXT NOT NULL DEFAULT 'simple',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_module_questions_module ON module_questions(course_id, module_id);
ALTER TABLE module_questions
    ADD COLUMN IF NOT EXISTS concepts_tested JSONB NOT NULL DEFAULT '[]';
ALTER TABLE module_questions
    ADD COLUMN IF NOT EXISTS source_section TEXT NOT NULL DEFAULT '';
ALTER TABLE module_questions
    ADD COLUMN IF NOT EXISTS is_answerable_from_lesson BOOLEAN NOT NULL DEFAULT TRUE;

-- 19. student_skills  ← lightweight skill graph/list nodes
CREATE TABLE IF NOT EXISTS student_skills (
    id              SERIAL PRIMARY KEY,
    student_id      TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    concept         TEXT NOT NULL,
    mastery_score   FLOAT NOT NULL DEFAULT 0.0,
    depth_score     FLOAT NOT NULL DEFAULT 0.0,
    source          TEXT NOT NULL DEFAULT 'course',
    status          TEXT NOT NULL DEFAULT 'learning',
    evidence_json   JSONB NOT NULL DEFAULT '{}',
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (student_id, concept, source)
);
CREATE INDEX IF NOT EXISTS idx_student_skills_student ON student_skills(student_id);

-- 20. course_roadmaps  ← personalized roadmap/study plan JSON per course
CREATE TABLE IF NOT EXISTS course_roadmaps (
    course_id       TEXT PRIMARY KEY REFERENCES courses(id) ON DELETE CASCADE,
    roadmap_json    JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 21. master_roadmaps  ← source-of-truth roadmap before modules are derived
CREATE TABLE IF NOT EXISTS master_roadmaps (
    course_id       TEXT PRIMARY KEY REFERENCES courses(id) ON DELETE CASCADE,
    roadmap_json    JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 22. evaluation_sessions — one per module evaluation run
CREATE TABLE IF NOT EXISTS evaluation_sessions (
    session_id          TEXT PRIMARY KEY,
    course_id           TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    module_id           TEXT NOT NULL,
    student_id          TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    status              TEXT NOT NULL DEFAULT 'active',
    questions_asked     INT NOT NULL DEFAULT 0,
    questions_json      JSONB NOT NULL DEFAULT '[]',
    answers_json        JSONB NOT NULL DEFAULT '[]',
    final_report_json   JSONB NOT NULL DEFAULT '{}',
    decision            TEXT NOT NULL DEFAULT '',
    motivational_feedback TEXT NOT NULL DEFAULT '',
    transition_feedback TEXT NOT NULL DEFAULT '',
    reteach_data_json   JSONB NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_eval_sessions_course_module
    ON evaluation_sessions(course_id, module_id);
CREATE INDEX IF NOT EXISTS idx_eval_sessions_student
    ON evaluation_sessions(student_id);

-- 23. adaptation_summaries — compact adaptation context per module
CREATE TABLE IF NOT EXISTS adaptation_summaries (
    id                  SERIAL PRIMARY KEY,
    student_id          TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    course_id           TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    module_id           TEXT NOT NULL,
    summary_json        JSONB NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (student_id, course_id, module_id)
);
CREATE INDEX IF NOT EXISTS idx_adaptation_summaries_student_course
    ON adaptation_summaries(student_id, course_id);

-- 24. course_completion_reports — final performance report after all modules done
CREATE TABLE IF NOT EXISTS course_completion_reports (
    id              SERIAL PRIMARY KEY,
    course_id       TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    student_id      TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    report_json     JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (course_id, student_id)
);
CREATE INDEX IF NOT EXISTS idx_course_completion_reports_student
    ON course_completion_reports(student_id);

-- 25. course_schedules — AI-generated learning timetable per course per student
CREATE TABLE IF NOT EXISTS course_schedules (
    id              TEXT PRIMARY KEY,
    course_id       TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    student_id      TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    schedule_json   JSONB NOT NULL DEFAULT '{}',
    total_days      INT NOT NULL DEFAULT 1,
    hours_per_day   FLOAT NOT NULL DEFAULT 1.0,
    study_slots     JSONB NOT NULL DEFAULT '[]',
    start_date      DATE,
    end_date        DATE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (course_id, student_id)
);
CREATE INDEX IF NOT EXISTS idx_course_schedules_student
    ON course_schedules(student_id);
CREATE INDEX IF NOT EXISTS idx_course_schedules_course
    ON course_schedules(course_id);
"""


async def init_db(max_retries: int = 5, retry_delay: float = 2.0) -> None:
    """
    Create the connection pool and run the schema DDL.
    Retries up to max_retries times with exponential backoff —
    handles the common case where PostgreSQL is still starting
    when the application launches (e.g. Docker Compose startup order).
    Call once at application startup.
    """
    import asyncio as _asyncio
    global _pool

    raw_url = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Connecting to DB (attempt {}/{}, pool_size={})…",
                attempt, max_retries, settings.db_pool_size
            )
            _pool = await asyncpg.create_pool(
                dsn=raw_url,
                min_size=2,
                max_size=settings.db_pool_size,
                command_timeout=10,                    # fail DB commands after 10s
                max_inactive_connection_lifetime=300,  # recycle idle connections after 5min
            )
            async with get_conn() as conn:
                await conn.execute(_SCHEMA_SQL)
            logger.info("DB pool ready and schema applied.")
            return
        except Exception as e:
            logger.warning("DB connection attempt {}/{} failed: {}", attempt, max_retries, e)
            if attempt == max_retries:
                logger.error("All {} DB connection attempts failed. Giving up.", max_retries)
                raise
            wait = retry_delay * (2 ** (attempt - 1))  # exponential backoff
            logger.info("Retrying in {:.1f}s…", wait)
            await _asyncio.sleep(wait)


async def close_db() -> None:
    """Gracefully close the pool at shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("DB pool closed.")


# ── Convenience helpers ───────────────────────────────────────────────────────

async def upsert_student(
    student_id: str,
    name: str,
    domain: str,
    goal: str,
    pace: str,
) -> None:
    """Insert or update a student's identity and current course preferences."""
    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO students (student_id, name, domain, goal, pace)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (student_id) DO UPDATE
              SET name=$2, domain=$3, goal=$4, pace=$5, updated_at=NOW()
            """,
            student_id, name, domain, goal, pace,
        )


async def write_evaluation(record: dict[str, Any]) -> None:
    """Persist an EvaluationReport immediately after evaluation (mid-session)."""
    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO evaluation_history
              (student_id, session_id, concept, correctness_score, depth_score,
               mastery_score, misconception_type, misconception_detail,
               confidence_stated, calibration_delta, questions_asked,
               recommended_action, raw_json)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
            """,
            record["student_id"],
            record["session_id"],
            record["concept"],
            record["correctness_score"],
            record["depth_score"],
            record["mastery_score"],
            record.get("misconception_type"),
            record.get("misconception_detail", ""),
            record["confidence_stated"],
            record["calibration_delta"],
            record["questions_asked"],
            record["recommended_action"],
            json.dumps(record),
        )


async def upsert_concept_mastery(
    student_id: str, concept: str, correctness: float, depth: float
) -> None:
    """Persist the latest mastery calculation for one student/concept pair."""
    mastery = round(0.6 * correctness + 0.4 * depth, 4)
    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO concept_mastery
              (student_id, concept, mastery_score, correctness, depth, sessions_seen)
            VALUES ($1, $2, $3, $4, $5, 1)
            ON CONFLICT (student_id, concept) DO UPDATE
              SET mastery_score=$3, correctness=$4, depth=$5,
                  sessions_seen = concept_mastery.sessions_seen + 1,
                  updated_at = NOW()
            """,
            student_id, concept, mastery, correctness, depth,
        )


async def save_metacognition(student_id: str, profile_json: dict) -> None:
    """Persist the student's metacognition profile JSON."""
    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO metacognition (student_id, profile_json)
            VALUES ($1, $2)
            ON CONFLICT (student_id) DO UPDATE
              SET profile_json=$2, updated_at=NOW()
            """,
            student_id, json.dumps(profile_json),
        )


async def load_metacognition(student_id: str) -> dict | None:
    """Load a student's metacognition profile JSON if it exists."""
    async with get_conn() as conn:
        row = await conn.fetchrow(
            "SELECT profile_json FROM metacognition WHERE student_id=$1", student_id
        )
        return _json_value(row["profile_json"], {}) if row else None


async def write_session_memory(
    student_id: str,
    session_id: str,
    summary: str,
    modules_covered: list,
    started_at,
) -> None:
    """Persist an end-of-session memory summary."""
    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO session_memory
              (student_id, session_id, summary_text, modules_covered, started_at)
            VALUES ($1,$2,$3,$4,$5)
            ON CONFLICT (session_id) DO NOTHING
            """,
            student_id, session_id, summary,
            json.dumps(modules_covered), started_at,
        )


async def bulk_write_decisions(decisions: list[dict], course_id: str = "") -> None:
    """Persist multiple agent decision-log records."""
    if not decisions:
        return
    async with get_conn() as conn:
        await conn.executemany(
            """
            INSERT INTO decision_log
              (student_id, session_id, course_id, agent, action, rationale, payload_json)
            VALUES ($1,$2,$3,$4,$5,$6,$7)
            """,
            [
                (
                    d["student_id"], d["session_id"],
                    d.get("course_id") or course_id or "",
                    d["agent"], d["action"], d.get("rationale", ""),
                    json.dumps(d.get("payload", {})),
                )
                for d in decisions
            ],
        )


async def flush_session_to_db(
    student_id: str,
    session_id: str,
    summary: str,
    modules_covered: list,
    started_at,
    decisions: list[dict],
    metacognition_json: dict,
    mastery_updates: list[dict],
    doubt_records: list[dict] | None = None,
) -> None:
    """
    Atomically write all session-end data in a single transaction.
    If any write fails, everything rolls back — no partial data.
    Called by orchestrator._end_session() instead of separate writes.
    """
    async with get_conn() as conn:
        async with conn.transaction():
            # Session summary
            await conn.execute(
                """
                INSERT INTO session_memory
                  (student_id, session_id, summary_text, modules_covered, started_at)
                VALUES ($1,$2,$3,$4,$5)
                ON CONFLICT (session_id) DO NOTHING
                """,
                student_id, session_id, summary,
                json.dumps(modules_covered), started_at,
            )
            # Decision log
            if decisions:
                await conn.executemany(
                    """
                    INSERT INTO decision_log
                      (student_id, session_id, course_id, agent, action, rationale, payload_json)
                    VALUES ($1,$2,$3,$4,$5,$6,$7)
                    """,
                    [
                        (
                            d["student_id"], d["session_id"],
                            d.get("course_id", ""),
                            d["agent"], d["action"], d.get("rationale", ""),
                            json.dumps(d.get("payload", {})),
                        )
                        for d in decisions
                    ],
                )
            # Metacognition
            await conn.execute(
                """
                INSERT INTO metacognition (student_id, profile_json)
                VALUES ($1, $2)
                ON CONFLICT (student_id) DO UPDATE
                  SET profile_json=$2, updated_at=NOW()
                """,
                student_id, json.dumps(metacognition_json),
            )
            # Mastery scores
            for m in mastery_updates:
                await conn.execute(
                    """
                    INSERT INTO concept_mastery
                      (student_id, concept, mastery_score, correctness, depth, sessions_seen)
                    VALUES ($1,$2,$3,$4,$5,1)
                    ON CONFLICT (student_id, concept) DO UPDATE
                      SET mastery_score=$3, correctness=$4, depth=$5,
                          sessions_seen = concept_mastery.sessions_seen + 1,
                          updated_at = NOW()
                    """,
                    student_id, m["concept"],
                    m["mastery_score"], m["correctness"], m["depth"],
                )
            # Doubt log
            for d in doubt_records or []:
                await conn.execute(
                    """
                    INSERT INTO doubt_log
                      (student_id, session_id, concept, count, doubt_type)
                    VALUES ($1,$2,$3,$4,$5)
                    """,
                    student_id,
                    session_id,
                    d["concept"],
                    d.get("count", 1),
                    d.get("doubt_type", "general"),
                )


# ── Frontend course/auth helpers ─────────────────────────────────────────────

def _json_value(value: Any, default: Any = None) -> Any:
    """Decode JSON strings while passing through already-decoded values."""
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def _course_id_for_curriculum(curriculum_id: int) -> str:
    """Return the stable frontend course id for a curriculum row id."""
    return f"course-{curriculum_id}"


def _course_title_from_profile(plan, personalization_profile: dict[str, Any] | None, pace: str) -> str:
    """Build the saved course title from profile metadata and plan topic."""
    profile = personalization_profile or {}
    topic = profile.get("course_scope") or profile.get("exact_subject") or plan.topic
    pace_label = {"fast": "Fast", "medium": "Balanced", "deep": "Deep"}.get(
        pace,
        str(pace or "Balanced").title(),
    )
    learner = str(profile.get("learner_level") or "").lower()
    learner_label = "Beginner" if "beginner" in learner or "fresh" in learner else ""
    labels = [pace_label]
    if learner_label and learner_label.lower() not in str(topic).lower():
        labels.append(learner_label)
    labels.append("Course")
    return f"{str(topic).strip()}: {' '.join(labels)}"


async def ensure_student(
    student_id: str,
    name: str = "Student",
    domain: str = "",
    goal: str = "",
    pace: str = "medium",
) -> None:
    """Create a minimal student row without erasing existing learning fields."""
    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO students (student_id, name, domain, goal, pace)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (student_id) DO UPDATE
              SET name=COALESCE(NULLIF($2, ''), students.name),
                  domain=COALESCE(NULLIF($3, ''), students.domain),
                  goal=COALESCE(NULLIF($4, ''), students.goal),
                  pace=COALESCE(NULLIF($5, ''), students.pace),
                  updated_at=NOW()
            """,
            student_id,
            name,
            domain,
            goal,
            pace,
        )


async def upsert_dev_user(
    email: str,
    name: str,
    avatar_url: str = "",
    google_sub: str | None = None,
) -> dict[str, Any]:
    """Dev-auth compatible user upsert. Google OAuth can reuse this shape."""
    email = (email or "student@edumind.dev").strip().lower()
    name = (name or "EduMind Student").strip()

    async with get_conn() as conn:
        existing = await conn.fetchrow(
            "SELECT * FROM users WHERE email=$1", email
        )
        if existing:
            student_id = existing["student_id"]
            await conn.execute(
                """
                UPDATE users
                   SET name=$2, avatar_url=$3,
                       google_sub=COALESCE($4, google_sub),
                       updated_at=NOW()
                 WHERE email=$1
                """,
                email,
                name,
                avatar_url,
                google_sub,
            )
        else:
            student_id = "dev-" + uuid.uuid5(
                uuid.NAMESPACE_URL, "edumind:" + email
            ).hex[:16]
            user_id = str(uuid.uuid4())
            await conn.execute(
                """
                INSERT INTO students (student_id, name)
                VALUES ($1, $2)
                ON CONFLICT (student_id) DO UPDATE
                  SET name=$2, updated_at=NOW()
                """,
                student_id,
                name,
            )
            await conn.execute(
                """
                INSERT INTO users
                  (id, google_sub, email, name, avatar_url, student_id)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                user_id,
                google_sub,
                email,
                name,
                avatar_url,
                student_id,
            )

        await conn.execute(
            """
            INSERT INTO students (student_id, name)
            VALUES ($1, $2)
            ON CONFLICT (student_id) DO UPDATE
              SET name=$2, updated_at=NOW()
            """,
            student_id,
            name,
        )
        row = await conn.fetchrow("SELECT * FROM users WHERE email=$1", email)

    return {
        "id": row["id"],
        "student_id": row["student_id"],
        "email": row["email"],
        "name": row["name"],
        "avatar_url": row["avatar_url"],
    }


async def upsert_google_user(
    google_sub: str,
    email: str,
    name: str,
    avatar_url: str = "",
) -> dict[str, Any]:
    """Create or update a user from a verified Google ID token."""
    google_sub = (google_sub or "").strip()
    email = (email or "").strip().lower()
    name = (name or "EduMind Student").strip()
    avatar_url = (avatar_url or "").strip()
    if not google_sub:
        raise ValueError("google_sub is required")
    if not email:
        raise ValueError("email is required")

    async with get_conn() as conn:
        existing = await conn.fetchrow(
            "SELECT * FROM users WHERE google_sub=$1", google_sub
        )
        if not existing:
            existing = await conn.fetchrow(
                "SELECT * FROM users WHERE email=$1", email
            )

        if existing:
            student_id = existing["student_id"]
            await conn.execute(
                """
                UPDATE users
                   SET google_sub=$2,
                       email=$3,
                       name=$4,
                       avatar_url=$5,
                       updated_at=NOW()
                 WHERE id=$1
                """,
                existing["id"],
                google_sub,
                email,
                name,
                avatar_url,
            )
        else:
            student_id = "google-" + uuid.uuid5(
                uuid.NAMESPACE_URL, "edumind-google:" + google_sub
            ).hex[:16]
            await conn.execute(
                """
                INSERT INTO students (student_id, name)
                VALUES ($1, $2)
                ON CONFLICT (student_id) DO UPDATE
                  SET name=$2, updated_at=NOW()
                """,
                student_id,
                name,
            )
            await conn.execute(
                """
                INSERT INTO users
                  (id, google_sub, email, name, avatar_url, student_id)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                str(uuid.uuid4()),
                google_sub,
                email,
                name,
                avatar_url,
                student_id,
            )

        await conn.execute(
            """
            INSERT INTO students (student_id, name)
            VALUES ($1, $2)
            ON CONFLICT (student_id) DO UPDATE
              SET name=$2, updated_at=NOW()
            """,
            student_id,
            name,
        )
        row = await conn.fetchrow("SELECT * FROM users WHERE google_sub=$1", google_sub)

    return {
        "id": row["id"],
        "student_id": row["student_id"],
        "email": row["email"],
        "name": row["name"],
        "avatar_url": row["avatar_url"],
    }


async def get_user_by_student_id(student_id: str) -> dict[str, Any] | None:
    """Return the auth user linked to a student id, if one exists."""
    async with get_conn() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM users WHERE student_id=$1", student_id
        )
    if not row:
        return None
    return {
        "id": row["id"],
        "student_id": row["student_id"],
        "email": row["email"],
        "name": row["name"],
        "avatar_url": row["avatar_url"],
    }


async def latest_curriculum_for_student(student_id: str):
    """Return the newest curriculum row for a student, regardless of active flag."""
    async with get_conn() as conn:
        return await conn.fetchrow(
            """
            SELECT id, student_id, topic, plan_json, current_index, version,
                   is_active, created_at, updated_at
              FROM curricula
             WHERE student_id=$1
             ORDER BY id DESC
             LIMIT 1
            """,
            student_id,
        )


async def create_course_from_plan(
    student_id: str,
    curriculum_id: int,
    plan,
    pace: str,
    prior_knowledge: str = "",
    personalization_profile: dict[str, Any] | None = None,
    web_search_enabled: bool | None = None,
) -> dict[str, Any]:
    """
    Create/update frontend course rows from a CurriculumPlan.

    `web_search_enabled=None` means "leave as-is" so background resyncs never
    clobber a course's toggle; pass an explicit bool only on real creation.
    """
    course_id = _course_id_for_curriculum(curriculum_id)
    title = _course_title_from_profile(plan, personalization_profile, pace)
    profile_json = json.dumps(personalization_profile or {})
    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO courses
              (id, student_id, curriculum_id, topic, goal, pace, title,
               status, prior_knowledge, personalization_profile, web_search_enabled)
            VALUES ($1,$2,$3,$4,$5,$6,$7,'active',$8,$9,COALESCE($10, FALSE))
            ON CONFLICT (id) DO UPDATE
              SET topic=$4, goal=$5, pace=$6, title=$7,
                  status='active', prior_knowledge=$8,
                  personalization_profile=CASE
                    WHEN $9::jsonb = '{}'::jsonb THEN courses.personalization_profile
                    ELSE $9::jsonb
                  END,
                  web_search_enabled=COALESCE($10, courses.web_search_enabled),
                  updated_at=NOW()
            """,
            course_id,
            student_id,
            curriculum_id,
            plan.topic,
            plan.goal,
            pace,
            title,
            prior_knowledge,
            profile_json,
            web_search_enabled,
        )

        for idx, module in enumerate(plan.modules):
            status = "in_progress" if idx == int(plan.current_index or 0) else "not_started"
            if idx < int(plan.current_index or 0):
                status = "completed"
            await conn.execute(
                """
                INSERT INTO course_modules
                  (course_id, id, module_index, title, concept, description,
                   prerequisites, estimated_minutes, depth_level, difficulty,
                   roadmap_step_id, module_metadata, status)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
                ON CONFLICT (course_id, id) DO UPDATE
                  SET module_index=$3, title=$4, concept=$5,
                      description=COALESCE(NULLIF($6, ''), course_modules.description),
                      prerequisites=$7, estimated_minutes=$8, depth_level=$9,
                      difficulty=$10,
                      roadmap_step_id=$11,
                      module_metadata=$12,
                      status=CASE
                        WHEN course_modules.status='completed' THEN 'completed'
                        ELSE $13
                      END,
                      updated_at=NOW()
                """,
                course_id,
                module.id,
                idx,
                module.title,
                module.concept,
                module.domain_framing,
                json.dumps(module.prerequisites),
                module.estimated_minutes,
                module.depth_level,
                module.difficulty or (
                    "advanced" if module.depth_level == "deep"
                    else "focused" if module.depth_level == "standard"
                    else "introductory"
                ),
                getattr(module, "roadmap_step_id", "") or "",
                json.dumps({
                    "purpose": module.purpose,
                    "why_it_matters_for_goal": module.why_it_matters_for_goal,
                    "must_teach": module.must_teach,
                    "examples_to_include": module.examples_to_include,
                    "practice_type": module.practice_type,
                    "concepts_taught": module.concepts_taught,
                    "depends_on_concepts": module.depends_on_concepts,
                    "unlocks_concepts": module.unlocks_concepts,
                    "module_goal": module.module_goal,
                    "why_now": module.why_now,
                    "what_this_module_will_not_cover": module.what_this_module_will_not_cover,
                    "lesson_requirements": module.lesson_requirements,
                    "practice_requirements": module.practice_requirements,
                    "question_scope": module.question_scope,
                }),
                status,
            )

    await recalculate_course_progress(course_id)
    return await get_course(course_id) or {}


async def sync_courses_for_student(student_id: str) -> None:
    """Expose old curricula rows as frontend courses/modules."""
    from core.student_model import CurriculumPlan

    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT c.*
              FROM curricula c
             WHERE c.student_id=$1
             ORDER BY c.id DESC
            """,
            student_id,
        )

    for row in rows:
        plan_data = _json_value(row["plan_json"], {})
        if not plan_data:
            continue
        plan_data["current_index"] = row["current_index"]
        plan_data["version"] = row["version"]
        try:
            plan = CurriculumPlan(**plan_data)
        except Exception as exc:
            logger.warning("Skipping curriculum sync id={}: {}", row["id"], exc)
            continue
        # Use the pace stored on the curriculum row itself (not the student's current
        # global pace) so that changing pace for a new course never retroactively
        # relabels older courses.
        curriculum_pace = row["pace"] if row["pace"] else "medium"
        await create_course_from_plan(
            student_id=student_id,
            curriculum_id=int(row["id"]),
            plan=plan,
            pace=curriculum_pace,
        )


async def recalculate_course_progress(course_id: str) -> float:
    """Recompute and persist course progress from module completion state."""
    async with get_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT COUNT(*)::FLOAT AS total,
                   SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END)::FLOAT AS done
              FROM course_modules
             WHERE course_id=$1
            """,
            course_id,
        )
        total = float(row["total"] or 0)
        done = float(row["done"] or 0)
        progress = round(done / total, 4) if total else 0.0
        status = "completed" if total and done >= total else "active"
        await conn.execute(
            "UPDATE courses SET progress=$2, status=$3, updated_at=NOW() WHERE id=$1",
            course_id,
            progress,
            status,
        )
    return progress


async def list_courses(student_id: str) -> list[dict[str, Any]]:
    """Return frontend course summaries for one student."""
    await sync_courses_for_student(student_id)
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT c.*,
                   COUNT(cm.id)::INT AS module_count,
                   SUM(CASE WHEN cm.status='completed' THEN 1 ELSE 0 END)::INT AS completed_modules
              FROM courses c
              LEFT JOIN course_modules cm ON cm.course_id=c.id
             WHERE c.student_id=$1
             GROUP BY c.id
             ORDER BY c.updated_at DESC
            """,
            student_id,
        )
    courses = []
    for row in rows:
        data = dict(row)
        data["personalization_profile"] = _json_value(
            data.get("personalization_profile"), {}
        )
        courses.append(data)
    return courses


async def get_course(course_id: str, student_id: str | None = None) -> dict[str, Any] | None:
    """Return a course summary, optionally scoped to one student."""
    async with get_conn() as conn:
        where_student = "AND c.student_id=$2" if student_id else ""
        args = (course_id, student_id) if student_id else (course_id,)
        row = await conn.fetchrow(
            f"""
            SELECT c.*,
                   COUNT(cm.id)::INT AS module_count,
                   SUM(CASE WHEN cm.status='completed' THEN 1 ELSE 0 END)::INT AS completed_modules
              FROM courses c
              LEFT JOIN course_modules cm ON cm.course_id=c.id
             WHERE c.id=$1 {where_student}
             GROUP BY c.id
            """,
            *args,
        )
    if not row:
        return None
    data = dict(row)
    data["personalization_profile"] = _json_value(
        data.get("personalization_profile"), {}
    )
    return data


async def get_course_for_student(course_id: str, student_id: str) -> dict[str, Any] | None:
    """Return a course only when it belongs to the given student."""
    return await get_course(course_id, student_id)


async def delete_course(course_id: str, student_id: str | None = None) -> bool:
    """
    Delete a course and linked curriculum when it is visible to the caller.

    Also purges any web-search RAG chunks stored under this course's namespace
    in the MCP search server — otherwise they accumulate there indefinitely
    with no owning course left to reference them. Purge failure (e.g. the MCP
    server is unreachable) never blocks the course deletion itself.
    """
    async with get_conn() as conn:
        async with conn.transaction():
            where_student = "AND student_id=$2" if student_id else ""
            args = (course_id, student_id) if student_id else (course_id,)
            row = await conn.fetchrow(
                f"""
                SELECT id, student_id, curriculum_id
                  FROM courses
                 WHERE id=$1 {where_student}
                """,
                *args,
            )
            if not row:
                return False

            await conn.execute("DELETE FROM decision_log WHERE course_id=$1", course_id)
            await conn.execute("DELETE FROM courses WHERE id=$1", course_id)

            if row["curriculum_id"] is not None:
                await conn.execute(
                    "DELETE FROM curricula WHERE id=$1 AND student_id=$2",
                    row["curriculum_id"],
                    row["student_id"],
                )

    try:
        # Lazy import: db/postgres.py is core infra and shouldn't hard-depend on
        # the MCP client SDK for a codepath most requests never touch.
        from clients.mcp_search_client import purge_namespace
        await purge_namespace(course_id)
    except Exception as exc:
        logger.warning("Web-search chunk purge failed for course_id={}: {}", course_id, exc)

    return True


async def save_course_roadmap(course_id: str, roadmap: dict[str, Any]) -> dict[str, Any]:
    """Upsert the frontend roadmap JSON for a course."""
    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO course_roadmaps (course_id, roadmap_json)
            VALUES ($1, $2)
            ON CONFLICT (course_id) DO UPDATE
              SET roadmap_json=$2, updated_at=NOW()
            """,
            course_id,
            json.dumps(roadmap),
        )
    return await get_course_roadmap(course_id) or roadmap


async def get_course_roadmap(course_id: str) -> dict[str, Any] | None:
    """Load the saved frontend roadmap JSON for a course."""
    async with get_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT course_id, roadmap_json, created_at, updated_at
              FROM course_roadmaps
             WHERE course_id=$1
            """,
            course_id,
        )
    if not row:
        return None
    roadmap = _json_value(row["roadmap_json"], {})
    if not isinstance(roadmap, dict):
        roadmap = {}
    roadmap.setdefault("course_id", row["course_id"])
    roadmap.setdefault("created_at", row["created_at"])
    roadmap.setdefault("updated_at", row["updated_at"])

    # Patch module_timeline with live status and recommended_next from DB.
    # The saved roadmap JSON is written once at generation time and never
    # updated, so recommended_next is always stale. We fix it here so the
    # frontend always sees the correct next module to study.
    timeline = roadmap.get("module_timeline")
    if timeline:
        live_modules = await list_course_modules(course_id)
        live_by_id = {m["id"]: m for m in live_modules}
        # First non-completed module id is the recommended one
        recommended_id = next(
            (m["id"] for m in live_modules if m["status"] != "completed"),
            live_modules[-1]["id"] if live_modules else None,
        )
        for item in timeline:
            mid = item.get("module_id")
            if mid and mid in live_by_id:
                item["status"] = live_by_id[mid]["status"]
                item["recommended_next"] = (mid == recommended_id)

    return roadmap


async def save_master_roadmap(course_id: str, roadmap: dict[str, Any]) -> dict[str, Any]:
    """Upsert the source roadmap used before course modules are derived."""
    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO master_roadmaps (course_id, roadmap_json)
            VALUES ($1, $2)
            ON CONFLICT (course_id) DO UPDATE
              SET roadmap_json=$2, updated_at=NOW()
            """,
            course_id,
            json.dumps(roadmap),
        )
    return await get_master_roadmap(course_id) or roadmap


async def get_master_roadmap(course_id: str) -> dict[str, Any] | None:
    """Load the saved source roadmap JSON for a course."""
    async with get_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT course_id, roadmap_json, created_at, updated_at
              FROM master_roadmaps
             WHERE course_id=$1
            """,
            course_id,
        )
    if not row:
        return None
    roadmap = _json_value(row["roadmap_json"], {})
    if not isinstance(roadmap, dict):
        roadmap = {}
    roadmap.setdefault("course_id", row["course_id"])
    roadmap.setdefault("created_at", row["created_at"])
    roadmap.setdefault("updated_at", row["updated_at"])
    return roadmap


def _module_payload(row, recommended_id: str | None = None) -> dict[str, Any]:
    """Convert a course_modules row into the frontend module response shape."""
    data = dict(row)
    data["prerequisites"] = _json_value(data.get("prerequisites"), [])
    data["module_metadata"] = _json_value(data.get("module_metadata"), {})
    if isinstance(data["module_metadata"], dict):
        for key, value in data["module_metadata"].items():
            data.setdefault(key, value)
    data["content_exists"] = bool(data.get("content_markdown"))
    data["recommended"] = data["id"] == recommended_id
    return data


async def list_course_modules(course_id: str) -> list[dict[str, Any]]:
    """Return all modules for a course, marking the next unfinished module."""
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM course_modules
             WHERE course_id=$1
             ORDER BY module_index ASC
            """,
            course_id,
        )
    recommended_id = next(
        (r["id"] for r in rows if r["status"] != "completed"),
        rows[-1]["id"] if rows else None,
    )
    return [_module_payload(r, recommended_id) for r in rows]


async def get_eval_summaries_for_course(
    course_id: str, student_id: str
) -> dict[str, dict[str, Any]]:
    """
    Bulk-fetch the latest completed evaluation summary per module for one student.
    Returns a dict keyed by module_id with mastery_score, has_report, session_id, decision.
    Uses a single query — safe to call on every module-list request.
    """
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (es.module_id)
                   es.module_id,
                   es.session_id,
                   es.decision,
                   (es.final_report_json->>'mastery_score')::float AS mastery_score
              FROM evaluation_sessions es
              JOIN courses c ON c.id = es.course_id
             WHERE es.course_id = $1
               AND es.student_id = $2
               AND c.student_id = $2
               AND es.status = 'completed'
             ORDER BY es.module_id, es.created_at DESC
            """,
            course_id,
            student_id,
        )
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        result[row["module_id"]] = {
            "session_id": row["session_id"],
            "decision": row["decision"] or "",
            "mastery_score": row["mastery_score"],
            "has_report": True,
        }
    return result


async def list_course_modules_for_student(course_id: str, student_id: str) -> list[dict[str, Any]]:
    """Return course modules only after confirming the course belongs to the student."""
    course = await get_course_for_student(course_id, student_id)
    if not course:
        return []
    modules = await list_course_modules(course_id)
    eval_summaries = await get_eval_summaries_for_course(course_id, student_id)
    for mod in modules:
        summary = eval_summaries.get(mod["id"])
        if summary:
            mod["latest_mastery_score"] = summary["mastery_score"]
            mod["has_eval_report"] = True
            mod["latest_eval_session_id"] = summary["session_id"]
            mod["latest_eval_decision"] = summary["decision"]
        else:
            mod["latest_mastery_score"] = None
            mod["has_eval_report"] = False
            mod["latest_eval_session_id"] = None
            mod["latest_eval_decision"] = None
    return modules


async def get_course_module(course_id: str, module_id: str) -> dict[str, Any] | None:
    """Return one course module with decoded metadata and recommendation state."""
    async with get_conn() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM course_modules WHERE course_id=$1 AND id=$2",
            course_id,
            module_id,
        )
    if not row:
        return None
    modules = await list_course_modules(course_id)
    recommended_id = next((m["id"] for m in modules if m["recommended"]), None)
    return _module_payload(row, recommended_id)


async def get_course_module_for_student(
    course_id: str,
    module_id: str,
    student_id: str,
) -> dict[str, Any] | None:
    """Return one module only when the enclosing course belongs to the student."""
    course = await get_course_for_student(course_id, student_id)
    if not course:
        return None
    return await get_course_module(course_id, module_id)


async def set_module_status(course_id: str, module_id: str, status: str) -> None:
    """Update a module status and recalculate overall course progress."""
    async with get_conn() as conn:
        await conn.execute(
            """
            UPDATE course_modules
               SET status=$3, updated_at=NOW()
             WHERE course_id=$1 AND id=$2
            """,
            course_id,
            module_id,
            status,
        )
    await recalculate_course_progress(course_id)


async def save_module_content(
    course_id: str,
    module_id: str,
    content_markdown: str,
    questions: list[dict[str, Any]] | None = None,
    videos: list[dict[str, Any]] | None = None,
) -> None:
    """Persist generated lesson markdown, optional videos, and optional questions."""
    async with get_conn() as conn:
        await conn.execute(
            """
            UPDATE course_modules
               SET content_markdown=$3,
                   content_generated_at=NOW(),
                   status=CASE
                     WHEN status='not_started' THEN 'in_progress'
                     ELSE status
                   END,
                   updated_at=NOW()
             WHERE course_id=$1 AND id=$2
            """,
            course_id,
            module_id,
            content_markdown,
        )
        if videos is not None:
            await conn.execute(
                """
                UPDATE course_modules
                   SET module_metadata=jsonb_set(
                         COALESCE(module_metadata, '{}'::jsonb),
                         '{lesson_videos}',
                         $3::jsonb,
                         true
                       ),
                       updated_at=NOW()
                 WHERE course_id=$1 AND id=$2
                """,
                course_id,
                module_id,
                json.dumps(videos),
            )
    if questions is not None:
        await save_module_questions(course_id, module_id, questions)
    await recalculate_course_progress(course_id)


async def save_module_questions(
    course_id: str,
    module_id: str,
    questions: list[dict[str, Any]],
) -> None:
    """Replace saved grounded questions for one course module."""
    async with get_conn() as conn:
        await conn.execute(
            "DELETE FROM module_questions WHERE course_id=$1 AND module_id=$2",
            course_id,
            module_id,
        )
        for idx, q in enumerate(questions):
	            await conn.execute(
	                """
	                INSERT INTO module_questions
	                  (id, course_id, module_id, order_index, question_text,
	                   expected_answer, source_quote, concepts_tested,
	                   source_section, is_answerable_from_lesson, difficulty)
	                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
	                """,
	                q.get("id") or f"{course_id}:{module_id}:q{idx + 1}",
	                course_id,
	                module_id,
	                idx,
	                q.get("question_text", ""),
	                q.get("expected_answer", ""),
	                q.get("source_quote", ""),
	                json.dumps(q.get("concepts_tested") or []),
	                q.get("source_section", ""),
	                bool(q.get("is_answerable_from_lesson", True)),
	                q.get("difficulty", "simple"),
	            )


async def get_module_questions(course_id: str, module_id: str) -> list[dict[str, Any]]:
    """Return saved grounded questions for one course module."""
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM module_questions
             WHERE course_id=$1 AND module_id=$2
             ORDER BY order_index ASC
            """,
            course_id,
            module_id,
        )
    result = []
    for row in rows:
        item = dict(row)
        item["concepts_tested"] = _json_value(item.get("concepts_tested"), [])
        result.append(item)
    return result


async def record_module_chat_message(
    student_id: str,
    course_id: str,
    module_id: str,
    role: str,
    message: str,
    doubt_type: str | None = None,
    related_concepts: list[str] | None = None,
    possible_missing_prerequisites: list[str] | None = None,
) -> dict[str, Any]:
    """Persist one module chat message and return the saved payload shape."""
    msg_id = str(uuid.uuid4())
    related_concepts = related_concepts or []
    possible_missing_prerequisites = possible_missing_prerequisites or []
    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO module_chat_messages
              (id, student_id, course_id, module_id, role, message, doubt_type,
               related_concepts, possible_missing_prerequisites)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
            """,
            msg_id,
            student_id,
            course_id,
            module_id,
            role,
            message,
            doubt_type,
            json.dumps(related_concepts),
            json.dumps(possible_missing_prerequisites),
        )
    return {
        "id": msg_id,
        "student_id": student_id,
        "course_id": course_id,
        "module_id": module_id,
        "role": role,
        "message": message,
        "doubt_type": doubt_type,
        "related_concepts": related_concepts,
        "possible_missing_prerequisites": possible_missing_prerequisites,
    }


async def list_module_chat_history(course_id: str, module_id: str) -> list[dict[str, Any]]:
    """Return all chat messages saved for one course module."""
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM module_chat_messages
             WHERE course_id=$1 AND module_id=$2
             ORDER BY created_at ASC
            """,
            course_id,
            module_id,
        )
    history = []
    for row in rows:
        data = dict(row)
        data["related_concepts"] = _json_value(data.get("related_concepts"), [])
        data["possible_missing_prerequisites"] = _json_value(
            data.get("possible_missing_prerequisites"), []
        )
        history.append(data)
    return history


async def list_module_chat_history_for_student(
    course_id: str,
    module_id: str,
    student_id: str,
) -> list[dict[str, Any]]:
    """Return module chat history filtered to the owning student."""
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM module_chat_messages
             WHERE course_id=$1 AND module_id=$2 AND student_id=$3
             ORDER BY created_at ASC
            """,
            course_id,
            module_id,
            student_id,
        )
    history = []
    for row in rows:
        data = dict(row)
        data["related_concepts"] = _json_value(data.get("related_concepts"), [])
        data["possible_missing_prerequisites"] = _json_value(
            data.get("possible_missing_prerequisites"), []
        )
        history.append(data)
    return history


async def record_doubt(
    student_id: str,
    course_id: str,
    concept: str,
    doubt_text: str,
    doubt_type: str,
) -> None:
    """Persist one side-chat or lesson doubt as adaptation evidence."""
    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO doubt_log
              (student_id, session_id, concept, doubt_text, doubt_type, count)
            VALUES ($1,$2,$3,$4,$5,1)
            """,
            student_id,
            "course:" + course_id,
            concept,
            doubt_text,
            doubt_type,
        )


async def upsert_student_skill(
    student_id: str,
    concept: str,
    mastery_score: float,
    depth_score: float,
    source: str,
    status: str,
    evidence: dict[str, Any] | None = None,
) -> None:
    """Upsert one skill-graph node from evaluation or course evidence."""
    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO student_skills
              (student_id, concept, mastery_score, depth_score, source,
               status, evidence_json)
            VALUES ($1,$2,$3,$4,$5,$6,$7)
            ON CONFLICT (student_id, concept, source) DO UPDATE
              SET mastery_score=$3, depth_score=$4, status=$6,
                  evidence_json=$7, updated_at=NOW()
            """,
            student_id,
            concept,
            mastery_score,
            depth_score,
            source,
            status,
            json.dumps(evidence or {}),
        )


async def get_student_dashboard(student_id: str) -> dict[str, Any]:
    """Build the dashboard summary from courses, mastery, doubts, and metacognition."""
    courses = await list_courses(student_id)
    async with get_conn() as conn:
        mastery_rows = await conn.fetch(
            """
            SELECT concept, mastery_score, depth
              FROM concept_mastery
             WHERE student_id=$1
             ORDER BY updated_at DESC
            """,
            student_id,
        )
        doubt_rows = await conn.fetch(
            """
            SELECT concept, doubt_type, doubt_text, created_at
              FROM doubt_log
             WHERE student_id=$1
             ORDER BY created_at DESC
             LIMIT 10
            """,
            student_id,
        )
        meta_row = await conn.fetchrow(
            "SELECT profile_json FROM metacognition WHERE student_id=$1",
            student_id,
        )
    mastered = [
        dict(r) for r in mastery_rows if float(r["mastery_score"] or 0) >= 0.75
    ]
    weak = [
        dict(r) for r in mastery_rows if float(r["mastery_score"] or 0) < 0.55
    ]
    return {
        "student_id": student_id,
        "courses": courses,
        "summary": {
            "courses_created": len(courses),
            "completed_modules": sum(int(c.get("completed_modules") or 0) for c in courses),
            "active_courses": len([c for c in courses if c.get("status") != "completed"]),
            "mastered_concepts": len(mastered),
            "weak_concepts": len(weak),
        },
        "mastered_concepts": mastered[:8],
        "weak_concepts": weak[:8],
        "recent_doubts": [dict(r) for r in doubt_rows],
        "metacognition": _json_value(meta_row["profile_json"], {}) if meta_row else {},
    }


async def get_student_skills(student_id: str) -> dict[str, Any]:
    """Return mastery and skill rows as graph-ready skill nodes."""
    async with get_conn() as conn:
        mastery_rows = await conn.fetch(
            """
            SELECT concept, mastery_score, depth, 'evaluation' AS source, updated_at
              FROM concept_mastery
             WHERE student_id=$1
            """,
            student_id,
        )
        skill_rows = await conn.fetch(
            """
            SELECT concept, mastery_score, depth_score, source, status,
                   evidence_json, updated_at
              FROM student_skills
             WHERE student_id=$1
            """,
            student_id,
        )
    nodes = []
    for r in mastery_rows:
        score = float(r["mastery_score"] or 0)
        nodes.append({
            "concept": r["concept"],
            "mastery_score": score,
            "depth_score": float(r["depth"] or 0),
            "source": r["source"],
            "status": "mastered" if score >= 0.75 else "weak" if score < 0.55 else "learning",
            "updated_at": r["updated_at"],
        })
    for r in skill_rows:
        data = dict(r)
        data["evidence_json"] = _json_value(data.get("evidence_json"), {})
        nodes.append(data)
    return {
        "student_id": student_id,
        "nodes": nodes,
        "edges": [],
        "weak_concepts": [n for n in nodes if n.get("status") == "weak"],
        "strong_concepts": [n for n in nodes if n.get("status") == "mastered"],
    }


async def get_student_doubts(student_id: str) -> list[dict[str, Any]]:
    """Return the student's doubt log in newest-first order."""
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT id, session_id, concept, doubt_text, doubt_type, count, created_at
              FROM doubt_log
             WHERE student_id=$1
             ORDER BY created_at DESC
            """,
            student_id,
        )
    return [dict(r) for r in rows]


async def get_course_decision_log(course_id: str) -> list[dict[str, Any]]:
    """Return recent agent decisions associated with a course or its session."""
    async with get_conn() as conn:
        course = await conn.fetchrow(
            "SELECT student_id, curriculum_id FROM courses WHERE id=$1",
            course_id,
        )
        if not course:
            return []
        rows = await conn.fetch(
            """
            SELECT *
              FROM decision_log
             WHERE course_id=$3
                OR session_id=$2
                OR (student_id=$1 AND session_id=$2)
             ORDER BY created_at DESC
             LIMIT 100
            """,
            course["student_id"],
            "course:" + course_id,
            course_id,
        )
    logs = []
    for row in rows:
        data = dict(row)
        data["payload_json"] = _json_value(data.get("payload_json"), {})
        logs.append(data)
    return logs


async def save_evaluation_session(session: dict[str, Any]) -> None:
    """Upsert the mutable state for one module evaluation session."""
    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO evaluation_sessions
              (session_id, course_id, module_id, student_id, status,
               questions_asked, questions_json, answers_json,
               final_report_json, decision, motivational_feedback,
               transition_feedback, reteach_data_json)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
            ON CONFLICT (session_id) DO UPDATE
              SET status=$5, questions_asked=$6, questions_json=$7,
                  answers_json=$8, final_report_json=$9, decision=$10,
                  motivational_feedback=$11, transition_feedback=$12,
                  reteach_data_json=$13, updated_at=NOW()
            """,
            session["session_id"], session["course_id"], session["module_id"],
            session["student_id"], session.get("status", "active"),
            session.get("questions_asked", 0),
            json.dumps(session.get("questions", [])),
            json.dumps(session.get("answers", [])),
            json.dumps(session.get("final_report", {})),
            session.get("decision", ""),
            session.get("motivational_feedback", ""),
            session.get("transition_feedback", ""),
            json.dumps(session.get("reteach_data", {})),
        )


async def get_evaluation_session(session_id: str) -> dict[str, Any] | None:
    """Load an evaluation session and decode its JSON columns."""
    async with get_conn() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM evaluation_sessions WHERE session_id=$1", session_id
        )
    if not row:
        return None
    data = dict(row)
    for key in ("questions_json", "answers_json", "final_report_json", "reteach_data_json"):
        data[key] = _json_value(data.get(key), {} if "report" in key or "reteach" in key else [])
    return data


async def get_evaluation_session_for_student(
    session_id: str,
    student_id: str,
) -> dict[str, Any] | None:
    """Load an evaluation session only when it belongs to the student."""
    async with get_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM evaluation_sessions
             WHERE session_id=$1 AND student_id=$2
            """,
            session_id,
            student_id,
        )
    if not row:
        return None
    data = dict(row)
    for key in ("questions_json", "answers_json", "final_report_json", "reteach_data_json"):
        data[key] = _json_value(data.get(key), {} if "report" in key or "reteach" in key else [])
    return data


async def get_latest_evaluation_session(
    course_id: str, module_id: str
) -> dict[str, Any] | None:
    """Return the newest evaluation session saved for a course module."""
    async with get_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM evaluation_sessions
             WHERE course_id=$1 AND module_id=$2
             ORDER BY created_at DESC LIMIT 1
            """,
            course_id, module_id,
        )
    if not row:
        return None
    data = dict(row)
    for key in ("questions_json", "answers_json", "final_report_json", "reteach_data_json"):
        data[key] = _json_value(data.get(key), {} if "report" in key or "reteach" in key else [])
    return data


async def get_latest_evaluation_session_for_student(
    course_id: str,
    module_id: str,
    student_id: str,
) -> dict[str, Any] | None:
    """Return the newest module evaluation visible to the owning student."""
    async with get_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT es.*
              FROM evaluation_sessions es
              JOIN courses c ON c.id=es.course_id
             WHERE es.course_id=$1
               AND es.module_id=$2
               AND es.student_id=$3
               AND c.student_id=$3
             ORDER BY es.created_at DESC LIMIT 1
            """,
            course_id,
            module_id,
            student_id,
        )
    if not row:
        return None
    data = dict(row)
    for key in ("questions_json", "answers_json", "final_report_json", "reteach_data_json"):
        data[key] = _json_value(data.get(key), {} if "report" in key or "reteach" in key else [])
    return data



async def save_adaptation_summary(
    student_id: str, course_id: str, module_id: str, summary: dict[str, Any]
) -> None:
    """Persist compact adaptation notes for future lessons in this module."""
    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO adaptation_summaries
              (student_id, course_id, module_id, summary_json)
            VALUES ($1,$2,$3,$4)
            ON CONFLICT (student_id, course_id, module_id) DO UPDATE
              SET summary_json=$4, updated_at=NOW()
            """,
            student_id, course_id, module_id, json.dumps(summary),
        )


async def get_adaptation_summary(
    student_id: str, course_id: str, module_id: str | None = None
) -> dict[str, Any] | None:
    """Load adaptation notes for one module or the latest course-level summary."""
    async with get_conn() as conn:
        if module_id:
            row = await conn.fetchrow(
                """
                SELECT summary_json FROM adaptation_summaries
                 WHERE student_id=$1 AND course_id=$2 AND module_id=$3
                """,
                student_id, course_id, module_id,
            )
        else:
            row = await conn.fetchrow(
                """
                SELECT summary_json FROM adaptation_summaries
                 WHERE student_id=$1 AND course_id=$2
                 ORDER BY updated_at DESC LIMIT 1
                """,
                student_id, course_id,
            )
    return _json_value(row["summary_json"], {}) if row else None


async def get_compact_doubt_summary(
    course_id: str,
    module_id: str,
    student_id: str | None = None,
) -> str:
    """Return a compact 1-3 sentence summary of student doubts for this module."""
    async with get_conn() as conn:
        where_student = "AND student_id=$3" if student_id else ""
        args = (course_id, module_id, student_id) if student_id else (course_id, module_id)
        rows = await conn.fetch(
            f"""
            SELECT role, message, doubt_type, created_at
              FROM module_chat_messages
             WHERE course_id=$1 AND module_id=$2 AND role='user' {where_student}
             ORDER BY created_at ASC
            """,
            *args,
        )
    if not rows:
        return ""
    doubt_types: dict[str, int] = {}
    messages = []
    for row in rows:
        dtype = row.get("doubt_type") or "general"
        doubt_types[dtype] = doubt_types.get(dtype, 0) + 1
        messages.append(str(row["message"] or "").strip())
    total = len(messages)
    if total == 0:
        return ""
    parts = [f"Student asked {total} question(s) in the side chat."]
    if doubt_types:
        top = sorted(doubt_types.items(), key=lambda x: -x[1])[:2]
        parts.append("Main doubt types: " + ", ".join(f"{k} ({v})" for k, v in top) + ".")
    if total >= 3:
        parts.append("Repeated doubts suggest the student needs more worked examples or clearer explanations.")
    return " ".join(parts)


async def get_next_module(course_id: str, current_module_id: str) -> dict[str, Any] | None:
    """Return the module immediately after current_module_id by module_index, or None."""
    async with get_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM course_modules
             WHERE course_id=$1
               AND module_index > (
                   SELECT module_index FROM course_modules
                    WHERE course_id=$1 AND id=$2
               )
             ORDER BY module_index ASC
             LIMIT 1
            """,
            course_id, current_module_id,
        )
    if not row:
        return None
    modules = await list_course_modules(course_id)
    recommended_id = next((m["id"] for m in modules if m.get("recommended")), None)
    return _module_payload(row, recommended_id)


async def get_prev_module(course_id: str, current_module_id: str) -> dict[str, Any] | None:
    """Return the module immediately before current_module_id by module_index, or None."""
    async with get_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM course_modules
             WHERE course_id=$1
               AND module_index < (
                   SELECT module_index FROM course_modules
                    WHERE course_id=$1 AND id=$2
               )
             ORDER BY module_index DESC
             LIMIT 1
            """,
            course_id, current_module_id,
        )
    if not row:
        return None
    modules = await list_course_modules(course_id)
    recommended_id = next((m["id"] for m in modules if m.get("recommended")), None)
    return _module_payload(row, recommended_id)


async def save_course_completion_report(
    course_id: str, student_id: str, report: dict[str, Any]
) -> None:
    """Save the final report generated after a student completes a course."""
    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO course_completion_reports (course_id, student_id, report_json)
            VALUES ($1, $2, $3)
            ON CONFLICT (course_id, student_id) DO UPDATE
              SET report_json=$3, created_at=NOW()
            """,
            course_id, student_id, json.dumps(report),
        )


async def get_course_completion_report(
    course_id: str, student_id: str
) -> dict[str, Any] | None:
    """Return the saved course completion report for the owning student."""
    async with get_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT report_json, created_at FROM course_completion_reports
             WHERE course_id=$1 AND student_id=$2
            """,
            course_id, student_id,
        )
    if not row:
        return None
    return {
        "report": _json_value(row["report_json"], {}),
        "created_at": row["created_at"],
    }


# ── Learning Schedule helpers (table 25) ──────────────────────────────────────

async def save_course_schedule(
    schedule_id: str,
    course_id: str,
    student_id: str,
    schedule: dict[str, Any],
) -> None:
    """Upsert the AI-generated learning schedule (one per course per student)."""
    # asyncpg requires datetime.date objects for DATE columns, not ISO strings
    from datetime import date as _date
    def _to_date(val):
        if val is None:
            return None
        if isinstance(val, _date):
            return val
        try:
            return _date.fromisoformat(str(val))
        except ValueError:
            return None

    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO course_schedules
              (id, course_id, student_id, schedule_json,
               total_days, hours_per_day, study_slots, start_date, end_date)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (course_id, student_id) DO UPDATE
              SET id=$1,
                  schedule_json=$4,
                  total_days=$5,
                  hours_per_day=$6,
                  study_slots=$7,
                  start_date=$8,
                  end_date=$9,
                  updated_at=NOW()
            """,
            schedule_id,
            course_id,
            student_id,
            json.dumps(schedule),
            schedule.get("total_days", 1),
            float(schedule.get("hours_per_day", 1.0)),
            json.dumps(schedule.get("study_slots", [])),
            _to_date(schedule.get("start_date")),
            _to_date(schedule.get("end_date")),
        )


async def get_course_schedule(
    course_id: str,
    student_id: str,
) -> dict[str, Any] | None:
    """Return the saved learning schedule for a course, or None."""
    async with get_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT schedule_json, created_at, updated_at
              FROM course_schedules
             WHERE course_id=$1 AND student_id=$2
            """,
            course_id, student_id,
        )
    if not row:
        return None
    return {
        **_json_value(row["schedule_json"], {}),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


async def update_schedule_module_completion(
    course_id: str,
    student_id: str,
    module_id: str,
    completed: bool,
) -> bool:
    """
    Toggle the completed flag on a specific module inside the schedule JSONB.
    Returns True if schedule was found and updated.
    """
    async with get_conn() as conn:
        row = await conn.fetchrow(
            "SELECT schedule_json FROM course_schedules WHERE course_id=$1 AND student_id=$2",
            course_id, student_id,
        )
        if not row:
            return False

        schedule = _json_value(row["schedule_json"], {})
        for day in schedule.get("days", []):
            for item in day.get("timetable_items", []):
                if item.get("module_id") == module_id:
                    item["completed"] = completed

        await conn.execute(
            """
            UPDATE course_schedules
               SET schedule_json=$1, updated_at=NOW()
             WHERE course_id=$2 AND student_id=$3
            """,
            json.dumps(schedule), course_id, student_id,
        )
    return True