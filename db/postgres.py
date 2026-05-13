"""
db/postgres.py
Connection pool (size 5) + init_db() + all 9 EduMind tables.

Tables:
  1. students          — identity + onboarding fields
  2. concept_mastery   — per-concept mastery scores
  3. metacognition     — MetacognitionProfile JSON
  4. curricula         — CurriculumPlan JSON + current_index
  5. evaluation_history— every EvaluationReport (written mid-session)
  6. session_memory    — end-of-session summaries
  7. decision_log      — every agent decision per session
  8. doubt_log         — doubt events with concept + type + count
  9. module_embeddings — chromadb ids for per-curriculum embeddings
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

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


# ── Schema ──────────────────────────────────────────────────────────────
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
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

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

-- 7. decision_log   ← written at session END (bulk insert of all decisions)
CREATE TABLE IF NOT EXISTS decision_log (
    id              SERIAL PRIMARY KEY,
    student_id      TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    session_id      TEXT NOT NULL,
    agent           TEXT NOT NULL,          -- orchestrator | evaluator | adaptation_engine etc.
    action          TEXT NOT NULL,
    rationale       TEXT NOT NULL DEFAULT '',
    payload_json    JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

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

    raw_url = settings.database_url.replace(
        "postgresql+asyncpg://", "postgresql://")

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Connecting to DB (attempt {}/{}, pool_size={})…",
                attempt, max_retries, settings.db_pool_size
            )
            _pool = await asyncpg.create_pool(
                dsn=raw_url,
                min_size=1,
                max_size=settings.db_pool_size,
            )
            async with get_conn() as conn:
                await conn.execute(_SCHEMA_SQL)
            logger.info("DB pool ready and schema applied.")
            return
        except Exception as e:
            logger.warning(
                "DB connection attempt {}/{} failed: {}",
                attempt,
                max_retries,
                e)
            if attempt == max_retries:
                logger.error(
                    "All {} DB connection attempts failed. Giving up.",
                    max_retries)
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


# ── Convenience helpers ─────────────────────────────────────────────────

async def upsert_student(
    student_id: str,
    name: str,
    domain: str,
    goal: str,
    pace: str,
) -> None:
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
    async with get_conn() as conn:
        row = await conn.fetchrow(
            "SELECT profile_json FROM metacognition WHERE student_id=$1", student_id
        )
        return json.loads(row["profile_json"]) if row else None


async def write_session_memory(
    student_id: str,
    session_id: str,
    summary: str,
    modules_covered: list,
    started_at,
) -> None:
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


async def bulk_write_decisions(decisions: list[dict]) -> None:
    if not decisions:
        return
    async with get_conn() as conn:
        await conn.executemany(
            """
            INSERT INTO decision_log
              (student_id, session_id, agent, action, rationale, payload_json)
            VALUES ($1,$2,$3,$4,$5,$6)
            """,
            [
                (
                    d["student_id"], d["session_id"], d["agent"],
                    d["action"], d.get("rationale", ""),
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
                decision_records = []
                for d in decisions:
                    data = d if isinstance(d, dict) else d.model_dump()
                    payload = data.get("payload", data)
                    if not isinstance(payload, dict):
                        payload = {"value": payload}
                    decision_records.append((
                        data.get("student_id", student_id),
                        data.get("session_id", session_id),
                        data.get(
                            "agent", payload.get(
                                "agent", "adaptation_engine")),
                        data.get("action", payload.get("action", "UNKNOWN")),
                        data.get("rationale", payload.get("reason", "")),
                        json.dumps(payload),
                    ))
                await conn.executemany(
                    """
                    INSERT INTO decision_log
                      (student_id, session_id, agent, action, rationale, payload_json)
                    VALUES ($1,$2,$3,$4,$5,$6)
                    """,
                    decision_records,
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
