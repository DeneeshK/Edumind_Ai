"""
db/institution.py
Repository functions for the "My Institution" module: classrooms, members,
classroom courses, per-student course assignments (clones), AI tests,
learning events, cached AI artifacts, assistant chat, and classroom posts.

All functions use the shared asyncpg pool from db.postgres.get_conn().
This module is purely additive — it never modifies existing tables.
"""

from __future__ import annotations

import json
import secrets
import string
import uuid
from typing import Any

from loguru import logger

from db.postgres import get_conn


# ── Helpers ───────────────────────────────────────────────────────────────────

def _json_value(value: Any, default: Any) -> Any:
    """Parse a JSONB column value that may arrive as str, dict, or None."""
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _row_dict(row: Any, json_fields: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convert an asyncpg Record to a dict, parsing listed JSONB fields."""
    data = dict(row)
    for field, default in (json_fields or {}).items():
        data[field] = _json_value(data.get(field), default)
    return data


def new_id(prefix: str) -> str:
    """Generate a short unique id with a readable prefix."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def generate_join_code(length: int = 7) -> str:
    """Generate a human-friendly join code (no ambiguous characters)."""
    alphabet = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(length))


# ── Classrooms ────────────────────────────────────────────────────────────────

async def create_classroom(
    owner_student_id: str,
    name: str,
    subject: str = "",
    grade_level: str = "",
    description: str = "",
    join_policy: str = "approval",
) -> dict[str, Any]:
    """Create a classroom owned by the given student (who becomes its teacher)."""
    classroom_id = new_id("cls")
    async with get_conn() as conn:
        # Retry join-code collisions (astronomically rare, but cheap to guard)
        for _ in range(5):
            code = generate_join_code()
            exists = await conn.fetchval(
                "SELECT 1 FROM classrooms WHERE join_code=$1", code
            )
            if not exists:
                break
        await conn.execute(
            """
            INSERT INTO classrooms
              (id, owner_student_id, name, subject, grade_level, description,
               join_code, join_policy)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
            """,
            classroom_id, owner_student_id, name, subject, grade_level,
            description, code, join_policy,
        )
    return await get_classroom(classroom_id) or {}


async def get_classroom(classroom_id: str) -> dict[str, Any] | None:
    """Return one classroom with member counts, or None."""
    async with get_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT c.*,
                   COUNT(m.id) FILTER (WHERE m.status='active')::INT  AS member_count,
                   COUNT(m.id) FILTER (WHERE m.status='pending')::INT AS pending_count
              FROM classrooms c
              LEFT JOIN classroom_members m ON m.classroom_id = c.id
             WHERE c.id=$1
             GROUP BY c.id
            """,
            classroom_id,
        )
    return _row_dict(row, {"settings_json": {}}) if row else None


async def get_classroom_by_join_code(join_code: str) -> dict[str, Any] | None:
    """Look up an active classroom by its join code."""
    async with get_conn() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM classrooms WHERE join_code=$1 AND status='active'",
            join_code.strip().upper(),
        )
    return _row_dict(row, {"settings_json": {}}) if row else None


async def update_classroom(classroom_id: str, fields: dict[str, Any]) -> dict[str, Any] | None:
    """Update editable classroom fields (name/subject/grade/description/policy/status)."""
    allowed = {"name", "subject", "grade_level", "description", "join_policy", "status"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if updates:
        sets = ", ".join(f"{col}=${i + 2}" for i, col in enumerate(updates))
        async with get_conn() as conn:
            await conn.execute(
                f"UPDATE classrooms SET {sets}, updated_at=NOW() WHERE id=$1",
                classroom_id, *updates.values(),
            )
    return await get_classroom(classroom_id)


async def regenerate_join_code(classroom_id: str) -> str:
    """Replace the classroom's join code and return the new code."""
    async with get_conn() as conn:
        for _ in range(5):
            code = generate_join_code()
            exists = await conn.fetchval("SELECT 1 FROM classrooms WHERE join_code=$1", code)
            if not exists:
                await conn.execute(
                    "UPDATE classrooms SET join_code=$2, updated_at=NOW() WHERE id=$1",
                    classroom_id, code,
                )
                return code
    raise RuntimeError("Could not generate a unique join code")


async def list_classrooms_for_student(student_id: str) -> dict[str, list[dict[str, Any]]]:
    """Return classrooms where the student teaches and where they are a member."""
    async with get_conn() as conn:
        teaching = await conn.fetch(
            """
            SELECT c.*,
                   COUNT(m.id) FILTER (WHERE m.status='active')::INT  AS member_count,
                   COUNT(m.id) FILTER (WHERE m.status='pending')::INT AS pending_count
              FROM classrooms c
              LEFT JOIN classroom_members m ON m.classroom_id=c.id
             WHERE c.owner_student_id=$1
             GROUP BY c.id
             ORDER BY c.status='active' DESC, c.updated_at DESC
            """,
            student_id,
        )
        joined = await conn.fetch(
            """
            SELECT c.*, m.status AS membership_status, m.role AS member_role,
                   COUNT(m2.id) FILTER (WHERE m2.status='active')::INT AS member_count
              FROM classroom_members m
              JOIN classrooms c ON c.id=m.classroom_id
              LEFT JOIN classroom_members m2 ON m2.classroom_id=c.id
             WHERE m.student_id=$1 AND m.status IN ('pending','active')
               AND c.status='active'
             GROUP BY c.id, m.status, m.role
             ORDER BY c.updated_at DESC
            """,
            student_id,
        )
    return {
        "teaching": [_row_dict(r, {"settings_json": {}}) for r in teaching],
        "joined": [_row_dict(r, {"settings_json": {}}) for r in joined],
    }


# ── Members ───────────────────────────────────────────────────────────────────

async def get_membership(classroom_id: str, student_id: str) -> dict[str, Any] | None:
    """Return the membership row for a student in a classroom, if any."""
    async with get_conn() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM classroom_members WHERE classroom_id=$1 AND student_id=$2",
            classroom_id, student_id,
        )
    return dict(row) if row else None


async def upsert_membership(
    classroom_id: str,
    student_id: str,
    status: str,
    display_name: str = "",
    role: str = "student",
) -> dict[str, Any]:
    """Create or update a membership row (used by join / approve / remove flows)."""
    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO classroom_members
              (classroom_id, student_id, role, status, display_name, joined_at)
            VALUES ($1,$2,$3,$4,$5, CASE WHEN $4='active' THEN NOW() ELSE NULL END)
            ON CONFLICT (classroom_id, student_id) DO UPDATE
              SET status=$4,
                  role=$3,
                  display_name=CASE WHEN $5='' THEN classroom_members.display_name ELSE $5 END,
                  joined_at=CASE
                    WHEN $4='active' AND classroom_members.joined_at IS NULL THEN NOW()
                    ELSE classroom_members.joined_at
                  END
            """,
            classroom_id, student_id, role, status, display_name,
        )
    return await get_membership(classroom_id, student_id) or {}


async def list_members(classroom_id: str, statuses: list[str] | None = None) -> list[dict[str, Any]]:
    """List classroom members joined with user identity, filtered by status."""
    statuses = statuses or ["active", "pending"]
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT m.*, u.name AS user_name, u.email, u.avatar_url
              FROM classroom_members m
              LEFT JOIN users u ON u.student_id=m.student_id
             WHERE m.classroom_id=$1 AND m.status = ANY($2::text[])
             ORDER BY m.status DESC, m.joined_at NULLS LAST, m.created_at
            """,
            classroom_id, statuses,
        )
    members = []
    for row in rows:
        data = dict(row)
        data["name"] = data.get("display_name") or data.get("user_name") or "Student"
        members.append(data)
    return members


async def active_member_ids(classroom_id: str) -> list[str]:
    """Return student_ids of all active (non-teacher) members."""
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT student_id FROM classroom_members
             WHERE classroom_id=$1 AND status='active'
            """,
            classroom_id,
        )
    return [r["student_id"] for r in rows]


# ── Invitations (email allowlist) ─────────────────────────────────────────────

def normalize_email(email: str) -> str:
    """Lowercase and trim an email for consistent allowlist matching."""
    return str(email or "").strip().lower()


async def add_invitations(
    classroom_id: str, students: list[dict[str, Any]], invited_by: str
) -> list[dict[str, Any]]:
    """
    Insert (or update) email invitations from {email, name, phone} entries.
    Re-invites a previously revoked email, and refreshes name/phone when given.
    """
    async with get_conn() as conn:
        for student in students:
            await conn.execute(
                """
                INSERT INTO classroom_invitations
                  (id, classroom_id, email, name, phone, invited_by, status)
                VALUES ($1,$2,$3,$4,$5,$6,'invited')
                ON CONFLICT (classroom_id, email) DO UPDATE
                  SET name  = CASE WHEN $4 <> '' THEN $4 ELSE classroom_invitations.name END,
                      phone = CASE WHEN $5 <> '' THEN $5 ELSE classroom_invitations.phone END,
                      invited_by = $6,
                      status = CASE WHEN classroom_invitations.status = 'revoked'
                                    THEN 'invited' ELSE classroom_invitations.status END,
                      accepted_at = CASE WHEN classroom_invitations.status = 'revoked'
                                    THEN NULL ELSE classroom_invitations.accepted_at END
                """,
                new_id("inv"), classroom_id,
                student["email"], student.get("name", ""), student.get("phone", ""),
                invited_by,
            )
    return await list_invitations(classroom_id)


async def list_invitations(classroom_id: str) -> list[dict[str, Any]]:
    """List non-revoked invitations with roster details and account status."""
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT ci.email, ci.name, ci.phone, ci.status, ci.created_at, ci.accepted_at,
                   u.name AS account_name,
                   (u.id IS NOT NULL) AS has_account
              FROM classroom_invitations ci
              LEFT JOIN users u ON lower(u.email) = ci.email
             WHERE ci.classroom_id=$1 AND ci.status <> 'revoked'
             ORDER BY ci.status, ci.created_at DESC
            """,
            classroom_id,
        )
    return [dict(r) for r in rows]


async def get_invitation(classroom_id: str, email: str) -> dict[str, Any] | None:
    """Return one invitation row by classroom + email, if present."""
    async with get_conn() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM classroom_invitations WHERE classroom_id=$1 AND email=$2",
            classroom_id, normalize_email(email),
        )
    return dict(row) if row else None


async def revoke_invitation(classroom_id: str, email: str) -> None:
    """Revoke an email's invitation (does not remove an already-joined student)."""
    async with get_conn() as conn:
        await conn.execute(
            """
            UPDATE classroom_invitations SET status='revoked'
             WHERE classroom_id=$1 AND email=$2
            """,
            classroom_id, normalize_email(email),
        )


async def mark_invitation_accepted(classroom_id: str, email: str) -> None:
    """Flag an invitation accepted after the student joins."""
    async with get_conn() as conn:
        await conn.execute(
            """
            UPDATE classroom_invitations
               SET status='accepted', accepted_at=NOW()
             WHERE classroom_id=$1 AND email=$2
            """,
            classroom_id, normalize_email(email),
        )


async def list_invitations_for_email(email: str) -> list[dict[str, Any]]:
    """
    Classrooms an email is invited to but hasn't joined yet — powers the
    student's "You've been invited" section. Excludes classrooms where the
    student is already an active member.
    """
    normalized = normalize_email(email)
    if not normalized:
        return []
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT ci.classroom_id, ci.created_at,
                   c.name, c.subject, c.grade_level, c.description,
                   COALESCE(u.name, 'A teacher') AS teacher_name
              FROM classroom_invitations ci
              JOIN classrooms c ON c.id = ci.classroom_id
              LEFT JOIN users u ON u.student_id = c.owner_student_id
             WHERE ci.email = $1
               AND ci.status = 'invited'
               AND c.status = 'active'
               AND NOT EXISTS (
                   SELECT 1 FROM classroom_members m
                    WHERE m.classroom_id = ci.classroom_id
                      AND m.student_id = (
                          SELECT student_id FROM users WHERE lower(email)=$1 LIMIT 1
                      )
                      AND m.status = 'active'
               )
             ORDER BY ci.created_at DESC
            """,
            normalized,
        )
    return [dict(r) for r in rows]


# ── Classroom courses & assignments ──────────────────────────────────────────

async def create_classroom_course(
    classroom_id: str,
    template_course_id: str,
    title: str,
) -> dict[str, Any]:
    """Register a teacher's course as a draft classroom course."""
    cc_id = new_id("cc")
    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO classroom_courses (id, classroom_id, template_course_id, title)
            VALUES ($1,$2,$3,$4)
            """,
            cc_id, classroom_id, template_course_id, title,
        )
    return await get_classroom_course(cc_id) or {}


async def get_classroom_course(cc_id: str) -> dict[str, Any] | None:
    """Return one classroom course with assignment counts."""
    async with get_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT cc.*, c.topic, c.goal, c.pace,
                   COUNT(ca.id)::INT AS assigned_count
              FROM classroom_courses cc
              JOIN courses c ON c.id=cc.template_course_id
              LEFT JOIN course_assignments ca ON ca.classroom_course_id=cc.id
             WHERE cc.id=$1
             GROUP BY cc.id, c.topic, c.goal, c.pace
            """,
            cc_id,
        )
    return dict(row) if row else None


async def list_classroom_courses(classroom_id: str) -> list[dict[str, Any]]:
    """List classroom courses with template info and assignment counts."""
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT cc.*, c.topic, c.goal, c.pace,
                   COUNT(ca.id)::INT AS assigned_count,
                   COUNT(cm.id)::INT AS module_count
              FROM classroom_courses cc
              JOIN courses c ON c.id=cc.template_course_id
              LEFT JOIN course_assignments ca ON ca.classroom_course_id=cc.id
              LEFT JOIN course_modules cm ON cm.course_id=cc.template_course_id
             WHERE cc.classroom_id=$1
             GROUP BY cc.id, c.topic, c.goal, c.pace
             ORDER BY cc.created_at DESC
            """,
            classroom_id,
        )
    return [dict(r) for r in rows]


async def set_classroom_course_status(cc_id: str, status: str) -> None:
    """Update a classroom course lifecycle status."""
    async with get_conn() as conn:
        await conn.execute(
            """
            UPDATE classroom_courses
               SET status=$2,
                   assigned_at=CASE WHEN $2='assigned' THEN NOW() ELSE assigned_at END,
                   updated_at=NOW()
             WHERE id=$1
            """,
            cc_id, status,
        )


async def get_assignment(cc_id: str, student_id: str) -> dict[str, Any] | None:
    """Return an existing assignment row for (classroom course, student)."""
    async with get_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM course_assignments
             WHERE classroom_course_id=$1 AND student_id=$2
            """,
            cc_id, student_id,
        )
    return dict(row) if row else None


async def clone_course_for_student(
    template_course_id: str,
    student_id: str,
    classroom_id: str,
    classroom_course_id: str,
) -> str:
    """
    Clone a template course's metadata for one student and record the assignment.

    Copies: courses row (new id, NULL curriculum_id so the legacy curricula sync
    never touches clones), course_modules metadata (content_markdown is NOT
    copied — lessons generate lazily, personalized per student), and both
    roadmap JSON rows. Returns the new course id. Idempotent per student.
    """
    existing = await get_assignment(classroom_course_id, student_id)
    if existing:
        return existing["course_id"]

    clone_id = new_id("crs")
    async with get_conn() as conn:
        async with conn.transaction():
            template = await conn.fetchrow(
                "SELECT * FROM courses WHERE id=$1", template_course_id
            )
            if not template:
                raise ValueError(f"Template course {template_course_id} not found")

            await conn.execute(
                """
                INSERT INTO courses
                  (id, student_id, curriculum_id, topic, goal, pace, title, status,
                   prior_knowledge, personalization_profile, progress, web_search_enabled)
                VALUES ($1,$2,NULL,$3,$4,$5,$6,'active','',$7,0.0,$8)
                """,
                clone_id, student_id,
                template["topic"], template["goal"], template["pace"],
                template["title"] or template["topic"],
                json.dumps(_json_value(template["personalization_profile"], {})),
                bool(template["web_search_enabled"]),
            )

            modules = await conn.fetch(
                "SELECT * FROM course_modules WHERE course_id=$1 ORDER BY module_index",
                template_course_id,
            )
            for idx, mod in enumerate(modules):
                await conn.execute(
                    """
                    INSERT INTO course_modules
                      (course_id, id, module_index, title, concept, description,
                       prerequisites, estimated_minutes, depth_level, difficulty,
                       roadmap_step_id, module_metadata, status)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
                    """,
                    clone_id, mod["id"], mod["module_index"], mod["title"],
                    mod["concept"], mod["description"],
                    json.dumps(_json_value(mod["prerequisites"], [])),
                    mod["estimated_minutes"], mod["depth_level"], mod["difficulty"],
                    mod["roadmap_step_id"],
                    json.dumps(_json_value(mod["module_metadata"], {})),
                    "in_progress" if idx == 0 else "not_started",
                )

            for table in ("master_roadmaps", "course_roadmaps"):
                roadmap = await conn.fetchrow(
                    f"SELECT roadmap_json FROM {table} WHERE course_id=$1",
                    template_course_id,
                )
                if roadmap:
                    await conn.execute(
                        f"""
                        INSERT INTO {table} (course_id, roadmap_json)
                        VALUES ($1,$2)
                        ON CONFLICT (course_id) DO UPDATE
                          SET roadmap_json=$2, updated_at=NOW()
                        """,
                        clone_id,
                        json.dumps(_json_value(roadmap["roadmap_json"], {})),
                    )

            await conn.execute(
                """
                INSERT INTO course_assignments
                  (classroom_course_id, classroom_id, student_id, course_id)
                VALUES ($1,$2,$3,$4)
                ON CONFLICT (classroom_course_id, student_id) DO NOTHING
                """,
                classroom_course_id, classroom_id, student_id, clone_id,
            )

    logger.info(
        "Cloned course {} -> {} for student {} (classroom {})",
        template_course_id, clone_id, student_id, classroom_id,
    )
    return clone_id


async def list_assignments_for_classroom_course(cc_id: str) -> list[dict[str, Any]]:
    """Per-student assignment rows with live progress from the cloned course."""
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT ca.*, c.progress, c.status AS course_status, c.updated_at AS last_activity,
                   COALESCE(NULLIF(m.display_name,''), u.name, 'Student') AS student_name,
                   u.email,
                   COUNT(cm.id)::INT AS module_count,
                   COUNT(cm.id) FILTER (WHERE cm.status='completed')::INT AS completed_modules
              FROM course_assignments ca
              JOIN courses c ON c.id=ca.course_id
              LEFT JOIN classroom_members m
                     ON m.classroom_id=ca.classroom_id AND m.student_id=ca.student_id
              LEFT JOIN users u ON u.student_id=ca.student_id
              LEFT JOIN course_modules cm ON cm.course_id=c.id
             WHERE ca.classroom_course_id=$1
             GROUP BY ca.id, c.progress, c.status, c.updated_at, m.display_name, u.name, u.email
             ORDER BY student_name
            """,
            cc_id,
        )
    return [dict(r) for r in rows]


async def list_assignments_for_student(classroom_id: str, student_id: str) -> list[dict[str, Any]]:
    """Assigned courses (clones) for one student in one classroom."""
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT ca.*, cc.title AS classroom_course_title, cc.due_date,
                   c.progress, c.topic, c.title AS course_title,
                   COUNT(cm.id)::INT AS module_count,
                   COUNT(cm.id) FILTER (WHERE cm.status='completed')::INT AS completed_modules
              FROM course_assignments ca
              JOIN classroom_courses cc ON cc.id=ca.classroom_course_id
              JOIN courses c ON c.id=ca.course_id
              LEFT JOIN course_modules cm ON cm.course_id=c.id
             WHERE ca.classroom_id=$1 AND ca.student_id=$2
             GROUP BY ca.id, cc.title, cc.due_date, c.progress, c.topic, c.title
             ORDER BY ca.created_at DESC
            """,
            classroom_id, student_id,
        )
    return [dict(r) for r in rows]


# ── Tests ─────────────────────────────────────────────────────────────────────

async def create_test(
    classroom_id: str,
    created_by: str,
    title: str,
    topic: str,
    config: dict[str, Any],
    classroom_course_id: str | None = None,
    duration_minutes: int = 30,
    instructions: str = "",
) -> str:
    """Insert a draft test shell and return its id."""
    test_id = new_id("tst")
    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO classroom_tests
              (id, classroom_id, classroom_course_id, created_by, title, topic,
               instructions, config_json, duration_minutes)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
            """,
            test_id, classroom_id, classroom_course_id, created_by,
            title, topic, instructions, json.dumps(config), duration_minutes,
        )
    return test_id


async def get_test(test_id: str) -> dict[str, Any] | None:
    """Return one test row with question and attempt counts."""
    async with get_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT t.*,
                   COUNT(DISTINCT q.id)::INT AS question_count,
                   COUNT(DISTINCT a.id) FILTER (WHERE a.status IN ('submitted','graded'))::INT
                       AS attempt_count
              FROM classroom_tests t
              LEFT JOIN test_questions q ON q.test_id=t.id
              LEFT JOIN test_attempts a ON a.test_id=t.id
             WHERE t.id=$1
             GROUP BY t.id
            """,
            test_id,
        )
    return _row_dict(row, {"config_json": {}}) if row else None


async def list_tests(classroom_id: str, statuses: list[str] | None = None) -> list[dict[str, Any]]:
    """List classroom tests, optionally filtered to a status set."""
    async with get_conn() as conn:
        if statuses:
            rows = await conn.fetch(
                """
                SELECT t.*, COUNT(DISTINCT q.id)::INT AS question_count,
                       COUNT(DISTINCT a.id) FILTER (WHERE a.status IN ('submitted','graded'))::INT
                           AS attempt_count
                  FROM classroom_tests t
                  LEFT JOIN test_questions q ON q.test_id=t.id
                  LEFT JOIN test_attempts a ON a.test_id=t.id
                 WHERE t.classroom_id=$1 AND t.status = ANY($2::text[])
                 GROUP BY t.id ORDER BY t.created_at DESC
                """,
                classroom_id, statuses,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT t.*, COUNT(DISTINCT q.id)::INT AS question_count,
                       COUNT(DISTINCT a.id) FILTER (WHERE a.status IN ('submitted','graded'))::INT
                           AS attempt_count
                  FROM classroom_tests t
                  LEFT JOIN test_questions q ON q.test_id=t.id
                  LEFT JOIN test_attempts a ON a.test_id=t.id
                 WHERE t.classroom_id=$1
                 GROUP BY t.id ORDER BY t.created_at DESC
                """,
                classroom_id,
            )
    return [_row_dict(r, {"config_json": {}}) for r in rows]


async def update_test(test_id: str, fields: dict[str, Any]) -> None:
    """Update editable test fields."""
    allowed = {
        "title", "topic", "instructions", "status",
        "scheduled_start", "scheduled_end", "duration_minutes",
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return
    sets = ", ".join(f"{col}=${i + 2}" for i, col in enumerate(updates))
    async with get_conn() as conn:
        await conn.execute(
            f"UPDATE classroom_tests SET {sets}, updated_at=NOW() WHERE id=$1",
            test_id, *updates.values(),
        )


async def replace_test_questions(test_id: str, questions: list[dict[str, Any]]) -> None:
    """Replace all questions for a test (used by generate/regenerate/edit)."""
    async with get_conn() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM test_questions WHERE test_id=$1", test_id)
            for idx, q in enumerate(questions):
                await conn.execute(
                    """
                    INSERT INTO test_questions
                      (id, test_id, order_index, question_type, question_text,
                       options_json, correct_answer, explanation, concepts_tested,
                       difficulty, points)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                    """,
                    q.get("id") or new_id("q"),
                    test_id, idx,
                    q.get("question_type") or "mcq",
                    q.get("question_text") or "",
                    json.dumps(q.get("options") or []),
                    str(q.get("correct_answer") or ""),
                    q.get("explanation") or "",
                    json.dumps(q.get("concepts_tested") or []),
                    q.get("difficulty") or "medium",
                    float(q.get("points") or 1.0),
                )


async def list_test_questions(test_id: str, include_answers: bool) -> list[dict[str, Any]]:
    """Return ordered questions; strips answers/explanations for students."""
    async with get_conn() as conn:
        rows = await conn.fetch(
            "SELECT * FROM test_questions WHERE test_id=$1 ORDER BY order_index",
            test_id,
        )
    questions = []
    for row in rows:
        q = _row_dict(row, {"options_json": [], "concepts_tested": []})
        q["options"] = q.pop("options_json")
        if not include_answers:
            q.pop("correct_answer", None)
            q.pop("explanation", None)
        questions.append(q)
    return questions


async def get_or_create_attempt(test_id: str, student_id: str) -> dict[str, Any]:
    """Return the student's attempt, creating an in_progress one if missing."""
    async with get_conn() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM test_attempts WHERE test_id=$1 AND student_id=$2",
            test_id, student_id,
        )
        if not row:
            attempt_id = new_id("att")
            await conn.execute(
                "INSERT INTO test_attempts (id, test_id, student_id) VALUES ($1,$2,$3)",
                attempt_id, test_id, student_id,
            )
            row = await conn.fetchrow(
                "SELECT * FROM test_attempts WHERE id=$1", attempt_id
            )
    return _row_dict(row, {
        "answers_json": [], "per_question_json": [], "concept_scores_json": {},
    })


async def get_attempt(attempt_id: str) -> dict[str, Any] | None:
    """Return one attempt row by id."""
    async with get_conn() as conn:
        row = await conn.fetchrow("SELECT * FROM test_attempts WHERE id=$1", attempt_id)
    return _row_dict(row, {
        "answers_json": [], "per_question_json": [], "concept_scores_json": {},
    }) if row else None


async def save_attempt_answers(attempt_id: str, answers: list[dict[str, Any]]) -> None:
    """Persist the current (possibly partial) answers for an in-progress attempt."""
    async with get_conn() as conn:
        await conn.execute(
            "UPDATE test_attempts SET answers_json=$2 WHERE id=$1 AND status='in_progress'",
            attempt_id, json.dumps(answers),
        )


async def finalize_attempt(
    attempt_id: str,
    score: float,
    max_score: float,
    per_question: list[dict[str, Any]],
    concept_scores: dict[str, float],
) -> None:
    """Write grading results and mark the attempt graded."""
    async with get_conn() as conn:
        await conn.execute(
            """
            UPDATE test_attempts
               SET status='graded', score=$2, max_score=$3,
                   per_question_json=$4, concept_scores_json=$5,
                   submitted_at=COALESCE(submitted_at, NOW()), graded_at=NOW()
             WHERE id=$1
            """,
            attempt_id, score, max_score,
            json.dumps(per_question), json.dumps(concept_scores),
        )


async def list_attempts_for_test(test_id: str) -> list[dict[str, Any]]:
    """All attempts for a test with student identity (teacher results view)."""
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT a.*, COALESCE(NULLIF(m.display_name,''), u.name, 'Student') AS student_name,
                   u.email
              FROM test_attempts a
              JOIN classroom_tests t ON t.id=a.test_id
              LEFT JOIN classroom_members m
                     ON m.classroom_id=t.classroom_id AND m.student_id=a.student_id
              LEFT JOIN users u ON u.student_id=a.student_id
             WHERE a.test_id=$1
             ORDER BY a.score DESC NULLS LAST
            """,
            test_id,
        )
    return [
        _row_dict(r, {"answers_json": [], "per_question_json": [], "concept_scores_json": {}})
        for r in rows
    ]


# ── Learning events ───────────────────────────────────────────────────────────

async def record_learning_events(events: list[dict[str, Any]]) -> int:
    """Bulk-insert learning events. Returns number written."""
    if not events:
        return 0
    async with get_conn() as conn:
        await conn.executemany(
            """
            INSERT INTO learning_events
              (student_id, classroom_id, course_id, module_id, event_type, payload_json)
            VALUES ($1,$2,$3,$4,$5,$6)
            """,
            [
                (
                    e["student_id"],
                    e.get("classroom_id"),
                    e.get("course_id"),
                    e.get("module_id"),
                    e.get("event_type") or "unknown",
                    json.dumps(e.get("payload") or {}),
                )
                for e in events
            ],
        )
    return len(events)


# ── AI artifacts ──────────────────────────────────────────────────────────────

async def save_ai_artifact(
    classroom_id: str,
    artifact_type: str,
    content: dict[str, Any],
    generated_by: str,
    scope_key: str = "",
) -> dict[str, Any]:
    """Persist a generated AI artifact and return the saved row."""
    artifact_id = new_id("art")
    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO classroom_ai_artifacts
              (id, classroom_id, artifact_type, scope_key, content_json, generated_by)
            VALUES ($1,$2,$3,$4,$5,$6)
            """,
            artifact_id, classroom_id, artifact_type, scope_key,
            json.dumps(content), generated_by,
        )
    return {
        "id": artifact_id,
        "classroom_id": classroom_id,
        "artifact_type": artifact_type,
        "scope_key": scope_key,
        "content_json": content,
        "generated_by": generated_by,
    }


async def latest_ai_artifact(
    classroom_id: str, artifact_type: str, scope_key: str = ""
) -> dict[str, Any] | None:
    """Return the most recent artifact of a type (optionally scoped)."""
    async with get_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM classroom_ai_artifacts
             WHERE classroom_id=$1 AND artifact_type=$2 AND scope_key=$3
             ORDER BY created_at DESC LIMIT 1
            """,
            classroom_id, artifact_type, scope_key,
        )
    return _row_dict(row, {"content_json": {}}) if row else None


async def mark_artifacts_stale(classroom_id: str) -> None:
    """Flag all cached artifacts stale after new data lands (attempts, evals)."""
    async with get_conn() as conn:
        await conn.execute(
            "UPDATE classroom_ai_artifacts SET stale=TRUE WHERE classroom_id=$1",
            classroom_id,
        )


# ── Teacher assistant chat ────────────────────────────────────────────────────

async def record_assistant_message(
    classroom_id: str,
    student_id: str,
    role: str,
    message: str,
    tool_trace: list[dict] | None = None,
) -> None:
    """Append one assistant chat message."""
    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO teacher_assistant_messages
              (id, classroom_id, student_id, role, message, tool_trace_json)
            VALUES ($1,$2,$3,$4,$5,$6)
            """,
            new_id("tam"), classroom_id, student_id, role, message,
            json.dumps(tool_trace or []),
        )


async def list_assistant_messages(classroom_id: str, student_id: str, limit: int = 60) -> list[dict[str, Any]]:
    """Return recent assistant chat history for one teacher, oldest first."""
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM (
                SELECT * FROM teacher_assistant_messages
                 WHERE classroom_id=$1 AND student_id=$2
                 ORDER BY created_at DESC LIMIT $3
            ) sub ORDER BY created_at ASC
            """,
            classroom_id, student_id, limit,
        )
    return [_row_dict(r, {"tool_trace_json": []}) for r in rows]


# ── Posts ─────────────────────────────────────────────────────────────────────

async def create_post(
    classroom_id: str,
    author_id: str,
    post_type: str,
    title: str,
    body_markdown: str,
    audience: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a classroom stream post (announcement / note / live class / task)."""
    post_id = new_id("post")
    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO classroom_posts
              (id, classroom_id, author_id, post_type, title, body_markdown,
               audience_json, meta_json)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
            """,
            post_id, classroom_id, author_id, post_type, title, body_markdown,
            json.dumps(audience or {"all": True}),
            json.dumps(meta or {}),
        )
    return {"id": post_id}


async def list_posts(classroom_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """Return classroom posts newest first, with author identity."""
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT p.*, COALESCE(u.name, 'Teacher') AS author_name, u.avatar_url
              FROM classroom_posts p
              LEFT JOIN users u ON u.student_id=p.author_id
             WHERE p.classroom_id=$1
             ORDER BY p.created_at DESC LIMIT $2
            """,
            classroom_id, limit,
        )
    return [_row_dict(r, {"audience_json": {}, "meta_json": {}}) for r in rows]
