"""
core/classroom_analytics.py
Deterministic classroom analytics — pure SQL aggregation over existing
per-student tables (concept_mastery, doubt_log, evaluation_sessions, courses)
plus the institution tables (test_attempts, course_assignments,
learning_events). No LLM calls live here; AI agents consume these numbers.
"""

from __future__ import annotations

from typing import Any

from db.postgres import get_conn
from db import institution as repo


# ── Roster helpers ────────────────────────────────────────────────────────────

async def _member_names(classroom_id: str) -> dict[str, str]:
    """Map active member student_id → display name."""
    members = await repo.list_members(classroom_id, statuses=["active"])
    return {m["student_id"]: m["name"] for m in members}


def _classroom_course_ids_subquery() -> str:
    """SQL fragment selecting cloned course ids for a classroom (as $1)."""
    return "(SELECT course_id FROM course_assignments WHERE classroom_id=$1)"


# ── Overview KPIs ─────────────────────────────────────────────────────────────

async def overview(classroom_id: str) -> dict[str, Any]:
    """Headline KPIs for the teacher dashboard."""
    async with get_conn() as conn:
        members = await conn.fetchrow(
            """
            SELECT COUNT(*) FILTER (WHERE status='active')::INT  AS active_members,
                   COUNT(*) FILTER (WHERE status='pending')::INT AS pending_members
              FROM classroom_members WHERE classroom_id=$1
            """,
            classroom_id,
        )
        progress = await conn.fetchrow(
            f"""
            SELECT COALESCE(AVG(c.progress), 0)                                    AS avg_progress,
                   COUNT(*) FILTER (WHERE c.progress >= 0.999)::INT                AS completed_courses,
                   COUNT(*)::INT                                                   AS assigned_courses
              FROM courses c WHERE c.id IN {_classroom_course_ids_subquery()}
            """,
            classroom_id,
        )
        tests = await conn.fetchrow(
            """
            SELECT COALESCE(AVG(a.score / NULLIF(a.max_score,0)), 0) AS avg_test_ratio,
                   COUNT(DISTINCT a.test_id)::INT                    AS tests_taken,
                   COUNT(a.id)::INT                                  AS attempts
              FROM test_attempts a
              JOIN classroom_tests t ON t.id=a.test_id
             WHERE t.classroom_id=$1 AND a.status='graded'
            """,
            classroom_id,
        )
        mastery = await conn.fetchrow(
            """
            SELECT COALESCE(AVG(cm.mastery_score), 0) AS avg_mastery
              FROM concept_mastery cm
              JOIN classroom_members m ON m.student_id=cm.student_id
             WHERE m.classroom_id=$1 AND m.status='active'
            """,
            classroom_id,
        )
        active_week = await conn.fetchval(
            f"""
            SELECT COUNT(DISTINCT student_id) FROM (
                SELECT student_id FROM learning_events
                 WHERE classroom_id=$1 AND created_at > NOW() - INTERVAL '7 days'
                UNION
                SELECT c.student_id FROM courses c
                 WHERE c.id IN {_classroom_course_ids_subquery()}
                   AND c.updated_at > NOW() - INTERVAL '7 days'
            ) recent
            """,
            classroom_id,
        )
        weekly_trend = await conn.fetch(
            """
            SELECT date_trunc('week', a.graded_at) AS week,
                   AVG(a.score / NULLIF(a.max_score,0)) AS avg_ratio,
                   COUNT(*)::INT AS attempts
              FROM test_attempts a
              JOIN classroom_tests t ON t.id=a.test_id
             WHERE t.classroom_id=$1 AND a.status='graded'
               AND a.graded_at > NOW() - INTERVAL '12 weeks'
             GROUP BY 1 ORDER BY 1
            """,
            classroom_id,
        )

    return {
        "active_members": int(members["active_members"] or 0),
        "pending_members": int(members["pending_members"] or 0),
        "avg_progress": round(float(progress["avg_progress"] or 0), 4),
        "completed_courses": int(progress["completed_courses"] or 0),
        "assigned_courses": int(progress["assigned_courses"] or 0),
        "avg_test_score": round(float(tests["avg_test_ratio"] or 0), 4),
        "tests_taken": int(tests["tests_taken"] or 0),
        "attempts": int(tests["attempts"] or 0),
        "avg_mastery": round(float(mastery["avg_mastery"] or 0), 4),
        "active_last_7_days": int(active_week or 0),
        "score_trend": [
            {"week": r["week"].date().isoformat(),
             "avg_score": round(float(r["avg_ratio"] or 0), 4),
             "attempts": int(r["attempts"])}
            for r in weekly_trend if r["week"]
        ],
    }


# ── Concept heatmap ───────────────────────────────────────────────────────────

async def concept_heatmap(classroom_id: str, max_concepts: int = 20) -> dict[str, Any]:
    """students × concepts mastery matrix limited to the most-seen concepts."""
    names = await _member_names(classroom_id)
    if not names:
        return {"concepts": [], "students": []}
    async with get_conn() as conn:
        top_concepts = await conn.fetch(
            """
            SELECT cm.concept, COUNT(*)::INT AS n, AVG(cm.mastery_score) AS avg_m
              FROM concept_mastery cm
              JOIN classroom_members m ON m.student_id=cm.student_id
             WHERE m.classroom_id=$1 AND m.status='active'
             GROUP BY cm.concept ORDER BY n DESC, avg_m ASC LIMIT $2
            """,
            classroom_id, max_concepts,
        )
        concepts = [r["concept"] for r in top_concepts]
        if not concepts:
            return {"concepts": [], "students": []}
        rows = await conn.fetch(
            """
            SELECT cm.student_id, cm.concept, cm.mastery_score
              FROM concept_mastery cm
              JOIN classroom_members m ON m.student_id=cm.student_id
             WHERE m.classroom_id=$1 AND m.status='active'
               AND cm.concept = ANY($2::text[])
            """,
            classroom_id, concepts,
        )

    by_student: dict[str, dict[str, float]] = {}
    for r in rows:
        by_student.setdefault(r["student_id"], {})[r["concept"]] = round(
            float(r["mastery_score"] or 0), 3
        )
    students = [
        {"student_id": sid, "name": name,
         "scores": [by_student.get(sid, {}).get(c) for c in concepts]}
        for sid, name in names.items()
    ]
    class_avg = [
        round(sum(v for v in (s["scores"][i] for s in students) if v is not None)
              / max(1, sum(1 for s in students if s["scores"][i] is not None)), 3)
        if any(s["scores"][i] is not None for s in students) else None
        for i in range(len(concepts))
    ]
    return {"concepts": concepts, "students": students, "class_avg": class_avg}


# ── Student table & feature vectors ───────────────────────────────────────────

async def student_table(classroom_id: str) -> list[dict[str, Any]]:
    """
    Per-student analytics row: progress, mastery, tests, doubts, activity,
    consistency, calibration, and a deterministic risk flag. Doubles as the
    feature-vector source for the clustering agent.
    """
    names = await _member_names(classroom_id)
    if not names:
        return []
    ids = list(names.keys())
    async with get_conn() as conn:
        progress = await conn.fetch(
            """
            SELECT ca.student_id, AVG(c.progress) AS avg_progress,
                   MAX(c.updated_at) AS last_course_activity
              FROM course_assignments ca JOIN courses c ON c.id=ca.course_id
             WHERE ca.classroom_id=$1 GROUP BY ca.student_id
            """,
            classroom_id,
        )
        mastery = await conn.fetch(
            """
            SELECT student_id, AVG(mastery_score) AS avg_mastery,
                   COUNT(*) FILTER (WHERE mastery_score < 0.5)::INT AS weak_concepts
              FROM concept_mastery WHERE student_id = ANY($1::text[])
             GROUP BY student_id
            """,
            ids,
        )
        tests = await conn.fetch(
            """
            SELECT a.student_id,
                   AVG(a.score / NULLIF(a.max_score,0)) AS avg_test,
                   COUNT(*)::INT AS tests_taken
              FROM test_attempts a JOIN classroom_tests t ON t.id=a.test_id
             WHERE t.classroom_id=$1 AND a.status='graded'
             GROUP BY a.student_id
            """,
            classroom_id,
        )
        doubts = await conn.fetch(
            """
            SELECT student_id, COUNT(*)::INT AS doubt_count
              FROM doubt_log
             WHERE student_id = ANY($1::text[])
               AND created_at > NOW() - INTERVAL '60 days'
             GROUP BY student_id
            """,
            ids,
        )
        calibration = await conn.fetch(
            """
            SELECT student_id, AVG(calibration_delta) AS avg_calibration
              FROM evaluation_history WHERE student_id = ANY($1::text[])
             GROUP BY student_id
            """,
            ids,
        )
        activity = await conn.fetch(
            """
            SELECT student_id,
                   COUNT(DISTINCT DATE(created_at))::INT AS active_days_30
              FROM learning_events
             WHERE classroom_id=$1 AND created_at > NOW() - INTERVAL '30 days'
             GROUP BY student_id
            """,
            classroom_id,
        )

    def _index(rows: Any, key: str = "student_id") -> dict[str, dict]:
        return {r[key]: dict(r) for r in rows}

    p, m, t = _index(progress), _index(mastery), _index(tests)
    d, cal, act = _index(doubts), _index(calibration), _index(activity)

    table = []
    for sid, name in names.items():
        avg_progress = float((p.get(sid) or {}).get("avg_progress") or 0)
        avg_mastery = float((m.get(sid) or {}).get("avg_mastery") or 0)
        avg_test = ((t.get(sid) or {}).get("avg_test"))
        avg_test = round(float(avg_test), 4) if avg_test is not None else None
        weak = int((m.get(sid) or {}).get("weak_concepts") or 0)
        tests_taken = int((t.get(sid) or {}).get("tests_taken") or 0)
        doubt_count = int((d.get(sid) or {}).get("doubt_count") or 0)
        calibration_delta = (cal.get(sid) or {}).get("avg_calibration")
        active_days = int((act.get(sid) or {}).get("active_days_30") or 0)
        last_activity = (p.get(sid) or {}).get("last_course_activity")

        # Deterministic risk heuristic: struggling AND/OR disengaged.
        risk_points = 0
        if avg_progress < 0.25:
            risk_points += 1
        if avg_mastery and avg_mastery < 0.45:
            risk_points += 1
        if avg_test is not None and avg_test < 0.4:
            risk_points += 1
        if active_days == 0 and avg_progress < 0.9:
            risk_points += 1
        risk = "high" if risk_points >= 3 else "medium" if risk_points == 2 else "low"

        table.append({
            "student_id": sid,
            "name": name,
            "avg_progress": round(avg_progress, 4),
            "avg_mastery": round(avg_mastery, 4),
            "avg_test_score": avg_test,
            "tests_taken": tests_taken,
            "weak_concepts": weak,
            "doubt_count": doubt_count,
            "avg_calibration_delta": round(float(calibration_delta), 3)
                if calibration_delta is not None else None,
            "active_days_30": active_days,
            "last_activity": last_activity.isoformat() if last_activity else None,
            "risk": risk,
        })

    # Rank by composite performance (mastery + tests + progress).
    def _composite(row: dict[str, Any]) -> float:
        parts = [row["avg_mastery"], row["avg_progress"]]
        if row["avg_test_score"] is not None:
            parts.append(row["avg_test_score"])
        return sum(parts) / len(parts)

    table.sort(key=_composite, reverse=True)
    for rank, row in enumerate(table, start=1):
        row["rank"] = rank
        row["composite_score"] = round(_composite(row), 4)
    return table


# ── Doubts ────────────────────────────────────────────────────────────────────

async def doubt_analytics(classroom_id: str) -> dict[str, Any]:
    """Most-asked concepts and recent doubt samples across the class."""
    async with get_conn() as conn:
        by_concept = await conn.fetch(
            """
            SELECT dl.concept, COUNT(*)::INT AS count
              FROM doubt_log dl
              JOIN classroom_members m ON m.student_id=dl.student_id
             WHERE m.classroom_id=$1 AND m.status='active'
               AND dl.created_at > NOW() - INTERVAL '60 days'
             GROUP BY dl.concept ORDER BY count DESC LIMIT 15
            """,
            classroom_id,
        )
        recent = await conn.fetch(
            """
            SELECT dl.concept, dl.doubt_text, dl.doubt_type, dl.created_at
              FROM doubt_log dl
              JOIN classroom_members m ON m.student_id=dl.student_id
             WHERE m.classroom_id=$1 AND m.status='active'
               AND dl.doubt_text != ''
             ORDER BY dl.created_at DESC LIMIT 25
            """,
            classroom_id,
        )
    return {
        "by_concept": [dict(r) for r in by_concept],
        "recent": [
            {"concept": r["concept"], "doubt_text": r["doubt_text"][:280],
             "doubt_type": r["doubt_type"], "created_at": r["created_at"].isoformat()}
            for r in recent
        ],
    }


# ── Per-student drilldown ─────────────────────────────────────────────────────

async def student_drilldown(classroom_id: str, student_id: str) -> dict[str, Any]:
    """Classroom-scoped view of one student for teachers (or the student)."""
    names = await _member_names(classroom_id)
    if student_id not in names:
        return {}
    async with get_conn() as conn:
        weak = await conn.fetch(
            """
            SELECT concept, mastery_score FROM concept_mastery
             WHERE student_id=$1 ORDER BY mastery_score ASC LIMIT 8
            """,
            student_id,
        )
        strong = await conn.fetch(
            """
            SELECT concept, mastery_score FROM concept_mastery
             WHERE student_id=$1 AND mastery_score >= 0.7
             ORDER BY mastery_score DESC LIMIT 8
            """,
            student_id,
        )
        attempts = await conn.fetch(
            """
            SELECT a.test_id, t.title, a.score, a.max_score, a.graded_at
              FROM test_attempts a JOIN classroom_tests t ON t.id=a.test_id
             WHERE t.classroom_id=$1 AND a.student_id=$2 AND a.status='graded'
             ORDER BY a.graded_at DESC LIMIT 10
            """,
            classroom_id, student_id,
        )
        misconceptions = await conn.fetch(
            """
            SELECT misconception_type, misconception_detail, concept, created_at
              FROM evaluation_history
             WHERE student_id=$1 AND misconception_type IS NOT NULL
             ORDER BY created_at DESC LIMIT 8
            """,
            student_id,
        )
    assignments = await repo.list_assignments_for_student(classroom_id, student_id)
    table = await student_table(classroom_id)
    summary = next((r for r in table if r["student_id"] == student_id), {})
    return {
        "student_id": student_id,
        "name": names[student_id],
        "summary": summary,
        "weak_concepts": [dict(r) for r in weak],
        "strong_concepts": [dict(r) for r in strong],
        "test_history": [
            {"test_id": r["test_id"], "title": r["title"],
             "score": float(r["score"] or 0), "max_score": float(r["max_score"] or 0),
             "graded_at": r["graded_at"].isoformat() if r["graded_at"] else None}
            for r in attempts
        ],
        "misconceptions": [
            {"type": r["misconception_type"], "detail": r["misconception_detail"][:240],
             "concept": r["concept"],
             "created_at": r["created_at"].isoformat() if r["created_at"] else None}
            for r in misconceptions
        ],
        "assignments": assignments,
    }


# ── Snapshot for AI agents ────────────────────────────────────────────────────

async def analytics_snapshot(classroom_id: str) -> dict[str, Any]:
    """Compact JSON bundle handed to the insight/cluster/revision agents."""
    overview_data = await overview(classroom_id)
    heatmap = await concept_heatmap(classroom_id, max_concepts=12)
    students = await student_table(classroom_id)
    doubts = await doubt_analytics(classroom_id)
    weakest = [
        {"concept": heatmap["concepts"][i], "class_avg": heatmap["class_avg"][i]}
        for i in range(len(heatmap.get("concepts") or []))
        if heatmap["class_avg"][i] is not None
    ]
    weakest.sort(key=lambda x: x["class_avg"])
    return {
        "overview": overview_data,
        "weakest_concepts": weakest[:8],
        "strongest_concepts": sorted(weakest, key=lambda x: -x["class_avg"])[:5],
        "students": [
            {k: v for k, v in s.items() if k != "last_activity"} for s in students
        ],
        "top_doubt_concepts": doubts["by_concept"][:8],
        "recent_doubts_sample": [d["doubt_text"] for d in doubts["recent"][:10]],
    }
