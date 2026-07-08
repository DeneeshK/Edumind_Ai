"""
core/test_service.py
Test lifecycle orchestration for the institution module:
generate → edit → approve → schedule → take → grade → feed analytics.

Grading strategy:
- MCQ: deterministic index comparison — never an LLM call.
- short_answer / conceptual: one batched LLM rubric call per submission.
- Results update the existing concept_mastery table so classroom tests feed
  the same adaptive signals the rest of the platform already uses.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException
from loguru import logger

from agents.institution.test_generation_agent import (
    generate_test_questions,
    regenerate_single_question,
)
from clients.groq_client import generate
from config import settings
from db import institution as repo
from db.postgres import get_conn, upsert_concept_mastery


# ── Context builders ──────────────────────────────────────────────────────────

async def _course_context_for_test(classroom_course_id: str | None) -> str:
    """Compact module summary from the linked classroom course's template."""
    if not classroom_course_id:
        return ""
    cc = await repo.get_classroom_course(classroom_course_id)
    if not cc:
        return ""
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT title, concept, description
              FROM course_modules WHERE course_id=$1 ORDER BY module_index
            """,
            cc["template_course_id"],
        )
    lines = [f"Course: {cc['title']} (topic: {cc.get('topic','')})"]
    for r in rows:
        lines.append(f"- {r['title']} — concept: {r['concept']}. {r['description'][:160]}")
    return "\n".join(lines)


async def _class_weak_concepts(classroom_id: str, limit: int = 8) -> list[str]:
    """Concepts with the lowest average mastery across active members."""
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT cm.concept, AVG(cm.mastery_score) AS avg_mastery, COUNT(*) AS n
              FROM concept_mastery cm
              JOIN classroom_members m ON m.student_id=cm.student_id
             WHERE m.classroom_id=$1 AND m.status='active'
             GROUP BY cm.concept
            HAVING COUNT(*) >= 2
             ORDER BY avg_mastery ASC
             LIMIT $2
            """,
            classroom_id, limit,
        )
    return [r["concept"] for r in rows if float(r["avg_mastery"] or 1.0) < 0.6]


# ── Lifecycle ─────────────────────────────────────────────────────────────────

async def generate_test(
    *,
    classroom_id: str,
    teacher_student_id: str,
    topic: str,
    title: str = "",
    classroom_course_id: str | None = None,
    num_mcq: int = 5,
    num_short: int = 3,
    num_conceptual: int = 2,
    difficulty_mix: str = "balanced",
    duration_minutes: int = 30,
    instructions: str = "",
) -> dict[str, Any]:
    """Generate a draft test with AI questions and persist it."""
    if not topic.strip():
        raise HTTPException(status_code=422, detail="Topic is required")

    classroom = await repo.get_classroom(classroom_id)
    course_context = await _course_context_for_test(classroom_course_id)
    weak_concepts = await _class_weak_concepts(classroom_id)

    try:
        generated = await generate_test_questions(
            topic=topic.strip(),
            subject=(classroom or {}).get("subject", ""),
            grade_level=(classroom or {}).get("grade_level", ""),
            num_mcq=num_mcq,
            num_short=num_short,
            num_conceptual=num_conceptual,
            difficulty_mix=difficulty_mix,
            course_context=course_context,
            weak_concepts=weak_concepts,
            extra_instructions=instructions,
        )
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    config = {
        "num_mcq": num_mcq, "num_short": num_short, "num_conceptual": num_conceptual,
        "difficulty_mix": difficulty_mix, "weak_concepts_used": weak_concepts,
    }
    test_id = await repo.create_test(
        classroom_id=classroom_id,
        created_by=teacher_student_id,
        title=title.strip() or generated["title"],
        topic=topic.strip(),
        config=config,
        classroom_course_id=classroom_course_id,
        duration_minutes=max(5, min(int(duration_minutes), 240)),
        instructions=instructions,
    )
    await repo.replace_test_questions(test_id, generated["questions"])
    return await test_detail_for_teacher(test_id)


async def regenerate_question(test_id: str, question_id: str) -> dict[str, Any]:
    """Regenerate one question in a draft test, keeping the rest untouched."""
    test = await repo.get_test(test_id)
    if not test:
        raise HTTPException(status_code=404, detail="Test not found")
    if test["status"] not in ("draft", "approved"):
        raise HTTPException(status_code=409, detail="Cannot edit a published test")

    questions = await repo.list_test_questions(test_id, include_answers=True)
    target = next((q for q in questions if q["id"] == question_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Question not found")

    course_context = await _course_context_for_test(test.get("classroom_course_id"))
    try:
        replacement = await regenerate_single_question(
            topic=test["topic"],
            question_type=target["question_type"],
            difficulty=target["difficulty"],
            course_context=course_context,
            avoid_texts=[q["question_text"] for q in questions],
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    updated = [replacement if q["id"] == question_id else q for q in questions]
    await repo.replace_test_questions(test_id, updated)
    return await test_detail_for_teacher(test_id)


async def update_test_questions(test_id: str, questions: list[dict[str, Any]]) -> dict[str, Any]:
    """Teacher manual edit of the full question list (draft/approved only)."""
    test = await repo.get_test(test_id)
    if not test:
        raise HTTPException(status_code=404, detail="Test not found")
    if test["status"] not in ("draft", "approved"):
        raise HTTPException(status_code=409, detail="Cannot edit a published test")
    if not questions:
        raise HTTPException(status_code=422, detail="A test needs at least one question")
    await repo.replace_test_questions(test_id, questions)
    return await test_detail_for_teacher(test_id)


async def transition_test(
    test_id: str,
    action: str,
    scheduled_start: str | None = None,
    scheduled_end: str | None = None,
) -> dict[str, Any]:
    """Move a test through its lifecycle: approve | schedule | publish | close."""
    test = await repo.get_test(test_id)
    if not test:
        raise HTTPException(status_code=404, detail="Test not found")

    valid = {
        "approve": ({"draft"}, "approved"),
        "schedule": ({"approved", "scheduled"}, "scheduled"),
        "publish": ({"approved", "scheduled"}, "live"),
        "close": ({"live", "scheduled"}, "closed"),
    }
    if action not in valid:
        raise HTTPException(status_code=422, detail=f"Unknown action '{action}'")
    allowed_from, target = valid[action]
    if test["status"] not in allowed_from:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot {action} a test in status '{test['status']}'",
        )

    fields: dict[str, Any] = {"status": target}
    if action == "schedule":
        if not scheduled_start:
            raise HTTPException(status_code=422, detail="scheduled_start is required")
        fields["scheduled_start"] = _parse_ts(scheduled_start)
        if scheduled_end:
            fields["scheduled_end"] = _parse_ts(scheduled_end)
    await repo.update_test(test_id, fields)
    return await test_detail_for_teacher(test_id)


def _parse_ts(value: str) -> datetime:
    """Parse an ISO timestamp, defaulting to UTC when no zone is given."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid timestamp '{value}'") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _effective_status(test: dict[str, Any]) -> str:
    """Resolve scheduled tests into live/closed based on the current time."""
    status = test["status"]
    now = datetime.now(timezone.utc)
    if status == "scheduled":
        start, end = test.get("scheduled_start"), test.get("scheduled_end")
        if start and start <= now and (not end or now <= end):
            return "live"
        if end and now > end:
            return "closed"
    if status == "live":
        end = test.get("scheduled_end")
        if end and now > end:
            return "closed"
    return status


# ── Read models ───────────────────────────────────────────────────────────────

async def test_detail_for_teacher(test_id: str) -> dict[str, Any]:
    """Full test payload including answers — teacher view."""
    test = await repo.get_test(test_id)
    if not test:
        raise HTTPException(status_code=404, detail="Test not found")
    test["effective_status"] = _effective_status(test)
    test["questions"] = await repo.list_test_questions(test_id, include_answers=True)
    return test


async def test_detail_for_student(test_id: str, student_id: str) -> dict[str, Any]:
    """Student view: questions without answers; includes own attempt state."""
    test = await repo.get_test(test_id)
    if not test:
        raise HTTPException(status_code=404, detail="Test not found")
    effective = _effective_status(test)
    if effective not in ("live", "closed") and test["status"] != "scheduled":
        raise HTTPException(status_code=403, detail="This test is not available yet")

    test["effective_status"] = effective
    attempt = None
    async with get_conn() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM test_attempts WHERE test_id=$1 AND student_id=$2",
            test_id, student_id,
        )
    if row:
        attempt = dict(row)
        attempt["answers_json"] = repo._json_value(attempt["answers_json"], [])
        attempt["per_question_json"] = repo._json_value(attempt["per_question_json"], [])

    if effective == "scheduled":
        # Upcoming test: metadata only, never the questions.
        test["questions"] = []
    else:
        include_answers = bool(attempt and attempt["status"] == "graded")
        test["questions"] = await repo.list_test_questions(test_id, include_answers=include_answers)
    test["my_attempt"] = attempt
    return test


async def visible_tests_for_student(classroom_id: str, student_id: str) -> list[dict[str, Any]]:
    """Scheduled/live/closed tests with the student's attempt summary."""
    tests = await repo.list_tests(classroom_id, statuses=["scheduled", "live", "closed"])
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT a.test_id, a.status, a.score, a.max_score
              FROM test_attempts a
              JOIN classroom_tests t ON t.id=a.test_id
             WHERE t.classroom_id=$1 AND a.student_id=$2
            """,
            classroom_id, student_id,
        )
    attempts = {r["test_id"]: dict(r) for r in rows}
    for test in tests:
        test["effective_status"] = _effective_status(test)
        test["my_attempt"] = attempts.get(test["id"])
    return tests


# ── Attempts & grading ────────────────────────────────────────────────────────

async def start_attempt(test_id: str, student_id: str) -> dict[str, Any]:
    """Start (or resume) the student's attempt on a live test."""
    test = await repo.get_test(test_id)
    if not test:
        raise HTTPException(status_code=404, detail="Test not found")
    if _effective_status(test) != "live":
        raise HTTPException(status_code=403, detail="This test is not live right now")
    attempt = await repo.get_or_create_attempt(test_id, student_id)
    if attempt["status"] != "in_progress":
        raise HTTPException(status_code=409, detail="You have already submitted this test")
    return attempt


async def save_answers(attempt_id: str, student_id: str, answers: list[dict[str, Any]]) -> None:
    """Autosave partial answers for the owning student's in-progress attempt."""
    attempt = await repo.get_attempt(attempt_id)
    if not attempt or attempt["student_id"] != student_id:
        raise HTTPException(status_code=404, detail="Attempt not found")
    if attempt["status"] != "in_progress":
        raise HTTPException(status_code=409, detail="Attempt already submitted")
    await repo.save_attempt_answers(attempt_id, answers)


_GRADING_SYSTEM = """
You are EduMind's strict but fair answer grader.
Grade each student answer against the ideal answer on a 0.0-1.0 scale:
1.0 fully correct, 0.7 mostly correct with minor gaps, 0.4 partially correct,
0.1 attempted but wrong, 0.0 blank or irrelevant.
Give one short constructive feedback sentence per answer.

Return ONLY valid JSON:
{"grades": [{"question_id": "...", "score": 0.0, "feedback": "..."}]}
"""


async def submit_attempt(attempt_id: str, student_id: str) -> dict[str, Any]:
    """
    Submit and grade an attempt: deterministic MCQ grading, one batched LLM
    call for subjective answers, mastery updates, and analytics invalidation.
    """
    attempt = await repo.get_attempt(attempt_id)
    if not attempt or attempt["student_id"] != student_id:
        raise HTTPException(status_code=404, detail="Attempt not found")
    if attempt["status"] != "in_progress":
        raise HTTPException(status_code=409, detail="Attempt already submitted")

    test = await repo.get_test(attempt["test_id"])
    questions = await repo.list_test_questions(attempt["test_id"], include_answers=True)
    answers = {a.get("question_id"): str(a.get("answer") or "") for a in attempt["answers_json"]}

    per_question: list[dict[str, Any]] = []
    subjective: list[dict[str, Any]] = []

    for q in questions:
        answer_text = answers.get(q["id"], "")
        entry = {
            "question_id": q["id"],
            "question_type": q["question_type"],
            "answer": answer_text,
            "points": q["points"],
            "concepts": q.get("concepts_tested") or [],
            "score": 0.0,
            "feedback": "",
        }
        if q["question_type"] == "mcq":
            entry["score"] = 1.0 if answer_text.strip() == str(q.get("correct_answer", "")) else 0.0
            entry["feedback"] = q.get("explanation") or ""
        elif answer_text.strip():
            subjective.append({
                "question_id": q["id"],
                "question": q["question_text"],
                "ideal_answer": q.get("correct_answer") or "",
                "student_answer": answer_text[:1500],
            })
        per_question.append(entry)

    if subjective:
        grades = await _grade_subjective(subjective)
        by_id = {g["question_id"]: g for g in grades}
        for entry in per_question:
            grade = by_id.get(entry["question_id"])
            if grade:
                entry["score"] = grade["score"]
                entry["feedback"] = grade["feedback"]

    max_score = sum(float(e["points"]) for e in per_question)
    score = sum(float(e["score"]) * float(e["points"]) for e in per_question)

    # Aggregate per-concept accuracy and feed the platform's mastery signals.
    concept_totals: dict[str, list[float]] = {}
    for entry in per_question:
        for concept in entry["concepts"]:
            concept_totals.setdefault(concept, []).append(float(entry["score"]))
    concept_scores = {c: round(sum(v) / len(v), 4) for c, v in concept_totals.items()}
    for concept, value in concept_scores.items():
        try:
            await upsert_concept_mastery(student_id, concept, correctness=value, depth=value * 0.8)
        except Exception as exc:
            logger.warning("Mastery update failed for {}: {}", concept, exc)

    await repo.finalize_attempt(attempt_id, round(score, 2), round(max_score, 2),
                                per_question, concept_scores)
    if test:
        await repo.record_learning_events([{
            "student_id": student_id,
            "classroom_id": test["classroom_id"],
            "event_type": "test_submitted",
            "payload": {"test_id": test["id"], "score": round(score, 2),
                        "max_score": round(max_score, 2)},
        }])
        await repo.mark_artifacts_stale(test["classroom_id"])

    return await repo.get_attempt(attempt_id) or {}


async def _grade_subjective(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One batched rubric-grading LLM call; degrades to 0.5 partial credit on failure."""
    prompt = "Grade these answers:\n" + json.dumps(items, indent=1)
    try:
        raw = await generate(
            messages=[{"role": "user", "content": prompt}],
            model=settings.eval_judge_model,
            system=_GRADING_SYSTEM,
            json_mode=True,
            max_tokens=2500,
            _caller="test_grading",
        )
        grades = []
        for g in (json.loads(raw).get("grades") or []):
            try:
                grades.append({
                    "question_id": str(g.get("question_id") or ""),
                    "score": max(0.0, min(float(g.get("score") or 0.0), 1.0)),
                    "feedback": str(g.get("feedback") or "")[:400],
                })
            except (TypeError, ValueError):
                continue
        if grades:
            return grades
    except Exception as exc:
        logger.error("Subjective grading failed: {}", exc)
    # Fallback: partial credit so a grading outage never zeroes a student.
    return [
        {"question_id": it["question_id"], "score": 0.5,
         "feedback": "Auto-graded with partial credit — AI grading was unavailable."}
        for it in items
    ]
