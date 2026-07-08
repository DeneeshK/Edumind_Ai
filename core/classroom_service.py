"""
core/classroom_service.py
Business logic for the institution module: classroom lifecycle, the join flow,
role checks, and the clone-and-assign pipeline that turns one approved teacher
course into N per-student adaptive courses.

Roles are contextual (Google Classroom style): the classroom owner is its
teacher; anyone else interacts as a member. No role column exists on users.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator

from fastapi import HTTPException
from loguru import logger

from db import institution as repo
from db.postgres import ensure_student, get_course


# ── Authorization helpers (single choke point for the whole module) ──────────

async def require_classroom(classroom_id: str) -> dict[str, Any]:
    """Return the classroom or raise 404."""
    classroom = await repo.get_classroom(classroom_id)
    if not classroom:
        raise HTTPException(status_code=404, detail="Classroom not found")
    return classroom


async def require_teacher(classroom_id: str, student_id: str) -> dict[str, Any]:
    """Allow only the classroom owner or an active co-teacher."""
    classroom = await require_classroom(classroom_id)
    if classroom["owner_student_id"] == student_id:
        return classroom
    membership = await repo.get_membership(classroom_id, student_id)
    if membership and membership["role"] == "co_teacher" and membership["status"] == "active":
        return classroom
    raise HTTPException(status_code=403, detail="Teacher access required")


async def require_member(classroom_id: str, student_id: str) -> dict[str, Any]:
    """Allow the teacher or any active member; returns classroom with `viewer_role`."""
    classroom = await require_classroom(classroom_id)
    if classroom["owner_student_id"] == student_id:
        classroom["viewer_role"] = "teacher"
        return classroom
    membership = await repo.get_membership(classroom_id, student_id)
    if membership and membership["status"] == "active":
        classroom["viewer_role"] = (
            "teacher" if membership["role"] == "co_teacher" else "student"
        )
        return classroom
    raise HTTPException(status_code=403, detail="You are not a member of this classroom")


def viewer_role(classroom: dict[str, Any], student_id: str) -> str:
    """Return 'teacher' or 'student' for a classroom already fetched."""
    if classroom["owner_student_id"] == student_id:
        return "teacher"
    return classroom.get("viewer_role") or "student"


# ── Classroom lifecycle ───────────────────────────────────────────────────────

async def create_classroom(
    owner_student_id: str,
    owner_name: str,
    name: str,
    subject: str = "",
    grade_level: str = "",
    description: str = "",
    join_policy: str = "approval",
) -> dict[str, Any]:
    """Create a classroom; the creator becomes its teacher."""
    if not name.strip():
        raise HTTPException(status_code=422, detail="Classroom name is required")
    if join_policy not in ("open", "approval", "invite_only"):
        join_policy = "approval"
    await ensure_student(owner_student_id, owner_name)
    return await repo.create_classroom(
        owner_student_id=owner_student_id,
        name=name.strip(),
        subject=subject.strip(),
        grade_level=grade_level.strip(),
        description=description.strip(),
        join_policy=join_policy,
    )


async def join_classroom(join_code: str, student_id: str, display_name: str) -> dict[str, Any]:
    """
    Join a classroom by code. Returns {classroom, membership_status}.

    open        → immediately active
    approval    → pending until the teacher approves
    invite_only → rejected (teacher must add by approval of a pending request)
    """
    classroom = await repo.get_classroom_by_join_code(join_code)
    if not classroom:
        raise HTTPException(status_code=404, detail="No classroom found for this code")
    if classroom["owner_student_id"] == student_id:
        raise HTTPException(status_code=400, detail="You are the teacher of this classroom")
    if classroom["join_policy"] == "invite_only":
        raise HTTPException(
            status_code=403, detail="This classroom is invite-only. Ask your teacher to add you."
        )

    existing = await repo.get_membership(classroom["id"], student_id)
    if existing and existing["status"] == "active":
        return {"classroom": classroom, "membership_status": "active", "already_member": True}
    if existing and existing["status"] == "removed":
        raise HTTPException(status_code=403, detail="You were removed from this classroom")

    status = "active" if classroom["join_policy"] == "open" else "pending"
    membership = await repo.upsert_membership(
        classroom["id"], student_id, status=status, display_name=display_name
    )

    # New active members receive all already-assigned courses automatically.
    if status == "active":
        await assign_all_courses_to_student(classroom["id"], student_id)

    return {"classroom": classroom, "membership_status": membership["status"]}


async def approve_member(classroom_id: str, student_id: str) -> dict[str, Any]:
    """Teacher approves a pending request; back-assigns existing courses."""
    membership = await repo.get_membership(classroom_id, student_id)
    if not membership or membership["status"] != "pending":
        raise HTTPException(status_code=404, detail="No pending request for this student")
    result = await repo.upsert_membership(classroom_id, student_id, status="active")
    await assign_all_courses_to_student(classroom_id, student_id)
    return result


async def remove_member(classroom_id: str, student_id: str) -> None:
    """Teacher removes a member (their cloned courses remain theirs)."""
    membership = await repo.get_membership(classroom_id, student_id)
    if not membership:
        raise HTTPException(status_code=404, detail="Member not found")
    await repo.upsert_membership(classroom_id, student_id, status="removed")


async def leave_classroom(classroom_id: str, student_id: str) -> None:
    """Student leaves a classroom voluntarily."""
    membership = await repo.get_membership(classroom_id, student_id)
    if not membership or membership["status"] not in ("active", "pending"):
        raise HTTPException(status_code=404, detail="You are not in this classroom")
    await repo.upsert_membership(classroom_id, student_id, status="left")


# ── Course publish / approve / assign ─────────────────────────────────────────

async def register_classroom_course(
    classroom_id: str,
    teacher_student_id: str,
    template_course_id: str,
    title: str = "",
) -> dict[str, Any]:
    """
    Register a course the teacher already built (via the untouched AI Course
    Creator) as a draft classroom course awaiting review/approval.
    """
    course = await get_course(template_course_id)
    if not course or course["student_id"] != teacher_student_id:
        raise HTTPException(
            status_code=404,
            detail="Course not found among your courses. Build it with the AI Course Creator first.",
        )
    existing = await repo.list_classroom_courses(classroom_id)
    if any(cc["template_course_id"] == template_course_id and cc["status"] != "archived"
           for cc in existing):
        raise HTTPException(status_code=409, detail="This course is already in the classroom")
    return await repo.create_classroom_course(
        classroom_id=classroom_id,
        template_course_id=template_course_id,
        title=title.strip() or course.get("title") or course.get("topic") or "Course",
    )


async def approve_classroom_course(cc_id: str, classroom_id: str) -> dict[str, Any]:
    """Mark a draft classroom course as reviewed and approved."""
    cc = await repo.get_classroom_course(cc_id)
    if not cc or cc["classroom_id"] != classroom_id:
        raise HTTPException(status_code=404, detail="Classroom course not found")
    if cc["status"] not in ("draft", "approved"):
        raise HTTPException(status_code=409, detail=f"Cannot approve from status '{cc['status']}'")
    await repo.set_classroom_course_status(cc_id, "approved")
    return await repo.get_classroom_course(cc_id) or {}


async def assign_course_events(
    cc_id: str, classroom_id: str
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Clone the approved template to every active member, yielding progress
    events for SSE. Idempotent — already-assigned students are skipped.
    """
    cc = await repo.get_classroom_course(cc_id)
    if not cc or cc["classroom_id"] != classroom_id:
        raise HTTPException(status_code=404, detail="Classroom course not found")
    if cc["status"] not in ("approved", "assigned"):
        raise HTTPException(status_code=409, detail="Approve the course before assigning it")

    member_ids = await repo.active_member_ids(classroom_id)
    yield {"type": "start", "total": len(member_ids)}

    done, failed = 0, 0
    for student_id in member_ids:
        try:
            course_id = await repo.clone_course_for_student(
                template_course_id=cc["template_course_id"],
                student_id=student_id,
                classroom_id=classroom_id,
                classroom_course_id=cc_id,
            )
            done += 1
            yield {"type": "assigned", "student_id": student_id,
                   "course_id": course_id, "done": done, "total": len(member_ids)}
        except Exception as exc:  # keep assigning the rest of the class
            failed += 1
            logger.error("Assignment failed for student {}: {}", student_id, exc)
            yield {"type": "error", "student_id": student_id, "detail": str(exc)}

    await repo.set_classroom_course_status(cc_id, "assigned")
    await repo.mark_artifacts_stale(classroom_id)
    yield {"type": "complete", "assigned": done, "failed": failed, "total": len(member_ids)}


async def assign_course_to_classroom(cc_id: str, classroom_id: str) -> dict[str, Any]:
    """Non-streaming assignment used by the plain POST endpoint."""
    summary: dict[str, Any] = {}
    async for event in assign_course_events(cc_id, classroom_id):
        if event["type"] == "complete":
            summary = event
    return summary


async def assign_all_courses_to_student(classroom_id: str, student_id: str) -> int:
    """Give a newly approved/joined student every already-assigned course."""
    count = 0
    for cc in await repo.list_classroom_courses(classroom_id):
        if cc["status"] != "assigned":
            continue
        try:
            await repo.clone_course_for_student(
                template_course_id=cc["template_course_id"],
                student_id=student_id,
                classroom_id=classroom_id,
                classroom_course_id=cc["id"],
            )
            count += 1
        except Exception as exc:
            logger.error(
                "Late assignment failed (cc={} student={}): {}", cc["id"], student_id, exc
            )
    return count
