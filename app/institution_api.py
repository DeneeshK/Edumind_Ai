"""
app/institution_api.py
HTTP API for the "My Institution" module under /api/institution/*.

Every route is cookie-authenticated via require_current_user, and every
classroom-scoped route passes through classroom_service.require_teacher or
require_member — the single authorization choke point for the module.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from app.auth import require_current_user
from agents.institution.clustering_agent import cluster_students
from agents.institution.insight_agent import generate_insights
from agents.institution.recommendation_agent import recommend_for_student
from agents.institution.revision_planner_agent import (
    generate_revision_plan,
    plan_to_markdown,
)
from agents.institution.teacher_assistant_agent import ask_teacher_assistant
from core import classroom_analytics as analytics
from core import classroom_service as service
from core import test_service
from db import institution as repo

router = APIRouter(prefix="/api/institution", tags=["institution"])


def _student_id(current_user: dict[str, Any]) -> str:
    """Extract the authenticated student's id from the auth payload."""
    return str(current_user["student_id"])


def _display_name(current_user: dict[str, Any]) -> str:
    """Best-effort display name from the auth payload."""
    return str(current_user.get("name") or current_user.get("email") or "Student")


# ── Request models ────────────────────────────────────────────────────────────

class ClassroomCreateRequest(BaseModel):
    name: str
    subject: str = ""
    grade_level: str = ""
    description: str = ""


class ClassroomUpdateRequest(BaseModel):
    name: str | None = None
    subject: str | None = None
    grade_level: str | None = None
    description: str | None = None


class InviteRequest(BaseModel):
    emails: str | list[str]


class RevokeInviteRequest(BaseModel):
    email: str


class RegisterCourseRequest(BaseModel):
    template_course_id: str
    title: str = ""


class GenerateTestRequest(BaseModel):
    topic: str
    title: str = ""
    classroom_course_id: str | None = None
    num_mcq: int = Field(default=5, ge=0, le=20)
    num_short: int = Field(default=3, ge=0, le=10)
    num_conceptual: int = Field(default=2, ge=0, le=10)
    difficulty_mix: str = "balanced"
    duration_minutes: int = Field(default=30, ge=5, le=240)
    instructions: str = ""


class TestQuestionPayload(BaseModel):
    id: str | None = None
    question_type: str = "mcq"
    question_text: str
    options: list[str] = []
    correct_answer: str = ""
    explanation: str = ""
    concepts_tested: list[str] = []
    difficulty: str = "medium"
    points: float = 1.0


class UpdateTestRequest(BaseModel):
    title: str | None = None
    topic: str | None = None
    instructions: str | None = None
    duration_minutes: int | None = Field(default=None, ge=5, le=240)
    questions: list[TestQuestionPayload] | None = None


class TransitionTestRequest(BaseModel):
    action: str  # approve | schedule | publish | close
    scheduled_start: str | None = None
    scheduled_end: str | None = None


class SaveAnswersRequest(BaseModel):
    answers: list[dict[str, Any]] = []


class AssistantChatRequest(BaseModel):
    message: str


class PostCreateRequest(BaseModel):
    post_type: str = "announcement"
    title: str = ""
    body_markdown: str
    student_ids: list[str] | None = None  # None/empty → everyone


class RevisionPlanRequest(BaseModel):
    days: int = Field(default=7, ge=3, le=30)
    publish: bool = False


class LearningEventPayload(BaseModel):
    classroom_id: str | None = None
    course_id: str | None = None
    module_id: str | None = None
    event_type: str
    payload: dict[str, Any] = {}


class EventsRequest(BaseModel):
    events: list[LearningEventPayload] = []


# ── Home ──────────────────────────────────────────────────────────────────────

@router.get("/me/home")
async def institution_home(current_user: dict[str, Any] = Depends(require_current_user)):
    """Everything the institution home needs: teaching, joined, and pending
    email invitations for the signed-in user."""
    sid = _student_id(current_user)
    data = await repo.list_classrooms_for_student(sid)
    invitations = await repo.list_invitations_for_email(current_user.get("email") or "")
    return {**data, "invitations": invitations, "student_id": sid}


# ── Classrooms ────────────────────────────────────────────────────────────────

@router.post("/classrooms")
async def create_classroom(
    req: ClassroomCreateRequest,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Create a classroom; the caller becomes its teacher."""
    return await service.create_classroom(
        owner_student_id=_student_id(current_user),
        owner_name=_display_name(current_user),
        name=req.name,
        subject=req.subject,
        grade_level=req.grade_level,
        description=req.description,
    )


@router.get("/classrooms/{classroom_id}")
async def classroom_detail(
    classroom_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Role-aware classroom detail."""
    sid = _student_id(current_user)
    classroom = await service.require_member(classroom_id, sid)
    role = service.viewer_role(classroom, sid)
    classroom.pop("join_code", None)  # codes are not used; joining is by email invite
    return {**classroom, "viewer_role": role}


@router.patch("/classrooms/{classroom_id}")
async def update_classroom(
    classroom_id: str,
    req: ClassroomUpdateRequest,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Edit classroom fields (teacher only)."""
    await service.require_teacher(classroom_id, _student_id(current_user))
    return await repo.update_classroom(classroom_id, req.model_dump(exclude_none=True))


@router.post("/classrooms/{classroom_id}/archive")
async def archive_classroom(
    classroom_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Archive (or re-activate) a classroom."""
    classroom = await service.require_teacher(classroom_id, _student_id(current_user))
    new_status = "archived" if classroom["status"] == "active" else "active"
    return await repo.update_classroom(classroom_id, {"status": new_status})


# ── Invitations (email allowlist) ─────────────────────────────────────────────

@router.get("/classrooms/{classroom_id}/invitations")
async def list_invitations(
    classroom_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Teacher: the classroom's email allowlist and who has joined."""
    await service.require_teacher(classroom_id, _student_id(current_user))
    return {"invitations": await repo.list_invitations(classroom_id)}


@router.post("/classrooms/{classroom_id}/invitations")
async def invite_students(
    classroom_id: str,
    req: InviteRequest,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Add one or more emails to the classroom allowlist."""
    await service.require_teacher(classroom_id, _student_id(current_user))
    return await service.invite_students(
        classroom_id=classroom_id,
        raw_emails=req.emails,
        invited_by=_student_id(current_user),
        teacher_email=current_user.get("email") or "",
    )


@router.post("/classrooms/{classroom_id}/invitations/revoke")
async def revoke_invitation(
    classroom_id: str,
    req: RevokeInviteRequest,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Remove an email from the allowlist (joined students are unaffected)."""
    await service.require_teacher(classroom_id, _student_id(current_user))
    await service.revoke_invitation(classroom_id, req.email)
    return {"success": True}


@router.post("/classrooms/{classroom_id}/accept")
async def accept_invitation(
    classroom_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Student one-tap accepts an email invitation to this classroom."""
    result = await service.accept_invitation(
        classroom_id=classroom_id,
        student_id=_student_id(current_user),
        email=current_user.get("email") or "",
        display_name=_display_name(current_user),
    )
    classroom = dict(result["classroom"])
    classroom.pop("join_code", None)
    return {**result, "classroom": classroom}


# ── Members ───────────────────────────────────────────────────────────────────

@router.get("/classrooms/{classroom_id}/members")
async def list_members(
    classroom_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Members list — teachers see emails, students see names of active members."""
    sid = _student_id(current_user)
    classroom = await service.require_member(classroom_id, sid)
    members = await repo.list_members(classroom_id, statuses=["active"])
    if service.viewer_role(classroom, sid) != "teacher":
        members = [
            {"student_id": m["student_id"], "name": m["name"], "status": m["status"]}
            for m in members
        ]
    return {"members": members}


@router.post("/classrooms/{classroom_id}/members/{member_student_id}/remove")
async def remove_member(
    classroom_id: str,
    member_student_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Remove a student from the classroom."""
    await service.require_teacher(classroom_id, _student_id(current_user))
    await service.remove_member(classroom_id, member_student_id)
    return {"success": True}


@router.post("/classrooms/{classroom_id}/leave")
async def leave_classroom(
    classroom_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Leave a classroom voluntarily."""
    await service.leave_classroom(classroom_id, _student_id(current_user))
    return {"success": True}


# ── Courses ───────────────────────────────────────────────────────────────────

@router.get("/classrooms/{classroom_id}/courses")
async def classroom_courses(
    classroom_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Teacher: all classroom courses. Student: their own assignments."""
    sid = _student_id(current_user)
    classroom = await service.require_member(classroom_id, sid)
    if service.viewer_role(classroom, sid) == "teacher":
        return {"role": "teacher", "courses": await repo.list_classroom_courses(classroom_id)}
    return {
        "role": "student",
        "assignments": await repo.list_assignments_for_student(classroom_id, sid),
    }


@router.post("/classrooms/{classroom_id}/courses")
async def register_course(
    classroom_id: str,
    req: RegisterCourseRequest,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Register one of the teacher's existing AI-built courses as a draft."""
    sid = _student_id(current_user)
    await service.require_teacher(classroom_id, sid)
    return await service.register_classroom_course(
        classroom_id=classroom_id,
        teacher_student_id=sid,
        template_course_id=req.template_course_id,
        title=req.title,
    )


@router.post("/classrooms/{classroom_id}/courses/{cc_id}/approve")
async def approve_course(
    classroom_id: str,
    cc_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Mark a reviewed course as approved for assignment."""
    await service.require_teacher(classroom_id, _student_id(current_user))
    return await service.approve_classroom_course(cc_id, classroom_id)


@router.post("/classrooms/{classroom_id}/courses/{cc_id}/assign")
async def assign_course(
    classroom_id: str,
    cc_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Clone the approved course to every active member (idempotent)."""
    await service.require_teacher(classroom_id, _student_id(current_user))
    return await service.assign_course_to_classroom(cc_id, classroom_id)


@router.get("/classrooms/{classroom_id}/courses/{cc_id}/progress")
async def course_progress_matrix(
    classroom_id: str,
    cc_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Per-student progress for one assigned classroom course."""
    await service.require_teacher(classroom_id, _student_id(current_user))
    cc = await repo.get_classroom_course(cc_id)
    if not cc or cc["classroom_id"] != classroom_id:
        raise HTTPException(status_code=404, detail="Classroom course not found")
    return {
        "classroom_course": cc,
        "students": await repo.list_assignments_for_classroom_course(cc_id),
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

@router.get("/classrooms/{classroom_id}/tests")
async def list_tests(
    classroom_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Teacher: all tests. Student: scheduled/live/closed with own attempts."""
    sid = _student_id(current_user)
    classroom = await service.require_member(classroom_id, sid)
    if service.viewer_role(classroom, sid) == "teacher":
        return {"role": "teacher", "tests": await repo.list_tests(classroom_id)}
    return {"role": "student", "tests": await test_service.visible_tests_for_student(classroom_id, sid)}


@router.post("/classrooms/{classroom_id}/tests/generate")
async def generate_test(
    classroom_id: str,
    req: GenerateTestRequest,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """AI-generate a draft test for the classroom."""
    sid = _student_id(current_user)
    await service.require_teacher(classroom_id, sid)
    return await test_service.generate_test(
        classroom_id=classroom_id,
        teacher_student_id=sid,
        topic=req.topic,
        title=req.title,
        classroom_course_id=req.classroom_course_id,
        num_mcq=req.num_mcq,
        num_short=req.num_short,
        num_conceptual=req.num_conceptual,
        difficulty_mix=req.difficulty_mix,
        duration_minutes=req.duration_minutes,
        instructions=req.instructions,
    )


@router.get("/classrooms/{classroom_id}/tests/{test_id}")
async def test_detail(
    classroom_id: str,
    test_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Role-aware test detail — students never see answers before grading."""
    sid = _student_id(current_user)
    classroom = await service.require_member(classroom_id, sid)
    test = await repo.get_test(test_id)
    if not test or test["classroom_id"] != classroom_id:
        raise HTTPException(status_code=404, detail="Test not found")
    if service.viewer_role(classroom, sid) == "teacher":
        return {"role": "teacher", **await test_service.test_detail_for_teacher(test_id)}
    return {"role": "student", **await test_service.test_detail_for_student(test_id, sid)}


@router.patch("/classrooms/{classroom_id}/tests/{test_id}")
async def update_test(
    classroom_id: str,
    test_id: str,
    req: UpdateTestRequest,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Edit test metadata and/or replace its question list."""
    await service.require_teacher(classroom_id, _student_id(current_user))
    test = await repo.get_test(test_id)
    if not test or test["classroom_id"] != classroom_id:
        raise HTTPException(status_code=404, detail="Test not found")
    meta = req.model_dump(exclude_none=True, exclude={"questions"})
    if meta:
        await repo.update_test(test_id, meta)
    if req.questions is not None:
        return await test_service.update_test_questions(
            test_id, [q.model_dump() for q in req.questions]
        )
    return await test_service.test_detail_for_teacher(test_id)


@router.post("/classrooms/{classroom_id}/tests/{test_id}/questions/{question_id}/regenerate")
async def regenerate_question(
    classroom_id: str,
    test_id: str,
    question_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Regenerate one question, keeping the rest of the test untouched."""
    await service.require_teacher(classroom_id, _student_id(current_user))
    test = await repo.get_test(test_id)
    if not test or test["classroom_id"] != classroom_id:
        raise HTTPException(status_code=404, detail="Test not found")
    return await test_service.regenerate_question(test_id, question_id)


@router.post("/classrooms/{classroom_id}/tests/{test_id}/transition")
async def transition_test(
    classroom_id: str,
    test_id: str,
    req: TransitionTestRequest,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Move a test through approve → schedule → publish → close."""
    await service.require_teacher(classroom_id, _student_id(current_user))
    test = await repo.get_test(test_id)
    if not test or test["classroom_id"] != classroom_id:
        raise HTTPException(status_code=404, detail="Test not found")
    return await test_service.transition_test(
        test_id, req.action, req.scheduled_start, req.scheduled_end
    )


@router.get("/classrooms/{classroom_id}/tests/{test_id}/results")
async def test_results(
    classroom_id: str,
    test_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """All graded attempts for a test — teacher results view."""
    await service.require_teacher(classroom_id, _student_id(current_user))
    test = await repo.get_test(test_id)
    if not test or test["classroom_id"] != classroom_id:
        raise HTTPException(status_code=404, detail="Test not found")
    attempts = await repo.list_attempts_for_test(test_id)
    graded = [a for a in attempts if a["status"] == "graded" and a.get("max_score")]
    ratios = [float(a["score"]) / float(a["max_score"]) for a in graded if a["max_score"]]
    return {
        "test": test,
        "attempts": attempts,
        "stats": {
            "graded": len(graded),
            "avg_score": round(sum(ratios) / len(ratios), 4) if ratios else None,
            "max": round(max(ratios), 4) if ratios else None,
            "min": round(min(ratios), 4) if ratios else None,
        },
    }


# ── Attempts (student) ────────────────────────────────────────────────────────

@router.post("/classrooms/{classroom_id}/tests/{test_id}/attempts/start")
async def start_attempt(
    classroom_id: str,
    test_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Start or resume the student's attempt on a live test."""
    sid = _student_id(current_user)
    await service.require_member(classroom_id, sid)
    test = await repo.get_test(test_id)
    if not test or test["classroom_id"] != classroom_id:
        raise HTTPException(status_code=404, detail="Test not found")
    return await test_service.start_attempt(test_id, sid)


@router.post("/attempts/{attempt_id}/answers")
async def save_answers(
    attempt_id: str,
    req: SaveAnswersRequest,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Autosave the student's partial answers."""
    await test_service.save_answers(attempt_id, _student_id(current_user), req.answers)
    return {"success": True}


@router.post("/attempts/{attempt_id}/submit")
async def submit_attempt(
    attempt_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Submit and grade the attempt (instant MCQ + AI rubric for the rest)."""
    return await test_service.submit_attempt(attempt_id, _student_id(current_user))


# ── Analytics ─────────────────────────────────────────────────────────────────

@router.get("/classrooms/{classroom_id}/analytics/overview")
async def analytics_overview(
    classroom_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Headline KPIs and weekly score trend."""
    await service.require_teacher(classroom_id, _student_id(current_user))
    return await analytics.overview(classroom_id)


@router.get("/classrooms/{classroom_id}/analytics/concepts")
async def analytics_concepts(
    classroom_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Concept mastery heatmap (students × concepts)."""
    await service.require_teacher(classroom_id, _student_id(current_user))
    return await analytics.concept_heatmap(classroom_id)


@router.get("/classrooms/{classroom_id}/analytics/students")
async def analytics_students(
    classroom_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Ranked per-student analytics table with risk flags."""
    await service.require_teacher(classroom_id, _student_id(current_user))
    return {"students": await analytics.student_table(classroom_id)}


@router.get("/classrooms/{classroom_id}/analytics/doubts")
async def analytics_doubts(
    classroom_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Top doubt concepts and recent doubt samples."""
    await service.require_teacher(classroom_id, _student_id(current_user))
    return await analytics.doubt_analytics(classroom_id)


@router.get("/classrooms/{classroom_id}/analytics/student/{target_student_id}")
async def analytics_student_detail(
    classroom_id: str,
    target_student_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Drilldown for one student — teacher, or the student viewing themself."""
    sid = _student_id(current_user)
    classroom = await service.require_member(classroom_id, sid)
    if service.viewer_role(classroom, sid) != "teacher" and sid != target_student_id:
        raise HTTPException(status_code=403, detail="You can only view your own progress")
    detail = await analytics.student_drilldown(classroom_id, target_student_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Student not found in this classroom")
    return detail


# ── AI layer ──────────────────────────────────────────────────────────────────

@router.post("/classrooms/{classroom_id}/ai/insights")
async def ai_insights(
    classroom_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Run the Insight Agent over fresh analytics and cache the result."""
    await service.require_teacher(classroom_id, _student_id(current_user))
    snapshot = await analytics.analytics_snapshot(classroom_id)
    try:
        content = await generate_insights(snapshot)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return await repo.save_ai_artifact(classroom_id, "insights", content, "insight_agent")


@router.post("/classrooms/{classroom_id}/ai/clusters")
async def ai_clusters(
    classroom_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Run the Clustering Agent over per-student feature vectors."""
    await service.require_teacher(classroom_id, _student_id(current_user))
    students = await analytics.student_table(classroom_id)
    try:
        content = await cluster_students(students)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return await repo.save_ai_artifact(classroom_id, "clusters", content, "clustering_agent")


@router.post("/classrooms/{classroom_id}/ai/revision-plan")
async def ai_revision_plan(
    classroom_id: str,
    req: RevisionPlanRequest,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Generate a class revision plan; optionally publish it to the stream."""
    sid = _student_id(current_user)
    await service.require_teacher(classroom_id, sid)
    snapshot = await analytics.analytics_snapshot(classroom_id)
    try:
        plan = await generate_revision_plan(snapshot, days=req.days)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    artifact = await repo.save_ai_artifact(
        classroom_id, "revision_plan", plan, "revision_planner_agent"
    )
    if req.publish:
        await repo.create_post(
            classroom_id=classroom_id,
            author_id=sid,
            post_type="revision_plan",
            title=plan["title"],
            body_markdown=plan_to_markdown(plan),
        )
    return artifact


@router.post("/classrooms/{classroom_id}/ai/recommendations/{target_student_id}")
async def ai_recommendations(
    classroom_id: str,
    target_student_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Personalized recommendations for one student (teacher or self)."""
    sid = _student_id(current_user)
    classroom = await service.require_member(classroom_id, sid)
    if service.viewer_role(classroom, sid) != "teacher" and sid != target_student_id:
        raise HTTPException(status_code=403, detail="You can only get your own recommendations")
    drilldown = await analytics.student_drilldown(classroom_id, target_student_id)
    if not drilldown:
        raise HTTPException(status_code=404, detail="Student not found in this classroom")
    try:
        content = await recommend_for_student(drilldown)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return await repo.save_ai_artifact(
        classroom_id, "recommendations", content, "recommendation_agent",
        scope_key=target_student_id,
    )


@router.get("/classrooms/{classroom_id}/ai/artifacts/{artifact_type}")
async def latest_artifact(
    classroom_id: str,
    artifact_type: str,
    scope_key: str = "",
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Read the latest cached AI artifact of a type (may be stale)."""
    sid = _student_id(current_user)
    classroom = await service.require_member(classroom_id, sid)
    if artifact_type not in ("insights", "clusters", "revision_plan", "recommendations"):
        raise HTTPException(status_code=404, detail="Unknown artifact type")
    # Students may only read their own recommendations and the revision plan.
    if service.viewer_role(classroom, sid) != "teacher":
        if artifact_type == "recommendations":
            scope_key = sid
        elif artifact_type != "revision_plan":
            raise HTTPException(status_code=403, detail="Teacher access required")
    artifact = await repo.latest_ai_artifact(classroom_id, artifact_type, scope_key)
    return artifact or {"artifact_type": artifact_type, "content_json": None}


# ── Teacher assistant ─────────────────────────────────────────────────────────

@router.post("/classrooms/{classroom_id}/assistant/chat")
async def assistant_chat(
    classroom_id: str,
    req: AssistantChatRequest,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Ask the analytics-grounded teacher assistant one question."""
    sid = _student_id(current_user)
    classroom = await service.require_teacher(classroom_id, sid)
    question = req.message.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Message is required")

    history = await repo.list_assistant_messages(classroom_id, sid, limit=8)
    await repo.record_assistant_message(classroom_id, sid, "user", question)
    try:
        result = await ask_teacher_assistant(
            classroom_id=classroom_id,
            classroom_name=classroom["name"],
            subject=classroom.get("subject", ""),
            question=question,
            history=history,
        )
    except Exception as exc:
        logger.error("Teacher assistant failed: {}", exc)
        raise HTTPException(
            status_code=502, detail="The assistant is unavailable right now. Please try again."
        ) from exc
    await repo.record_assistant_message(
        classroom_id, sid, "assistant", result["answer_markdown"], result["tool_trace"]
    )
    return result


@router.get("/classrooms/{classroom_id}/assistant/history")
async def assistant_history(
    classroom_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """The teacher's saved assistant conversation for this classroom."""
    sid = _student_id(current_user)
    await service.require_teacher(classroom_id, sid)
    return {"messages": await repo.list_assistant_messages(classroom_id, sid)}


# ── Posts / stream ────────────────────────────────────────────────────────────

@router.get("/classrooms/{classroom_id}/posts")
async def list_posts(
    classroom_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Classroom stream. Students only see posts addressed to them (or all)."""
    sid = _student_id(current_user)
    classroom = await service.require_member(classroom_id, sid)
    posts = await repo.list_posts(classroom_id)
    if service.viewer_role(classroom, sid) != "teacher":
        posts = [
            p for p in posts
            if p["audience_json"].get("all")
            or sid in (p["audience_json"].get("student_ids") or [])
        ]
    return {"posts": posts}


@router.post("/classrooms/{classroom_id}/posts")
async def create_post(
    classroom_id: str,
    req: PostCreateRequest,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Publish an announcement / task (optionally targeted to specific students)."""
    sid = _student_id(current_user)
    await service.require_teacher(classroom_id, sid)
    if not req.body_markdown.strip():
        raise HTTPException(status_code=422, detail="Post body is required")
    audience = (
        {"student_ids": req.student_ids} if req.student_ids else {"all": True}
    )
    post_type = req.post_type if req.post_type in ("announcement", "task", "revision_plan") else "announcement"
    return await repo.create_post(
        classroom_id=classroom_id,
        author_id=sid,
        post_type=post_type,
        title=req.title.strip(),
        body_markdown=req.body_markdown,
        audience=audience,
    )


# ── Learning events ───────────────────────────────────────────────────────────

@router.post("/events")
async def record_events(
    req: EventsRequest,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Batched engagement events from the frontend (always self-attributed)."""
    sid = _student_id(current_user)
    events = [
        {**e.model_dump(), "student_id": sid}
        for e in req.events[:100]
        if e.event_type
    ]
    written = await repo.record_learning_events(events)
    return {"written": written}
