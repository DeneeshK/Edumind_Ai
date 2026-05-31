"""
app/course_api.py
Frontend-friendly EduMind API.

The legacy /session endpoints remain in app/api.py. This router adds the
course/module contract used by the React app.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from agents.course_report_agent import generate_course_report
from agents.evaluation_agent import (
    get_session_report,
    start_session as start_eval_session,
    submit_answer as submit_eval_answer,
)
from app.auth import require_current_user
from core.course_service import (
    answer_module_chat,
    complete_module,
    create_course,
    create_course_events,
    evaluate_module_answer,
    generate_module_lesson,
    generate_module_lesson_events,
    get_or_create_module_questions,
    get_student_history_snapshot,
    lesson_videos_from_module,
)
from db.postgres import (
    get_course_for_student,
    get_course_module_for_student,
    get_next_module,
    get_prev_module,
    get_course_decision_log,
    get_course_roadmap,
    get_course_completion_report,
    get_student_dashboard,
    get_student_doubts,
    get_student_skills,
    get_user_by_student_id,
    get_evaluation_session_for_student,
    get_latest_evaluation_session_for_student,
    list_course_modules_for_student,
    list_courses,
    list_module_chat_history_for_student,
    upsert_dev_user,
)


router = APIRouter(prefix="/api", tags=["frontend"])

_course_creation_jobs: dict[str, dict[str, Any]] = {}


SAFE_PROFILE_KEYS = {
    "topic",
    "exact_subject",
    "learning_goal",
    "goal",
    "goal_description",
    "target_context",
    "current_level",
    "learner_level",
    "specialization",
    "course_scope",
    "pace",
    "depth_preference",
    "time_constraint",
    "time_commitment",
    "deadline",
    "prior_knowledge_summary",
    "prior_knowledge",
    "prior_experience",
    "known_concepts",
    "weak_concepts",
    "must_include",
    "should_skip",
    "preferred_teaching_style",
    "assessment_preference",
    "expected_outcome",
    "course_constraints",
    "setup_source",
    "known_concepts",
    "weak_concepts",
    "must_include",
    "do_not_include",
    "duration_value",
    "duration_unit",
    "hours_per_day",
}


def _safe_creation_profile(profile: dict[str, Any] | None) -> dict[str, Any]:
    """Keep only course-creation profile fields that are safe to persist and reuse."""
    if not isinstance(profile, dict):
        return {}
    source = profile.get("profile") if isinstance(profile.get("profile"), dict) else profile
    return {
        key: source[key]
        for key in SAFE_PROFILE_KEYS
        if key in source
    }


LEVEL_LABELS = {
    "complete_beginner": "complete beginner",
    "basic": "some basic knowledge",
    "intermediate": "intermediate",
    "advanced": "advanced",
    "not_sure": "not sure",
}

DEPTH_BY_PACE = {
    "fast": "quick overview and practical path",
    "medium": "balanced learning with practice",
    "deep": "detailed, rigorous, concept-heavy learning",
}


def _clean_text(value: Any) -> str:
    """Normalize optional user-facing text fields to stripped strings."""
    return str(value or "").strip()


def _valid_pace(value: Any) -> str:
    """Return a supported course pace, defaulting to medium."""
    pace = _clean_text(value).lower()
    return pace if pace in {"fast", "medium", "deep"} else "medium"


def _level_label(value: Any) -> str:
    """Convert frontend learner-level keys into display labels."""
    key = _clean_text(value).lower()
    return LEVEL_LABELS.get(key, _clean_text(value))


def _time_commitment_text(
    duration_value: int | None = None,
    duration_unit: str | None = None,
    hours_per_day: float | None = None,
    time_commitment: dict[str, Any] | None = None,
    deadline: str | None = None,
) -> str:
    """
    Build a plain-English time constraint string from structured time fields.

    Legacy `time_commitment` payloads are still accepted so existing clients can
    submit older setup data without changing the course-generation contract.
    """
    parts = []
    if duration_value and duration_unit:
        unit_label = {"days": "day", "weeks": "week", "months": "month"}.get(
            str(duration_unit).lower(), str(duration_unit)
        )
        parts.append(f"{duration_value} {unit_label}{'s' if duration_value != 1 else ''}")
    if hours_per_day:
        h = float(hours_per_day)
        label = f"{h:.0f}" if h == int(h) else f"{h}"
        parts.append(f"{label} hour{'s' if h != 1 else ''}/day")
    if not parts and isinstance(time_commitment, dict):
        value = _clean_text(time_commitment.get("value"))
        unit = _clean_text(time_commitment.get("unit")).lower()
        if value:
            if unit == "minutes_per_day":
                parts.append(f"{value} minutes per day")
            elif unit == "hours_per_week":
                parts.append(f"{value} hours per week")
            else:
                parts.append(value)
    text = ", ".join(parts)
    deadline_text = _clean_text(deadline)
    if deadline_text:
        text = f"{text}, target by {deadline_text}" if text else f"Target by {deadline_text}"
    return text


def course_payload_from_request(req: "CreateCourseRequest") -> dict[str, Any]:
    """Normalize the guided setup request into the course-generation service payload."""
    source = _safe_creation_profile(req.profile or {})
    topic = _clean_text(req.topic or source.get("topic") or source.get("exact_subject"))
    goal_description = _clean_text(
        req.goal_description
        or source.get("goal_description")
        or source.get("target_context")
    )
    explicit_goal = _clean_text(req.goal or source.get("learning_goal") or source.get("goal"))
    goal = explicit_goal or goal_description or (f"Learn {topic}" if topic else "")
    pace = _valid_pace(req.pace or source.get("pace"))
    current_level = _clean_text(req.current_level or source.get("current_level") or "not_sure")
    learner_level = _level_label(current_level or source.get("learner_level"))
    prior_experience = _clean_text(req.prior_experience or source.get("prior_experience"))

    # Prefer structured time fields, while preserving legacy dict compatibility.
    duration_value = req.duration_value or None
    duration_unit = _clean_text(req.duration_unit or source.get("duration_unit")).lower() or None
    hours_per_day = req.hours_per_day or None
    time_commitment = (
        req.time_commitment
        if isinstance(req.time_commitment, dict)
        else source.get("time_commitment")
        if isinstance(source.get("time_commitment"), dict)
        else {}
    )
    deadline = _clean_text(req.deadline or source.get("deadline"))
    time_constraint = _time_commitment_text(
        duration_value=duration_value,
        duration_unit=duration_unit,
        hours_per_day=hours_per_day,
        time_commitment=time_commitment,
        deadline=deadline,
    ) or _clean_text(source.get("time_constraint"))

    # Merge personalization lists from both direct request fields and profile data.
    known_concepts = list(dict.fromkeys(
        list(getattr(req, "known_concepts", None) or [])
        + list(source.get("known_concepts") or [])
        + list(source.get("assumed_known_concepts") or [])
    ))
    weak_concepts = list(dict.fromkeys(
        list(getattr(req, "weak_concepts", None) or [])
        + list(source.get("weak_concepts") or [])
    ))
    must_include = list(dict.fromkeys(
        list(getattr(req, "must_include", None) or [])
        + list(source.get("must_include") or [])
    ))
    do_not_include = list(dict.fromkeys(
        list(getattr(req, "do_not_include", None) or [])
        + list(source.get("do_not_include") or [])
    ))

    prior_parts = []
    if learner_level:
        prior_parts.append(f"Current level: {learner_level}.")
    if prior_experience:
        prior_parts.append(f"Prior experience: {prior_experience}.")
    if known_concepts:
        prior_parts.append(f"Already knows: {', '.join(known_concepts[:8])}.")
    if weak_concepts:
        prior_parts.append(f"Struggles with: {', '.join(weak_concepts[:8])}.")
    prior_knowledge = _clean_text(
        req.prior_knowledge
        or source.get("prior_knowledge_summary")
        or source.get("prior_knowledge")
        or " ".join(prior_parts)
    )

    profile = dict(source)
    profile.update({
        "topic": topic,
        "exact_subject": topic,
        "learning_goal": goal,
        "goal_description": goal_description,
        "target_context": goal_description or source.get("target_context") or "general learning",
        "current_level": current_level,
        "learner_level": learner_level,
        "pace": pace,
        "depth_preference": source.get("depth_preference") or DEPTH_BY_PACE[pace],
        "time_constraint": time_constraint,
        "time_commitment": time_commitment,
        "duration_value": duration_value,
        "duration_unit": duration_unit,
        "hours_per_day": hours_per_day,
        "deadline": deadline,
        "prior_experience": prior_experience,
        "prior_knowledge_summary": prior_knowledge,
        "known_concepts": known_concepts,
        "weak_concepts": weak_concepts,
        "must_include": must_include,
        "do_not_include": do_not_include,
        "expected_outcome": goal_description or explicit_goal or goal,
        "setup_source": "guided_course_setup",
    })

    return {
        "topic": topic,
        "goal": goal,
        "pace": pace,
        "prior_knowledge": prior_knowledge,
        "profile": _safe_creation_profile(profile),
    }


class DevLoginRequest(BaseModel):
    """Development-only login payload used when local auth bypass is enabled."""

    name: str = "EduMind Student"
    email: str = "student@edumind.dev"
    avatar_url: str = ""


class CreateCourseRequest(BaseModel):
    """Frontend payload for creating a personalized course from guided setup."""

    student_id: str | None = None
    topic: str | None = None
    goal: str | None = None
    goal_description: str | None = None
    current_level: str | None = None
    prior_experience: str = ""
    # Structured time commitment
    # duration_value + duration_unit = total time available (e.g. 3 weeks)
    # hours_per_day = daily study hours (e.g. 2.0)
    duration_value: int | None = None
    duration_unit: str | None = None           # "days" | "weeks" | "months"
    hours_per_day: float | None = None
    time_commitment: dict[str, Any] | None = None  # legacy compat
    deadline: str | None = None
    pace: str | None = "medium"
    prior_knowledge: str = ""
    # Personalisation lists
    known_concepts: list[str] = Field(default_factory=list)   # already mastered — skip
    weak_concepts: list[str] = Field(default_factory=list)    # struggles with — reinforce
    must_include: list[str] = Field(default_factory=list)     # must be in course
    do_not_include: list[str] = Field(default_factory=list)   # absolute exclusions
    name: str = "Student"
    profile: dict[str, Any] | None = None


class ChatRequest(BaseModel):
    """Request body for asking a question inside a generated module."""

    student_id: str | None = None
    message: str


class EvaluateRequest(BaseModel):
    """Request body for answering one module check question."""

    student_id: str | None = None
    answer: str
    question_id: str
    confidence: int = Field(default=3, ge=1, le=5)


def _sse(data: Any, event: str = "message") -> str:
    """Format data as a Server-Sent Event frame."""
    if not isinstance(data, str):
        data = json.dumps(data, default=str)
    lines = data.splitlines() or [""]
    payload = "".join(f"data: {line}\n" for line in lines)
    return f"event: {event}\n{payload}\n"


async def _event_stream(events):
    """Convert internal event dictionaries into SSE frames."""
    async for item in events:
        yield _sse(item.get("data", ""), item.get("event", "message"))


def _current_student_id(current_user: dict[str, Any]) -> str:
    """Extract the authenticated student's id from the auth dependency payload."""
    return str(current_user["student_id"])


def _require_matching_student(student_id: str, current_user: dict[str, Any]) -> str:
    """Reject requests for a student id that does not match the authenticated user."""
    current_student_id = _current_student_id(current_user)
    if student_id != current_student_id:
        raise HTTPException(status_code=404, detail="Student not found")
    return current_student_id


async def _require_owned_course(
    course_id: str,
    current_user: dict[str, Any],
) -> dict[str, Any]:
    """Load a course only when it belongs to the authenticated student."""
    course = await get_course_for_student(course_id, _current_student_id(current_user))
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    return course


async def _require_owned_module(
    course_id: str,
    module_id: str,
    current_user: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a course module only when both course and module belong to the student."""
    course = await _require_owned_course(course_id, current_user)
    module = await get_course_module_for_student(
        course_id,
        module_id,
        _current_student_id(current_user),
    )
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    return course, module


@router.post("/auth/dev-login")
async def dev_login(req: DevLoginRequest):
    """Create or update a development user profile and return it."""
    profile = await upsert_dev_user(
        email=req.email,
        name=req.name,
        avatar_url=req.avatar_url,
    )
    return {"user": profile, "auth_mode": "dev"}


@router.get("/auth/me")
async def auth_me(student_id: str = Query(default="")):
    """Return the development-mode user profile for the provided student id."""
    if not student_id:
        raise HTTPException(status_code=401, detail="student_id is required in dev mode")
    profile = await get_user_by_student_id(student_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user": profile, "auth_mode": "dev"}


@router.post("/auth/logout")
async def logout():
    """Return a no-op logout response for the development auth route."""
    return {"status": "ok"}


@router.get("/courses")
async def courses(
    student_id: str | None = Query(default=None),
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """List courses owned by the authenticated student."""
    return {"courses": await list_courses(_current_student_id(current_user))}


@router.post("/courses")
async def create_course_endpoint(
    req: CreateCourseRequest,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Create a course immediately and return its saved roadmap payload."""
    payload = course_payload_from_request(req)
    if not payload["topic"]:
        raise HTTPException(status_code=400, detail="topic is required")
    try:
        course = await create_course(
            student_id=_current_student_id(current_user),
            topic=payload["topic"],
            goal=payload["goal"],
            pace=payload["pace"],
            prior_knowledge=payload["prior_knowledge"],
            name=req.name,
            personalization_profile=payload["profile"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {
        "course": course,
        "course_id": course["id"],
        "roadmap_ready": bool(course.get("roadmap")),
        "roadmap": course.get("roadmap"),
        "redirect_url": course.get("redirect_url"),
    }


@router.post("/courses/create-intent")
async def create_course_intent(
    req: CreateCourseRequest,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Create a short-lived streaming job for course generation."""
    payload = course_payload_from_request(req)
    if not payload["topic"]:
        raise HTTPException(status_code=400, detail="topic is required")
    job_id = str(uuid.uuid4())
    _course_creation_jobs[job_id] = {
        "student_id": _current_student_id(current_user),
        "topic": payload["topic"],
        "goal": payload["goal"],
        "pace": payload["pace"],
        "prior_knowledge": payload["prior_knowledge"],
        "name": req.name,
        "profile": payload["profile"],
    }
    return {
        "job_id": job_id,
        "stream_url": f"/api/stream/courses/create/{job_id}",
    }


@router.get("/courses/{course_id}")
async def course_detail(
    course_id: str,
    student_id: str | None = Query(default=None),
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Return one course with modules, roadmap readiness, and completion-report state."""
    sid = _current_student_id(current_user)
    course = await _require_owned_course(course_id, current_user)
    course["modules"] = await list_course_modules_for_student(course_id, sid)
    course["roadmap"] = await get_course_roadmap(course_id)
    course["roadmap_ready"] = bool(course["roadmap"])
    cached_report = await get_course_completion_report(course_id, sid)
    course["has_completion_report"] = bool(cached_report and cached_report.get("report"))
    return {"course": course}


@router.get("/courses/{course_id}/roadmap")
async def course_roadmap(
    course_id: str,
    student_id: str | None = Query(default=None),
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Return the saved roadmap for a course owned by the authenticated student."""
    course = await _require_owned_course(course_id, current_user)
    roadmap = await get_course_roadmap(course_id)
    if not roadmap:
        raise HTTPException(status_code=404, detail="Roadmap not found")
    return {"course": course, "roadmap": roadmap}


@router.post("/courses/{course_id}/roadmap/regenerate")
async def regenerate_course_roadmap(
    course_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Regenerate and persist a course roadmap from existing modules."""
    from core.roadmap_service import CourseRoadmapService
    from db.postgres import save_course_roadmap

    course = await _require_owned_course(course_id, current_user)
    modules = await list_course_modules_for_student(
        course_id,
        _current_student_id(current_user),
    )
    history = await get_student_history_snapshot(course["student_id"])
    profile = course.get("personalization_profile") or {}
    roadmap = CourseRoadmapService().build(course, modules, profile, history)
    roadmap = await save_course_roadmap(course_id, roadmap)
    return {"course": course, "roadmap": roadmap}


@router.get("/courses/{course_id}/modules")
async def course_modules(
    course_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """List saved modules for a course."""
    await _require_owned_course(course_id, current_user)
    return {
        "modules": await list_course_modules_for_student(
            course_id,
            _current_student_id(current_user),
        )
    }


@router.get("/courses/{course_id}/modules/{module_id}")
async def course_module(
    course_id: str,
    module_id: str,
    student_id: str | None = Query(default=None),
    auto_generate: bool = Query(default=False),
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Return one module, optionally generating lesson content before responding."""
    current_student_id = _current_student_id(current_user)
    course, owned_module = await _require_owned_module(course_id, module_id, current_user)
    if auto_generate:
        module = await generate_module_lesson(course_id, module_id, current_student_id)
    else:
        module = owned_module
        module["questions"] = []
        module["videos"] = lesson_videos_from_module(module)
        logger.info(
            "module_get_response_videos course_id='{}' module_id='{}' response_video_count={}",
            course_id,
            module_id,
            len(module.get("videos") or []),
        )
    return {"module": module}


@router.post("/courses/{course_id}/modules/{module_id}/generate")
async def generate_module(
    course_id: str,
    module_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Generate and persist lesson content for a module."""
    await _require_owned_module(course_id, module_id, current_user)
    module = await generate_module_lesson(
        course_id,
        module_id,
        _current_student_id(current_user),
    )
    return {"module": module}


@router.post("/courses/{course_id}/modules/{module_id}/complete")
async def complete_module_endpoint(
    course_id: str,
    module_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Mark a module complete for the authenticated student."""
    await _require_owned_module(course_id, module_id, current_user)
    try:
        module = await complete_module(
            course_id,
            module_id,
            _current_student_id(current_user),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"module": module}


@router.post("/courses/{course_id}/modules/{module_id}/chat")
async def module_chat(
    course_id: str,
    module_id: str,
    req: ChatRequest,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Answer a student's module chat message and persist the doubt trace."""
    await _require_owned_module(course_id, module_id, current_user)
    try:
        return await answer_module_chat(
            course_id=course_id,
            module_id=module_id,
            student_id=_current_student_id(current_user),
            message=req.message,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/courses/{course_id}/modules/{module_id}/chat-history")
async def module_chat_history(
    course_id: str,
    module_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Return module chat history visible to the authenticated student."""
    await _require_owned_module(course_id, module_id, current_user)
    return {
        "messages": await list_module_chat_history_for_student(
            course_id,
            module_id,
            _current_student_id(current_user),
        )
    }


@router.get("/courses/{course_id}/modules/{module_id}/questions")
async def module_questions(
    course_id: str,
    module_id: str,
    student_id: str | None = Query(default=None),
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Return saved module questions, generating them from the lesson when needed."""
    course, module = await _require_owned_module(course_id, module_id, current_user)
    return {"questions": await get_or_create_module_questions(course, module)}


@router.post("/courses/{course_id}/modules/{module_id}/evaluate")
async def module_evaluate(
    course_id: str,
    module_id: str,
    req: EvaluateRequest,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Evaluate one answer against a generated module question."""
    await _require_owned_module(course_id, module_id, current_user)
    try:
        return await evaluate_module_answer(
            course_id=course_id,
            module_id=module_id,
            student_id=_current_student_id(current_user),
            question_id=req.question_id,
            answer=req.answer,
            confidence=req.confidence,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


class StartEvaluationRequest(BaseModel):
    """Request body for starting a module evaluation session."""

    student_id: str | None = None


class SubmitAnswerRequest(BaseModel):
    """Request body for submitting one answer during module evaluation."""

    question_id: str
    answer_text: str
    confidence: int = 3  # 1-5


@router.post("/courses/{course_id}/modules/{module_id}/evaluation/start")
async def evaluation_start(
    course_id: str,
    module_id: str,
    req: StartEvaluationRequest | None = None,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """
    Start evaluation after the student clicks Next or Complete.

    If lesson content is missing, the endpoint generates the lesson first so
    evaluation can still start from a grounded module body.
    """
    student_id = _current_student_id(current_user)
    try:
        # Evaluation requires lesson content; generate it lazily when the student
        # reaches the quiz before opening the module page.
        course, module = await _require_owned_module(course_id, module_id, current_user)
        if module and not module.get("content_markdown"):
            logger.info(
                "evaluation/start: lesson not yet generated for module '{}' — auto-generating before eval.",
                module_id,
            )
            try:
                if course:
                    await generate_module_lesson(
                        course_id=course_id,
                        module_id=module_id,
                        student_id=student_id,
                    )
            except Exception as gen_exc:
                logger.warning("Auto-lesson generation before eval failed: {}", gen_exc)
                # Keep evaluation as the source of truth for whether enough content exists.

        result = await start_eval_session(
            course_id=course_id,
            module_id=module_id,
            student_id=student_id,
        )
        return result
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Evaluation start failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/courses/{course_id}/modules/{module_id}/evaluation/{session_id}/answer"
)
async def evaluation_submit_answer(
    course_id: str,
    module_id: str,
    session_id: str,
    req: SubmitAnswerRequest,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """
    Submit one evaluation answer.

    The response either contains the next question or the final report when the
    session has completed.
    """
    try:
        await _require_owned_module(course_id, module_id, current_user)
        session = await get_evaluation_session_for_student(
            session_id,
            _current_student_id(current_user),
        )
        if (
            not session
            or session.get("course_id") != course_id
            or session.get("module_id") != module_id
        ):
            raise HTTPException(status_code=404, detail="Evaluation session not found")
        result = await submit_eval_answer(
            session_id=session_id,
            question_id=req.question_id,
            answer_text=req.answer_text,
            confidence=max(1, min(5, int(req.confidence or 3))),
        )

        # Auto-complete the module when the evaluation finishes successfully.
        # RETEACH decisions mean the student needs more work — do not mark complete.
        # Failure here is non-fatal: the eval result is still returned to the client.
        if result.get("session_complete") and result.get("decision", "") not in (
            "RETEACH_ALL",
            "RETEACH_WEAK_CONCEPTS",
            "REPEAT_MODULE",
        ):
            try:
                await complete_module(
                    course_id,
                    module_id,
                    _current_student_id(current_user),
                )
            except Exception:
                logger.warning(
                    "Auto-complete module failed after eval (non-fatal)",
                    exc_info=True,
                )

        return result
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Evaluation answer submission failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/courses/{course_id}/modules/{module_id}/evaluation/{session_id}/report"
)
async def evaluation_report(
    course_id: str,
    module_id: str,
    session_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """
    Return the completed evaluation report for a verified session.
    """
    try:
        await _require_owned_module(course_id, module_id, current_user)
        session = await get_evaluation_session_for_student(
            session_id,
            _current_student_id(current_user),
        )
        if (
            not session
            or session.get("course_id") != course_id
            or session.get("module_id") != module_id
        ):
            raise HTTPException(status_code=404, detail="Evaluation report not found")
        result = await get_session_report(session_id)
        return result
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Evaluation report fetch failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/courses/{course_id}/modules/{module_id}/evaluation/latest"
)
async def evaluation_latest(
    course_id: str,
    module_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """
    Return the most recent evaluation session summary for a module, if present.
    """
    try:
        await _require_owned_module(course_id, module_id, current_user)
        session = await get_latest_evaluation_session_for_student(
            course_id,
            module_id,
            _current_student_id(current_user),
        )
        if not session:
            return {"session": None, "message": "No evaluation done yet for this module."}
        return {
            "session": {
                "session_id": session["session_id"],
                "status": session["status"],
                "questions_asked": session["questions_asked"],
                "decision": session["decision"],
                "motivational_feedback": session["motivational_feedback"],
            }
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Evaluation latest fetch failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/courses/{course_id}/modules/{module_id}/evaluation/latest-full"
)
async def evaluation_latest_full(
    course_id: str,
    module_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """
    Return the latest completed evaluation report with full persisted details.

    The frontend uses this to rehydrate progress UI after reopening a module.
    """
    try:
        await _require_owned_module(course_id, module_id, current_user)
        session = await get_latest_evaluation_session_for_student(
            course_id,
            module_id,
            _current_student_id(current_user),
        )
        if not session or session.get("status") != "completed":
            return {"session": None, "has_report": False}
        return {
            "has_report": True,
            "session": {
                "session_id": session["session_id"],
                "status": session["status"],
                "questions_asked": session.get("questions_asked", 0),
                "decision": session.get("decision", ""),
                "final_report": session.get("final_report_json") or {},
                "motivational_feedback": session.get("motivational_feedback", ""),
                "transition_feedback": session.get("transition_feedback", ""),
                "reteach_data": session.get("reteach_data_json") or {},
                "questions": session.get("questions_json") or [],
                "answers": session.get("answers_json") or [],
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Evaluation latest-full fetch failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/courses/{course_id}/modules/{module_id}/next")
async def module_next(
    course_id: str,
    module_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Return the next module in sequence. Used by the Next button."""
    await _require_owned_module(course_id, module_id, current_user)
    mod = await get_next_module(course_id, module_id)
    return {"module": mod, "has_next": mod is not None}


@router.get("/courses/{course_id}/modules/{module_id}/previous")
async def module_previous(
    course_id: str,
    module_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Return the previous module in sequence. Used by the Previous button."""
    await _require_owned_module(course_id, module_id, current_user)
    mod = await get_prev_module(course_id, module_id)
    return {"module": mod, "has_previous": mod is not None}


@router.get("/courses/{course_id}/report")
async def course_completion_report(
    course_id: str,
    student_id: str | None = Query(default=None),
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """
    Get or generate the final course performance report.

    The report summarizes mastered skills, weak skills, mentor feedback, and
    suggested next steps.
    """
    try:
        await _require_owned_course(course_id, current_user)
        report = await generate_course_report(
            course_id=course_id,
            student_id=_current_student_id(current_user),
        )
        return {"report": report}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Course report generation failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/students/{student_id}/skills/categorized")
async def student_skills_categorized(
    student_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """
    Return student skills categorized into mastered, learning, and weak groups.
    """
    current_student_id = _require_matching_student(student_id, current_user)
    skills = await get_student_skills(current_student_id)
    nodes = skills.get("nodes") or []
    by_source: dict[str, list] = {}
    for n in nodes:
        src = n.get("source", "course")
        by_source.setdefault(src, []).append(n)
    return {
        "student_id": current_student_id,
        "mastered": [n for n in nodes if n.get("status") == "mastered"],
        "learning": [n for n in nodes if n.get("status") == "learning"],
        "weak": [n for n in nodes if n.get("status") == "weak"],
        "by_source": by_source,
        "total": len(nodes),
        "mastered_count": len([n for n in nodes if n.get("status") == "mastered"]),
        "weak_count": len([n for n in nodes if n.get("status") == "weak"]),
    }


@router.get("/students/me/progress")
async def my_progress(
    student_id: str | None = Query(default=None),
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Return dashboard progress for the authenticated student."""
    return await get_student_dashboard(_current_student_id(current_user))


@router.get("/students/{student_id}/dashboard")
async def student_dashboard(
    student_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Return dashboard progress for a matching student id."""
    return await get_student_dashboard(_require_matching_student(student_id, current_user))


@router.get("/students/{student_id}/skills")
async def student_skills(
    student_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Return skill graph data for a matching student id."""
    return await get_student_skills(_require_matching_student(student_id, current_user))


@router.get("/students/{student_id}/doubts")
async def student_doubts(
    student_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Return saved doubt records for a matching student id."""
    return {
        "doubts": await get_student_doubts(
            _require_matching_student(student_id, current_user)
        )
    }


@router.get("/students/{student_id}/courses")
async def student_courses(
    student_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Return course summaries for a matching student id."""
    return {"courses": await list_courses(_require_matching_student(student_id, current_user))}


@router.get("/debug/courses/{course_id}/decision-log")
async def course_decision_log(
    course_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Return the saved agent decision log for a course."""
    await _require_owned_course(course_id, current_user)
    return {"decision_log": await get_course_decision_log(course_id)}


@router.get("/debug/session/{session_id}/trace")
async def debug_session_trace(session_id: str):
    """Point frontend debug callers to the legacy in-memory session trace route."""
    # The legacy trace endpoint owns the in-memory session store; this route
    # avoids importing that store across routers.
    return {
        "session_id": session_id,
        "message": "Use /session/trace/{session_id} for active legacy sessions.",
    }


@router.get("/stream/courses/create")
async def stream_create_course(
    topic: str,
    current_user: dict[str, Any] = Depends(require_current_user),
    student_id: str | None = Query(default=None),
    goal: str = "",
    pace: str = "medium",
    prior_knowledge: str = "",
    name: str = "Student",
    profile_json: str = "",
):
    """Stream course creation events from query-string setup parameters."""
    profile: dict[str, Any] = {}
    if profile_json:
        try:
            parsed = json.loads(profile_json)
            profile = _safe_creation_profile(parsed if isinstance(parsed, dict) else {})
        except json.JSONDecodeError:
            profile = {}
    payload = course_payload_from_request(CreateCourseRequest(
        student_id=_current_student_id(current_user),
        topic=topic,
        goal=goal,
        pace=pace,
        prior_knowledge=prior_knowledge,
        name=name,
        profile=profile,
    ))

    async def events():
        """Proxy generated course-creation events into the SSE response."""
        async for item in create_course_events(
                student_id=_current_student_id(current_user),
                topic=payload["topic"],
                goal=payload["goal"],
                pace=payload["pace"],
                prior_knowledge=payload["prior_knowledge"],
                name=name,
                personalization_profile=payload["profile"],
        ):
            yield item

    return StreamingResponse(
        _event_stream(events()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/stream/courses/create/{job_id}")
async def stream_create_course_job(
    job_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Stream course creation events for a previously registered creation job."""
    job = _course_creation_jobs.get(job_id)
    if not job or job.get("student_id") != _current_student_id(current_user):
        raise HTTPException(status_code=404, detail="course creation job not found")

    async def events():
        """Stream a registered course-creation job and clear it afterward."""
        try:
            async for item in create_course_events(
                    student_id=job["student_id"],
                    topic=job["topic"],
                    goal=job["goal"],
                    pace=job["pace"],
                    prior_knowledge=job["prior_knowledge"],
                    name=job["name"],
                    personalization_profile=_safe_creation_profile(job.get("profile")),
            ):
                yield item
        finally:
            _course_creation_jobs.pop(job_id, None)

    return StreamingResponse(
        _event_stream(events()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/stream/courses/{course_id}/create")
async def stream_existing_course_create(
    course_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Replay existing course creation state as a stream for reconnecting clients."""
    await _require_owned_course(course_id, current_user)

    async def events():
        """Replay saved course and module state for an existing course."""
        course = await get_course_for_student(course_id, _current_student_id(current_user))
        if not course:
            yield {"event": "error", "data": {"message": "Course not found"}}
            return
        yield {"event": "connected", "data": {"message": "connected"}}
        course["roadmap"] = await get_course_roadmap(course_id)
        course["roadmap_ready"] = bool(course["roadmap"])
        for module in await list_course_modules_for_student(
            course_id,
            _current_student_id(current_user),
        ):
            yield {"event": "module_planned", "data": module}
        yield {"event": "done", "data": course}

    return StreamingResponse(
        _event_stream(events()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/stream/courses/{course_id}/modules/{module_id}/generate")
async def stream_generate_module(
    course_id: str,
    module_id: str,
    student_id: str | None = Query(default=None),
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Stream module lesson generation events for one owned module."""
    await _require_owned_module(course_id, module_id, current_user)
    return StreamingResponse(
        _event_stream(
            generate_module_lesson_events(
                course_id,
                module_id,
                _current_student_id(current_user),
            )
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
