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

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from agents.course_report_agent import generate_course_report
from agents.evaluation_agent import (
    get_session_report,
    start_session as start_eval_session,
    submit_answer as submit_eval_answer,
)
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
    get_next_module,
    get_prev_module,
    get_course,
    get_course_decision_log,
    get_course_module,
    get_course_roadmap,
    get_student_dashboard,
    get_student_doubts,
    get_student_skills,
    get_user_by_student_id,
    list_course_modules,
    list_courses,
    list_module_chat_history,
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
}


def _safe_creation_profile(profile: dict[str, Any] | None) -> dict[str, Any]:
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
    return str(value or "").strip()


def _valid_pace(value: Any) -> str:
    pace = _clean_text(value).lower()
    return pace if pace in {"fast", "medium", "deep"} else "medium"


def _level_label(value: Any) -> str:
    key = _clean_text(value).lower()
    return LEVEL_LABELS.get(key, _clean_text(value))


def _time_commitment_text(
    time_commitment: dict[str, Any] | None,
    deadline: str | None = None,
) -> str:
    commitment = time_commitment if isinstance(time_commitment, dict) else {}
    value = _clean_text(commitment.get("value"))
    unit = _clean_text(commitment.get("unit")).lower()
    deadline_text = _clean_text(deadline)

    if value:
        if unit == "minutes_per_day":
            text = f"{value} minutes per day"
        elif unit == "hours_per_week":
            text = f"{value} hours per week"
        else:
            text = value
    else:
        text = ""

    if deadline_text:
        text = f"{text}, target by {deadline_text}" if text else f"Target by {deadline_text}"
    return text


def course_payload_from_request(req: "CreateCourseRequest") -> dict[str, Any]:
    """Normalize the guided setup request into course-generation inputs."""
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
    time_commitment = (
        req.time_commitment
        if isinstance(req.time_commitment, dict)
        else source.get("time_commitment")
        if isinstance(source.get("time_commitment"), dict)
        else {}
    )
    deadline = _clean_text(req.deadline or source.get("deadline"))
    time_constraint = _time_commitment_text(time_commitment, deadline) or _clean_text(source.get("time_constraint"))

    prior_parts = []
    if learner_level:
        prior_parts.append(f"Current level: {learner_level}.")
    if prior_experience:
        prior_parts.append(f"Prior experience: {prior_experience}.")
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
        "deadline": deadline,
        "prior_experience": prior_experience,
        "prior_knowledge_summary": prior_knowledge,
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
    name: str = "EduMind Student"
    email: str = "student@edumind.dev"
    avatar_url: str = ""


class CreateCourseRequest(BaseModel):
    student_id: str
    topic: str | None = None
    goal: str | None = None
    goal_description: str | None = None
    current_level: str | None = None
    prior_experience: str = ""
    time_commitment: dict[str, Any] | None = None
    deadline: str | None = None
    pace: str | None = "medium"
    prior_knowledge: str = ""
    name: str = "Student"
    profile: dict[str, Any] | None = None


class ChatRequest(BaseModel):
    student_id: str
    message: str


class EvaluateRequest(BaseModel):
    student_id: str
    answer: str
    question_id: str
    confidence: int = Field(default=3, ge=1, le=5)


def _sse(data: Any, event: str = "message") -> str:
    if not isinstance(data, str):
        data = json.dumps(data, default=str)
    lines = data.splitlines() or [""]
    payload = "".join(f"data: {line}\n" for line in lines)
    return f"event: {event}\n{payload}\n"


async def _event_stream(events):
    async for item in events:
        yield _sse(item.get("data", ""), item.get("event", "message"))


@router.post("/auth/dev-login")
async def dev_login(req: DevLoginRequest):
    profile = await upsert_dev_user(
        email=req.email,
        name=req.name,
        avatar_url=req.avatar_url,
    )
    return {"user": profile, "auth_mode": "dev"}


@router.get("/auth/me")
async def auth_me(student_id: str = Query(default="")):
    if not student_id:
        raise HTTPException(status_code=401, detail="student_id is required in dev mode")
    profile = await get_user_by_student_id(student_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user": profile, "auth_mode": "dev"}


@router.post("/auth/logout")
async def logout():
    return {"status": "ok"}


@router.get("/courses")
async def courses(student_id: str = Query(...)):
    return {"courses": await list_courses(student_id)}


@router.post("/courses")
async def create_course_endpoint(req: CreateCourseRequest):
    payload = course_payload_from_request(req)
    if not payload["topic"]:
        raise HTTPException(status_code=400, detail="topic is required")
    try:
        course = await create_course(
            student_id=req.student_id,
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
async def create_course_intent(req: CreateCourseRequest):
    payload = course_payload_from_request(req)
    if not payload["topic"]:
        raise HTTPException(status_code=400, detail="topic is required")
    job_id = str(uuid.uuid4())
    _course_creation_jobs[job_id] = {
        "student_id": req.student_id,
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
async def course_detail(course_id: str, student_id: str | None = Query(default=None)):
    course = await get_course(course_id, student_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    course["modules"] = await list_course_modules(course_id)
    course["roadmap"] = await get_course_roadmap(course_id)
    course["roadmap_ready"] = bool(course["roadmap"])
    return {"course": course}


@router.get("/courses/{course_id}/roadmap")
async def course_roadmap(course_id: str, student_id: str | None = Query(default=None)):
    course = await get_course(course_id, student_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    roadmap = await get_course_roadmap(course_id)
    if not roadmap:
        raise HTTPException(status_code=404, detail="Roadmap not found")
    return {"course": course, "roadmap": roadmap}


@router.post("/courses/{course_id}/roadmap/regenerate")
async def regenerate_course_roadmap(course_id: str, req: Request):
    from core.roadmap_service import CourseRoadmapService
    from db.postgres import save_course_roadmap

    body = await req.json() if req.headers.get("content-length") else {}
    student_id = body.get("student_id")
    course = await get_course(course_id, student_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    modules = await list_course_modules(course_id)
    history = await get_student_history_snapshot(course["student_id"])
    profile = course.get("personalization_profile") or {}
    roadmap = CourseRoadmapService().build(course, modules, profile, history)
    roadmap = await save_course_roadmap(course_id, roadmap)
    return {"course": course, "roadmap": roadmap}


@router.get("/courses/{course_id}/modules")
async def course_modules(course_id: str):
    return {"modules": await list_course_modules(course_id)}


@router.get("/courses/{course_id}/modules/{module_id}")
async def course_module(
    course_id: str,
    module_id: str,
    student_id: str | None = Query(default=None),
    auto_generate: bool = Query(default=False),
):
    if auto_generate:
        module = await generate_module_lesson(course_id, module_id, student_id)
    else:
        module = await get_course_module(course_id, module_id)
        if not module:
            raise HTTPException(status_code=404, detail="Module not found")
        course = await get_course(course_id, student_id)
        if course:
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
async def generate_module(course_id: str, module_id: str, req: Request):
    body = await req.json() if req.headers.get("content-length") else {}
    student_id = body.get("student_id")
    module = await generate_module_lesson(course_id, module_id, student_id)
    return {"module": module}


@router.post("/courses/{course_id}/modules/{module_id}/complete")
async def complete_module_endpoint(course_id: str, module_id: str, req: Request):
    body = await req.json() if req.headers.get("content-length") else {}
    try:
        module = await complete_module(course_id, module_id, body.get("student_id"))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"module": module}


@router.post("/courses/{course_id}/modules/{module_id}/chat")
async def module_chat(course_id: str, module_id: str, req: ChatRequest):
    try:
        return await answer_module_chat(
            course_id=course_id,
            module_id=module_id,
            student_id=req.student_id,
            message=req.message,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/courses/{course_id}/modules/{module_id}/chat-history")
async def module_chat_history(course_id: str, module_id: str):
    return {"messages": await list_module_chat_history(course_id, module_id)}


@router.get("/courses/{course_id}/modules/{module_id}/questions")
async def module_questions(course_id: str, module_id: str, student_id: str | None = Query(default=None)):
    course = await get_course(course_id, student_id)
    module = await get_course_module(course_id, module_id)
    if not course or not module:
        raise HTTPException(status_code=404, detail="Course or module not found")
    return {"questions": await get_or_create_module_questions(course, module)}


@router.post("/courses/{course_id}/modules/{module_id}/evaluate")
async def module_evaluate(course_id: str, module_id: str, req: EvaluateRequest):
    try:
        return await evaluate_module_answer(
            course_id=course_id,
            module_id=module_id,
            student_id=req.student_id,
            question_id=req.question_id,
            answer=req.answer,
            confidence=req.confidence,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


class StartEvaluationRequest(BaseModel):
    student_id: str | None = None


class SubmitAnswerRequest(BaseModel):
    question_id: str
    answer_text: str
    confidence: int = 3  # 1-5


@router.post("/courses/{course_id}/modules/{module_id}/evaluation/start")
async def evaluation_start(
    course_id: str,
    module_id: str,
    req: StartEvaluationRequest,
):
    """
    Start evaluation after the student clicks Next/Complete.
    Returns session_id and the first batch of questions.

    If the lesson has not been generated yet (no content_markdown), this endpoint
    will auto-generate the lesson first, then start the evaluation.
    This fixes the "Take a Quiz" button doing nothing when the module was never opened.
    """
    student_id = req.student_id or ""
    try:
        # Auto-generate lesson if content is missing — this is the root cause of the
        # "Take a Quiz" button doing nothing: evaluation/start throws ValueError when
        # content_markdown is empty, and the frontend silently swallows it.
        from db.postgres import get_course_module
        module = await get_course_module(course_id, module_id)
        if module and not module.get("content_markdown"):
            logger.info(
                "evaluation/start: lesson not yet generated for module '{}' — auto-generating before eval.",
                module_id,
            )
            try:
                course = await get_course(course_id, student_id)
                if course:
                    await generate_module_lesson(
                        course_id=course_id,
                        module_id=module_id,
                        student_id=student_id,
                    )
            except Exception as gen_exc:
                logger.warning("Auto-lesson generation before eval failed: {}", gen_exc)
                # Don't abort — try to start eval anyway; it may have partial content

        result = await start_eval_session(
            course_id=course_id,
            module_id=module_id,
            student_id=student_id,
        )
        return result
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
):
    """
    Submit one answer. Returns diagnosis and either the next question or the final report.
    If session_complete=True, the response contains the full evaluation report.
    """
    try:
        result = await submit_eval_answer(
            session_id=session_id,
            question_id=req.question_id,
            answer_text=req.answer_text,
            confidence=max(1, min(5, int(req.confidence or 3))),
        )
        return result
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
):
    """
    Get the completed evaluation report for a session.
    """
    try:
        result = await get_session_report(session_id)
        return result
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
):
    """
    Get the most recent evaluation session for this module (if any).
    Returns null if no evaluation has been done yet.
    """
    from db.postgres import get_latest_evaluation_session
    try:
        session = await get_latest_evaluation_session(course_id, module_id)
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
    except Exception as exc:
        logger.exception("Evaluation latest fetch failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/courses/{course_id}/modules/{module_id}/next")
async def module_next(course_id: str, module_id: str):
    """Return the next module in sequence. Used by the Next button."""
    mod = await get_next_module(course_id, module_id)
    return {"module": mod, "has_next": mod is not None}


@router.get("/courses/{course_id}/modules/{module_id}/previous")
async def module_previous(course_id: str, module_id: str):
    """Return the previous module in sequence. Used by the Previous button."""
    mod = await get_prev_module(course_id, module_id)
    return {"module": mod, "has_previous": mod is not None}


@router.get("/courses/{course_id}/report")
async def course_completion_report(
    course_id: str,
    student_id: str = Query(...),
):
    """
    Get (or generate) the final course performance report.
    Shows mastered skills, weak skills, mentor feedback, and next steps.
    Call this when the course is fully completed.
    """
    try:
        report = await generate_course_report(course_id=course_id, student_id=student_id)
        return {"report": report}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Course report generation failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/students/{student_id}/skills/categorized")
async def student_skills_categorized(student_id: str):
    """
    Return student skills categorized into mastered / learning / weak.
    Used for the My Skills tab.
    """
    skills = await get_student_skills(student_id)
    nodes = skills.get("nodes") or []
    by_source: dict[str, list] = {}
    for n in nodes:
        src = n.get("source", "course")
        by_source.setdefault(src, []).append(n)
    return {
        "student_id": student_id,
        "mastered": [n for n in nodes if n.get("status") == "mastered"],
        "learning": [n for n in nodes if n.get("status") == "learning"],
        "weak": [n for n in nodes if n.get("status") == "weak"],
        "by_source": by_source,
        "total": len(nodes),
        "mastered_count": len([n for n in nodes if n.get("status") == "mastered"]),
        "weak_count": len([n for n in nodes if n.get("status") == "weak"]),
    }


@router.get("/students/me/progress")
async def my_progress(student_id: str = Query(...)):
    return await get_student_dashboard(student_id)


@router.get("/students/{student_id}/dashboard")
async def student_dashboard(student_id: str):
    return await get_student_dashboard(student_id)


@router.get("/students/{student_id}/skills")
async def student_skills(student_id: str):
    return await get_student_skills(student_id)


@router.get("/students/{student_id}/doubts")
async def student_doubts(student_id: str):
    return {"doubts": await get_student_doubts(student_id)}


@router.get("/students/{student_id}/courses")
async def student_courses(student_id: str):
    return {"courses": await list_courses(student_id)}


@router.get("/debug/courses/{course_id}/decision-log")
async def course_decision_log(course_id: str):
    return {"decision_log": await get_course_decision_log(course_id)}


@router.get("/debug/session/{session_id}/trace")
async def debug_session_trace(session_id: str):
    # The legacy trace endpoint lives at /session/trace/{session_id}. This
    # frontend route documents the debug surface but avoids importing the
    # in-memory session store across modules.
    return {
        "session_id": session_id,
        "message": "Use /session/trace/{session_id} for active legacy sessions.",
    }


@router.get("/stream/courses/create")
async def stream_create_course(
    student_id: str,
    topic: str,
    goal: str = "",
    pace: str = "medium",
    prior_knowledge: str = "",
    name: str = "Student",
    profile_json: str = "",
):
    profile: dict[str, Any] = {}
    if profile_json:
        try:
            parsed = json.loads(profile_json)
            profile = _safe_creation_profile(parsed if isinstance(parsed, dict) else {})
        except json.JSONDecodeError:
            profile = {}
    payload = course_payload_from_request(CreateCourseRequest(
        student_id=student_id,
        topic=topic,
        goal=goal,
        pace=pace,
        prior_knowledge=prior_knowledge,
        name=name,
        profile=profile,
    ))

    async def events():
        async for item in create_course_events(
                student_id=student_id,
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
async def stream_create_course_job(job_id: str):
    job = _course_creation_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="course creation job not found")

    async def events():
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
async def stream_existing_course_create(course_id: str):
    async def events():
        course = await get_course(course_id)
        if not course:
            yield {"event": "error", "data": {"message": "Course not found"}}
            return
        yield {"event": "connected", "data": {"message": "connected"}}
        course["roadmap"] = await get_course_roadmap(course_id)
        course["roadmap_ready"] = bool(course["roadmap"])
        for module in await list_course_modules(course_id):
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
):
    return StreamingResponse(
        _event_stream(generate_module_lesson_events(course_id, module_id, student_id)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
