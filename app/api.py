"""
app/api.py
EduMind FastAPI server.

Replaces the CLI input() loop with HTTP endpoints + SSE streaming.

Endpoints:
  POST /session/start          — start new student or load returning student
  POST /session/chat           — send a message, receive SSE stream of response
  POST /session/answer         — submit answer to evaluator question
  POST /session/confidence     — submit confidence rating (1-5)
  GET  /session/status         — get current session state
  POST /session/end            — end session, flush to DB
  GET  /student/{id}/progress  — get mastery + metacognition summary

Session state is kept in memory (dict keyed by session_id) during a session.
Persistent state lives in PostgreSQL (loaded/saved by StudentState).

Run with:
  uvicorn app.api:app --reload --port 8000
"""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from fastapi import Depends, FastAPI, HTTPException, BackgroundTasks, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

from db.postgres import init_db, close_db
from core.metrics import prometheus_response
from core.metrics_middleware import PrometheusMiddleware
from core.student_model import StudentState, CurriculumPlan
from config import settings


# ── In-memory session store ───────────────────────────────────────────────────
# Maps session_id -> SessionContext (holds state + queues for I/O)

class SessionContext:
    """
    Holds everything needed to run an async session.

    The session runs as a background asyncio Task.
    Communication between the task and HTTP requests goes through asyncio Queues:
      - output_queue: task puts strings/events here → SSE sends them to client
      - input_queue:  client posts answers here → task reads them to continue
    """
    def __init__(self, state: StudentState):
        self.state = state
        self.output_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=512)
        self.input_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=64)
        self.task: asyncio.Task | None = None
        self.ended: bool = False
        self.current_question: str | None = None
        self.awaiting_confidence: bool = False
        self.awaiting_answer: bool = False
        self.awaiting_input_type: str | None = None


_sessions: dict[str, SessionContext] = {}

# Per-student lock — prevents two concurrent sessions corrupting the same state
_student_locks: dict[str, asyncio.Lock] = {}


# ── Auth ──────────────────────────────────────────────────────────────────────


async def verify_api_key(x_api_key: str = Header(default="")) -> None:
    """
    Basic API key check. Set EDUMIND_API_KEY in .env to enable.
    If not set, auth is disabled (dev mode).
    """
    expected = getattr(settings, "edumind_api_key", None)
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


# ── Session cleanup ───────────────────────────────────────────────────────────

async def _cleanup_dead_sessions() -> None:
    """Remove finished sessions from memory every 5 minutes to prevent leaks."""
    while True:
        await asyncio.sleep(300)
        dead = [
            sid for sid, ctx in list(_sessions.items())
            if ctx.task and ctx.task.done()
        ]
        for sid in dead:
            _sessions.pop(sid, None)
            logger.debug("Cleaned up dead session: {}", sid)
        if dead:
            logger.info("Session cleanup: removed {} dead sessions", len(dead))


# ── Rate limiter ──────────────────────────────────────────────────────────────
# Per-IP rate limiting via slowapi. Prevents a single client from exhausting
# the Groq API quota for all users.
limiter = Limiter(key_func=get_remote_address)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared services at startup and flush in-flight sessions at shutdown."""
    await init_db()
    if settings.eval_enabled:
        from evaluation.scheduler import start_scheduler

        start_scheduler()
    asyncio.create_task(_cleanup_dead_sessions())
    logger.info("EduMind API ready.")
    yield
    # ── Graceful shutdown: save in-flight sessions ────────────────────────────
    logger.info("Shutting down — saving {} active sessions…", len(_sessions))
    shutdown_tasks = []
    for sid, ctx in list(_sessions.items()):
        if ctx.task and not ctx.task.done():
            ctx.task.cancel()
            shutdown_tasks.append(ctx.task)
    # Wait briefly for tasks to cancel
    if shutdown_tasks:
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
    # Flush state for any sessions that completed initial setup.
    for sid, ctx in list(_sessions.items()):
        if ctx.state.domain:
            try:
                await ctx.state.save()
                logger.info("Saved in-flight session student={}", ctx.state.student_id)
            except Exception as e:
                logger.error("Failed to save session on shutdown: {}", e)
    if settings.eval_enabled:
        from evaluation.scheduler import stop_scheduler

        stop_scheduler()
    await close_db()
    logger.info("EduMind API shut down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="EduMind API", version="1.0.0", lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(PrometheusMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from evaluation.api_router import router as eval_router
from app.auth import router as auth_router
from app.course_api import router as course_router

app.include_router(eval_router)
app.include_router(auth_router)
app.include_router(course_router)


# ── Request / Response models ─────────────────────────────────────────────────

class StartRequest(BaseModel):
    """Legacy session-start request for creating or resuming a student session."""

    student_id: str | None = None   # None = new student
    name: str = "Student"
    domain: str = ""
    goal: str = ""
    topic: str = ""
    pace: str = "medium"            # fast | medium | deep


class StartResponse(BaseModel):
    """Response returned after a legacy session is created."""

    session_id: str
    student_id: str
    is_new: bool
    message: str


class ChatRequest(BaseModel):
    """Free-form message request for an active legacy session."""

    session_id: str
    message: str


class AnswerRequest(BaseModel):
    """Evaluator answer request for an active legacy session."""

    session_id: str
    answer: str


class ConfidenceRequest(BaseModel):
    """Confidence-rating request for an active legacy session."""

    session_id: str
    confidence: int                 # 1-5


class ModuleStatus(BaseModel):
    """Frontend-visible module status for a legacy session."""

    id: str
    index: int
    title: str
    concept: str
    domain_framing: str
    prerequisites: list[str]
    estimated_minutes: int
    depth_level: str
    status: str                     # completed | current | locked
    unlocked: bool
    content: str = ""


class SessionStatus(BaseModel):
    """Current state snapshot for an active legacy session."""

    session_id: str
    student_id: str
    domain: str
    pace: str
    current_module: str | None
    modules_done: int
    total_modules: int
    ended: bool
    awaiting_answer: bool
    awaiting_confidence: bool
    awaiting_input_type: str | None
    current_question: str | None
    modules: list[ModuleStatus] = Field(default_factory=list)


# ── SSE helpers ───────────────────────────────────────────────────────────────

def _sse_event(data: str, event: str = "message") -> str:
    """Format a Server-Sent Event."""
    lines = str(data).splitlines() or [""]
    payload = "".join(f"data: {line}\n" for line in lines)
    return f"event: {event}\n{payload}\n"


async def _sse_stream(ctx: SessionContext) -> AsyncGenerator[str, None]:
    """
    Drain output_queue and yield SSE events.

    Items can be:
      - Plain str           → event: message
      - "\x00{type}\x00{data}" → event: {type}  (used by ask() for question/confidence_request)
      - None                → event: done
    """
    try:
        yield _sse_event("connected", event="ready")
        while True:
            try:
                item = await asyncio.wait_for(ctx.output_queue.get(), timeout=45)
            except asyncio.TimeoutError:
                if ctx.ended or (ctx.task and ctx.task.done()):
                    yield _sse_event("", event="done")
                    break
                yield _sse_event("keepalive", event="ping")
                continue

            if item is None:
                yield _sse_event("", event="done")
                break

            # Typed event from ask()
            if isinstance(item, str) and item.startswith("\x00"):
                parts = item.split("\x00", 2)
                event_type = parts[1]
                data = parts[2] if len(parts) > 2 else ""
                yield _sse_event(data, event=event_type)
            else:
                yield _sse_event(item)

    except GeneratorExit:
        logger.warning("SSE client disconnected — session task kept alive")


# ── Session runner (runs as background task) ──────────────────────────────────

async def _run_session(
    ctx: SessionContext,
    is_new: bool,
    topic: str,
    student_lock: asyncio.Lock | None = None,
) -> None:
    """
    Replaces the CLI session loop. All output goes to ctx.output_queue.
    All input (student answers) comes from ctx.input_queue.
    The student_lock is held for the duration of the session to prevent
    concurrent sessions for the same student corrupting shared state.
    """
    from agents.orchestrator import OrchestratorAgent

    # Acquire per-student lock for entire session duration
    if student_lock:
        await student_lock.acquire()

    async def emit(text: str) -> None:
        """Send text to the SSE stream. Uses await put() so no messages are dropped."""
        try:
            await asyncio.wait_for(
                ctx.output_queue.put(text),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("output_queue blocked for 10s — client may be disconnected for student={}", ctx.state.student_id)

    async def ask(question: str, is_confidence: bool = False) -> str:
        """
        Send a question to the client via a typed SSE event and wait for answer.

        Three distinct input types:
          - confidence_request : student rates 1-5 via /session/confidence
          - doubt_request      : optional free-text; times out after 20 s
                                 awaiting_answer stays False so /session/answer
                                 won't accept a premature answer here
          - question           : required eval answer via /session/answer
        """
        ctx.current_question = question
        ctx.awaiting_confidence = is_confidence
        is_doubt = (not is_confidence) and (
            "Any questions" in question or question.lstrip().startswith("❓")
        )
        # Only set awaiting_answer=True for required eval questions.
        # Doubt prompts are optional — keeping it False prevents a client
        # submitting an answer to a doubt prompt from consuming the slot
        # intended for the next real evaluation question.
        ctx.awaiting_answer = (not is_confidence) and (not is_doubt)
        ctx.awaiting_input_type = (
            "confidence" if is_confidence else "doubt" if is_doubt else "answer"
        )

        event_type = (
            "confidence_request" if is_confidence
            else "doubt_request" if is_doubt
            else "question"
        )
        # Emit as a typed SSE event so the frontend sees event: question / confidence_request
        await ctx.output_queue.put(f"\x00{event_type}\x00{question}")
        # Compatibility fallback for simple HTML clients that only use
        # EventSource.onmessage (no custom event listeners).
        await ctx.output_queue.put(f"\x00message\x00{question}")

        # Optional doubt prompts: short timeout, empty string on expiry.
        if is_doubt:
            try:
                answer = await asyncio.wait_for(ctx.input_queue.get(), timeout=20)
            except asyncio.TimeoutError:
                await ctx.output_queue.put("No doubt received, continuing.")
                answer = ""
        else:
            # Required inputs (confidence rating, eval answers): block until received.
            answer = await ctx.input_queue.get()

        ctx.awaiting_answer = False
        ctx.awaiting_confidence = False
        ctx.awaiting_input_type = None
        ctx.current_question = None
        return answer

    # Inject the async ask/emit into the orchestrator via the session context
    # so agents can output and receive input without any CLI dependency
    ctx.state.start_session()

    orchestrator = None
    try:
        orchestrator = OrchestratorAgent(ctx.state, emit_fn=emit, ask_fn=ask)
        orchestrator._ensure_eval_runner(topic)

        if not is_new and ctx.state.curriculum:
            await orchestrator._check_cross_session_doubts()

        if is_new or not ctx.state.curriculum:
            await orchestrator._run_initial_setup_api(topic)
        else:
            await emit(
                f"👋 Welcome back, {ctx.state.name}! "
                f"Resuming: {ctx.state.curriculum.topic} "
                f"(module {ctx.state.curriculum.current_index + 1}"
                f"/{len(ctx.state.curriculum.modules)})"
            )

        completed = await orchestrator._run_module_loop()

        if (ctx.state.curriculum and
                ctx.state.curriculum.current_index >= len(ctx.state.curriculum.modules)):
            await emit("🎉 Curriculum complete! You've mastered all modules!")

        await orchestrator._end_session(completed)

    except Exception as exc:
        logger.error("Session error for student={}: {}", ctx.state.student_id, exc)
        await emit(f"⚠️ Session error: {exc}")
    finally:
        if orchestrator is not None:
            orchestrator._clear_eval_runner()
        # Release per-student lock so the student can start a new session
        if student_lock and student_lock.locked():
            student_lock.release()
        ctx.ended = True
        await ctx.output_queue.put(None)   # signal SSE stream to close


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    """Prometheus scrape endpoint. Scraped by prometheus every 15s."""
    from fastapi.responses import Response
    body, content_type = prometheus_response()
    return Response(content=body, media_type=content_type)


@app.get("/health")
async def health():
    """Lightweight liveness check for Docker, CI, and load balancers."""
    return {"status": "ok"}


@app.get("/ready")
async def readiness():
    """
    Readiness check for dependencies required to serve stateful requests.
    This intentionally stays separate from /health so liveness does not depend
    on Postgres, Groq, Tavily, or other external services.
    """
    try:
        from db.postgres import get_pool
        pool = await get_pool()
        await pool.fetchval("SELECT 1")
        return {"status": "ok", "db": "ok", "sessions": len(_sessions)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {e}")


@app.post("/session/start", response_model=StartResponse,
          dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def start_session(request: Request, req: StartRequest, background_tasks: BackgroundTasks):
    """
    Start a new session.
    - If student_id is None: create new student
    - If student_id is provided: load returning student
    Raises 409 if the student already has an active session.
    """
    is_new = req.student_id is None

    # Determine student_id early so we can check for active sessions
    student_id_check = req.student_id or "new"

    # Per-student lock — prevent two concurrent sessions for same student
    if not is_new:
        lock = _student_locks.setdefault(req.student_id, asyncio.Lock())
        if lock.locked():
            raise HTTPException(
                status_code=409,
                detail=f"Student '{req.student_id}' already has an active session. "
                       f"End it first via POST /session/end/{{session_id}}"
            )

    if is_new:
        # Validate required fields for new student
        if not req.domain or not req.goal or not req.topic:
            raise HTTPException(
                status_code=400,
                detail="New students must provide: domain, goal, topic"
            )

        student_id = str(uuid.uuid4())
        state = StudentState(
            student_id=student_id,
            name=req.name,
            domain=req.domain,
            goal=req.goal,
            pace=req.pace if req.pace in ("fast", "medium", "deep") else "medium",
        )
        # Save identity to DB immediately
        from db.postgres import upsert_student
        await upsert_student(student_id, req.name, req.domain, req.goal, state.pace)
        message = f"Welcome {req.name}! Starting your course on '{req.topic}'."

    else:
        # Load returning student
        try:
            state = await StudentState.load(req.student_id)
        except ValueError:
            raise HTTPException(
                status_code=404,
                detail=f"Student '{req.student_id}' not found. Use student_id=null for new students."
            )
        message = f"Welcome back, {state.name}!"

    # Create session context
    ctx = SessionContext(state)
    session_id = str(uuid.uuid4())
    _sessions[session_id] = ctx

    # Start session as background task
    ctx.task = asyncio.create_task(
        _run_session(ctx, is_new=is_new, topic=req.topic,
                     student_lock=_student_locks.get(state.student_id))
    )

    return StartResponse(
        session_id=session_id,
        student_id=state.student_id,
        is_new=is_new,
        message=message,
    )


@app.get("/session/stream/{session_id}")
async def stream_session(
    session_id: str,
    request: Request,
    x_api_key: str = "",   # query param fallback for EventSource (can't send headers)
):
    """
    SSE stream for a session. Connect to this after /session/start.
    Pass API key as query param ?x-api-key=... since EventSource doesn't support headers.
    """
    # Auth check — accept from header OR query param
    expected = getattr(settings, "edumind_api_key", None)
    if expected:
        header_key = request.headers.get("x-api-key", "")
        query_key = (
            x_api_key
            or request.query_params.get("x-api-key", "")
            or request.query_params.get("x_api_key", "")
        )
        if header_key != expected and query_key != expected:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Invalid API key"}, status_code=401)
    ctx = _sessions.get(session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="Session not found")

    return StreamingResponse(
        _sse_stream(ctx),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )


@app.post("/session/answer")
@limiter.limit("60/minute")
async def submit_answer(request: Request, req: AnswerRequest):
    """
    Submit the student's answer to an evaluator question.
    Call this after receiving a 'question' event from the SSE stream.
    """
    ctx = _sessions.get(req.session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="Session not found")
    if not ctx.awaiting_answer:
        raise HTTPException(
            status_code=400,
            detail="Session is not currently awaiting an answer"
        )
    await ctx.input_queue.put(req.answer)
    return {"status": "ok", "message": "Answer received"}


@app.post("/session/confidence")
@limiter.limit("60/minute")
async def submit_confidence(request: Request, req: ConfidenceRequest):
    """
    Submit the student's confidence rating (1-5).
    Call this after receiving a 'confidence_request' event from the SSE stream.
    """
    ctx = _sessions.get(req.session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="Session not found")
    if not ctx.awaiting_confidence:
        raise HTTPException(
            status_code=400,
            detail="Session is not currently awaiting a confidence rating"
        )

    confidence = max(1, min(5, req.confidence))
    await ctx.input_queue.put(str(confidence))
    return {"status": "ok", "confidence": confidence}


@app.post("/session/chat")
@limiter.limit("30/minute")
async def chat(request: Request, req: ChatRequest):
    """
    Send a free-form message during a session (for doubts, comments).
    This goes into the input queue and the tutor/orchestrator will handle it.
    """
    ctx = _sessions.get(req.session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="Session not found")
    if ctx.ended:
        raise HTTPException(status_code=400, detail="Session has ended")

    if ctx.awaiting_confidence:
        raise HTTPException(
            status_code=400,
            detail="Session is awaiting a confidence rating; use /session/confidence"
        )
    # Accept input when awaiting an answer OR when awaiting a doubt response
    if ctx.awaiting_answer or ctx.awaiting_input_type == "doubt":
        await ctx.input_queue.put(req.message)
        return {"status": "ok"}

    await ctx.output_queue.put(
        "💬 Message received. I will ask for input when the tutor is ready."
    )
    return {"status": "ok", "message": "Message noted; session was not awaiting input"}


@app.post("/session/end/{session_id}")
async def end_session(session_id: str):
    """
    Manually end a session (e.g. student closes the browser).
    Cancels the background task and cleans up.
    """
    ctx = _sessions.get(session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="Session not found")

    if ctx.task and not ctx.task.done():
        ctx.task.cancel()
    ctx.ended = True
    await ctx.output_queue.put(None)
    _sessions.pop(session_id, None)
    return {"status": "ok", "message": "Session ended"}


@app.get("/session/status/{session_id}", response_model=SessionStatus)
async def get_status(session_id: str):
    """Return current in-memory session state for legacy live sessions."""
    ctx = _sessions.get(session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="Session not found")

    state = ctx.state
    curriculum = state.curriculum
    current_module = None
    total = 0
    modules = []
    if curriculum:
        total = len(curriculum.modules)
        if curriculum.current_index < total:
            current_module = curriculum.modules[curriculum.current_index].concept
        for idx, module in enumerate(curriculum.modules):
            status = (
                "completed" if idx < curriculum.current_index
                else "current" if idx == curriculum.current_index
                else "locked"
            )
            modules.append(ModuleStatus(
                id=module.id,
                index=idx,
                title=module.title,
                concept=module.concept,
                domain_framing=module.domain_framing,
                prerequisites=module.prerequisites,
                estimated_minutes=module.estimated_minutes,
                depth_level=module.depth_level,
                status=status,
                unlocked=idx <= curriculum.current_index,
                content=(
                    state.get_module_content(module.id)
                    if idx <= curriculum.current_index else ""
                ),
            ))

    return SessionStatus(
        session_id=session_id,
        student_id=state.student_id,
        domain=state.domain,
        pace=state.pace,
        current_module=current_module,
        modules_done=curriculum.current_index if curriculum else 0,
        total_modules=total,
        ended=ctx.ended,
        awaiting_answer=ctx.awaiting_answer,
        awaiting_confidence=ctx.awaiting_confidence,
        awaiting_input_type=ctx.awaiting_input_type,
        current_question=ctx.current_question,
        modules=modules,
    )


@app.get("/student/{student_id}/progress", dependencies=[Depends(verify_api_key)])
async def get_progress(student_id: str):
    """Get a student's full mastery history and metacognition profile."""
    try:
        state = await StudentState.load(student_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Student '{student_id}' not found")

    return {
        "student_id": student_id,
        "name": state.name,
        "domain": state.domain,
        "pace": state.pace,
        "concept_mastery": state.concept_mastery,
        "concept_depth": state.concept_depth,
        "preferred_style": state.metacognition.preferred_style,
        "calibration_pattern": state.metacognition.calibration_pattern,
        "weak_concept_types": state.metacognition.weak_concept_types,
        "total_concepts_seen": len(state.concept_mastery),
        "advance_threshold": state.advance_threshold,
    }


@app.get("/session/trace/{session_id}", dependencies=[Depends(verify_api_key)])
async def get_session_trace(session_id: str):
    """
    Return the agent execution trace for a session.
    Shows every agent run: which tools were called, in what order,
    and how long each took. Useful for debugging agent decisions.
    """
    ctx = _sessions.get(session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="Session not found or already cleaned up")

    # Collect all decisions logged to state (each agent logs its own)
    decisions = [
        {
            "action": d.action,
            "reason": d.reason,
            "payload": d.metacognition_updates,
        }
        for d in ctx.state.session_decisions
    ]

    return {
        "session_id": session_id,
        "student_id": ctx.state.student_id,
        "ended": ctx.ended,
        "decisions": decisions,
        "modules_done": ctx.state.curriculum.current_index if ctx.state.curriculum else 0,
        "total_modules": len(ctx.state.curriculum.modules) if ctx.state.curriculum else 0,
    }
