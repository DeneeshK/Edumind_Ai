"""
core/metrics.py
───────────────
EduMind Prometheus metrics — single source of truth.

Import `metrics` anywhere you need to record something.
The /metrics HTTP endpoint is mounted in app/api.py.

Usage:
    from core.metrics import metrics

    metrics.http_requests.labels(method="POST", endpoint="/api/courses", status=200).inc()
    with metrics.llm_latency.labels(model="llama-3.3-70b").time():
        raw = await generate(...)
"""

from __future__ import annotations

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    CollectorRegistry,
    CONTENT_TYPE_LATEST,
    generate_latest,
)

# Use the default registry so it also picks up process/runtime metrics
# (memory, CPU, open file descriptors) automatically via prometheus_client internals.

# ── Buckets ───────────────────────────────────────────────────────────────────
# LLM calls can take 2–120 s; HTTP requests should be under 2 s normally.
_LLM_BUCKETS  = (0.5, 1, 2, 5, 10, 20, 40, 60, 90, 120)
_HTTP_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10)
_EVAL_BUCKETS = (0.1, 0.5, 1, 2, 5, 10, 30)


class EduMindMetrics:
    """All application metrics in one place."""

    def __init__(self) -> None:

        # ── 1. HTTP layer ─────────────────────────────────────────────────────
        self.http_requests = Counter(
            "edumind_http_requests_total",
            "Total HTTP requests by method, endpoint, and status code.",
            ["method", "endpoint", "status"],
        )
        self.http_latency = Histogram(
            "edumind_http_request_duration_seconds",
            "HTTP request latency in seconds.",
            ["method", "endpoint"],
            buckets=_HTTP_BUCKETS,
        )

        # ── 2. LLM calls (Groq) ───────────────────────────────────────────────
        self.llm_requests = Counter(
            "edumind_llm_requests_total",
            "Total LLM calls by model and caller (which agent/function).",
            ["model", "caller"],
        )
        self.llm_errors = Counter(
            "edumind_llm_errors_total",
            "LLM call failures by model and error type (timeout, rate_limit, parse_fail).",
            ["model", "error_type"],
        )
        self.llm_latency = Histogram(
            "edumind_llm_duration_seconds",
            "LLM call latency in seconds by model.",
            ["model"],
            buckets=_LLM_BUCKETS,
        )
        self.llm_retries = Counter(
            "edumind_llm_retries_total",
            "Number of LLM retry attempts (429 rate-limit backoffs).",
            ["model"],
        )

        # ── 3. Course & curriculum ────────────────────────────────────────────
        self.courses_created = Counter(
            "edumind_courses_created_total",
            "Total courses successfully created.",
            ["topic_category"],   # e.g. programming, science, math, other
        )
        self.course_creation_failures = Counter(
            "edumind_course_creation_failures_total",
            "Course creation failures by reason (validation, llm_fail, db_error).",
            ["reason"],
        )
        self.roadmap_fallback_used = Counter(
            "edumind_roadmap_fallback_total",
            "Times the hardcoded fallback skeleton was used instead of LLM-generated roadmap.",
            ["topic"],
        )
        self.curriculum_validation_failures = Counter(
            "edumind_curriculum_validation_failures_total",
            "Curriculum validation failures by issue type.",
            ["issue_type"],   # module_count, delayed_concept, missing_concept, etc.
        )

        # ── 4. Lesson generation ──────────────────────────────────────────────
        self.lessons_generated = Counter(
            "edumind_lessons_generated_total",
            "Total lesson content generated successfully.",
            ["pace"],   # fast, medium, deep
        )
        self.lesson_generation_errors = Counter(
            "edumind_lesson_generation_errors_total",
            "Lesson generation failures.",
            ["pace", "error_type"],
        )
        self.lesson_latency = Histogram(
            "edumind_lesson_generation_duration_seconds",
            "End-to-end lesson generation latency in seconds.",
            ["pace"],
            buckets=_LLM_BUCKETS,
        )

        # ── 5. Evaluation / quiz ──────────────────────────────────────────────
        self.eval_sessions_started = Counter(
            "edumind_eval_sessions_started_total",
            "Evaluation sessions started.",
            ["pace"],
        )
        self.eval_sessions_completed = Counter(
            "edumind_eval_sessions_completed_total",
            "Evaluation sessions completed (student answered all questions).",
            ["pace", "decision"],   # decision: ADVANCE, REPEAT_MODULE, etc.
        )
        self.eval_questions_asked = Counter(
            "edumind_eval_questions_asked_total",
            "Total questions asked across all sessions.",
            ["pace", "question_type"],   # recall, conceptual, probe, bonus
        )
        self.eval_probe_triggered = Counter(
            "edumind_eval_probe_triggered_total",
            "Times a chained probe question was triggered (weakness detected).",
            ["pace"],
        )
        self.eval_latency = Histogram(
            "edumind_eval_answer_duration_seconds",
            "Time to diagnose a student answer and generate next question.",
            ["pace"],
            buckets=_EVAL_BUCKETS,
        )
        self.mastery_score = Histogram(
            "edumind_eval_mastery_score",
            "Distribution of mastery scores at end of evaluation session.",
            ["pace"],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0),
        )

        # ── 6. External API health ────────────────────────────────────────────
        self.tavily_requests = Counter(
            "edumind_tavily_requests_total",
            "Total Tavily search requests.",
            ["cache_hit"],   # "true" or "false"
        )
        self.tavily_errors = Counter(
            "edumind_tavily_errors_total",
            "Tavily search failures.",
        )

        # ── 7. Database ───────────────────────────────────────────────────────
        self.db_query_latency = Histogram(
            "edumind_db_query_duration_seconds",
            "PostgreSQL query latency in seconds by operation.",
            ["operation"],   # e.g. save_course, get_module, save_eval_session
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1),
        )
        self.db_errors = Counter(
            "edumind_db_errors_total",
            "Database operation failures by operation.",
            ["operation"],
        )

        # ── 8. Active sessions (gauges — current state, not cumulative) ───────
        self.active_students = Gauge(
            "edumind_active_students",
            "Number of students with an active session right now.",
        )
        self.active_course_creations = Gauge(
            "edumind_active_course_creations",
            "Number of course creation jobs currently in flight.",
        )


# Module-level singleton — import this everywhere
metrics = EduMindMetrics()


def prometheus_response() -> tuple[bytes, str]:
    """Return (body_bytes, content_type) for the /metrics endpoint."""
    return generate_latest(), CONTENT_TYPE_LATEST
