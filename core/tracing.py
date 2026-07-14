"""
core/tracing.py
───────────────
OpenTelemetry setup for EduMind — request-scoped distributed tracing.

Design goals (see docs/ARCHITECTURE.md → Observability):
  • OPT-IN. When settings.otel_enabled is False (the default), no exporter and no
    TracerProvider are installed. The OTel API then hands back non-recording
    (no-op) spans, so every `with tracer.start_as_current_span(...)` in the code
    base is a zero-cost no-op and prod is completely unaffected until opted in.
  • Exporter failures must NEVER break a request. The BatchSpanProcessor ships
    spans on a background thread and swallows transport errors; nothing here is
    on a request's critical path once a span is created.

Trace hierarchy produced across the codebase:
  HTTP span (FastAPI auto-instrumentation)
    └─ workflow span      (course_service.create_course / evaluation session …)
         └─ agent span    (base_agent.run — agent name, student_id)
              ├─ LLM span (groq_client.generate / tool_call_loop iter / stream)
              └─ tool span (tool executor / mcp_search_client._call)

Everything routes through get_tracer(); callers never touch the SDK directly.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from opentelemetry import trace
from config import settings


# ── GenAI semantic-convention attribute keys (OTel gen_ai.*) ───────────────────
# Kept as constants so span-emitting call sites stay consistent.
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_COST_USD = "gen_ai.usage.cost_usd"
GEN_AI_USAGE_IS_ESTIMATE = "gen_ai.usage.is_estimate"

# Prompt-registry traceability: which versioned prompt artifact drove this call.
GEN_AI_PROMPT_NAME = "gen_ai.prompt.name"
GEN_AI_PROMPT_VERSION = "gen_ai.prompt.version"

# EduMind-specific attributes (not part of the GenAI convention).
EDUMIND_CALLER = "edumind.caller"

# Learner privacy: never attach full answer/lesson text to spans — excerpts only.
MAX_EXCERPT_CHARS = 200

_TRACER_NAME = "edumind"

# Kept so shutdown can flush; None when tracing is disabled.
_provider: Any | None = None
_initialised = False


def get_tracer() -> trace.Tracer:
    """Return the EduMind tracer.

    Safe to call whether or not tracing is enabled — when disabled the global
    OTel API returns a tracer that yields non-recording spans (no-ops).
    """
    return trace.get_tracer(_TRACER_NAME)


def excerpt(text: str | None, limit: int = MAX_EXCERPT_CHARS) -> str:
    """Return a privacy-safe excerpt (first `limit` chars) of learner text."""
    if not text:
        return ""
    return text[:limit]


def current_trace_id_hex() -> str | None:
    """Return the active span's 32-char hex trace id, or None if not tracing.

    Used to cross-reference log lines with traces without duplicating span data
    into logs.
    """
    span = trace.get_current_span()
    ctx = span.get_span_context() if span else None
    if not ctx or not ctx.is_valid:
        return None
    return format(ctx.trace_id, "032x")


def init_tracing(app: Any | None = None) -> bool:
    """Initialise OpenTelemetry if enabled. Idempotent. Returns True if active.

    When settings.otel_enabled is False this is a no-op and returns False, so no
    exporter is configured and no spans are emitted.
    """
    global _provider, _initialised

    if _initialised:
        return _provider is not None

    if not settings.otel_enabled:
        _initialised = True
        logger.info("OpenTelemetry disabled (OTEL_ENABLED=false) — tracing is a no-op.")
        return False

    # Import the SDK lazily so a deployment that never enables tracing does not
    # pay the import cost and does not hard-depend on the exporter package.
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        resource = Resource.create({"service.name": settings.otel_service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _provider = provider

        if app is not None:
            try:
                from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

                FastAPIInstrumentor.instrument_app(app)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("FastAPI OTel instrumentation failed: {}", exc)

        logger.info(
            "OpenTelemetry enabled — exporting spans to {} (service={}).",
            settings.otel_exporter_endpoint, settings.otel_service_name,
        )
    except Exception as exc:
        # A misconfigured/unavailable exporter must never break startup.
        _provider = None
        logger.warning("OpenTelemetry init failed ({}) — continuing without tracing.", exc)
    finally:
        _initialised = True

    return _provider is not None


def shutdown_tracing() -> None:
    """Flush and shut down the tracer provider (best-effort)."""
    global _provider
    if _provider is not None:
        try:
            _provider.shutdown()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Tracer provider shutdown error: {}", exc)
        _provider = None
