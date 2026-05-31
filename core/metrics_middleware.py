"""
core/metrics_middleware.py
──────────────────────────
Starlette middleware that automatically records every HTTP request
into Prometheus without requiring any per-route changes.

Mounted in app/api.py via:
    from core.metrics_middleware import PrometheusMiddleware
    app.add_middleware(PrometheusMiddleware)
"""

from __future__ import annotations

import time
import re
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from core.metrics import metrics

# Normalise dynamic path segments so cardinality stays low.
# e.g. /api/courses/course-72/modules/m3 → /api/courses/{course_id}/modules/{module_id}
_PATH_PATTERNS = [
    (re.compile(r"/courses/[^/]+"), "/courses/{course_id}"),
    (re.compile(r"/modules/[^/]+"), "/modules/{module_id}"),
    (re.compile(r"/evaluation/[^/]+/answer"), "/evaluation/{session_id}/answer"),
    (re.compile(r"/students/[^/]+"),  "/students/{student_id}"),
]

# Skip /metrics itself — no point tracking the scraper endpoint
_SKIP_PATHS = {"/metrics", "/health", "/favicon.ico"}


def _normalise(path: str) -> str:
    """Collapse dynamic API path segments to low-cardinality metric labels."""
    for pattern, replacement in _PATH_PATTERNS:
        path = pattern.sub(replacement, path)
    return path


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Record request count and latency metrics for each non-skipped HTTP request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        """Measure one request/response cycle and update Prometheus metrics."""
        path = request.url.path
        if path in _SKIP_PATHS:
            return await call_next(request)

        endpoint = _normalise(path)
        method = request.method
        start = time.perf_counter()

        response = await call_next(request)

        duration = time.perf_counter() - start
        status = str(response.status_code)

        metrics.http_requests.labels(method=method, endpoint=endpoint, status=status).inc()
        metrics.http_latency.labels(method=method, endpoint=endpoint).observe(duration)

        return response
