"""Unit tests: with OTEL disabled, no exporter is configured and no spans emit.

These guard the core constraint: OTEL_ENABLED unset/false must leave runtime
behaviour identical to baseline — every tracing path is a zero-cost no-op.
"""

from __future__ import annotations

import pytest

import core.tracing as tracing


pytestmark = pytest.mark.unit


@pytest.fixture
def fresh_tracing(monkeypatch):
    """Reset the init-once module state so init_tracing() runs cleanly."""
    monkeypatch.setattr(tracing, "_initialised", False)
    monkeypatch.setattr(tracing, "_provider", None)
    yield tracing


def test_init_tracing_noop_when_disabled(monkeypatch, fresh_tracing):
    monkeypatch.setattr(tracing.settings, "otel_enabled", False)

    active = fresh_tracing.init_tracing(app=None)

    assert active is False
    # No exporter/provider installed by us.
    assert fresh_tracing._provider is None


def test_no_spans_recorded_when_disabled(monkeypatch, fresh_tracing):
    monkeypatch.setattr(tracing.settings, "otel_enabled", False)
    fresh_tracing.init_tracing(app=None)

    tracer = fresh_tracing.get_tracer()
    with tracer.start_as_current_span("unit.disabled.span") as span:
        # A no-op span is non-recording; nothing is exported.
        assert span.is_recording() is False
        # And there is no valid trace id to correlate against.
        assert fresh_tracing.current_trace_id_hex() is None


def test_excerpt_truncates_to_privacy_limit():
    long_text = "x" * 5000
    assert len(tracing.excerpt(long_text)) == tracing.MAX_EXCERPT_CHARS
    assert tracing.excerpt("") == ""
    assert tracing.excerpt(None) == ""
