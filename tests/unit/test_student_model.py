from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from core.student_model import StudentState


pytestmark = pytest.mark.unit


def test_student_state_validates_pace_and_defaults(sample_student_state):
    state = StudentState(**sample_student_state)

    assert state.pace == "medium"
    assert state.metacognition.preferred_style == "formal"
    assert state.get_mastery("variables") == 0.82


def test_student_state_rejects_unknown_pace(sample_student_state):
    sample_student_state["pace"] = "warp-speed"

    with pytest.raises(ValidationError):
        StudentState(**sample_student_state)


def test_mastery_update_sets_depth_and_dirty_field(sample_student_state):
    state = StudentState(**sample_student_state)

    state.update_mastery("functions", correctness=0.8, depth=0.5)

    assert state.concept_mastery["functions"] == 0.68
    assert state.concept_depth["functions"] == 0.5
    assert "concept_mastery" in state._dirty


def test_ready_to_advance_uses_pace_thresholds(monkeypatch, sample_student_state):
    import core.student_model as student_model

    monkeypatch.setattr(student_model.settings, "mastery_threshold_fast", 0.6)
    monkeypatch.setattr(student_model.settings, "mastery_threshold_medium", 0.72)
    monkeypatch.setattr(student_model.settings, "mastery_threshold_deep", 0.85)

    state = StudentState(**sample_student_state)

    assert state.ready_to_advance("variables") is True
    assert state.ready_to_advance("loops") is False


def test_record_module_content_appends_non_empty_text(sample_student_state):
    state = StudentState(**sample_student_state)

    state.record_module_content("M1", "First paragraph.")
    state.record_module_content("M1", "Second paragraph.")
    state.record_module_content("M1", "   ")

    assert state.get_module_content("M1") == "First paragraph.\n\nSecond paragraph."
    assert "session_module_content" in state._dirty


def test_start_session_resets_session_scoped_fields(monkeypatch, sample_student_state):
    from clients import tavily_client

    monkeypatch.setattr(tavily_client, "clear_cache", lambda *args, **kwargs: None)
    state = StudentState(**sample_student_state)
    state.session_doubt_counts = {"loops": 2}
    state.session_module_content = {"M1": "old content"}

    session_id = state.start_session()

    assert session_id
    assert state.session_doubt_counts == {}
    assert state.session_module_content == {}
    assert isinstance(state.session_started_at, datetime)
