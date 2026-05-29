import pytest
import os
import sys
from unittest.mock import AsyncMock
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core import course_service

@pytest.mark.asyncio
async def test_lesson_generation_does_not_generate_questions(monkeypatch):
    course = {"id": "c1", "student_id": "s1", "topic": "Calculus", "goal": "learn", "pace": "fast"}
    module = {"id": "m1", "title": "Derivatives", "concept": "derivatives", "prerequisites": [], "content_markdown": ""}
    saved = {}

    async def fake_get_course(*args, **kwargs): return course
    async def fake_get_module(*args, **kwargs): return module
    async def fake_save(course_id, module_id, content, questions, videos=None):
        saved["content"] = content
        saved["questions"] = questions

    monkeypatch.setattr(course_service, "get_course", fake_get_course)
    monkeypatch.setattr(course_service, "get_course_module", fake_get_module)
    monkeypatch.setattr(course_service, "set_module_status", AsyncMock())
    monkeypatch.setattr(course_service, "adaptation_context_for_module", AsyncMock(return_value={}))
    monkeypatch.setattr(course_service, "retrieve", AsyncMock(return_value=[]))
    monkeypatch.setattr(course_service, "generate", AsyncMock(return_value="# Lesson"))
    monkeypatch.setattr(course_service, "validate_lesson_quality", lambda *args, **kwargs: {"passed": True, "issues": []})
    monkeypatch.setattr(course_service, "save_module_content", fake_save)
    monkeypatch.setattr(course_service, "search_youtube_videos_for_module", AsyncMock(return_value=[]))

    result = await course_service.generate_module_lesson("c1", "m1", "s1")
    assert saved["questions"] == []
    assert result["questions"] == []

@pytest.mark.asyncio
async def test_stream_lesson_does_not_emit_question_generated(monkeypatch):
    course = {"id": "c1", "student_id": "s1", "topic": "Calculus", "goal": "learn", "pace": "fast"}
    module = {"id": "m1", "title": "Derivatives", "concept": "derivatives", "prerequisites": [], "content_markdown": ""}

    async def fake_stream(*args, **kwargs):
        yield "# Les"
        yield "son"

    monkeypatch.setattr(course_service, "get_course", AsyncMock(return_value=course))
    monkeypatch.setattr(course_service, "get_course_module", AsyncMock(return_value=module))
    monkeypatch.setattr(course_service, "set_module_status", AsyncMock())
    monkeypatch.setattr(course_service, "adaptation_context_for_module", AsyncMock(return_value={}))
    monkeypatch.setattr(course_service, "validate_lesson_quality", lambda *args, **kwargs: {"passed": True, "issues": []})
    monkeypatch.setattr(course_service, "retrieve", AsyncMock(return_value=[]))
    monkeypatch.setattr(course_service, "stream", fake_stream)
    monkeypatch.setattr(course_service, "save_module_content", AsyncMock())
    monkeypatch.setattr(course_service, "search_youtube_videos_for_module", AsyncMock(return_value=[]))

    events = [e async for e in course_service.generate_module_lesson_events("c1", "m1", "s1")]
    assert not any(e.get("event") == "question_generated" for e in events)
    assert any(e.get("event") == "chunk" for e in events)

def _get_react_source(filename):
    path = os.path.join(os.path.dirname(__file__), f"../../edumind_frontend/src/{filename}")
    with open(path, encoding="utf-8") as f:
        return f.read()

def test_module_page_does_not_render_lesson_questions():
    source = _get_react_source("pages/ModuleReaderPage.jsx")
    assert "QuestionCard" not in source
    assert "Check questions" not in source

def test_next_button_opens_evaluation_choice_modal():
    source = _get_react_source("pages/ModuleReaderPage.jsx")
    assert "onClick={handleNextClick}" in source
    assert "setEvalModalOpen(true)" in source
    assert "<EvaluationModal" in source

def test_skip_evaluation_navigates_without_starting_session():
    source = _get_react_source("pages/ModuleReaderPage.jsx")
    assert "handleSkipOrContinue" in source
    assert "setEvalModalOpen(false)" in source
    assert "getNextModule" in source
    assert "navigate(" in source

def test_evaluate_myself_calls_evaluation_start():
    source = _get_react_source("pages/ModuleReaderPage.jsx")
    assert "handleEvalStart" in source
    assert "startEvaluation(courseId, moduleId" in source
    assert "setEvalSession(res)" in source

def test_evaluation_questions_render_one_by_one():
    source = _get_react_source("components/lesson/EvaluationModal.jsx")
    assert "currentQuestion.question_text" in source
    assert "Question {sessionData?.questions_asked" in source
    assert "textarea" in source

def test_answer_submission_shows_next_question_until_complete():
    source = _get_react_source("pages/ModuleReaderPage.jsx")
    assert "handleEvalSubmit" in source
    assert "submitEvaluationAnswer(" in source
    assert "res.session_complete" in source
    assert "setEvalReport(res)" in source
    assert "setEvalQuestion(res.next_question)" in source

def test_final_feedback_modal_shows_motivational_and_transition_feedback():
    source = _get_react_source("components/lesson/EvaluationModal.jsx")
    assert "Evaluation Complete" in source
    assert "finalReport.motivational_feedback" in source
    assert "finalReport.transition_feedback" in source

def test_course_creation_does_not_generate_questions():
    pass # Covered by test_no_questions_generated_during_course_creation in test_frontend_course_api.py

def test_youtube_iframe_still_renders_on_module_page():
    page_source = _get_react_source("pages/ModuleReaderPage.jsx")
    video_source = _get_react_source("components/lesson/VideoResources.jsx")
    assert "<VideoResources videos={videos} />" in page_source
    assert "<iframe" in video_source

