from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

import core.course_service as course_service
from core.student_model import CurriculumPlan, Module, StudentState


pytestmark = pytest.mark.unit


def test_normalise_profile_drops_untrusted_known_concepts_for_no_prior_python():
    profile = course_service.normalise_personalization_profile(
        topic="Python",
        goal="Understand Python programming",
        pace="fast",
        prior_knowledge="no prior programming experience",
        profile={
            "known_concepts": ["thermodynamics", "linear algebra for machine learning"],
            "target_context": "general pure Python",
            "learner_level": "complete beginner",
        },
    )

    assert profile["assumed_known_concepts"] == []
    assert "machine learning" in profile["do_not_include"]
    assert "thermodynamics" not in profile["recommended_strategy"].lower()


@pytest.mark.asyncio
async def test_create_course_wraps_planner_result_and_saves_roadmap(monkeypatch):
    plan = CurriculumPlan(
        topic="Python",
        domain="software engineering",
        goal="Learn Python fundamentals",
        modules=[
            Module(
                id="M1",
                title="Values and Variables",
                concept="variables",
                domain_framing="variables for Python scripts",
                prerequisites=[],
                estimated_minutes=30,
                depth_level="standard",
                roadmap_step_id="step_01",
            )
        ],
        roadmap_steps=["First, learn values and variables."],
        validation_result={"passed": True, "issues": []},
    )
    captured = {}

    class FakeArchitect:
        def __init__(self, state):
            self.state = state
            self._curriculum_id = 42

        async def build_curriculum(self, topic):
            captured["planner_topic"] = topic
            return plan

    async def fake_create_course_from_plan(**kwargs):
        captured["create_kwargs"] = kwargs
        return {
            "id": "course-python-1",
            "student_id": kwargs["student_id"],
            "topic": kwargs["plan"].topic,
            "goal": kwargs["plan"].goal,
            "pace": kwargs["pace"],
            "personalization_profile": kwargs["personalization_profile"],
        }

    module_rows = [
        {
            "id": "M1",
            "module_index": 0,
            "title": "Values and Variables",
            "concept": "variables",
            "estimated_minutes": 30,
            "status": "not_started",
        }
    ]

    monkeypatch.setattr(course_service, "upsert_student", AsyncMock())
    monkeypatch.setattr(
        course_service.StudentState,
        "load",
        AsyncMock(
            return_value=StudentState(
                student_id="student-test-1",
                domain="software engineering",
                goal="Learn Python fundamentals",
                pace="medium",
            )
        ),
    )
    monkeypatch.setattr(course_service, "CurriculumArchitectAgent", FakeArchitect)
    monkeypatch.setattr(course_service, "get_student_history_snapshot", AsyncMock(return_value={}))
    monkeypatch.setattr(course_service, "create_course_from_plan", fake_create_course_from_plan)
    monkeypatch.setattr(course_service, "list_course_modules", AsyncMock(return_value=module_rows))
    monkeypatch.setattr(
        course_service,
        "save_course_roadmap",
        AsyncMock(side_effect=lambda course_id, roadmap: roadmap),
    )

    course = await course_service.create_course(
        student_id="student-test-1",
        topic="Python",
        goal="Learn Python fundamentals",
        pace="medium",
        name="Test Student",
    )

    assert captured["planner_topic"] == "Python"
    assert captured["create_kwargs"]["curriculum_id"] == 42
    assert course["roadmap_ready"] is True
    assert course["redirect_url"] == "/courses/course-python-1/roadmap"
    assert course["roadmap"]["module_timeline"][0]["module_id"] == "M1"


@pytest.mark.asyncio
async def test_generate_module_lesson_saves_mocked_content_without_questions(monkeypatch):
    course = {
        "id": "course-python-1",
        "student_id": "student-test-1",
        "topic": "Python",
        "goal": "Learn Python fundamentals",
        "pace": "fast",
        "personalization_profile": {},
    }
    module = {
        "id": "M1",
        "module_index": 0,
        "title": "Values and Variables",
        "concept": "variables",
        "concepts_taught": ["values", "variables"],
        "question_scope": ["values", "variables"],
        "content_markdown": "",
    }
    saved = {}
    prompt_seen = {}

    async def fake_get_module(course_id, module_id):
        if saved:
            return {**module, "content_markdown": saved["content"], "lesson_videos": saved["videos"]}
        return dict(module)

    async def fake_generate(messages, **kwargs):
        prompt_seen["content"] = messages[0]["content"]
        return "# Values and Variables\n\nVariables store values for later reuse."

    async def fake_save(course_id, module_id, content, questions, videos=None):
        saved["course_id"] = course_id
        saved["module_id"] = module_id
        saved["content"] = content
        saved["questions"] = questions
        saved["videos"] = videos or []

    monkeypatch.setattr(course_service, "get_course", AsyncMock(return_value=course))
    monkeypatch.setattr(course_service, "get_course_module", fake_get_module)
    monkeypatch.setattr(course_service, "set_module_status", AsyncMock())
    monkeypatch.setattr(course_service, "list_course_modules", AsyncMock(return_value=[module]))
    monkeypatch.setattr(course_service, "adaptation_context_for_module", AsyncMock(return_value={}))
    monkeypatch.setattr(course_service, "retrieve", AsyncMock(return_value=[]))
    monkeypatch.setattr(course_service, "generate", fake_generate)
    monkeypatch.setattr(
        course_service,
        "validate_lesson_quality",
        lambda *args, **kwargs: {"passed": True, "issues": []},
    )
    monkeypatch.setattr(course_service, "search_youtube_videos_for_module", AsyncMock(return_value=[]))
    monkeypatch.setattr(course_service, "save_module_content", fake_save)

    result = await course_service.generate_module_lesson(
        "course-python-1",
        "M1",
        "student-test-1",
    )

    assert "Module title: Values and Variables" in prompt_seen["content"]
    assert saved["questions"] == []
    assert saved["content"].startswith("# Values and Variables")
    assert result["content_markdown"] == saved["content"]
    assert result["videos"] == []
