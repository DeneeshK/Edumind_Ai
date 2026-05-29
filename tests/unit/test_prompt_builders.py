from __future__ import annotations

import pytest

from core.course_service import lesson_prompt, question_generation_prompt


pytestmark = pytest.mark.unit


def test_lesson_prompt_includes_course_module_and_student_context():
    course = {
        "topic": "Python",
        "goal": "Learn Python for automation",
        "pace": "fast",
        "personalization_profile": {
            "learner_level": "complete beginner",
            "roadmap_steps": ["First, learn variables."],
        },
    }
    module = {
        "id": "M1",
        "title": "Values and Variables",
        "concept": "variables",
        "concepts_taught": ["values", "variables"],
        "question_scope": ["values", "variables"],
    }
    adaptation_context = {
        "weak_concepts": ["assignment"],
        "recommended_teaching_adjustments": ["Use one short runnable code example."],
    }

    prompt = lesson_prompt(course, module, [], adaptation_context, [])

    assert "Course topic: Python" in prompt
    assert "Student goal: Learn Python for automation" in prompt
    assert "Pace: fast" in prompt
    assert "Module title: Values and Variables" in prompt
    assert "Teach every required concept by name: values, variables" in prompt
    assert "Use one short runnable code example." in prompt


def test_lesson_prompt_handles_minimal_valid_input():
    prompt = lesson_prompt(
        {"topic": "Calculus", "goal": "Understand slope", "pace": "medium"},
        {"id": "M1", "title": "Derivatives", "concept": "derivatives"},
        [],
        {},
        [],
    )

    assert "Derivatives" in prompt
    assert "derivatives" in prompt


def test_question_generation_prompt_contains_grounding_rules():
    prompt = question_generation_prompt(
        {"id": "course-1", "topic": "Python", "goal": "Learn programming", "pace": "medium"},
        {
            "id": "M1",
            "title": "Values and Variables",
            "concept": "variables",
            "question_scope": ["variables"],
        },
        "Variables store values so a program can reuse them later.",
        [],
    )

    assert "Course: Python" in prompt
    assert "Module title: Values and Variables" in prompt
    assert "source_quote must be copied verbatim" in prompt
    assert "Ban placeholder/meta-question patterns" in prompt
