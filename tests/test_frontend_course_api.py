import os
import sys
import json
from unittest.mock import AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi import HTTPException

import app.course_api as course_api
from app.course_api import CreateCourseRequest, course_payload_from_request
from core.curriculum_quality import (
    filter_relevant_student_history,
    fallback_scope_analysis,
    validate_curriculum_quality,
    validate_questions_grounded,
)
from core.roadmap_service import CourseRoadmapService
from core.student_model import CurriculumPlan, Module
import core.course_service as course_service


def test_guided_setup_payload_uses_topic_when_goal_is_optional():
    req = CreateCourseRequest(
        student_id="s1",
        topic="World History",
        current_level="not_sure",
        pace="medium",
    )

    payload = course_payload_from_request(req)

    assert payload["topic"] == "World History"
    assert payload["goal"] == "Learn World History"
    assert payload["pace"] == "medium"
    assert payload["profile"]["setup_source"] == "guided_course_setup"
    assert payload["profile"]["current_level"] == "not_sure"
    assert payload["profile"]["target_context"] == "general learning"


def test_guided_setup_payload_preserves_level_time_and_prior_experience():
    req = CreateCourseRequest(
        student_id="s1",
        topic="Thermodynamics",
        goal_description="I want to understand thermodynamics for my exam.",
        current_level="basic",
        prior_experience="I studied this before but forgot PV diagrams.",
        time_commitment={"value": "45", "unit": "minutes_per_day"},
        deadline="2026-06-20",
        pace="fast",
    )

    payload = course_payload_from_request(req)
    profile = payload["profile"]

    assert payload["goal"] == "I want to understand thermodynamics for my exam."
    assert profile["learner_level"] == "some basic knowledge"
    assert profile["prior_experience"] == "I studied this before but forgot PV diagrams."
    assert profile["time_constraint"] == "45 minutes per day, target by 2026-06-20"
    assert profile["depth_preference"] == "quick overview and practical path"


def test_guided_setup_profile_reaches_course_service_current_intent():
    req = CreateCourseRequest(
        student_id="s1",
        topic="Public Speaking",
        goal_description="I want to speak confidently at work.",
        current_level="complete_beginner",
        prior_experience="I avoid presentations.",
        time_commitment={"value": "3", "unit": "hours_per_week"},
        pace="deep",
    )
    payload = course_payload_from_request(req)

    profile = course_service.normalise_personalization_profile(
        topic=payload["topic"],
        goal=payload["goal"],
        pace=payload["pace"],
        prior_knowledge=payload["prior_knowledge"],
        profile=payload["profile"],
    )

    assert profile["current_intent"]["topic"] == "Public Speaking"
    assert profile["current_intent"]["goal_description"] == "I want to speak confidently at work."
    assert profile["current_intent"]["current_level"] == "complete_beginner"
    assert profile["current_intent"]["prior_experience"] == "I avoid presentations."
    assert profile["time_constraint"] == "3 hours per week"


def test_grounded_questions_use_module_content_only():
    module = {
        "id": "m1",
        "title": "Gradient Descent",
        "concept": "gradient descent",
        "description": "gradient descent as iterative improvement",
    }
    content = (
        "Gradient descent updates a parameter by moving against the derivative. "
        "The derivative tells us the local slope of the loss. "
        "A learning rate controls how large each update step is."
    )

    questions = course_service.grounded_questions_from_content(
        content, module, pace="medium"
    )

    assert len(questions) == 3
    assert all(q["source_quote"] in content for q in questions)
    assert not any("softmax" in q["question_text"].lower() for q in questions)
    assert not any("according to the lesson" in q["question_text"].lower() for q in questions)


def test_lesson_prompt_uses_mentor_style_structure():
    course = {
        "topic": "Python",
        "goal": "learn programming",
        "pace": "fast",
        "personalization_profile": {"learner_level": "complete beginner"},
    }
    module = {
        "title": "Running Your First Python Program",
        "concept": "running a Python file",
        "description": "run a first .py file and see output",
        "content_markdown": "",
    }

    prompt = course_service.lesson_prompt(course, module, [], {}, [])

    assert "Mentor-style opening / hook" in prompt
    assert "Mental model" in prompt
    assert "Worked example / demonstration" in prompt
    assert "Common beginner mistake" in prompt
    assert "Mini practice task" in prompt
    assert "Short recap" in prompt


def test_lesson_prompt_does_not_expose_internal_metadata_headings():
    course = {"topic": "Python", "goal": "learn programming", "pace": "fast"}
    module = {
        "title": "Running Your First Python Program",
        "concept": "running a Python file",
        "module_metadata": {
            "must_teach": ["running a file"],
            "lesson_requirements": ["show output"],
            "practice_requirements": ["edit a print statement"],
        },
    }

    prompt = course_service.lesson_prompt(course, module, [], {}, [])

    assert "do not expose raw metadata labels" in prompt.lower()
    assert 'Do not use student-facing headings named "Must Teach"' in prompt
    assert '"Lesson Requirements"' in prompt
    assert '"Concepts Taught in this Module"' in prompt
    assert '"Practice Requirements"' in prompt


def test_fast_pace_concise_not_shallow():
    course = {"topic": "Python", "goal": "learn programming", "pace": "fast"}
    module = {"title": "First Program", "concept": "running code"}

    prompt = course_service.lesson_prompt(course, module, [], {}, [])

    assert "Fast pace: concise, practical, focused" in prompt
    assert "Fast pace must not be shallow" in prompt
    assert "generic" in prompt


def test_question_prompt_bans_placeholder_questions():
    course = {"id": "c1", "topic": "Python", "goal": "learn programming", "pace": "fast"}
    module = {
        "id": "m1",
        "title": "First Program",
        "concept": "running code",
        "question_scope": ["running code"],
    }

    prompt = course_service.question_generation_prompt(
        course,
        module,
        "Running code means asking Python to execute the file.",
        [],
    )

    assert "According to the lesson" in prompt
    assert "What is the key idea" in prompt
    assert "What detail from the lesson explains" in prompt
    assert "Ban placeholder/meta-question patterns" in prompt


def test_coding_lesson_requires_runnable_code_output_and_line_explanation():
    course = {"topic": "Python", "goal": "learn programming", "pace": "fast"}
    module = {"title": "First Program", "concept": "running a Python file"}

    prompt = course_service.lesson_prompt(course, module, [], {}, [])

    assert "runnable code blocks" in prompt
    assert "expected output" in prompt
    assert "line-by-line explanation" in prompt
    assert "output prediction" in prompt


def test_lesson_covers_all_required_module_concepts():
    course = {"topic": "Python", "goal": "learn programming", "pace": "medium"}
    module = {
        "id": "m8",
        "title": "Control Structures: for loops, and while loops",
        "concept": "control structures",
        "concepts_taught": ["for loops", "while loops"],
        "question_scope": ["for loops", "while loops"],
    }

    prompt = course_service.lesson_prompt(course, module, [], {}, [])

    assert "Teach every required concept by name: for loops, while loops" in prompt
    assert "For each one, include what it is, when to use it, a simple example or demonstration" in prompt
    assert "teaching only \"for loops\" does not cover \"while loops\"" in prompt


def test_question_generation_does_not_reference_untaught_concepts():
    module = {
        "id": "m8",
        "title": "Control Structures: for loops, and while loops",
        "concept": "control structures",
        "concepts_taught": ["for loops", "while loops"],
        "question_scope": ["for loops", "while loops"],
    }
    lesson = (
        "For loops repeat code for each item in a list. "
        "A for loop is useful when you already have a collection to walk through. "
        "A common beginner mistake with for loops is changing the list while looping over it. "
        "Practice by writing a for loop that prints each name in a list."
    )

    questions = course_service.grounded_questions_from_content(lesson, module, pace="medium")

    assert questions
    assert not any("while loop" in q["question_text"].lower() for q in questions)
    assert not any("while loops" in [c.lower() for c in q["concepts_tested"]] for q in questions)

    invalid = [{
        "question_text": "In the worked example, what happens when you use while loops?",
        "expected_answer": "For loops repeat code for each item in a list.",
        "source_quote": "For loops repeat code for each item in a list.",
        "concepts_tested": ["while loops"],
        "source_section": "Lesson",
    }]
    validation = validate_questions_grounded(invalid, lesson, module)
    assert not validation["passed"]
    assert any("while loops" in issue for issue in validation["issues"])


def test_youtube_watch_url_to_embed_url():
    assert (
        course_service.youtube_watch_url_to_embed_url("https://www.youtube.com/watch?v=abc123")
        == "https://www.youtube.com/embed/abc123"
    )


def test_youtube_result_extraction_from_tavily():
    course = {
        "topic": "Python",
        "goal": "learn programming",
        "pace": "medium",
        "personalization_profile": {"learner_level": "complete beginner"},
    }
    module = {
        "id": "m8",
        "title": "Control Structures: for loops, and while loops",
        "concept": "control structures",
    }
    results = [{
        "title": "For Loops and While Loops in Python",
        "url": "https://www.youtube.com/watch?v=abc123",
        "content": "beginner explanation of for loops and while loops",
    }]

    videos, stats = course_service.youtube_videos_from_tavily_results(course, module, results)

    assert stats["raw_result_count"] == 1
    assert stats["youtube_url_count"] == 1
    assert stats["selected_video_count"] == 1
    assert videos == [{
        "title": "For Loops and While Loops in Python",
        "url": "https://www.youtube.com/watch?v=abc123",
        "embed_url": "https://www.youtube.com/embed/abc123",
        "source": "youtube",
        "reason": "Relevant beginner explanation for this module.",
    }]


@pytest.mark.asyncio
async def test_youtube_search_filters_playlist_and_advanced_for_beginner(monkeypatch):
    course = {
        "topic": "Python",
        "goal": "learn programming",
        "pace": "fast",
        "personalization_profile": {"learner_level": "complete beginner"},
    }
    module = {
        "title": "Running Your First Python Program",
        "concept": "running a Python file",
    }
    results = [
        {
            "title": "Advanced Python Runtime Internals",
            "url": "https://www.youtube.com/watch?v=adv999",
            "content": "advanced interpreter internals",
        },
        {
            "title": "Python Beginner Playlist",
            "url": "https://www.youtube.com/playlist?list=pl123",
            "content": "playlist",
        },
        {
            "title": "Run Your First Python File",
            "url": "https://www.youtube.com/watch?v=abc123",
            "content": "beginner explanation under 10 minutes",
        },
    ]

    monkeypatch.setattr(course_service, "tavily_search", lambda *args, **kwargs: results)

    videos = await course_service.search_youtube_videos_for_module(course, module)

    assert len(videos) == 1
    assert videos[0]["embed_url"] == "https://www.youtube.com/embed/abc123"


def test_roadmap_service_personalizes_study_plan():
    course = {
        "id": "course-1",
        "topic": "thermodynamics",
        "goal": "prepare for JEE",
        "pace": "fast",
    }
    modules = [
        {
            "id": "m1",
            "title": "First Law and Energy Accounting",
            "concept": "first law of thermodynamics",
            "estimated_minutes": 15,
            "status": "in_progress",
            "prerequisites": [],
            "recommended": True,
        },
        {
            "id": "m2",
            "title": "PV Diagrams and Processes",
            "concept": "PV diagrams",
            "estimated_minutes": 20,
            "status": "not_started",
            "prerequisites": ["first law of thermodynamics"],
        },
    ]
    profile = {
        "topic": "thermodynamics",
        "learning_goal": "prepare for JEE",
        "target_context": "jee",
        "pace": "fast",
        "time_constraint": "very limited time",
        "weak_concepts": ["PV diagrams"],
        "recommended_strategy": "Focus on formulas, common traps, and JEE problem patterns.",
    }

    roadmap = CourseRoadmapService().build(course, modules, profile, {})

    assert roadmap["course_id"] == "course-1"
    assert "JEE" in roadmap["title"]
    assert roadmap["estimated_total_time_minutes"] >= 100
    assert roadmap["recommended_schedule"]
    assert roadmap["module_timeline"][1]["recommended_next"] is False
    assert any("Common exam traps" in item for item in roadmap["emphasized_topics"])


def test_roadmap_filters_unrelated_history_from_skip_list():
    course = {"id": "course-2", "topic": "Python", "goal": "machine learning", "pace": "fast"}
    modules = [
        {"id": "m1", "title": "Python Variables for Data Work", "concept": "Python variables", "estimated_minutes": 40},
        {"id": "m2", "title": "NumPy Arrays for ML", "concept": "NumPy arrays", "estimated_minutes": 45},
    ]
    profile = {
        "topic": "Python",
        "learning_goal": "machine learning",
        "target_context": "machine learning",
        "pace": "fast",
        "scope_analysis": fallback_scope_analysis({"topic": "Python", "learning_goal": "machine learning"}).to_dict(),
        "validation_result": {"passed": True, "quality_score": 0.95, "issues": []},
    }
    history = {"mastered_concepts": [{"concept": "thermodynamics"}], "previous_courses": [{"topic": "thermodynamics"}]}

    roadmap = CourseRoadmapService().build(course, modules, profile, history)

    assert "thermodynamics" not in str(roadmap).lower()


def test_history_filter_keeps_thermodynamics_out_of_beginner_python():
    intent = {
        "topic": "Python",
        "exact_subject": "Python programming",
        "goal": "understand Python programming and learn to code with Python",
        "target_context": "general pure Python",
        "learner_level": "complete beginner",
        "pace": "fast",
        "prior_knowledge_summary": "no prior knowledge of programming languages",
    }
    history = {
        "mastered_concepts": [{"concept": "thermodynamics basics", "mastery_score": 0.95}],
        "previous_courses": [
            {"topic": "thermodynamics"},
            {"topic": "python purpose and mental model for machine learning"},
            {"topic": "core mechanics of python"},
        ],
        "skill_graph": {
            "nodes": [{"concept": "linear algebra for machine learning", "mastery_score": 0.8}]
        },
    }

    result = filter_relevant_student_history(intent, history)

    assert result["assumed_known"] == []
    assert not any("thermodynamics" in item.lower() for item in result["concepts"])
    assert any(
        "python purpose and mental model" in item["concept"].lower()
        for item in result["possibly_related_but_not_assumed_known"]
    )


def test_normalized_profile_drops_stale_frontend_known_concepts_for_no_prior_python():
    profile = course_service.normalise_personalization_profile(
        topic="python",
        goal="understand python programming and learn to code with python",
        pace="fast",
        prior_knowledge="no prior knowledge of programming languages",
        profile={
            "known_concepts": [
                "thermodynamics basics",
                "python purpose and mental model for machine learning",
                "core mechanics of python",
            ],
            "student_history_relevant_concepts": ["linear algebra for machine learning"],
            "recommended_strategy": "Emphasize transformations and reduce thermodynamics basics.",
        },
    )

    assert profile["assumed_known_concepts"] == []
    assert profile["known_concepts"] == []
    assert "thermodynamics" not in profile["recommended_strategy"].lower()
    assert "machine learning" in profile["do_not_include"]
    assert profile["current_intent"]["target_context"] == "general pure Python"


def test_apply_history_filter_does_not_treat_generated_modules_as_mastery():
    profile = course_service.normalise_personalization_profile(
        topic="Python",
        goal="understand Python programming",
        pace="fast",
        prior_knowledge="no prior programming experience",
        profile={"learner_level": "complete beginner", "target_context": "general pure Python"},
    )
    history = {
        "mastered_concepts": [
            {"concept": "python purpose and mental model for machine learning", "mastery_score": 0.8},
            {"concept": "core mechanics of python", "mastery_score": 0.8},
        ],
        "previous_courses": [{"topic": "Python for machine learning"}],
    }

    filtered = course_service.apply_relevant_history_filter(profile, history)

    assert profile["assumed_known_concepts"] == []
    assert profile["known_concepts"] == []
    assert filtered["possibly_related_but_not_assumed_known"]
    assert "machine learning" not in profile["recommended_strategy"].lower()


def test_validator_rejects_profile_state_contamination_before_modules_save():
    scope = fallback_scope_analysis({
        "topic": "Python programming",
        "learning_goal": "pure Python fundamentals",
        "target_context": "pure Python",
        "learner_level": "complete beginner",
        "pace": "fast",
    })
    module = Module(
        id="m1",
        title="Variables and Values",
        concept="variables",
        domain_framing="variables in pure Python",
        prerequisites=[],
        estimated_minutes=20,
        depth_level="surface",
        concepts_taught=["values", "variables"],
        depends_on_concepts=[],
        question_scope=["values", "variables"],
    )

    result = validate_curriculum_quality(
        topic="Python programming",
        modules=[module],
        profile={
            "topic": "Python programming",
            "target_context": "pure Python",
            "learner_level": "complete beginner",
            "prior_knowledge_summary": "no prior programming experience",
            "known_concepts": ["thermodynamics basics"],
            "recommended_strategy": "Emphasize transformations and reduce review of thermodynamics basics.",
        },
        scope_analysis=scope,
        concept_inventory={"core_concepts": ["variables"], "concepts_to_delay": []},
        prerequisite_graph={"variables": []},
        roadmap_steps=["First, learn variables and values."],
        schedule=[{"day": 1, "items": [{"module_id": "m1", "estimated_minutes": 20}], "total_minutes": 30}],
    )

    assert result["passed"] is False
    assert any("state contamination" in issue.lower() for issue in result["issues"])


def test_scope_fallback_distinguishes_huge_and_narrow_topics():
    huge = fallback_scope_analysis({
        "topic": "complete Python",
        "learning_goal": "deep level from scratch to advanced",
        "pace": "deep",
    })
    narrow = fallback_scope_analysis({
        "topic": "linear regression",
        "learning_goal": "deep conceptual understanding",
        "pace": "deep",
    })

    assert huge.topic_breadth in {"broad", "huge"}
    assert huge.recommended_module_count > narrow.recommended_module_count
    assert narrow.topic_breadth in {"narrow", "medium"}


def test_curriculum_validator_rejects_contamination_and_garbage():
    scope = fallback_scope_analysis({"topic": "Python", "learning_goal": "machine learning"})
    modules = [
        Module(
            id="m1",
            title="Python Variables for ML",
            concept="Python variables",
            domain_framing="Python variables for machine learning scripts",
            prerequisites=[],
            estimated_minutes=20,
            depth_level="surface",
        ),
        Module(
            id="m2",
            title="Reduce review of thermodynamics basics",
            concept="thermodynamics basics",
            domain_framing="unrelated",
            prerequisites=[],
            estimated_minutes=20,
            depth_level="surface",
        ),
    ]

    result = validate_curriculum_quality(
        topic="Python",
        modules=modules,
        profile={"topic": "Python", "learning_goal": "machine learning", "target_context": "machine learning"},
        scope_analysis=scope,
        student_history={"previous_courses": [{"topic": "thermodynamics"}]},
    )

    assert result["passed"] is False
    assert any("thermodynamics" in issue.lower() for issue in result["issues"])


def test_curriculum_validator_rejects_placeholder_modules():
    scope = fallback_scope_analysis({
        "topic": "Python programming",
        "learning_goal": "pure Python fundamentals",
        "target_context": "pure Python",
        "learner_level": "complete beginner",
        "pace": "fast",
    })
    module = Module(
        id="m1",
        title="Python: Focused Learning Unit 12",
        concept="variables",
        domain_framing="variables in pure Python",
        prerequisites=[],
        estimated_minutes=20,
        depth_level="surface",
        concepts_taught=["variables"],
        depends_on_concepts=[],
        question_scope=["variables"],
    )

    result = validate_curriculum_quality(
        topic="Python programming",
        modules=[module],
        profile={"topic": "Python programming", "target_context": "pure Python", "learner_level": "complete beginner"},
        scope_analysis=scope,
        concept_inventory={"core_concepts": ["variables"], "concepts_to_delay": []},
        prerequisite_graph={"variables": []},
        roadmap_steps=["First, learn variables."],
        schedule=[{"day": 1, "items": [{"module_id": "m1", "estimated_minutes": 20}], "total_minutes": 30, "break_minutes": 0}],
    )

    assert result["passed"] is False
    assert any("placeholder" in issue.lower() for issue in result["issues"])


def test_curriculum_validator_rejects_pure_python_ml_drift():
    scope = fallback_scope_analysis({
        "topic": "Python programming",
        "learning_goal": "pure Python fundamentals",
        "target_context": "pure Python",
        "learner_level": "complete beginner",
        "pace": "fast",
    })
    module = Module(
        id="m1",
        title="Python in Machine Learning",
        concept="sklearn model.fit",
        domain_framing="train_test_split and LinearRegression in Python",
        prerequisites=[],
        estimated_minutes=20,
        depth_level="surface",
        concepts_taught=["sklearn model.fit"],
        depends_on_concepts=[],
        question_scope=["sklearn model.fit"],
    )

    result = validate_curriculum_quality(
        topic="Python programming",
        modules=[module],
        profile={"topic": "Python programming", "target_context": "pure Python", "learner_level": "complete beginner"},
        scope_analysis=scope,
        concept_inventory={"core_concepts": ["sklearn model.fit"], "concepts_to_delay": []},
        prerequisite_graph={"sklearn model.fit": []},
        roadmap_steps=["First, learn sklearn model.fit."],
        schedule=[{"day": 1, "items": [{"module_id": "m1", "estimated_minutes": 20}], "total_minutes": 30, "break_minutes": 0}],
    )

    assert result["passed"] is False
    assert any("machine-learning" in issue.lower() or "excluded" in issue.lower() for issue in result["issues"])


def test_question_validator_rejects_untaught_concepts():
    lesson = "# Variables\n\nVariables store values with assignment."
    module = {
        "id": "m1",
        "title": "Variables and Values",
        "concept": "variables",
        "concepts_taught": ["values", "variables", "assignment"],
        "depends_on_concepts": [],
        "question_scope": ["values", "variables", "assignment"],
    }
    questions = [{
        "question_text": "How does a loop repeat work?",
        "expected_answer": "Variables store values with assignment.",
        "source_quote": "Variables store values with assignment.",
        "concepts_tested": ["loops"],
        "source_section": "Variables",
        "is_answerable_from_lesson": True,
        "difficulty": "simple",
    }]

    result = validate_questions_grounded(questions, lesson, module)

    assert result["passed"] is False
    assert any("outside" in issue.lower() for issue in result["issues"])


def test_schedule_validator_rejects_beginner_overload():
    modules = [
        Module(
            id=f"m{i}",
            title=f"Specific Concept {i}",
            concept=f"concept {i}",
            domain_framing=f"concept {i} in Python programming",
            prerequisites=[f"concept {i - 1}"] if i > 1 else [],
            estimated_minutes=25,
            depth_level="surface",
            concepts_taught=[f"concept {i}"],
            depends_on_concepts=[f"concept {i - 1}"] if i > 1 else [],
            question_scope=([f"concept {i - 1}"] if i > 1 else []) + [f"concept {i}"],
        )
        for i in range(1, 18)
    ]
    schedule = [{
        "day": 1,
        "items": [
            {"module_id": f"m{i}", "module_title": f"Specific Concept {i}", "estimated_minutes": 25}
            for i in range(1, 18)
        ],
        "total_minutes": 425,
        "break_minutes": 0,
    }]

    result = validate_curriculum_quality(
        topic="Python programming",
        modules=modules,
        profile={"topic": "Python programming", "target_context": "pure Python", "learner_level": "complete beginner", "pace": "fast"},
        scope_analysis={"recommended_module_count": 17, "topic_breadth": "broad", "pace": "fast", "learner_level": "complete beginner"},
        concept_inventory={"core_concepts": [f"concept {i}" for i in range(1, 18)], "concepts_to_delay": []},
        prerequisite_graph={f"concept {i}": ([f"concept {i - 1}"] if i > 1 else []) for i in range(1, 18)},
        roadmap_steps=[f"Step {i}: learn concept {i}." for i in range(1, 18)],
        schedule=schedule,
    )

    assert result["passed"] is False
    assert any("schedule overload" in issue.lower() for issue in result["issues"])


def test_roadmap_schedule_limits_beginner_daily_modules():
    course = {"id": "course-python", "topic": "Python programming", "goal": "pure Python fundamentals", "pace": "fast"}
    modules = [
        {
            "id": f"m{i}",
            "title": f"Specific Python Concept {i}",
            "concept": f"concept {i}",
            "estimated_minutes": 25,
            "concepts_taught": [f"concept {i}"],
            "depends_on_concepts": [f"concept {i - 1}"] if i > 1 else [],
            "question_scope": ([f"concept {i - 1}"] if i > 1 else []) + [f"concept {i}"],
        }
        for i in range(1, 18)
    ]
    profile = {
        "topic": "Python programming",
        "learning_goal": "pure Python fundamentals",
        "target_context": "pure Python",
        "learner_level": "complete beginner",
        "pace": "fast",
        "scope_analysis": {"recommended_module_count": 17},
        "validation_result": {"passed": True, "quality_score": 1.0, "issues": []},
    }

    roadmap = CourseRoadmapService().build(course, modules, profile, {})
    max_modules = max(len(day["items"]) for day in roadmap["recommended_schedule"])

    assert max_modules <= 2


@pytest.mark.asyncio
async def test_create_course_wraps_curriculum_plan(monkeypatch):
    plan = CurriculumPlan(
        topic="Vectors",
        domain="machine learning",
        goal="understand embeddings",
        modules=[
            Module(
                id="m1",
                title="Vector Mental Model",
                concept="vectors",
                domain_framing="vectors as embedding coordinates",
                prerequisites=[],
                estimated_minutes=15,
                depth_level="standard",
            )
        ],
    )

    class FakeState:
        domain = "machine learning"
        goal = "understand embeddings"
        pace = "medium"

    class FakeArchitect:
        def __init__(self, state):
            self.state = state

        async def build_curriculum(self, topic):
            return plan, 42

    async def fake_upsert_student(*args, **kwargs):
        return None

    async def fake_load(student_id):
        return FakeState()

    async def fake_create_record(**kwargs):
        assert kwargs["curriculum_id"] == 42
        assert kwargs["plan"].topic == "Vectors"
        return {"id": "course-42", "topic": "Vectors", "progress": 0.0}

    async def fake_modules(course_id):
        return [{"id": "m1", "title": "Vector Mental Model"}]

    async def fake_history(student_id):
        return {}

    async def fake_save_roadmap(course_id, roadmap):
        assert roadmap["course_id"] == "course-42"
        return roadmap

    monkeypatch.setattr(course_service, "upsert_student", fake_upsert_student)
    monkeypatch.setattr(course_service.StudentState, "load", fake_load)
    monkeypatch.setattr(course_service, "CurriculumArchitectAgent", FakeArchitect)
    monkeypatch.setattr(course_service, "create_course_from_plan", fake_create_record)
    monkeypatch.setattr(course_service, "list_course_modules", fake_modules)
    monkeypatch.setattr(course_service, "get_student_history_snapshot", fake_history)
    monkeypatch.setattr(course_service, "save_course_roadmap", fake_save_roadmap)

    course = await course_service.create_course(
        student_id="s1",
        topic="Vectors",
        goal="understand embeddings",
        pace="medium",
        name="Dana",
    )

    assert course["id"] == "course-42"
    assert course["modules"][0]["id"] == "m1"
    assert course["roadmap_ready"] is True
    assert course["redirect_url"] == "/courses/course-42/roadmap"


@pytest.mark.asyncio
async def test_course_creation_does_not_generate_content_markdown(monkeypatch):
    plan = CurriculumPlan(
        topic="Python",
        domain="general pure Python",
        goal="learn programming",
        modules=[
            Module(
                id="m1",
                title="Python Values",
                concept="Python values",
                domain_framing="Python values for beginner scripts",
                prerequisites=[],
                estimated_minutes=12,
                depth_level="surface",
                roadmap_step_id="step_01",
            )
        ],
    )

    class FakeState:
        domain = "general pure Python"
        goal = "learn programming"
        pace = "fast"

    class FakeArchitect:
        def __init__(self, state):
            self.state = state

        async def build_curriculum(self, topic):
            return plan, 101

    save_content = AsyncMock()

    monkeypatch.setattr(course_service, "upsert_student", AsyncMock(return_value=None))
    monkeypatch.setattr(course_service.StudentState, "load", AsyncMock(return_value=FakeState()))
    monkeypatch.setattr(course_service, "CurriculumArchitectAgent", FakeArchitect)
    monkeypatch.setattr(course_service, "create_course_from_plan", fake_create_record)
    monkeypatch.setattr(course_service, "list_course_modules", fake_modules)
    monkeypatch.setattr(course_service, "get_student_history_snapshot", AsyncMock(return_value={}))
    monkeypatch.setattr(course_service, "save_course_roadmap", AsyncMock(side_effect=lambda course_id, roadmap: roadmap))
    monkeypatch.setattr(course_service, "save_module_content", save_content)

    course = await course_service.create_course(
        student_id="s-python",
        topic="Python",
        goal="learn programming",
        pace="fast",
    )

    assert course["modules"][0]["content_markdown"] == ""
    assert course["modules"][0]["content_exists"] is False
    save_content.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_questions_generated_during_course_creation(monkeypatch):
    plan = CurriculumPlan(
        topic="Python",
        domain="general pure Python",
        goal="learn programming",
        modules=[
            Module(
                id="m1",
                title="Python Values",
                concept="Python values",
                domain_framing="Python values for beginner scripts",
                prerequisites=[],
                estimated_minutes=12,
                depth_level="surface",
                roadmap_step_id="step_01",
            )
        ],
    )

    class FakeState:
        domain = "general pure Python"
        goal = "learn programming"
        pace = "fast"

    class FakeArchitect:
        def __init__(self, state):
            self.state = state

        async def build_curriculum(self, topic):
            return plan, 102

    save_questions = AsyncMock()

    monkeypatch.setattr(course_service, "upsert_student", AsyncMock(return_value=None))
    monkeypatch.setattr(course_service.StudentState, "load", AsyncMock(return_value=FakeState()))
    monkeypatch.setattr(course_service, "CurriculumArchitectAgent", FakeArchitect)
    monkeypatch.setattr(course_service, "create_course_from_plan", AsyncMock(return_value={"id": "course-python", "topic": "Python"}))
    monkeypatch.setattr(course_service, "list_course_modules", AsyncMock(return_value=[{"id": "m1", "title": "Python Values", "content_markdown": ""}]))
    monkeypatch.setattr(course_service, "get_student_history_snapshot", AsyncMock(return_value={}))
    monkeypatch.setattr(course_service, "save_course_roadmap", AsyncMock(side_effect=lambda course_id, roadmap: roadmap))
    monkeypatch.setattr(course_service, "save_module_questions", save_questions)

    await course_service.create_course(
        student_id="s-python-questions",
        topic="Python",
        goal="learn programming",
        pace="fast",
    )

    save_questions.assert_not_awaited()


@pytest.mark.asyncio
async def test_youtube_video_search_only_runs_on_module_open(monkeypatch):
    plan = CurriculumPlan(
        topic="Python",
        domain="general pure Python",
        goal="learn programming",
        modules=[
            Module(
                id="m1",
                title="Running Your First Python Program",
                concept="running a Python file",
                domain_framing="run a first .py file",
                prerequisites=[],
                estimated_minutes=12,
                depth_level="surface",
                roadmap_step_id="step_01",
            )
        ],
    )

    class FakeState:
        domain = "general pure Python"
        goal = "learn programming"
        pace = "fast"

    class FakeArchitect:
        def __init__(self, state):
            self.state = state

        async def build_curriculum(self, topic):
            return plan, 103

    video_search = AsyncMock(return_value=[{
        "title": "Python First Program",
        "url": "https://www.youtube.com/watch?v=abc123",
        "embed_url": "https://www.youtube.com/embed/abc123",
        "source": "youtube",
        "reason": "Shows how to run a first Python program.",
    }])

    monkeypatch.setattr(course_service, "upsert_student", AsyncMock(return_value=None))
    monkeypatch.setattr(course_service.StudentState, "load", AsyncMock(return_value=FakeState()))
    monkeypatch.setattr(course_service, "CurriculumArchitectAgent", FakeArchitect)
    monkeypatch.setattr(course_service, "create_course_from_plan", AsyncMock(return_value={"id": "course-python", "topic": "Python"}))
    monkeypatch.setattr(course_service, "list_course_modules", AsyncMock(return_value=[{"id": "m1", "title": "Running Your First Python Program", "content_markdown": ""}]))
    monkeypatch.setattr(course_service, "get_student_history_snapshot", AsyncMock(return_value={}))
    monkeypatch.setattr(course_service, "save_course_roadmap", AsyncMock(side_effect=lambda course_id, roadmap: roadmap))
    monkeypatch.setattr(course_service, "search_youtube_videos_for_module", video_search)

    await course_service.create_course(
        student_id="s-python-videos",
        topic="Python",
        goal="learn programming",
        pace="fast",
    )

    video_search.assert_not_awaited()

    course = {
        "id": "course-python",
        "student_id": "s-python-videos",
        "topic": "Python",
        "goal": "learn programming",
        "pace": "fast",
    }
    module = {
        "id": "m1",
        "title": "Running Your First Python Program",
        "concept": "running a Python file",
        "description": "run a first .py file",
        "prerequisites": [],
        "content_markdown": "",
    }
    saved = {}

    async def fake_get_module(course_id, module_id):
        if saved.get("content"):
            return {**module, "content_markdown": saved["content"]}
        return module

    async def fake_save(course_id, module_id, content, questions, videos=None):
        saved["content"] = content
        saved["questions"] = questions
        saved["videos"] = videos

    async def fake_questions(course_arg, module_arg, content):
        assert content
        saved["questions_after_content"] = True
        return ([{
            "id": "course-python:m1:q1",
            "question_text": "What command runs the Python file?",
            "expected_answer": "python hello.py",
            "source_quote": "python hello.py",
            "concepts_tested": ["running a Python file"],
            "source_section": "Lesson",
        }], {"status": "passed"})

    monkeypatch.setattr(course_service, "get_course", AsyncMock(return_value=course))
    monkeypatch.setattr(course_service, "get_course_module", fake_get_module)
    monkeypatch.setattr(course_service, "set_module_status", AsyncMock(return_value=None))
    monkeypatch.setattr(course_service, "list_course_modules", AsyncMock(return_value=[module]))
    monkeypatch.setattr(course_service, "adaptation_context_for_module", AsyncMock(return_value={}))
    monkeypatch.setattr(course_service, "retrieve", AsyncMock(return_value=[]))
    monkeypatch.setattr(course_service, "generate", AsyncMock(return_value="# Lesson\nRun the file with `python hello.py` and read the output."))
    monkeypatch.setattr(course_service, "validate_lesson_quality", lambda *args, **kwargs: {"passed": True, "issues": []})
    monkeypatch.setattr(course_service, "generate_validated_questions_for_lesson", fake_questions)
    monkeypatch.setattr(course_service, "save_module_content", fake_save)

    result = await course_service.generate_module_lesson("course-python", "m1", "s-python-videos")

    video_search.assert_awaited_once()
    assert saved["questions_after_content"] is True
    assert saved["videos"][0]["embed_url"] == "https://www.youtube.com/embed/abc123"
    assert result["videos"][0]["source"] == "youtube"


@pytest.mark.asyncio
async def test_module_lesson_response_includes_videos(monkeypatch):
    course = {
        "id": "course-1",
        "student_id": "s1",
        "topic": "Python",
        "goal": "learn programming",
        "pace": "medium",
    }
    module = {
        "id": "m8",
        "title": "Control Structures: for loops, and while loops",
        "concept": "control structures",
        "concepts_taught": ["for loops", "while loops"],
        "question_scope": ["for loops", "while loops"],
        "content_markdown": "",
    }
    video = {
        "title": "For Loops and While Loops in Python",
        "url": "https://www.youtube.com/watch?v=abc123",
        "embed_url": "https://www.youtube.com/embed/abc123",
        "source": "youtube",
        "reason": "Relevant beginner explanation for this module.",
    }
    saved = {}

    async def fake_get_module(course_id, module_id):
        if saved.get("content"):
            return {**module, "content_markdown": saved["content"], "lesson_videos": saved["videos"]}
        return module

    async def fake_save(course_id, module_id, content, questions, videos=None):
        saved["content"] = content
        saved["questions"] = questions
        saved["videos"] = videos

    monkeypatch.setattr(course_service, "get_course", AsyncMock(return_value=course))
    monkeypatch.setattr(course_service, "get_course_module", fake_get_module)
    monkeypatch.setattr(course_service, "set_module_status", AsyncMock(return_value=None))
    monkeypatch.setattr(course_service, "list_course_modules", AsyncMock(return_value=[module]))
    monkeypatch.setattr(course_service, "adaptation_context_for_module", AsyncMock(return_value={}))
    monkeypatch.setattr(course_service, "retrieve", AsyncMock(return_value=[]))
    monkeypatch.setattr(course_service, "generate", AsyncMock(return_value="# Lesson\nFor loops repeat over items. While loops repeat while a condition is true."))
    monkeypatch.setattr(course_service, "validate_lesson_quality", lambda *args, **kwargs: {"passed": True, "issues": []})
    monkeypatch.setattr(course_service, "generate_validated_questions_for_lesson", AsyncMock(return_value=([], {"status": "passed"})))
    monkeypatch.setattr(course_service, "search_youtube_videos_for_module", AsyncMock(return_value=[video]))
    monkeypatch.setattr(course_service, "save_module_content", fake_save)

    result = await course_service.generate_module_lesson("course-1", "m8", "s1")

    assert saved["videos"] == [video]
    assert result["videos"] == [video]


def test_frontend_renders_video_if_embed_url_present():
    component_path = os.path.join(
        os.path.dirname(__file__),
        "../../edumind_frontend/src/components/lesson/VideoResources.jsx",
    )
    page_path = os.path.join(
        os.path.dirname(__file__),
        "../../edumind_frontend/src/pages/ModuleReaderPage.jsx",
    )

    component_source = open(component_path, encoding="utf-8").read()
    page_source = open(page_path, encoding="utf-8").read()

    assert "<iframe" in component_source
    assert "src={video.embed_url}" in component_source
    assert "allowFullScreen" in component_source
    assert "getLessonVideos" in page_source
    assert "<VideoResources videos={videos} />" in page_source


@pytest.mark.asyncio
async def test_youtube_search_failure_does_not_break_lesson_generation(monkeypatch):
    course = {
        "id": "course-1",
        "student_id": "s1",
        "topic": "Python",
        "goal": "learn programming",
        "pace": "fast",
    }
    module = {
        "id": "m1",
        "title": "Running Your First Python Program",
        "concept": "running a Python file",
        "description": "run a first .py file",
        "prerequisites": [],
        "content_markdown": "",
    }
    saved = {}

    async def fake_get_module(course_id, module_id):
        if saved.get("content"):
            return {**module, "content_markdown": saved["content"]}
        return module

    async def fake_save(course_id, module_id, content, questions, videos=None):
        saved["content"] = content
        saved["questions"] = questions
        saved["videos"] = videos

    monkeypatch.setattr(course_service, "get_course", AsyncMock(return_value=course))
    monkeypatch.setattr(course_service, "get_course_module", fake_get_module)
    monkeypatch.setattr(course_service, "set_module_status", AsyncMock(return_value=None))
    monkeypatch.setattr(course_service, "list_course_modules", AsyncMock(return_value=[module]))
    monkeypatch.setattr(course_service, "adaptation_context_for_module", AsyncMock(return_value={}))
    monkeypatch.setattr(course_service, "retrieve", AsyncMock(return_value=[]))
    monkeypatch.setattr(course_service, "generate", AsyncMock(return_value="# Lesson\nRun the file with `python hello.py` and read the output."))
    monkeypatch.setattr(course_service, "validate_lesson_quality", lambda *args, **kwargs: {"passed": True, "issues": []})
    monkeypatch.setattr(course_service, "generate_validated_questions_for_lesson", AsyncMock(return_value=([], {"status": "passed"})))
    monkeypatch.setattr(course_service, "search_youtube_videos_for_module", AsyncMock(side_effect=RuntimeError("search down")))
    monkeypatch.setattr(course_service, "save_module_content", fake_save)

    result = await course_service.generate_module_lesson("course-1", "m1", "s1")

    assert saved["content"]
    assert saved["videos"] == []
    assert result["videos"] == []


@pytest.mark.asyncio
async def test_lazy_module_generation_saves_content_and_questions(monkeypatch):
    course = {
        "id": "course-1",
        "student_id": "s1",
        "topic": "Calculus",
        "goal": "gradient descent",
        "pace": "fast",
    }
    module = {
        "id": "m1",
        "title": "Derivatives",
        "concept": "derivatives",
        "description": "derivatives as local slope",
        "prerequisites": [],
        "content_markdown": "",
    }
    saved = {}

    async def fake_get_course(course_id, student_id=None):
        return course

    async def fake_get_module(course_id, module_id):
        if saved.get("content"):
            return {**module, "content_markdown": saved["content"]}
        return module

    async def fake_status(*args, **kwargs):
        return None

    async def fake_adaptation(*args, **kwargs):
        return {"weak_concepts": []}

    async def fake_retrieve(*args, **kwargs):
        return ["A derivative measures local slope."]

    async def fake_generate(*args, **kwargs):
        return (
            "Calculus uses derivatives to measure local slope. "
            "A derivative measures local slope, and that slope tells gradient descent which way to move. "
            "In this module, derivatives connect a changing function to an update direction. "
            "For gradient descent, the derivative is useful because it points toward how the loss changes. "
            "If the derivative is positive, moving the parameter down can reduce the value. "
            "If the derivative is negative, moving the parameter up can reduce the value. "
            "A learning rate controls how large each step should be. "
            "The key practice is to identify the function, compute or interpret the derivative, and explain what the slope says. "
            "This lesson stays focused on derivatives as the calculus idea needed for gradient descent. "
            "A simple check is to ask whether the slope is steep, flat, positive, or negative before choosing an update. "
            "Worked example: if a loss curve rises as a parameter increases, the derivative is positive at that point. "
            "Gradient descent responds by stepping in the opposite direction so the loss can decrease. "
            "If the curve is nearly flat, the derivative is small and the update should usually be smaller. "
            "This is why derivatives are not just a formula; they are a decision signal. "
            "Common mistake: memorizing derivative rules without explaining what the sign and size mean for the update. "
            "Practice by describing the slope first, then choosing the movement direction."
        )

    async def fake_save(course_id, module_id, content, questions, videos=None):
        saved["content"] = content
        saved["questions"] = questions
        saved["videos"] = videos

    monkeypatch.setattr(course_service, "get_course", fake_get_course)
    monkeypatch.setattr(course_service, "get_course_module", fake_get_module)
    monkeypatch.setattr(course_service, "set_module_status", fake_status)
    monkeypatch.setattr(course_service, "adaptation_context_for_module", fake_adaptation)
    monkeypatch.setattr(course_service, "retrieve", fake_retrieve)
    monkeypatch.setattr(course_service, "generate", fake_generate)
    monkeypatch.setattr(course_service, "save_module_content", fake_save)
    monkeypatch.setattr(course_service, "search_youtube_videos_for_module", AsyncMock(return_value=[]))

    result = await course_service.generate_module_lesson("course-1", "m1", "s1")

    assert "derivative measures local slope" in saved["content"]
    assert len(saved["questions"]) == 2
    assert result["content_markdown"] == saved["content"]


@pytest.mark.asyncio
async def test_lazy_module_generation_retries_questions_then_saves(monkeypatch):
    course = {
        "id": "course-1",
        "student_id": "s1",
        "topic": "Calculus",
        "goal": "gradient descent",
        "pace": "fast",
    }
    module = {
        "id": "m1",
        "title": "Derivatives",
        "concept": "derivatives",
        "description": "derivatives as local slope",
        "prerequisites": [],
        "question_scope": ["derivatives"],
        "content_markdown": "",
    }
    lesson = (
        "Calculus uses derivatives to measure local slope. "
        "A derivative measures local slope, and that slope tells gradient descent which way to move. "
        "In this module, derivatives connect a changing function to an update direction. "
        "For gradient descent, the derivative is useful because it points toward how the loss changes. "
        "If the derivative is positive, moving the parameter down can reduce the value. "
        "If the derivative is negative, moving the parameter up can reduce the value. "
        "A learning rate controls how large each step should be. "
        "The key practice is to identify the function, compute or interpret the derivative, and explain what the slope says. "
        "This lesson stays focused on derivatives as the calculus idea needed for gradient descent. "
        "A simple check is to ask whether the slope is steep, flat, positive, or negative before choosing an update. "
        "Worked example: if a loss curve rises as a parameter increases, the derivative is positive at that point. "
        "Gradient descent responds by stepping in the opposite direction so the loss can decrease. "
        "If the curve is nearly flat, the derivative is small and the update should usually be smaller. "
        "This is why derivatives are not just a formula; they are a decision signal. "
        "Common mistake: memorizing derivative rules without explaining what the sign and size mean for the update. "
        "Practice by describing the slope first, then choosing the movement direction."
    )
    saved = {}
    validations = []

    async def fake_get_course(course_id, student_id=None):
        return course

    async def fake_get_module(course_id, module_id):
        if saved.get("content"):
            return {**module, "content_markdown": saved["content"]}
        return module

    async def fake_status(*args, **kwargs):
        return None

    async def fake_adaptation(*args, **kwargs):
        return {"weak_concepts": []}

    async def fake_retrieve(*args, **kwargs):
        return []

    async def fake_generate(messages, **kwargs):
        prompt = messages[0]["content"]
        if course_service.STRICT_QUESTION_RETRY_INSTRUCTION in prompt:
            return json.dumps({
                "questions": [{
                    "question_text": "What does the lesson say a derivative measures?",
                    "expected_answer": "A derivative measures local slope",
                    "source_quote": "A derivative measures local slope",
                    "concepts_tested": ["derivatives"],
                    "source_section": "Lesson",
                    "is_answerable_from_lesson": True,
                    "difficulty": "simple",
                }]
            })
        return lesson

    def fake_question_validation(questions, content, module_arg):
        validations.append(questions)
        if len(validations) == 1:
            return {"passed": False, "issues": ["bad source quote"]}
        return {"passed": True, "issues": []}

    async def fake_save(course_id, module_id, content, questions, videos=None):
        saved["content"] = content
        saved["questions"] = questions
        saved["videos"] = videos

    monkeypatch.setattr(course_service, "get_course", fake_get_course)
    monkeypatch.setattr(course_service, "get_course_module", fake_get_module)
    monkeypatch.setattr(course_service, "set_module_status", fake_status)
    monkeypatch.setattr(course_service, "adaptation_context_for_module", fake_adaptation)
    monkeypatch.setattr(course_service, "retrieve", fake_retrieve)
    monkeypatch.setattr(course_service, "generate", fake_generate)
    monkeypatch.setattr(course_service, "validate_questions_grounded", fake_question_validation)
    monkeypatch.setattr(course_service, "save_module_content", fake_save)
    monkeypatch.setattr(course_service, "search_youtube_videos_for_module", AsyncMock(return_value=[]))

    result = await course_service.generate_module_lesson("course-1", "m1", "s1")

    assert saved["content"] == lesson
    assert saved["questions"][0]["question_text"] == "What does the lesson say a derivative measures?"
    assert saved["questions"][0]["id"] == "course-1:m1:q1"
    assert len(validations) == 2
    assert result["questions"] == saved["questions"]


@pytest.mark.asyncio
async def test_lazy_module_generation_saves_content_without_questions_after_retry_failure(monkeypatch):
    course = {
        "id": "course-1",
        "student_id": "s1",
        "topic": "Calculus",
        "goal": "gradient descent",
        "pace": "fast",
    }
    module = {
        "id": "m1",
        "title": "Derivatives",
        "concept": "derivatives",
        "description": "derivatives as local slope",
        "prerequisites": [],
        "content_markdown": "",
    }
    lesson = (
        "Calculus uses derivatives to measure local slope. "
        "A derivative measures local slope, and that slope tells gradient descent which way to move. "
        "In this module, derivatives connect a changing function to an update direction. "
        "For gradient descent, the derivative is useful because it points toward how the loss changes. "
        "If the derivative is positive, moving the parameter down can reduce the value. "
        "If the derivative is negative, moving the parameter up can reduce the value. "
        "A learning rate controls how large each step should be. "
        "The key practice is to identify the function, compute or interpret the derivative, and explain what the slope says. "
        "This lesson stays focused on derivatives as the calculus idea needed for gradient descent. "
        "A simple check is to ask whether the slope is steep, flat, positive, or negative before choosing an update. "
        "Worked example: if a loss curve rises as a parameter increases, the derivative is positive at that point. "
        "Gradient descent responds by stepping in the opposite direction so the loss can decrease. "
        "If the curve is nearly flat, the derivative is small and the update should usually be smaller. "
        "This is why derivatives are not just a formula; they are a decision signal. "
        "Common mistake: memorizing derivative rules without explaining what the sign and size mean for the update. "
        "Practice by describing the slope first, then choosing the movement direction."
    )
    saved = {}

    async def fake_get_course(course_id, student_id=None):
        return course

    async def fake_get_module(course_id, module_id):
        if saved.get("content"):
            return {**module, "content_markdown": saved["content"]}
        return module

    async def fake_status(*args, **kwargs):
        return None

    async def fake_adaptation(*args, **kwargs):
        return {"weak_concepts": []}

    async def fake_retrieve(*args, **kwargs):
        return []

    async def fake_generate(messages, **kwargs):
        prompt = messages[0]["content"]
        if course_service.STRICT_QUESTION_RETRY_INSTRUCTION in prompt:
            return json.dumps({
                "questions": [{
                    "question_text": "What external fact should be used?",
                    "expected_answer": "Use outside knowledge.",
                    "source_quote": "not in the lesson",
                    "concepts_tested": ["derivatives"],
                    "source_section": "Lesson",
                }]
            })
        return lesson

    def fake_question_validation(*args, **kwargs):
        return {"passed": False, "issues": ["question is not grounded"]}

    async def fake_save(course_id, module_id, content, questions, videos=None):
        saved["content"] = content
        saved["questions"] = questions
        saved["videos"] = videos

    monkeypatch.setattr(course_service, "get_course", fake_get_course)
    monkeypatch.setattr(course_service, "get_course_module", fake_get_module)
    monkeypatch.setattr(course_service, "set_module_status", fake_status)
    monkeypatch.setattr(course_service, "adaptation_context_for_module", fake_adaptation)
    monkeypatch.setattr(course_service, "retrieve", fake_retrieve)
    monkeypatch.setattr(course_service, "generate", fake_generate)
    monkeypatch.setattr(course_service, "validate_questions_grounded", fake_question_validation)
    monkeypatch.setattr(course_service, "save_module_content", fake_save)
    monkeypatch.setattr(course_service, "search_youtube_videos_for_module", AsyncMock(return_value=[]))

    result = await course_service.generate_module_lesson("course-1", "m1", "s1")

    assert saved["content"] == lesson
    assert saved["questions"] == []
    assert result["questions"] == []
    assert result["content_markdown"] == lesson


@pytest.mark.asyncio
async def test_lazy_module_generation_lesson_quality_failure_blocks_save(monkeypatch):
    course = {
        "id": "course-1",
        "student_id": "s1",
        "topic": "Calculus",
        "goal": "gradient descent",
        "pace": "fast",
    }
    module = {
        "id": "m1",
        "title": "Derivatives",
        "concept": "derivatives",
        "description": "derivatives as local slope",
        "prerequisites": [],
        "content_markdown": "",
    }
    saved = {"called": False}

    async def fake_get_course(course_id, student_id=None):
        return course

    async def fake_get_module(course_id, module_id):
        return module

    async def fake_status(*args, **kwargs):
        return None

    async def fake_adaptation(*args, **kwargs):
        return {"weak_concepts": []}

    async def fake_retrieve(*args, **kwargs):
        return []

    async def fake_generate(*args, **kwargs):
        return "Too short."

    def fake_lesson_validation(*args, **kwargs):
        return {"passed": False, "issues": ["Lesson is too short to teach the module."]}

    async def fake_save(*args, **kwargs):
        saved["called"] = True

    monkeypatch.setattr(course_service, "get_course", fake_get_course)
    monkeypatch.setattr(course_service, "get_course_module", fake_get_module)
    monkeypatch.setattr(course_service, "set_module_status", fake_status)
    monkeypatch.setattr(course_service, "adaptation_context_for_module", fake_adaptation)
    monkeypatch.setattr(course_service, "retrieve", fake_retrieve)
    monkeypatch.setattr(course_service, "generate", fake_generate)
    monkeypatch.setattr(course_service, "validate_lesson_quality", fake_lesson_validation)
    monkeypatch.setattr(course_service, "save_module_content", fake_save)

    with pytest.raises(ValueError, match="Lesson generation failed validation"):
        await course_service.generate_module_lesson("course-1", "m1", "s1")

    assert saved["called"] is False


@pytest.mark.asyncio
async def test_module_chat_saves_doubt_and_skill_signal(monkeypatch):
    course = {
        "id": "course-1",
        "student_id": "s1",
        "topic": "Optimization",
        "goal": "train ML models",
        "pace": "medium",
    }
    module = {
        "id": "m1",
        "title": "Gradient Descent",
        "concept": "gradient descent",
        "description": "gradient descent as stepwise loss reduction",
        "prerequisites": ["derivatives"],
        "content_markdown": "Gradient descent uses derivatives to choose an update direction.",
    }
    calls = {"messages": 0, "doubts": 0, "skills": 0}

    async def fake_get_course(course_id, student_id=None):
        return course

    async def fake_get_module(course_id, module_id):
        return module

    async def fake_history(*args):
        return []

    async def fake_retrieve(*args, **kwargs):
        return []

    async def fake_generate(*args, **kwargs):
        return "Derivatives point to the local slope, so they tell the update which direction reduces loss."

    async def fake_record_message(*args, **kwargs):
        calls["messages"] += 1

    async def fake_record_doubt(*args, **kwargs):
        calls["doubts"] += 1

    async def fake_skill(*args, **kwargs):
        calls["skills"] += 1

    async def fake_meta(*args, **kwargs):
        return None

    monkeypatch.setattr(course_service, "get_course", fake_get_course)
    monkeypatch.setattr(course_service, "get_course_module", fake_get_module)
    monkeypatch.setattr(course_service, "list_module_chat_history", fake_history)
    monkeypatch.setattr(course_service, "retrieve", fake_retrieve)
    monkeypatch.setattr(course_service, "generate", fake_generate)
    monkeypatch.setattr(course_service, "record_module_chat_message", fake_record_message)
    monkeypatch.setattr(course_service, "record_doubt", fake_record_doubt)
    monkeypatch.setattr(course_service, "upsert_student_skill", fake_skill)
    monkeypatch.setattr(course_service, "update_metacognition_from_doubt", fake_meta)

    result = await course_service.answer_module_chat(
        "course-1",
        "m1",
        "s1",
        "I don't understand why gradient descent uses derivatives",
    )

    assert result["saved"] is True
    assert result["doubt_type"] in {"conceptual", "prerequisite gap"}
    assert calls == {"messages": 2, "doubts": 1, "skills": 1}


def test_app_keeps_legacy_session_routes_and_adds_course_routes():
    from app.api import app

    paths = {route.path for route in app.routes}
    assert "/session/start" in paths
    assert "/session/stream/{session_id}" in paths
    assert "/api/courses" in paths
    assert "/api/courses/create-intent" in paths
    assert "/api/courses/{course_id}" in paths
    assert "/api/courses/{course_id}/roadmap" in paths
    assert any(
        route.path == "/api/courses/{course_id}" and "DELETE" in getattr(route, "methods", set())
        for route in app.routes
    )


@pytest.mark.asyncio
async def test_delete_course_endpoint_calls_delete_with_student_scope(monkeypatch):
    calls = {}

    async def fake_delete(course_id, student_id=None):
        calls["course_id"] = course_id
        calls["student_id"] = student_id
        return True

    monkeypatch.setattr(course_api, "delete_course", fake_delete)

    result = await course_api.delete_course_endpoint("course-1", student_id="student-1")

    assert result == {"deleted": True, "course_id": "course-1"}
    assert calls == {"course_id": "course-1", "student_id": "student-1"}


@pytest.mark.asyncio
async def test_delete_course_endpoint_raises_404_when_missing(monkeypatch):
    async def fake_delete(course_id, student_id=None):
        return False

    monkeypatch.setattr(course_api, "delete_course", fake_delete)

    with pytest.raises(HTTPException) as exc:
        await course_api.delete_course_endpoint("missing-course", student_id="student-1")

    assert exc.value.status_code == 404
