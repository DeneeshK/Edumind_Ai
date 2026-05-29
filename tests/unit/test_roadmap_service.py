from __future__ import annotations

import pytest

from core.roadmap_service import CourseRoadmapService


pytestmark = pytest.mark.unit


def test_roadmap_service_builds_ordered_module_timeline(sample_course):
    roadmap = CourseRoadmapService().build(
        sample_course,
        sample_course["modules"],
        sample_course["personalization_profile"],
        {},
    )

    assert roadmap["course_id"] == sample_course["id"]
    assert [item["module_id"] for item in roadmap["module_timeline"]] == ["M1", "M2"]
    assert roadmap["module_timeline"][0]["recommended_next"] is True
    assert roadmap["recommended_schedule"]


def test_roadmap_service_keeps_alias_like_topics_separate():
    course = {
        "id": "course-js",
        "topic": "JavaScript",
        "goal": "Build browser apps",
        "pace": "fast",
    }
    modules = [
        {
            "id": "M1",
            "title": "JavaScript Variables",
            "concept": "JavaScript variables",
            "estimated_minutes": 25,
        }
    ]
    profile = {
        "topic": "JavaScript",
        "learning_goal": "Build browser apps",
        "target_context": "frontend development",
        "pace": "fast",
        "known_concepts": ["HTML"],
    }

    roadmap = CourseRoadmapService().build(course, modules, profile, {})

    assert "JavaScript" in roadmap["title"]
    assert "NumPy" not in str(roadmap)
