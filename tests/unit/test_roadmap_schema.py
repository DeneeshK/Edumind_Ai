from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.student_model import MasterRoadmap, Module, RoadmapStep


pytestmark = pytest.mark.unit


def test_valid_master_roadmap_preserves_step_order():
    roadmap = MasterRoadmap(
        course_id="course-1",
        topic="Python",
        goal="Learn Python fundamentals",
        steps=[
            RoadmapStep(
                step_id="M1",
                title="Values and Variables",
                concept_cluster="variables",
                subtopics=["values", "assignment"],
                estimated_minutes=30,
            ),
            RoadmapStep(
                step_id="M2",
                title="Control Flow",
                concept_cluster="loops",
                prerequisites=["M1"],
                estimated_minutes=40,
            ),
        ],
    )

    assert [step.step_id for step in roadmap.steps] == ["M1", "M2"]
    assert roadmap.steps[1].prerequisites == ["M1"]


def test_missing_required_roadmap_step_fields_fail_clearly():
    with pytest.raises(ValidationError) as exc:
        RoadmapStep(title="Missing ID", concept_cluster="variables")

    assert "step_id" in str(exc.value)


def test_module_schema_accepts_m_style_ids_and_optional_metadata():
    module = Module(
        id="M3",
        title="Functions",
        concept="functions",
        domain_framing="functions as reusable Python behavior",
        prerequisites=["M1", "M2"],
        estimated_minutes=45,
        depth_level="standard",
    )

    assert module.id == "M3"
    assert module.must_teach == []
    assert module.roadmap_step_id == ""


def test_module_schema_rejects_invalid_depth_level():
    with pytest.raises(ValidationError):
        Module(
            id="M4",
            title="Invalid",
            concept="invalid",
            domain_framing="invalid",
            prerequisites=[],
            estimated_minutes=10,
            depth_level="expert",
        )
