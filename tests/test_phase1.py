from core.student_model import (
    StudentState, MetacognitionProfile, CurriculumPlan,
    Module, EvaluationReport, AdaptationDecision,
)
from db.postgres import upsert_student
import db.postgres as pg
from dotenv import load_dotenv
import uuid
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

load_dotenv()


@pytest.mark.asyncio
async def test_pydantic_models_instantiate():
    MetacognitionProfile()
    Module(id="m1", title="Dot Product", concept="dot_product",
           domain_framing="vectors for ML", prerequisites=[],
           estimated_minutes=10, depth_level="standard")
    CurriculumPlan(
        topic="Linear Algebra",
        domain="ML",
        goal="exam",
        modules=[])
    EvaluationReport(
        concept="dot_product", session_id="s1",
        correctness_score=0.8, depth_score=0.7, mastery_score=0.76,
        confidence_stated=4, calibration_delta=0.04,
        questions_asked=3, recommended_action="MOVE_FORWARD"
    )
    AdaptationDecision(action="MOVE_FORWARD", reason="threshold cleared")
    print("\n✅ All Pydantic models instantiate cleanly.")


@pytest.mark.asyncio
async def test_student_state_round_trip():
    # Init pool inside the same event loop as the test
    await pg.init_db()

    try:
        sid = f"test_{uuid.uuid4().hex[:8]}"
        state = StudentState(
            student_id=sid, name="Test Student",
            domain="machine learning",
            goal="understand attention mechanisms",
            pace="medium",
        )
        state.start_session()
        await upsert_student(sid, state.name, state.domain, state.goal, state.pace)

        state.update_mastery("dot_product", correctness=0.85, depth=0.70)
        expected_mastery = round(0.6 * 0.85 + 0.4 * 0.70, 3)

        state.metacognition.record_style_depth("analogy", 0.88)
        state.metacognition.record_style_depth("analogy", 0.84)
        state.metacognition.record_style_depth("formal", 0.62)
        state.metacognition.record_style_depth("formal", 0.65)

        state.metacognition.update_calibration(0.25)
        state.metacognition.update_calibration(0.30)
        state.metacognition.update_calibration(0.20)

        await state.save()
        reloaded = await StudentState.load(sid)

        assert abs(
            reloaded.concept_mastery["dot_product"] -
            expected_mastery) < 0.01
        assert reloaded.metacognition.preferred_style == "analogy"
        assert reloaded.metacognition.calibration_pattern == "overconfident"

        print(f"\n✅ Round-trip passed for {sid}")
        print(f"   mastery     = {reloaded.concept_mastery['dot_product']}")
        print(f"   style       = {reloaded.metacognition.preferred_style}")
        print(f"   calibration = {reloaded.metacognition.calibration_pattern}")

    finally:
        await pg.close_db()
