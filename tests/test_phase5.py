"""
tests/test_phase5.py
Phase 5 — smoke tests for all 5 agents (no DB required for most).

Run: pytest tests/test_phase5.py -v -s
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from core.student_model import (
    StudentState, EvaluationReport, CurriculumPlan, Module
)
from agents.base_agent import BaseAgent
from agents.evaluator import EvaluatorAgent
from agents.curriculum_architect import CurriculumArchitectAgent
from agents.tutor import TutorAgent
from agents.adaptation_engine import AdaptationEngine
from agents.orchestrator import OrchestratorAgent


@pytest.fixture
def student():
    s = StudentState(
        student_id="smoke_test_001",
        name="Arjun",
        domain="machine learning",
        goal="understand transformers",
        pace="medium",
    )
    s.start_session()
    # Give a curriculum
    s.curriculum = CurriculumPlan(
        topic="Transformer Architecture",
        domain="machine learning",
        goal="understand transformers",
        modules=[
            Module(
                id="m1", title="Dot Product",
                concept="dot_product",
                domain_framing="dot product as similarity in embeddings",
                prerequisites=[], estimated_minutes=10,
                depth_level="standard",
            ),
            Module(
                id="m2", title="Attention Mechanism",
                concept="attention_mechanism",
                domain_framing="attention as selective focus in NLP",
                prerequisites=["dot_product"], estimated_minutes=15,
                depth_level="standard",
            ),
        ],
    )
    return s


# ── BaseAgent ─────────────────────────────────────────────────────────────────

def test_all_agents_instantiate(student):
    """All 5 agents must instantiate without error."""
    EvaluatorAgent(student)
    CurriculumArchitectAgent(student)
    TutorAgent(student)
    AdaptationEngine(student)
    OrchestratorAgent(student)
    print("\n✅ All 5 agents instantiate cleanly")


def test_all_agents_have_tools(student):
    """All agents must define at least 2 tools including their terminal tool."""
    agents = [
        EvaluatorAgent(student),
        CurriculumArchitectAgent(student),
        TutorAgent(student),
        AdaptationEngine(student),
        OrchestratorAgent(student),
    ]
    for agent in agents:
        assert len(agent.TOOLS) >= 2, f"{agent.NAME} has < 2 tools"
        tool_names = [t["function"]["name"] for t in agent.TOOLS]
        assert agent.TERMINAL_TOOL in tool_names, \
            f"{agent.NAME} terminal tool '{agent.TERMINAL_TOOL}' not in tools"
        print(f"   ✅ {agent.NAME}: {len(agent.TOOLS)} tools, terminal='{agent.TERMINAL_TOOL}'")
    print("✅ All agents have valid tools")


# ── AdaptationEngine ──────────────────────────────────────────────────────────

def test_adaptation_move_forward(student):
    """AdaptationEngine should recommend MOVE_FORWARD for high mastery."""
    student.update_mastery("dot_product", correctness=0.90, depth=0.85)
    report = EvaluationReport(
        concept="dot_product",
        session_id=student.session_id,
        correctness_score=0.90,
        depth_score=0.85,
        mastery_score=round(0.6*0.90 + 0.4*0.85, 3),
        confidence_stated=4,
        calibration_delta=0.06,
        questions_asked=4,
        recommended_action="MOVE_FORWARD",
    )
    engine = AdaptationEngine(student)
    decision = engine.decide(report)

    print(f"\n✅ AdaptationEngine decision: {decision.action} — {decision.reason}")
    assert decision.action in [
        "MOVE_FORWARD", "MOVE_FORWARD_WITH_FLAG", "COMPRESS"
    ], f"Expected forward action, got {decision.action}"


def test_adaptation_escalates_after_3_reteach(student):
    """After 3 reteach cycles, engine must ESCALATE regardless of LLM output."""
    student.metacognition.consecutive_reteach_count = 3
    student.update_mastery("dot_product", correctness=0.40, depth=0.30)
    report = EvaluationReport(
        concept="dot_product",
        session_id=student.session_id,
        correctness_score=0.40,
        depth_score=0.30,
        mastery_score=round(0.6*0.40 + 0.4*0.30, 3),
        confidence_stated=3,
        calibration_delta=0.0,
        questions_asked=4,
        recommended_action="RETEACH",
    )
    engine = AdaptationEngine(student)
    decision = engine.decide(report)

    print(f"\n✅ Escalation test: {decision.action} — {decision.reason}")
    assert decision.action == "ESCALATE", \
        f"Expected ESCALATE after 3 reteach cycles, got {decision.action}"


# ── EvaluatorAgent ────────────────────────────────────────────────────────────

def test_evaluator_has_7_recommended_actions(student):
    """submit_evaluation tool must have all 7 action enums."""
    evaluator = EvaluatorAgent(student)
    submit_tool = next(
        t for t in evaluator.TOOLS
        if t["function"]["name"] == "submit_evaluation"
    )
    actions = submit_tool["function"]["parameters"]["properties"][
        "recommended_action"
    ]["enum"]
    expected = {
        "MOVE_FORWARD", "MOVE_FORWARD_WITH_FLAG", "RETEACH",
        "DETOUR", "ESCALATE", "COMPRESS", "HOLD"
    }
    assert set(actions) == expected
    print(f"\n✅ EvaluatorAgent has all 7 recommended actions: {actions}")


# ── CurriculumArchitect ───────────────────────────────────────────────────────

def test_curriculum_architect_buffers_modules(student):
    """_execute_tool('add_module') should buffer modules correctly."""
    architect = CurriculumArchitectAgent(student)
    result = architect._execute_tool("add_module", {
        "id": "m1",
        "title": "Vectors",
        "concept": "vectors",
        "domain_framing": "vectors as word embeddings in NLP",
        "prerequisites": [],
        "estimated_minutes": 10,
        "depth_level": "standard",
    })
    assert len(architect._modules_buffer) == 1
    assert architect._modules_buffer[0]["concept"] == "vectors"
    print(f"\n✅ CurriculumArchitect buffered module: {result}")


# ── Metacognition updates ─────────────────────────────────────────────────────

  # below threshold


def test_metacognition_check_prerequisites_fixed(student):
    """check_prerequisites detects missing prereqs for m2 (attention_mechanism)."""
    student.concept_mastery["dot_product"] = 0.40  # below threshold
    student.curriculum.current_index = 1  # advance to m2 which has dot_product as prereq
    engine = AdaptationEngine(student)
    result = engine._execute_tool("check_prerequisites", {"concept": "attention_mechanism"})
    print(f"\n✅ Prerequisite check result:\n{result}")
    assert "dot_product" in result
    assert "❌" in result
