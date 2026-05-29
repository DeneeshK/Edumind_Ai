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
    StudentState, EvaluationReport, CurriculumPlan, Module, AdaptationDecision
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

@pytest.mark.asyncio
async def test_adaptation_move_forward(student):
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
    engine.run = AsyncMock(return_value={
        "action": "MOVE_FORWARD",
        "reason": "Mastery is above threshold",
        "metacognition_updates": {},
    })
    decision = await engine.decide(report)

    print(f"\n✅ AdaptationEngine decision: {decision.action} — {decision.reason}")
    assert decision.action in [
        "MOVE_FORWARD", "MOVE_FORWARD_WITH_FLAG", "COMPRESS"
    ], f"Expected forward action, got {decision.action}"


@pytest.mark.asyncio
async def test_adaptation_escalates_after_3_reteach(student):
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
    engine.run = AsyncMock(return_value={
        "action": "RETEACH",
        "reason": "Mocked low mastery reteach decision",
        "metacognition_updates": {},
    })
    decision = await engine.decide(report)

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


@pytest.mark.asyncio
async def test_tutor_records_delivered_module_content(student):
    """Delivered lesson text should be stored for grounded evaluator questions."""
    lesson = (
        "The module says dot product measures similarity between embedding vectors. "
        "A larger dot product means the vectors point in more similar directions."
    )
    tutor = TutorAgent(
        student,
        emit_fn=AsyncMock(return_value=None),
        ask_fn=AsyncMock(return_value=""),
    )

    result = await tutor._execute_tool(
        "deliver_lesson",
        {"lesson_text": lesson, "style_used": "formal"},
    )

    assert "Lesson delivered" in result
    assert lesson in student.get_module_content("m1")


def test_tutor_generic_fallback_has_planned_lesson_structure():
    """Fallback lessons must follow the domain-agnostic five-part structure."""
    state = StudentState(
        student_id="generic_fallback_001",
        name="Deneesh",
        domain="machine learning",
        goal="build practical AI apps",
        pace="fast",
    )
    state.curriculum = CurriculumPlan(
        topic="attention mechanisms",
        domain="machine learning",
        goal="build practical AI apps",
        modules=[
            Module(
                id="m1",
                title="Attention Scores",
                concept="attention scores",
                domain_framing="attention scores as relevance signals in AI apps",
                prerequisites=[],
                estimated_minutes=10,
                depth_level="surface",
            )
        ],
    )

    lesson = TutorAgent(state)._fallback_lesson_text(state.curriculum.modules[0])

    assert "Core definition" in lesson
    assert "Key rule or mechanism" in lesson
    assert "Worked example" in lesson
    assert "Why this matters" in lesson
    assert "Connection" in lesson
    assert "attention scores" in lesson
    assert "is the main concept for this module" not in lesson


def test_topic_specific_scope_guards_do_not_leak_into_ai_course(student):
    """Evaluator scope control should stay generic for AI/software courses."""
    student.curriculum = CurriculumPlan(
        topic="AI product design",
        domain="artificial intelligence",
        goal="build practical AI apps",
        modules=[
            Module(
                id="m1",
                title="First Law of Product Feedback",
                concept="first law of product feedback",
                domain_framing="feedback loops for AI product iteration",
                prerequisites=[],
                estimated_minutes=12,
                depth_level="standard",
            )
        ],
    )

    module = student.curriculum.modules[0]
    lesson = TutorAgent(student)._fallback_lesson_text(module)
    evaluator = EvaluatorAgent(student)

    assert evaluator._question_out_of_scope_reason("How does feedback improve an AI app?") is None
    assert "current module concept" in evaluator._module_boundary_rules(module)
    assert "feedback loops" in lesson


@pytest.mark.asyncio
async def test_orchestrator_emits_curriculum_overview(student):
    """The UI stream should receive the full module list as soon as it is built."""
    emitted = []

    async def emit(text):
        emitted.append(text)

    orchestrator = OrchestratorAgent(student, emit_fn=emit, ask_fn=AsyncMock())

    await orchestrator._emit_curriculum_overview()

    assert any("Curriculum built: 2 modules" in item for item in emitted)
    assert any("1. Dot Product" in item for item in emitted)
    assert any("2. Attention Mechanism" in item for item in emitted)


@pytest.mark.asyncio
async def test_orchestrator_reteaches_then_advances(student, monkeypatch):
    """Module loop should reteach without advancing, then move forward when mastery passes."""
    emitted = []
    taught = []
    decisions = [
        AdaptationDecision(
            action="RETEACH",
            reason="Below threshold; reteach once.",
            style_for_reteach="example_first",
        ),
        AdaptationDecision(
            action="MOVE_FORWARD",
            reason="Mastery passed after reteach.",
        ),
        AdaptationDecision(
            action="HOLD",
            reason="Stop test after first module advances.",
        ),
    ]

    async def emit(text):
        emitted.append(text)

    async def ask(question, **kwargs):
        return "4" if kwargs.get("is_confidence") else ""

    async def fake_teach(self):
        module = self.state.curriculum.modules[self.state.curriculum.current_index]
        taught.append(module.id)
        return {"style_used": "formal", "fatigue_detected": "no", "doubt_count": 0}

    async def fake_evaluate(self, concept, confidence):
        return EvaluationReport(
            concept=concept,
            session_id=self.state.session_id,
            correctness_score=0.8,
            depth_score=0.8,
            mastery_score=0.8,
            misconception_type=None,
            misconception_detail="",
            confidence_stated=confidence,
            calibration_delta=0.0,
            questions_asked=2,
            recommended_action="MOVE_FORWARD",
        )

    async def fake_decide(self, report):
        return decisions.pop(0)

    async def fake_gap_analysis(self):
        return None

    monkeypatch.setattr(TutorAgent, "teach", fake_teach)
    monkeypatch.setattr(EvaluatorAgent, "evaluate", fake_evaluate)
    monkeypatch.setattr(AdaptationEngine, "decide", fake_decide)
    monkeypatch.setattr(AdaptationEngine, "run_gap_analysis", fake_gap_analysis)

    orchestrator = OrchestratorAgent(student, emit_fn=emit, ask_fn=ask)
    completed = await orchestrator._run_module_loop()

    assert taught[:2] == ["m1", "m1"]
    assert completed == ["m1"]
    assert student.curriculum.current_index == 1
    assert any("RETEACH" in item for item in emitted)
    assert any("MOVE_FORWARD" in item for item in emitted)


@pytest.mark.asyncio
async def test_evaluator_rejects_ungrounded_questions(student):
    """Evaluator questions must cite exact text from the current module content."""
    student.record_module_content(
        "m1",
        "The module says dot product measures similarity between embedding vectors.",
    )
    ask = AsyncMock(return_value="It measures similarity.")
    evaluator = EvaluatorAgent(student, ask_fn=ask)

    rejected = await evaluator._execute_tool(
        "ask_question",
        {
            "question": "What does softmax do to attention scores?",
            "question_type": "recall",
            "source_quote": "softmax turns scores into probabilities",
        },
    )

    assert "Question rejected" in rejected
    ask.assert_not_called()
    assert evaluator._qa_log == []

    rejected_topic_drift = await evaluator._execute_tool(
        "ask_question",
        {
            "question": "What does softmax do to attention scores?",
            "question_type": "recall",
            "source_quote": "dot product measures similarity",
        },
    )

    assert "Question rejected" in rejected_topic_drift
    ask.assert_not_called()
    assert evaluator._qa_log == []

    accepted = await evaluator._execute_tool(
        "ask_question",
        {
            "question": "According to the module, what does dot product measure?",
            "question_type": "recall",
            "source_quote": "dot product measures similarity",
        },
    )

    assert "Student answered" in accepted
    assert evaluator._qa_log[0]["source_quote"] == "dot product measures similarity"


# ── CurriculumArchitect ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_curriculum_architect_buffers_modules(student):
    """_execute_tool('add_module') should buffer modules correctly."""
    architect = CurriculumArchitectAgent(student)
    result = await architect._execute_tool("add_module", {
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


@pytest.mark.asyncio
async def test_metacognition_check_prerequisites_fixed(student):
    """check_prerequisites detects missing prereqs for m2 (attention_mechanism)."""
    student.concept_mastery["dot_product"] = 0.40  # below threshold
    student.curriculum.current_index = 1  # advance to m2 which has dot_product as prereq
    engine = AdaptationEngine(student)
    result = await engine._execute_tool("check_prerequisites", {"concept": "attention_mechanism"})
    print(f"\n✅ Prerequisite check result:\n{result}")
    assert "dot_product" in result
    assert "❌" in result
