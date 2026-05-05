"""
tests/test_phase4.py
Phase 4 — BaseAgent verification:
  - build_tool() produces valid Groq spec
  - Subclass with 2 tools runs correctly via tool_call_loop
  - Terminal tool exits loop and returns correct args
  - _log_decision() records to session_decisions

Run: pytest tests/test_phase4.py -v -s
"""

import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

import pytest
from agents.base_agent import BaseAgent
from core.student_model import StudentState


# ── Toy subclass ──────────────────────────────────────────────────────────────

class GreeterAgent(BaseAgent):
    NAME = "greeter_agent"
    TERMINAL_TOOL = "conclude"

    def __init__(self, state: StudentState):
        super().__init__(state)
        self.TOOLS = [
            self.build_tool(
                name="get_student_name",
                description="Retrieve the student's name to personalise the greeting.",
                properties={
                    "student_id": {"type": "string", "description": "The student ID"},
                },
                required=["student_id"],
            ),
            self.build_tool(
                name="conclude",
                description="Conclude the greeting session. Call this after greeting.",
                properties={
                    "greeting": {"type": "string", "description": "The final greeting message"},
                    "ready":    {"type": "boolean", "description": "Is the student ready?"},
                },
                required=["greeting", "ready"],
            ),
        ]

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        if tool_name == "get_student_name":
            return f"Student name: {self.state.name or 'Arjun'}"
        return "done"

    def run_greeting(self) -> dict:
        return self.run(
            system=(
                "You are a friendly session starter. "
                "First call get_student_name to retrieve the student's name. "
                "Then call conclude with a personalised greeting and ready=true."
            ),
            user_message=f"Start a session for student_id={self.state.student_id}",
        )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def student():
    return StudentState(
        student_id="test_stu_001",
        name="Arjun",
        domain="machine learning",
        goal="understand transformers",
        pace="medium",
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_build_tool_structure():
    """build_tool() must produce a valid Groq tool spec."""
    tool = BaseAgent.build_tool(
        name="my_tool",
        description="A test tool",
        properties={
            "score": {"type": "number", "description": "A score"},
            "label": {"type": "string", "description": "A label",
                      "enum": ["good", "bad"]},
        },
        required=["score"],
    )
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "my_tool"
    assert "score" in tool["function"]["parameters"]["properties"]
    assert "label" in tool["function"]["parameters"]["properties"]
    assert tool["function"]["parameters"]["required"] == ["score"]
    print("\n✅ build_tool() produces valid Groq spec")


def test_greeter_agent_runs(student):
    """GreeterAgent should call get_student_name then conclude (terminal)."""
    agent = GreeterAgent(student)
    result = agent.run_greeting()

    print(f"\n✅ GreeterAgent result: {result}")
    assert "greeting" in result, f"Expected 'greeting' in result, got: {result}"
    assert "ready" in result, f"Expected 'ready' in result, got: {result}"
    assert isinstance(result["greeting"], str)
    assert len(result["greeting"]) > 0
    print(f"   greeting = {result['greeting']}")
    print(f"   ready    = {result['ready']}")


def test_log_decision_records(student):
    """_log_decision() should append to state.session_decisions."""
    student.start_session()
    agent = GreeterAgent(student)

    agent._log_decision(
        action="MOVE_FORWARD",
        reason="mastery cleared threshold",
        payload={"concept": "dot_product", "mastery": 0.82},
    )

    assert len(student.session_decisions) == 1
    d = student.session_decisions[0]
    assert d.action == "MOVE_FORWARD"
    assert "mastery" in d.metacognition_updates
    print(f"\n✅ _log_decision() recorded: {d.action} — {d.reason}")


def test_current_module_returns_none_without_curriculum(student):
    """_current_module() returns None when no curriculum is set."""
    agent = GreeterAgent(student)
    assert agent._current_module() is None
    print("\n✅ _current_module() returns None with no curriculum")


def test_student_context_contains_key_fields(student):
    """_student_context() must contain domain, pace, style."""
    agent = GreeterAgent(student)
    ctx = agent._student_context()
    assert "machine learning" in ctx
    assert "medium" in ctx
    assert "formal" in ctx  # default preferred_style
    print(f"\n✅ _student_context() output:\n{ctx}")
