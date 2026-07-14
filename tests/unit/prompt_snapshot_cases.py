"""
Shared representative inputs + capture logic for prompt snapshot tests.

`capture_prompts()` returns {snapshot_name: exact_string_sent_to_or_returned_by
the live prompt code}. It is used two ways:

  * once, to GENERATE the checked-in snapshots from the ORIGINAL inline strings
    (scripts run before the registry rewire), and
  * by tests/unit/test_prompt_snapshots.py, which re-captures against the CURRENT
    (registry-backed) code and asserts equality with the checked-in snapshots.

Because both paths call the *live* functions with identical inputs, a byte-for-byte
match proves the registry extraction is render-identical. For the LLM-call prompts
(curriculum system prompts, evaluation payloads) the captured string is the exact
`system` / user `content` handed to `clients.groq_client.generate` — intercepted so
no network call and no DB access happens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


class _StopCapture(BaseException):
    """Raised by the fake generate() once the prompt has been captured.

    Subclasses BaseException (not Exception) so it propagates cleanly past the
    ``except Exception`` handlers that wrap several live generate() calls, instead
    of triggering their fallback paths (and, for finalize, subsequent DB writes).
    """


# ── Representative inputs (stable — changing these changes the snapshots) ──────

TOPIC = "Python programming"

PROFILE: dict[str, Any] = {
    "topic": "Python programming",
    "exact_subject": "Python programming",
    "learning_goal": "Build web apps with FastAPI",
    "target_context": "Backend web development",
    "learner_level": "some basic knowledge",
    "pace": "medium",
    "prior_knowledge_summary": "Knows basic HTML and did one JavaScript tutorial.",
    "known_concepts": ["HTML basics", "CSS basics"],
    "weak_concepts": ["Functions", "Scope"],
    "must_include": ["FastAPI", "Pydantic"],
    "do_not_include": ["machine learning", "pandas"],
}

CONCEPTS: list[dict[str, Any]] = [
    {"name": "Variables and Assignment", "cluster": "basics", "importance": "essential", "why_needed": "foundation"},
    {"name": "Functions", "cluster": "functions", "importance": "essential", "why_needed": "reuse"},
    {"name": "Dictionaries", "cluster": "data structures", "importance": "important", "why_needed": "mapping"},
]

RAW_MODULES: list[dict[str, Any]] = [
    {"id": "m1", "title": "Variables", "concept": "Variables and Assignment",
     "concepts_taught": ["Variables and Assignment"], "prerequisites": []},
    {"id": "m2", "title": "Functions", "concept": "Functions",
     "concepts_taught": ["Functions"], "prerequisites": ["Variables and Assignment"]},
]

# Evaluation fixtures
MOD_CTX: dict[str, Any] = {
    "title": "Functions",
    "concept": "Functions",
    "concepts_taught": ["Functions", "Parameters", "Return values"],
    "question_scope": ["Functions", "Parameters"],
    "must_teach": ["Functions"],
    "depth_level": "standard",
}
LESSON_CONTENT = (
    "# Functions\n\nA function is a reusable block of code. You define it with `def`.\n"
    "Parameters let you pass data in; `return` sends a value back.\n\n"
    "```python\ndef add(a, b):\n    return a + b\n```\n"
)
QUESTION: dict[str, Any] = {
    "id": "q1",
    "question_text": "What does the `return` statement do in a function?",
    "type": "recall",
    "concepts_tested": ["Return values"],
    "difficulty": "simple",
    "is_base_question": True,
}
ANSWER_TEXT = "It sends a value back to the caller."
PREVIOUS_ANSWERS: list[dict[str, Any]] = []

TRIGGER_ANSWER: dict[str, Any] = {
    "question_text": "What does `return` do?",
    "answer_text": "It prints the value.",
    "diagnosis": {
        "weak_concepts": ["Return values"],
        "missing_reasoning": "Confused return with print.",
        "vague_parts": "",
        "suspicious_parts": "Thinks return prints output.",
    },
}

FINALIZE_ANSWERS: list[dict[str, Any]] = [
    {"question_id": "q1", "question_text": "What does return do?",
     "answer_text": "It sends a value back.", "confidence": 4,
     "diagnosis": {"correct_concepts": ["Return values"], "weak_concepts": [],
                   "mastery_signal": "clear"}, "is_probe": False},
    {"question_id": "q2", "question_text": "What is a parameter?",
     "answer_text": "Input to a function.", "confidence": 3,
     "diagnosis": {"correct_concepts": ["Parameters"], "weak_concepts": [],
                   "mastery_signal": "uncertain"}, "is_probe": False},
]

# Lesson / question / chat fixtures
COURSE: dict[str, Any] = {
    "id": "course-1",
    "topic": "Python programming",
    "goal": "Build web apps with FastAPI",
    "pace": "medium",
    "personalization_profile": {
        "learner_level": "some basic knowledge",
        "roadmap_steps": ["First, learn variables.", "Then functions."],
        "scope_analysis": {"what_to_exclude": ["machine learning"]},
    },
}
MODULE: dict[str, Any] = {
    "id": "M2",
    "title": "Functions",
    "concept": "Functions",
    "concepts_taught": ["Functions", "Parameters", "Return values"],
    "question_scope": ["Functions", "Parameters"],
    "must_teach": ["Functions"],
    "prerequisites": ["Variables and Assignment"],
    "description": "Reusable blocks of code.",
    "depends_on_concepts": ["Variables and Assignment"],
    "lesson_requirements": ["Show a def with parameters and a return."],
    "practice_requirements": ["Write a function that adds two numbers."],
}
ADAPTATION_CONTEXT: dict[str, Any] = {
    "weak_concepts": ["Scope"],
    "doubt_concepts": ["Return values"],
    "recommended_teaching_adjustments": [
        "Student has weak mastery of: Scope. Add a short prerequisite bridge.",
    ],
    "adaptation_summary": {"example_preference": "more"},
    "recent_doubt_messages": ["Why doesn't my function return anything?"],
}
PREVIOUS_MODULES: list[dict[str, Any]] = [
    {"title": "Variables", "concepts_taught": ["Variables and Assignment"]},
]
CHAT_HISTORY: list[dict[str, Any]] = [
    {"role": "user", "content": "What is a parameter?"},
    {"role": "assistant", "content": "A parameter is an input to a function."},
]
DOUBT_MESSAGE = "I don't understand how return is different from print."
VALIDATION_ISSUES = ["Question 2 not grounded in lesson text."]


async def capture_prompts() -> dict[str, str]:
    """Return {snapshot_name: exact live prompt string} for all moved prompts."""
    import agents.curriculum_architect as ca
    import agents.evaluation_agent as ea
    from agents.curriculum_architect import CurriculumArchitectAgent
    from core.student_model import StudentState
    from core import course_service as cs
    from prompts import get_prompt

    out: dict[str, str] = {}

    # ── Curriculum system prompts (intercept generate) ────────────────────────
    state = StudentState(student_id="snapshot", name="Snap", domain="d", goal="g", pace="medium")
    agent = CurriculumArchitectAgent(state)
    captured: dict[str, str] = {}

    async def fake_generate_ca(*, messages, system=None, **_kw):
        captured["system"] = system
        captured["user"] = messages[0]["content"]
        raise _StopCapture()

    orig_ca = ca.generate
    ca.generate = fake_generate_ca
    try:
        try:
            await agent._plan_coverage(TOPIC, PROFILE)
        except _StopCapture:
            pass
        out["curriculum_coverage_planner_system"] = captured["system"]

        for pace in ("fast", "medium", "deep"):
            agent.state.pace = pace
            try:
                await agent._sequence_modules(TOPIC, {**PROFILE, "pace": pace}, CONCEPTS)
            except _StopCapture:
                pass
            out[f"curriculum_sequencer_system_{pace}"] = captured["system"]

        try:
            await agent._audit_coverage(TOPIC, PROFILE, RAW_MODULES, CONCEPTS)
        except _StopCapture:
            pass
        out["curriculum_auditor_system"] = captured["system"]
    finally:
        ca.generate = orig_ca

    # ── Evaluation prompts ────────────────────────────────────────────────────
    out["evaluation_system"] = ea._EVAL_SYSTEM

    captured_ea: dict[str, str] = {}

    async def fake_generate_ea(*, messages, system=None, **_kw):
        captured_ea["system"] = system
        captured_ea["user"] = messages[0]["content"]
        raise _StopCapture()

    orig_ea = ea.generate
    ea.generate = fake_generate_ea
    try:
        try:
            await ea.diagnose_student_answer(
                mod_ctx=MOD_CTX, lesson_content=LESSON_CONTENT, question=QUESTION,
                answer_text=ANSWER_TEXT, confidence=3, previous_answers=PREVIOUS_ANSWERS,
            )
        except _StopCapture:
            pass
        out["evaluation_diagnose_payload"] = captured_ea["user"]

        try:
            await ea._generate_probe_question(MOD_CTX, LESSON_CONTENT, TRIGGER_ANSWER, 1)
        except _StopCapture:
            pass
        out["evaluation_probe_payload"] = captured_ea["user"]

    finally:
        ea.generate = orig_ea

    # Finalize: the full payload embeds list(DECISION_ENUM) whose order is
    # nondeterministic (set iteration), so snapshot the registry artifact the
    # finalize prompt actually owns — the instructions block — instead.
    out["evaluation_finalize_instructions"] = get_prompt(
        "evaluation_finalize_instructions"
    ).render()

    # ── Lesson / question / chat prompts (pure builders) ──────────────────────
    for pace in ("fast", "medium", "deep"):
        out[f"lesson_prompt_{pace}"] = cs.lesson_prompt(
            {**COURSE, "pace": pace}, MODULE, [], ADAPTATION_CONTEXT, PREVIOUS_MODULES
        )
    out["question_generation"] = cs.question_generation_prompt(
        COURSE, MODULE, LESSON_CONTENT, VALIDATION_ISSUES
    )
    out["module_chat_grounded"] = cs.module_chat_prompt(
        COURSE, MODULE, "conceptual", DOUBT_MESSAGE, LESSON_CONTENT, CHAT_HISTORY
    )
    out["module_chat_web_search_system"] = cs.module_chat_web_search_system(
        LESSON_CONTENT, CHAT_HISTORY
    )
    return out
