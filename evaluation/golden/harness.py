"""
evaluation/golden/harness.py
Invoke the LIVE prompt-driven flows for golden evaluation — no reimplementation,
no production DB.

Each helper calls the same code production uses:
  * curriculum → CurriculumArchitectAgent._build_plan (the LLM planning pipeline,
    minus the DB persist that _persist_curriculum does)
  * diagnosis  → agents.evaluation_agent.diagnose_student_answer
  * lesson     → core.course_service.generate_lesson_content

Metric scoring reuses the existing judges (evaluation/metrics/*). record_metric
(a DB write) is patched to a no-op so golden runs never touch the production DB;
the judges still call the real Groq judge model.
"""

from __future__ import annotations

from typing import Any, Callable

from core.student_model import CurriculumPlan, StudentState


def disable_metric_persistence() -> None:
    """Replace the metric collector's DB write with a no-op (golden runs are DB-free)."""
    async def _noop(*_a: Any, **_kw: Any) -> None:
        return None

    import evaluation.metrics.agent_metrics as am
    import evaluation.metrics.rag_metrics as rm

    am.record_metric = _noop      # type: ignore[assignment]
    rm.record_metric = _noop      # type: ignore[assignment]


# ── Curriculum ────────────────────────────────────────────────────────────────

def _pace(value: str | None) -> str:
    """Clamp a pace string to the supported set."""
    return value if value in ("fast", "medium", "deep") else "medium"


async def build_curriculum_plan(profile_case: dict[str, Any]) -> CurriculumPlan:
    """Build a CurriculumPlan through the live architect (DB-free)."""
    from agents.curriculum_architect import CurriculumArchitectAgent
    from core.course_service import (
        normalise_personalization_profile,
        sanitize_trusted_profile_for_course,
    )

    topic = str(profile_case["topic"])
    goal = str(profile_case.get("goal") or "")
    pace = _pace(profile_case.get("pace"))
    prior = str(profile_case.get("prior_knowledge") or "")
    known = list(profile_case.get("known_concepts") or [])

    profile = normalise_personalization_profile(
        topic=topic, goal=goal, pace=pace, prior_knowledge=prior,
        profile={
            "known_concepts": known,
            "assumed_known_concepts": known,
            "weak_concepts": list(profile_case.get("weak_concepts") or []),
            "must_include": list(profile_case.get("must_include") or []),
            "do_not_include": list(profile_case.get("do_not_include") or []),
            "learner_level": profile_case.get("learner_level"),
        },
    )
    profile = sanitize_trusted_profile_for_course(profile)

    state = StudentState(
        student_id="golden", name="Golden",
        domain=str(profile.get("target_context") or topic), goal=goal, pace=pace,
    )
    for concept in profile.get("assumed_known_concepts") or []:
        state.concept_mastery.setdefault(str(concept), 0.78)
        state.concept_depth.setdefault(str(concept), 0.55)
    for concept in profile.get("weak_concepts") or []:
        state.concept_mastery.setdefault(str(concept), 0.25)
        state.concept_depth.setdefault(str(concept), 0.2)

    agent = CurriculumArchitectAgent(state)
    setattr(agent, "personalization_profile", profile)
    setattr(agent, "student_history_snapshot", {})
    return await agent._build_plan(topic, profile)


def modules_as_dicts(plan: CurriculumPlan) -> list[dict[str, Any]]:
    """Flatten a plan's typed modules into the dict shape the metrics expect."""
    return [
        {
            "concept": m.concept,
            "title": m.title,
            "prerequisites": list(m.prerequisites or []),
            "concepts_taught": list(m.concepts_taught or []),
        }
        for m in plan.modules
    ]


def module_concepts(plan: CurriculumPlan) -> list[str]:
    """Return every concept name a plan's modules teach (concept + concepts_taught)."""
    names: list[str] = []
    for m in plan.modules:
        names.append(m.concept)
        names.extend(m.concepts_taught or [])
    return [str(n) for n in names if str(n).strip()]


# ── Diagnosis ─────────────────────────────────────────────────────────────────

async def run_diagnosis(case: dict[str, Any]) -> dict[str, Any]:
    """Diagnose a canned student answer through the live evaluation seam."""
    from agents.evaluation_agent import _module_context, diagnose_student_answer

    from clients.groq_client import GroqRateLimitError

    mod_ctx = _module_context(case["module"])
    question = case["question"]
    if isinstance(question, str):
        question = {"question_text": question}
    diagnosis = await diagnose_student_answer(
        mod_ctx=mod_ctx,
        lesson_content=str(case.get("lesson_excerpt") or ""),
        question=question,
        answer_text=str(case.get("answer") or ""),
        confidence=int(case.get("confidence") or 3),
        previous_answers=[],
    )
    # diagnose_student_answer swallows rate-limit/timeout errors and returns this
    # fixed fallback. Surface it as a rate-limit so the runner retries instead of
    # scoring the fallback "uncertain" as a quality failure.
    if diagnosis.get("missing_reasoning") == "Could not analyze answer.":
        raise GroqRateLimitError("diagnosis fell back (likely rate limit/timeout)")
    return diagnosis


# ── Lesson ────────────────────────────────────────────────────────────────────

async def run_lesson(case: dict[str, Any]) -> str:
    """Generate a lesson through the live prompt+generate seam (DB-free)."""
    from core.course_service import generate_lesson_content

    return await generate_lesson_content(
        course=case["course"],
        module=case["module"],
        adaptation_context=case.get("adaptation_context") or {},
        previous_modules=case.get("previous_modules") or [],
        context_chunks=[],
    )


# ── Judge helpers ─────────────────────────────────────────────────────────────

async def judge_mean(score_factory: Callable[[], Any], runs: int = 2) -> tuple[float, list[float]]:
    """Run a judge-based score `runs` times and return (mean, per_run) to damp noise."""
    scores: list[float] = []
    for _ in range(max(1, runs)):
        result = await score_factory()
        scores.append(float(result.get("score", 0.0)))
    return sum(scores) / len(scores), scores
