import contextlib
import json
import os
import sys
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import agents.curriculum_architect as curriculum_architect
import core.course_service as course_service
from agents.curriculum_architect import CurriculumArchitectAgent
from core.curriculum_quality import (
    CourseScopeAnalysis,
    validate_curriculum_quality,
    validate_master_roadmap,
    validate_modules_against_roadmap,
)
from core.student_model import (
    AdaptationDecision,
    CurriculumPlan,
    MasterRoadmap,
    Module,
    ResearchSummary,
    RoadmapStep,
    StudentState,
)


def _agent() -> CurriculumArchitectAgent:
    state = StudentState(
        student_id="s1",
        domain="machine learning",
        goal="learn machine learning",
        pace="medium",
    )
    return CurriculumArchitectAgent(state)


def _step(step_id: str, cluster: str, prereqs: list[str] | None = None) -> RoadmapStep:
    return RoadmapStep(
        step_id=step_id,
        title=cluster.replace("_", " ").title(),
        concept_cluster=cluster,
        subtopics=[cluster],
        prerequisites=prereqs or [],
        estimated_minutes=30,
        why_this_step_exists=f"{cluster} is needed next.",
        goal_alignment="It supports the stated goal.",
        depth_level="standard",
        module_generation_hint="Create one focused module.",
    )


def _module(module_id: str, concept: str, step_id: str) -> Module:
    return Module(
        id=module_id,
        title=concept.title(),
        concept=concept,
        domain_framing=f"{concept} for the learner goal",
        prerequisites=[],
        estimated_minutes=20,
        depth_level="standard",
        concepts_taught=[concept],
        depends_on_concepts=[],
        question_scope=[concept],
        roadmap_step_id=step_id,
    )


def _rich_module(
    module_id: str,
    title: str,
    concept: str,
    step_id: str,
    taught: list[str] | None = None,
) -> Module:
    taught = taught or [concept]
    return Module(
        id=module_id,
        title=title,
        concept=concept,
        domain_framing=f"{concept} in Python for the learner goal",
        prerequisites=[],
        estimated_minutes=20,
        depth_level="standard",
        purpose=f"Teach {concept}.",
        why_it_matters_for_goal=f"{concept} supports Python fluency.",
        difficulty="introductory",
        must_teach=taught,
        concepts_taught=taught,
        depends_on_concepts=[],
        question_scope=taught,
        roadmap_step_id=step_id,
    )


@contextlib.asynccontextmanager
async def _fake_get_conn():
    class FakeConn:
        async def execute(self, *args, **kwargs):
            return None

        async def fetchrow(self, *args, **kwargs):
            return {"id": 123}

    yield FakeConn()


def test_generate_research_queries_python_ml():
    queries = _agent()._generate_research_queries(
        "Python",
        {
            "learning_goal": "machine learning",
            "target_context": "machine learning",
            "learner_level": "beginner",
        },
    )

    text = " ".join(q.query for q in queries)
    assert "machine learning" in text
    assert any(q.category == "prerequisites" for q in queries)
    assert any(q.priority == 1 for q in queries)
    assert "Django" not in text
    assert "Flask" not in text


def test_generate_research_queries_thermodynamics_jee():
    queries = _agent()._generate_research_queries(
        "Thermodynamics",
        {
            "learning_goal": "JEE preparation",
            "target_context": "jee",
            "learner_level": "intermediate",
        },
    )

    assert any("JEE" in q.query or "jee" in q.query for q in queries)
    assert any(q.category == "exam_specifics" for q in queries)
    assert any(q.category == "general_roadmap" for q in queries)


def test_generate_research_queries_generic_python():
    queries = _agent()._generate_research_queries(
        "Python",
        {"learning_goal": "general programming", "target_context": "general"},
    )

    assert not any(q.category == "exam_specifics" for q in queries)
    assert any(q.category == "general_roadmap" for q in queries)


def test_validate_master_roadmap_passes():
    roadmap = MasterRoadmap(
        topic="Python",
        goal="programming",
        steps=[
            _step("step_01", "values"),
            _step("step_02", "control_flow", ["step_01"]),
            _step("step_03", "functions", ["step_02"]),
        ],
    )
    scope = CourseScopeAnalysis(what_to_exclude=[])

    result = validate_master_roadmap(roadmap, scope, {})

    assert result["passed"] is True


def test_validate_master_roadmap_fails_empty_hint():
    roadmap = MasterRoadmap(
        topic="Python",
        goal="programming",
        steps=[
            _step("step_01", "values"),
            _step("step_02", "functions", ["step_01"]),
        ],
    )
    roadmap.steps[0].module_generation_hint = ""

    result = validate_master_roadmap(roadmap, CourseScopeAnalysis(), {})

    assert result["passed"] is False
    assert "module_generation_hint" in result["issues"][0]


def test_validate_master_roadmap_rejects_excluded_topic():
    roadmap = MasterRoadmap(
        topic="Python",
        goal="programming",
        steps=[
            _step("step_01", "values"),
            _step("step_02", "Django web framework", ["step_01"]),
        ],
    )
    scope = CourseScopeAnalysis(what_to_exclude=["Django"])

    result = validate_master_roadmap(roadmap, scope, {})

    assert result["passed"] is False


@pytest.mark.asyncio
async def test_beginner_python_fast_roadmap_excludes_advanced_topics(monkeypatch):
    state = StudentState(
        student_id="s1",
        domain="general pure Python",
        goal="learn Python and code applications",
        pace="fast",
    )
    agent = CurriculumArchitectAgent(state)
    profile = {
        "learning_goal": "learn Python and code applications",
        "target_context": "general pure Python",
        "learner_level": "complete beginner",
        "pace": "fast",
        "do_not_include": [
            "decorators",
            "async",
            "deployment",
            "pandas",
            "sklearn",
            "machine learning",
        ],
    }
    scope = CourseScopeAnalysis(
        actual_course_focus="pure Python fundamentals",
        learner_level="complete beginner",
        pace="fast",
        recommended_module_count=4,
        initial_recommended_module_count=4,
        final_module_count_target=4,
        what_to_exclude=profile["do_not_include"],
    )
    invalid_roadmap = MasterRoadmap(
        topic="Python",
        goal=profile["learning_goal"],
        steps=[
            _step("step_01", "Python values"),
            _step("step_02", "Python functions"),
            _step("step_03", "Python decorators", ["Python functions"]),
            _step("step_04", "Python deployment", ["Python decorators"]),
        ],
    )
    repaired_roadmap = MasterRoadmap(
        topic="Python",
        goal=profile["learning_goal"],
        steps=[
            _step("step_01", "Python values"),
            _step("step_02", "Python control flow"),
            _step("step_03", "Python functions"),
            _step("step_04", "Python collections"),
        ],
    )
    monkeypatch.setattr(agent, "_run_research", AsyncMock(return_value=ResearchSummary(full_text="Python decorators and deployment appear in advanced research.")))
    monkeypatch.setattr(agent, "_analyze_scope", AsyncMock(return_value=scope))
    monkeypatch.setattr(agent, "_build_master_roadmap", AsyncMock(return_value=invalid_roadmap))
    repair = AsyncMock(return_value=repaired_roadmap)
    monkeypatch.setattr(agent, "_repair_master_roadmap_from_validation_issues", repair)
    monkeypatch.setattr(
        agent,
        "_review_curriculum_blueprint",
        AsyncMock(return_value={"passed": True, "issues": [], "coverage_gaps": []}),
    )
    monkeypatch.setattr(agent, "_embed_modules_to_chromadb", AsyncMock(return_value=None))
    monkeypatch.setattr(curriculum_architect, "get_conn", _fake_get_conn)

    plan, _ = await agent.build_curriculum("Python", profile=profile)

    repair.assert_awaited_once()
    assert plan.validation_result["passed"] is True
    assert len(plan.modules) == 4
    roadmap_text = json.dumps([step.model_dump() for step in agent._master_roadmap.steps]).lower()
    for forbidden in ("decorators", "deployment", "async", "pandas", "sklearn", "machine learning"):
        assert forbidden not in roadmap_text


@pytest.mark.asyncio
async def test_roadmap_repair_receives_exact_exclusion_issues(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_generate(messages, model, system):
        captured["system"] = system
        captured["payload"] = json.loads(messages[0]["content"])
        return json.dumps({
            "steps": [
                {
                    "step_id": "step_01",
                    "title": "Python Values",
                    "concept_cluster": "Python values",
                    "subtopics": ["variables", "assignment"],
                    "prerequisites": [],
                    "estimated_minutes": 30,
                    "why_this_step_exists": "Values come first.",
                    "goal_alignment": "Values support Python applications.",
                    "depth_level": "surface",
                    "module_generation_hint": "Create one focused module.",
                    "is_optional": False,
                },
                {
                    "step_id": "step_02",
                    "title": "Python Functions",
                    "concept_cluster": "Python functions",
                    "subtopics": ["parameters", "return values"],
                    "prerequisites": ["Python values"],
                    "estimated_minutes": 30,
                    "why_this_step_exists": "Functions organize code.",
                    "goal_alignment": "Functions support practical scripts.",
                    "depth_level": "surface",
                    "module_generation_hint": "Create one focused module.",
                    "is_optional": False,
                },
            ],
            "total_estimated_minutes": 60,
            "repair_rationale": "Removed excluded decorators.",
        })

    monkeypatch.setattr(curriculum_architect, "generate", fake_generate)
    agent = _agent()
    current = MasterRoadmap(
        topic="Python",
        goal="learn Python",
        steps=[_step("step_01", "Python values"), _step("step_02", "Python decorators", ["Python values"])],
    )
    scope = CourseScopeAnalysis(
        recommended_module_count=2,
        what_to_exclude=["decorators"],
    )
    issues = ["Roadmap step 2 uses excluded concept: decorators."]

    repaired = await agent._repair_master_roadmap_from_validation_issues(
        current_roadmap=current,
        scope=scope,
        profile={"do_not_include": ["decorators"], "target_context": "general pure Python"},
        validation_issues=issues,
        research_summary=ResearchSummary(full_text="Advanced Python roadmap mentions decorators."),
    )

    payload = captured["payload"]
    assert repaired.steps[1].concept_cluster == "Python functions"
    assert issues[0] in payload["validation_issues"]
    assert payload["do_not_include"] == ["decorators"]
    assert "invalid_roadmap_json" in payload
    assert "Python decorators" in json.dumps(payload["invalid_roadmap_json"])
    assert "Remove excluded concepts" in captured["system"]
    assert "Remove excluded concepts" in payload["repair_instruction"]


def test_scope_roadmap_reconciliation():
    agent = _agent()
    scope = CourseScopeAnalysis(
        recommended_module_count=15,
        initial_recommended_module_count=15,
        reason_for_module_count="Rough broad-course estimate.",
    )
    roadmap = MasterRoadmap(
        topic="Python",
        goal="learn Python",
        steps=[
            RoadmapStep(
                step_id=f"step_{idx:02d}",
                title=f"Python Skill Cluster {idx}",
                concept_cluster=f"Python skill cluster {idx}",
                subtopics=[f"subtopic {idx}.{n}" for n in range(1, 5)],
                prerequisites=[],
                estimated_minutes=60,
                why_this_step_exists="This cluster has several teachable ideas.",
                goal_alignment="It supports practical Python coding.",
                depth_level="surface",
                module_generation_hint="Split when module target requires granularity.",
            )
            for idx in range(1, 8)
        ],
    )

    reconciled = agent._reconcile_scope_with_roadmap(
        scope,
        roadmap,
        {"pace": "fast", "learner_level": "complete beginner"},
    )

    assert reconciled.final_module_count_target == 15
    assert reconciled.scope_roadmap_alignment
    assert reconciled.roadmap_split_hints
    assert any("Reconciliation:" in step.module_generation_hint for step in roadmap.steps)


@pytest.mark.asyncio
async def test_roadmap_finalizer_splits_broad_beginner_step_without_search(monkeypatch):
    agent = _agent()
    monkeypatch.setattr(
        curriculum_architect,
        "tavily_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Tavily should not run for obvious Python splitting")),
    )
    scope = CourseScopeAnalysis(
        actual_course_focus="Python fundamentals",
        learner_level="complete beginner",
        pace="fast",
        recommended_module_count=4,
        final_module_count_target=4,
    )
    roadmap = MasterRoadmap(
        topic="Python",
        goal="learn Python programming",
        steps=[
            RoadmapStep(
                step_id="step_01",
                title="Control Structures: Loops and Functions",
                concept_cluster="Control structures with loops and functions",
                subtopics=["conditionals", "loops", "functions"],
                prerequisites=[],
                estimated_minutes=90,
                why_this_step_exists="Control structures make programs useful.",
                goal_alignment="They support practical Python applications.",
                depth_level="surface",
                module_generation_hint="Split if needed.",
            ),
            _step("step_02", "Practice scripts", ["Control structures with loops and functions"]),
        ],
    )

    finalized = await agent._finalize_master_roadmap(
        topic="Python",
        scope=scope,
        draft_roadmap=roadmap,
        profile={
            "learning_goal": "learn Python programming",
            "target_context": "general pure Python",
            "learner_level": "complete beginner",
            "pace": "fast",
        },
        research_summary=ResearchSummary(coverage_confidence=0.7, full_text="Python beginner roadmap."),
    )

    titles = [step.title for step in finalized.steps]
    assert "Conditional Logic" in titles
    assert "Loops and Repetition" in titles
    assert "Functions and Reusable Code" in titles
    assert agent._roadmap_finalizer_result["searches_used"] == []
    assert agent._roadmap_finalizer_result["splitting_decisions"][0]["decision"] == "split"
    assert validate_master_roadmap(finalized, scope, {})["passed"] is True


@pytest.mark.asyncio
async def test_roadmap_finalizer_repairs_filler_forbidden_and_thin_setup():
    agent = _agent()
    scope = CourseScopeAnalysis(
        actual_course_focus="pure Python fundamentals",
        learner_level="complete beginner",
        pace="fast",
        recommended_module_count=3,
        final_module_count_target=3,
        what_to_exclude=["Django"],
    )
    roadmap = MasterRoadmap(
        topic="Python",
        goal="learn Python",
        steps=[
            RoadmapStep(
                step_id="step_01",
                title="Installing Python",
                concept_cluster="Installing Python",
                subtopics=["installation"],
                prerequisites=[],
                estimated_minutes=10,
                why_this_step_exists="The learner needs Python available.",
                goal_alignment="It supports running code.",
                depth_level="surface",
                module_generation_hint="Create one short module.",
            ),
            _step("step_02", "Django web framework", ["Installing Python"]),
            _step("step_03", "Conclusion", ["Django web framework"]),
        ],
    )

    finalized = await agent._finalize_master_roadmap(
        topic="Python",
        scope=scope,
        draft_roadmap=roadmap,
        profile={
            "learning_goal": "learn Python",
            "target_context": "general pure Python",
            "learner_level": "complete beginner",
            "pace": "fast",
            "do_not_include": ["Django"],
        },
        research_summary=ResearchSummary(coverage_confidence=0.8, full_text="Python beginner roadmap."),
    )

    roadmap_text = json.dumps(finalized.model_dump()).lower()
    report = agent._roadmap_finalizer_result
    assert "django" not in roadmap_text
    assert "conclusion" not in roadmap_text
    assert finalized.steps[0].title == "Running Your First Python Program"
    assert "Django" in report["coverage_check"]["forbidden_topics_found"]
    assert "Conclusion" in report["coverage_check"]["irrelevant_topics"]
    assert validate_master_roadmap(finalized, scope, {"do_not_include": ["Django"]})["passed"] is True


def test_roadmap_finalizer_tavily_queries_are_bounded_and_targeted():
    agent = _agent()
    generic_context = {
        "user_topic": "Python",
        "user_goal_motive": "general beginner programming",
        "target_context": "general pure Python",
    }
    generic_roadmap = MasterRoadmap(
        topic="Python",
        goal="learn Python",
        steps=[_step("step_01", "Python values"), _step("step_02", "Python loops", ["Python values"])],
    )

    assert agent._finalizer_uncertainty_queries(
        generic_context,
        generic_roadmap,
        ResearchSummary(coverage_confidence=0.2, full_text=""),
    ) == []

    exam_context = {
        "user_topic": "Thermodynamics",
        "user_goal_motive": "JEE exam preparation interview confidence",
        "target_context": "JEE",
    }
    exam_roadmap = MasterRoadmap(
        topic="Thermodynamics",
        goal="JEE exam preparation",
        steps=[
            _step("step_01", "First law of thermodynamics"),
            _step("step_02", "PV diagrams", ["First law of thermodynamics"]),
        ],
    )

    queries = agent._finalizer_uncertainty_queries(
        exam_context,
        exam_roadmap,
        ResearchSummary(coverage_confidence=0.2, full_text=""),
    )

    assert len(queries) <= 2
    assert all(item["query"] and item["reason"] for item in queries)
    assert any("syllabus" in item["query"].lower() for item in queries)


def test_question_scope_normalizer_removes_question_like_text():
    agent = _agent()
    step = _step("step_01", "Python review")
    module = _rich_module("m1", "Review Practice", "Python review", "step_01")
    module.question_scope = ["How to review the material?"]

    normalized, report = agent._normalize_assessment_scope(module, step)

    assert "How to review the material?" not in normalized.question_scope
    assert "review strategy" in normalized.question_scope
    assert "review strategy" in normalized.concepts_taught
    assert report["changed"] is True
    assert report["removed_question_like_values"] == ["How to review the material?"]


def test_question_scope_falls_back_to_concepts_taught():
    agent = _agent()
    step = RoadmapStep(
        step_id="step_01",
        title="Loops",
        concept_cluster="Python loops",
        subtopics=["for loops", "while loops"],
        estimated_minutes=30,
        why_this_step_exists="Loops teach repetition.",
        goal_alignment="Loops support practical programs.",
        module_generation_hint="Create one focused module.",
    )
    module = _rich_module(
        "m1",
        "Loop Practice",
        "Python loops",
        "step_01",
        taught=["Python loops", "for loops"],
    )
    module.question_scope = ["What is the key idea?"]

    normalized, _ = agent._normalize_assessment_scope(module, step)

    assert "What is the key idea?" not in normalized.question_scope
    assert "Python loops" in normalized.question_scope
    assert "for loops" in normalized.question_scope


def test_projection_does_not_copy_full_questions_to_question_scope():
    agent = _agent()
    roadmap = MasterRoadmap(
        topic="Python",
        goal="learn Python",
        steps=[
            RoadmapStep(
                step_id="step_01",
                title="Review Strategy",
                concept_cluster="review strategy",
                subtopics=["concept recap"],
                estimated_minutes=30,
                why_this_step_exists="Review helps consolidate learning.",
                goal_alignment="It supports beginner retention.",
                module_generation_hint="Create one focused module.",
                check_question_targets=["How to review the material?"],
            ),
            _step("step_02", "Practice scripts", ["review strategy"]),
        ],
    )
    scope = CourseScopeAnalysis(recommended_module_count=2, final_module_count_target=2)

    modules = agent._project_roadmap_to_modules(
        roadmap,
        scope,
        {"target_context": "Python"},
        "surface",
    )

    assert "How to review the material?" not in modules[0].question_scope
    assert all("?" not in item for item in modules[0].question_scope)
    assert "review strategy" in modules[0].question_scope


def test_validator_reports_question_like_scope_clearly():
    module = _rich_module("m1", "Review Practice", "Python review", "step_01")
    module.question_scope = ["How to review the material?"]

    result = validate_curriculum_quality(
        topic="Python",
        modules=[module],
        profile={},
        scope_analysis=CourseScopeAnalysis(recommended_module_count=1),
    )

    assert result["passed"] is False
    assert any(
        "expected concept/skill labels, not full questions" in issue
        for issue in result["issues"]
    )


@pytest.mark.asyncio
async def test_course_creation_continues_after_sanitizable_question_scope(monkeypatch):
    agent = _agent()
    scope = CourseScopeAnalysis(
        actual_course_focus="Python",
        recommended_module_count=3,
        final_module_count_target=3,
        reason_for_module_count="Three focused modules are enough.",
    )
    roadmap = MasterRoadmap(
        topic="Python",
        goal="learn Python",
        steps=[
            _step("step_01", "Python values"),
            _step("step_02", "Python review", ["Python values"]),
            _step("step_03", "Python practice", ["Python review"]),
        ],
    )
    bad_modules = [
        _rich_module("m1", "Python Values", "Python values", "step_01"),
        _rich_module("m2", "Review Practice", "Python review", "step_02"),
        _rich_module("m3", "Python Practice", "Python practice", "step_03"),
    ]
    bad_modules[1].question_scope = ["How to review the material?"]

    monkeypatch.setattr(agent, "_run_research", AsyncMock(return_value=ResearchSummary(full_text="Python research")))
    monkeypatch.setattr(agent, "_analyze_scope", AsyncMock(return_value=scope))
    monkeypatch.setattr(agent, "_build_master_roadmap", AsyncMock(return_value=roadmap))
    monkeypatch.setattr(agent, "_project_roadmap_to_modules", lambda *args, **kwargs: bad_modules)
    monkeypatch.setattr(
        agent,
        "_review_curriculum_blueprint",
        AsyncMock(return_value={"passed": True, "issues": [], "coverage_gaps": []}),
    )
    monkeypatch.setattr(agent, "_embed_modules_to_chromadb", AsyncMock(return_value=None))
    monkeypatch.setattr(curriculum_architect, "get_conn", _fake_get_conn)

    plan, _ = await agent.build_curriculum(
        "Python",
        profile={"learning_goal": "learn Python", "target_context": "Python"},
    )

    assert plan.validation_result["passed"] is True
    assert plan.validation_result["metadata_sanitizer"]["changed"] is True
    assert "How to review the material?" not in plan.modules[1].question_scope


@pytest.mark.asyncio
async def test_research_excluded_topics_not_copied(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_generate(messages, model, system):
        captured["system"] = system
        captured["payload"] = json.loads(messages[0]["content"])
        return json.dumps({
            "steps": [
                {
                    "step_id": "step_01",
                    "title": "Python Values",
                    "concept_cluster": "Python values",
                    "subtopics": ["variables"],
                    "prerequisites": [],
                    "estimated_minutes": 30,
                    "why_this_step_exists": "Values come first.",
                    "goal_alignment": "Values support Python scripts.",
                    "depth_level": "surface",
                    "module_generation_hint": "Create one focused module.",
                    "is_optional": False,
                },
                {
                    "step_id": "step_02",
                    "title": "Python Control Flow",
                    "concept_cluster": "Python control flow",
                    "subtopics": ["if statements", "loops"],
                    "prerequisites": ["Python values"],
                    "estimated_minutes": 30,
                    "why_this_step_exists": "Control flow makes scripts useful.",
                    "goal_alignment": "Control flow supports practical applications.",
                    "depth_level": "surface",
                    "module_generation_hint": "Create one focused module.",
                    "is_optional": False,
                },
            ],
            "total_estimated_minutes": 60,
            "rationale": "Ignored advanced research.",
        })

    monkeypatch.setattr(curriculum_architect, "generate", fake_generate)
    scope = CourseScopeAnalysis(
        actual_course_focus="pure Python fundamentals",
        learner_level="complete beginner",
        recommended_module_count=2,
        what_to_exclude=["deployment", "decorators"],
    )

    await _agent()._build_master_roadmap(
        topic="Python",
        scope=scope,
        research_summary=ResearchSummary(full_text="Python roadmaps often mention decorators and deployment."),
        profile={
            "learning_goal": "learn Python and code applications",
            "target_context": "general pure Python",
            "learner_level": "complete beginner",
            "do_not_include": ["deployment", "decorators"],
        },
    )

    payload = captured["payload"]
    assert set(payload["excluded_research_concepts_to_ignore"]) == {"deployment", "decorators"}
    assert set(payload["forbidden_topics"]) >= {"deployment", "decorators"}
    assert "If research mentions forbidden topics" in captured["system"]


def test_validate_modules_against_roadmap_passes():
    roadmap = MasterRoadmap(
        topic="Python",
        goal="programming",
        steps=[
            _step("step_01", "values"),
            _step("step_02", "control_flow", ["step_01"]),
            _step("step_03", "functions", ["step_02"]),
        ],
    )
    modules = [
        _module("m1", "values", "step_01"),
        _module("m2", "control flow", "step_02"),
        _module("m3", "functions", "step_03"),
    ]
    scope = CourseScopeAnalysis(recommended_module_count=3)

    result = validate_modules_against_roadmap(modules, roadmap, scope, {})

    assert result["passed"] is True


def test_project_roadmap_step_to_module():
    agent = _agent()
    roadmap = MasterRoadmap(
        topic="Python",
        goal="learn Python",
        steps=[
            RoadmapStep(
                step_id="step_01",
                title="Control Flow Fundamentals",
                concept_cluster="Control Flow",
                subtopics=["if statements", "loops", "branching"],
                prerequisites=["Basic Types"],
                estimated_minutes=45,
                why_this_step_exists="Control flow lets programs make decisions.",
                goal_alignment="It supports writing useful Python scripts.",
                depth_level="surface",
                module_generation_hint="Create one focused module.",
            )
        ],
    )
    scope = CourseScopeAnalysis(recommended_module_count=1, final_module_count_target=1)

    modules = agent._project_roadmap_to_modules(
        roadmap,
        scope,
        {"target_context": "general pure Python"},
        "surface",
    )

    assert len(modules) == 1
    module = modules[0]
    assert module.roadmap_step_id == "step_01"
    assert module.title == "Control Flow Fundamentals"
    assert module.concept == "Control Flow"
    assert module.concepts_taught == ["Control Flow", "if statements", "loops", "branching"]
    assert module.must_teach == ["if statements", "loops", "branching"]
    assert module.question_scope == ["if statements", "loops", "branching"]
    assert module.purpose == "Control flow lets programs make decisions."
    assert module.why_it_matters_for_goal == "It supports writing useful Python scripts."


def test_projected_modules_do_not_have_module_id_prerequisites():
    agent = _agent()
    roadmap = MasterRoadmap(
        topic="Python",
        goal="learn Python",
        steps=[
            _step("step_01", "Python values"),
            _step("step_02", "Python control flow", ["step_01", "m1"]),
        ],
    )
    roadmap.steps[1].question_scope = ["m2", "if statements"]
    scope = CourseScopeAnalysis(recommended_module_count=2, final_module_count_target=2)

    modules = agent._project_roadmap_to_modules(
        roadmap,
        scope,
        {"target_context": "Python"},
        "surface",
    )

    assert len(modules) == 2
    assert modules[1].prerequisites == ["Python values"]
    assert modules[1].depends_on_concepts == ["Python values"]
    assert modules[1].question_scope == ["if statements"]


def test_broad_roadmap_step_splits_into_multiple_modules():
    agent = _agent()
    scope = CourseScopeAnalysis(
        actual_course_focus="Python",
        recommended_module_count=4,
        final_module_count_target=4,
        rough_scope_recommendation=4,
    )
    roadmap = MasterRoadmap(
        topic="Python",
        goal="learn Python",
        steps=[
            RoadmapStep(
                step_id="step_01",
                title="Functions and Modules",
                concept_cluster="Functions and modules",
                subtopics=["defining functions", "parameters", "return values", "scope", "importing modules"],
                prerequisites=[],
                estimated_minutes=90,
                why_this_step_exists="This broad step has several teachable pieces.",
                goal_alignment="It supports practical code organization.",
                depth_level="surface",
                module_generation_hint="Split into multiple focused modules.",
            ),
            _step("step_02", "Practice scripts", ["Functions and modules"]),
        ],
    )

    modules = agent._project_roadmap_to_modules(
        roadmap,
        scope,
        {"target_context": "Python"},
        "surface",
    )
    step_01_modules = [module for module in modules if module.roadmap_step_id == "step_01"]

    assert len(modules) == 4
    assert len(step_01_modules) == 3
    assert all(module.concepts_taught for module in step_01_modules)
    assert step_01_modules[0].title.startswith("Functions and Modules:")


def test_projected_modules_cover_roadmap_concepts():
    agent = _agent()
    roadmap = MasterRoadmap(
        topic="Python",
        goal="learn Python",
        steps=[
            _step("step_01", "Python values"),
            _step("step_02", "Python control flow", ["Python values"]),
            _step("step_03", "Python functions", ["Python control flow"]),
        ],
    )
    scope = CourseScopeAnalysis(recommended_module_count=3, final_module_count_target=3)

    modules = agent._project_roadmap_to_modules(
        roadmap,
        scope,
        {"target_context": "Python"},
        "surface",
    )
    result = validate_modules_against_roadmap(modules, roadmap, scope, {})
    taught = [concept for module in modules for concept in module.concepts_taught]

    assert result["passed"] is True
    for step in roadmap.steps:
        assert step.concept_cluster in taught


def test_final_module_target_separate_from_rough_scope():
    agent = _agent()
    scope = CourseScopeAnalysis(
        recommended_module_count=20,
        initial_recommended_module_count=20,
        rough_scope_recommendation=20,
    )
    roadmap = MasterRoadmap(
        topic="Python",
        goal="learn Python",
        steps=[
            RoadmapStep(
                step_id=f"step_{idx:02d}",
                title=f"Python Concept {idx}",
                concept_cluster=f"Python concept {idx}",
                subtopics=[f"Python concept {idx}"],
                prerequisites=[],
                estimated_minutes=25,
                why_this_step_exists="This step is narrow.",
                goal_alignment="It supports Python fluency.",
                depth_level="surface",
                module_generation_hint="Create one focused module.",
            )
            for idx in range(1, 11)
        ],
    )

    reconciled = agent._reconcile_scope_with_roadmap(scope, roadmap, {"pace": "fast"})
    modules = [
        _rich_module(f"m{idx}", f"Python Concept {idx}", f"Python concept {idx}", f"step_{idx:02d}")
        for idx in range(1, 11)
    ]
    result = validate_modules_against_roadmap(modules, roadmap, reconciled, {})

    assert reconciled.rough_scope_recommendation == 20
    assert reconciled.final_module_count_target == 10
    assert result["passed"] is True


def test_roadmap_driven_split_for_broad_steps():
    agent = _agent()
    scope = CourseScopeAnalysis(
        recommended_module_count=4,
        final_module_count_target=4,
        rough_scope_recommendation=4,
    )
    roadmap = MasterRoadmap(
        topic="Python",
        goal="learn Python",
        steps=[
            RoadmapStep(
                step_id="step_01",
                title="Functions and Modules",
                concept_cluster="Functions and modules",
                subtopics=["defining functions", "parameters", "return values", "imports"],
                prerequisites=[],
                estimated_minutes=90,
                why_this_step_exists="This broad step has several teachable pieces.",
                goal_alignment="It supports practical code organization.",
                depth_level="surface",
                module_generation_hint="Split into multiple focused module plans.",
            ),
            _step("step_02", "Practice scripts", ["Functions and modules"]),
        ],
    )

    modules = agent._project_roadmap_to_modules(
        roadmap,
        scope,
        {"target_context": "Python"},
        "surface",
    )
    step_01_modules = [module for module in modules if module.roadmap_step_id == "step_01"]
    result = validate_modules_against_roadmap(modules, roadmap, scope, {})

    assert len(modules) == 4
    assert len(step_01_modules) > 1
    assert result["passed"] is True


@pytest.mark.asyncio
async def test_no_llm_module_generation_called_in_default_path(monkeypatch):
    agent = _agent()
    scope = CourseScopeAnalysis(
        actual_course_focus="Python",
        recommended_module_count=5,
        final_module_count_target=5,
        reason_for_module_count="Five focused metadata modules are needed.",
    )
    roadmap = MasterRoadmap(
        topic="Python",
        goal="learn Python",
        steps=[
            _step("step_01", "Python values"),
            _step("step_02", "Python expressions"),
            _step("step_03", "Python control flow"),
            _step("step_04", "Python functions"),
            _step("step_05", "Python collections"),
        ],
    )

    monkeypatch.setattr(agent, "_run_research", AsyncMock(return_value=ResearchSummary(full_text="Python research")))
    monkeypatch.setattr(agent, "_analyze_scope", AsyncMock(return_value=scope))
    monkeypatch.setattr(agent, "_build_master_roadmap", AsyncMock(return_value=roadmap))
    generate_modules = AsyncMock(side_effect=AssertionError("old LLM module planner should not run"))
    repair_modules = AsyncMock(side_effect=AssertionError("old LLM module repair should not run"))
    monkeypatch.setattr(agent, "_generate_modules_from_roadmap", generate_modules, raising=False)
    monkeypatch.setattr(agent, "_repair_modules_from_validation_issues", repair_modules, raising=False)
    monkeypatch.setattr(
        agent,
        "_review_curriculum_blueprint",
        AsyncMock(return_value={"passed": True, "issues": [], "coverage_gaps": []}),
    )
    monkeypatch.setattr(agent, "_embed_modules_to_chromadb", AsyncMock(return_value=None))
    monkeypatch.setattr(curriculum_architect, "get_conn", _fake_get_conn)

    plan, _ = await agent.build_curriculum(
        "Python",
        profile={"learning_goal": "learn Python", "target_context": "Python"},
    )

    generate_modules.assert_not_awaited()
    repair_modules.assert_not_awaited()
    assert len(plan.modules) == 5
    assert plan.validation_result["passed"] is True


@pytest.mark.asyncio
async def test_build_curriculum_falls_back_when_master_roadmap_json_is_unusable(monkeypatch):
    agent = _agent()
    profile = {
        "learning_goal": "learn Python programming as a first language",
        "target_context": "general pure Python",
        "learner_level": "complete beginner",
        "do_not_include": [
            "thermodynamics",
            "linear algebra",
            "machine learning",
            "pandas",
            "sklearn",
            "scikit-learn",
            "advanced OOP",
            "decorators",
            "async",
            "deployment",
        ],
    }
    scope = CourseScopeAnalysis(
        actual_course_focus="pure Python fundamentals",
        learner_level="complete beginner",
        target_outcome=profile["learning_goal"],
        pace="fast",
        recommended_module_count=15,
        initial_recommended_module_count=15,
        rough_scope_recommendation=15,
        final_module_count_target=15,
        what_to_exclude=profile["do_not_include"],
    )
    broken_builder = AsyncMock(side_effect=ValueError("master roadmap was not valid JSON or had no steps"))

    monkeypatch.setattr(agent, "_run_research", AsyncMock(return_value=ResearchSummary(full_text="Python beginner roadmap.")))
    monkeypatch.setattr(agent, "_analyze_scope", AsyncMock(return_value=scope))
    monkeypatch.setattr(agent, "_build_master_roadmap", broken_builder)
    monkeypatch.setattr(
        agent,
        "_finalize_master_roadmap",
        AsyncMock(side_effect=lambda **kwargs: kwargs["draft_roadmap"]),
    )
    monkeypatch.setattr(
        agent,
        "_review_curriculum_blueprint",
        AsyncMock(return_value={"passed": True, "issues": [], "coverage_gaps": []}),
    )
    monkeypatch.setattr(agent, "_embed_modules_to_chromadb", AsyncMock(return_value=None))
    monkeypatch.setattr(curriculum_architect, "get_conn", _fake_get_conn)

    plan, _ = await agent.build_curriculum("Python", profile=profile)

    assert broken_builder.await_count == 2
    assert any(item["stage"] == "master_roadmap_fallback_seed" for item in plan.repair_history)
    assert plan.validation_result["passed"] is True
    assert plan.modules
    roadmap_text = json.dumps([step.model_dump() for step in agent._master_roadmap.steps]).lower()
    for forbidden in profile["do_not_include"]:
        assert forbidden.lower() not in roadmap_text


@pytest.mark.asyncio
async def test_embed_modules_uses_deterministic_metadata_not_groq(monkeypatch):
    agent = _agent()
    plan = CurriculumPlan(
        topic="Python",
        domain="general pure Python",
        goal="learn programming",
        modules=[
            Module(
                id="m1",
                title="Functions",
                concept="functions",
                domain_framing="functions for writing reusable Python scripts",
                prerequisites=["variables"],
                estimated_minutes=15,
                depth_level="surface",
                purpose="Teach reusable code blocks.",
                why_it_matters_for_goal="Functions help organize practical Python programs.",
                must_teach=["defining functions", "return values"],
                concepts_taught=["functions", "defining functions", "return values"],
                question_scope=["defining functions", "return values"],
                roadmap_step_id="step_01",
            )
        ],
    )
    generate_mock = AsyncMock(side_effect=AssertionError("embedding must not call Groq"))
    captured: dict[str, object] = {}

    async def fake_chroma_insert(chunk_id, domain, full_text, metadata):
        captured["chunk_id"] = chunk_id
        captured["domain"] = domain
        captured["full_text"] = full_text
        captured["metadata"] = metadata

    monkeypatch.setattr(curriculum_architect, "generate", generate_mock)
    monkeypatch.setattr(curriculum_architect, "chroma_insert", fake_chroma_insert)
    monkeypatch.setattr(curriculum_architect, "get_conn", _fake_get_conn)

    await agent._embed_modules_to_chromadb(plan, curriculum_id=123)

    generate_mock.assert_not_awaited()
    text = str(captured["full_text"])
    assert "Course topic: Python" in text
    assert "Module title: Functions" in text
    assert "Concepts taught: functions, defining functions, return values" in text
    assert "Summary:" not in text


@pytest.mark.asyncio
async def test_optional_embedding_failure_does_not_block_course_creation(monkeypatch):
    agent = _agent()
    scope = CourseScopeAnalysis(
        actual_course_focus="Python",
        recommended_module_count=3,
        final_module_count_target=3,
    )
    roadmap = MasterRoadmap(
        topic="Python",
        goal="learn Python",
        steps=[
            _step("step_01", "Python values"),
            _step("step_02", "Python functions", ["Python values"]),
            _step("step_03", "Python practice", ["Python functions"]),
        ],
    )

    monkeypatch.setattr(agent, "_run_research", AsyncMock(return_value=ResearchSummary(full_text="Python research")))
    monkeypatch.setattr(agent, "_analyze_scope", AsyncMock(return_value=scope))
    monkeypatch.setattr(agent, "_build_master_roadmap", AsyncMock(return_value=roadmap))
    monkeypatch.setattr(
        agent,
        "_review_curriculum_blueprint",
        AsyncMock(return_value={"passed": True, "issues": [], "coverage_gaps": []}),
    )
    monkeypatch.setattr(agent, "_embed_modules_to_chromadb", AsyncMock(side_effect=RuntimeError("chroma down")))
    monkeypatch.setattr(curriculum_architect, "get_conn", _fake_get_conn)

    plan, _ = await agent.build_curriculum(
        "Python",
        profile={"learning_goal": "learn Python", "target_context": "Python"},
    )

    assert len(plan.modules) == 3
    assert plan.validation_result["passed"] is True


@pytest.mark.asyncio
async def test_optional_curriculum_review_rate_limit_does_not_block_course_creation(monkeypatch):
    agent = _agent()
    scope = CourseScopeAnalysis(
        actual_course_focus="Python",
        recommended_module_count=3,
        final_module_count_target=3,
    )
    roadmap = MasterRoadmap(
        topic="Python",
        goal="learn Python",
        steps=[
            _step("step_01", "Python values"),
            _step("step_02", "Python functions", ["Python values"]),
            _step("step_03", "Python practice", ["Python functions"]),
        ],
    )

    monkeypatch.setattr(agent, "_run_research", AsyncMock(return_value=ResearchSummary(full_text="Python research")))
    monkeypatch.setattr(agent, "_analyze_scope", AsyncMock(return_value=scope))
    monkeypatch.setattr(agent, "_build_master_roadmap", AsyncMock(return_value=roadmap))
    monkeypatch.setattr(
        agent,
        "_review_curriculum_blueprint",
        AsyncMock(side_effect=curriculum_architect.GroqRateLimitError("rate limit")),
    )
    monkeypatch.setattr(agent, "_embed_modules_to_chromadb", AsyncMock(return_value=None))
    monkeypatch.setattr(curriculum_architect, "get_conn", _fake_get_conn)

    plan, _ = await agent.build_curriculum(
        "Python",
        profile={"learning_goal": "learn Python", "target_context": "Python"},
    )

    assert plan.validation_result["passed"] is True
    assert plan.validation_result["curriculum_review"]["skipped"] is True


@pytest.mark.asyncio
async def test_projected_modules_cover_broad_control_flow_without_module_repair(monkeypatch):
    agent = _agent()
    scope = CourseScopeAnalysis(
        actual_course_focus="Python",
        recommended_module_count=4,
        final_module_count_target=4,
        reason_for_module_count="Control flow and functions should be split into focused module metadata.",
    )
    roadmap = MasterRoadmap(
        topic="Python",
        goal="learn Python",
        steps=[
            RoadmapStep(
                step_id="step_01",
                title="Python Values",
                concept_cluster="Python values",
                subtopics=["variables", "expressions"],
                estimated_minutes=30,
                why_this_step_exists="Values come first.",
                goal_alignment="Values support Python fluency.",
                module_generation_hint="Create one focused module.",
            ),
            RoadmapStep(
                step_id="step_02",
                title="Control Structures and Functions",
                concept_cluster="Control structures and functions",
                subtopics=["if statements", "loops", "function definition", "parameters", "return values"],
                prerequisites=["Python values"],
                estimated_minutes=90,
                why_this_step_exists="Programs need branching, repetition, and reusable logic.",
                goal_alignment="This enables writing useful Python programs.",
                module_generation_hint="Split into conditionals, loops, and functions.",
            ),
        ],
    )

    monkeypatch.setattr(agent, "_run_research", AsyncMock(return_value=ResearchSummary(full_text="Python control flow research")))
    monkeypatch.setattr(agent, "_analyze_scope", AsyncMock(return_value=scope))
    monkeypatch.setattr(agent, "_build_master_roadmap", AsyncMock(return_value=roadmap))
    repair = AsyncMock(side_effect=AssertionError("old LLM module repair should not run"))
    monkeypatch.setattr(agent, "_repair_modules_from_validation_issues", repair, raising=False)
    monkeypatch.setattr(
        agent,
        "_review_curriculum_blueprint",
        AsyncMock(return_value={"passed": True, "issues": [], "coverage_gaps": []}),
    )
    monkeypatch.setattr(agent, "_embed_modules_to_chromadb", AsyncMock(return_value=None))
    monkeypatch.setattr(curriculum_architect, "get_conn", _fake_get_conn)

    plan, _ = await agent.build_curriculum(
        "Python",
        profile={"learning_goal": "learn Python", "target_context": "Python"},
    )

    repair.assert_not_awaited()
    assert plan.validation_result["passed"] is True
    taught = {concept for module in plan.modules for concept in module.concepts_taught}
    assert {"Conditional logic", "Loops and repetition", "Functions and reusable code"} <= taught
    assert {"if statements", "for loops", "defining functions", "parameters", "return values"} <= taught
    assert plan.validation_result["roadmap_finalizer"]["splitting_decisions"]


def test_concept_alias_matching_control_structures():
    modules = [
        _rich_module(
            "m1",
            "Conditional Logic and Branching",
            "Control Flow",
            "step_01",
            ["Control Flow", "If statements", "Branching"],
        ),
        _rich_module(
            "m2",
            "Loops and Iteration Patterns",
            "Loops and Iteration",
            "step_01",
            ["Loops", "Iteration"],
        ),
        _rich_module(
            "m3",
            "Function Basics",
            "Function Basics",
            "step_01",
            ["Function Basics", "Parameters", "Return Values"],
        ),
    ]

    result = validate_curriculum_quality(
        topic="Python",
        modules=modules,
        profile={"topic": "Python", "learning_goal": "learn Python", "target_context": "Python"},
        scope_analysis=CourseScopeAnalysis(recommended_module_count=3),
        student_history={},
        concept_inventory={
            "core_concepts": ["Control structures and functions"],
            "concepts_to_delay": [],
            "concepts_to_skip": [],
        },
        prerequisite_graph={"Control structures and functions": []},
        roadmap_steps=[
            "Control structures and functions",
            "Conditional logic and branching",
            "Functions with parameters and return values",
        ],
        schedule=None,
    )

    assert result["passed"] is True


def test_validate_modules_against_roadmap_fails_unmapped_step():
    roadmap = MasterRoadmap(
        topic="Python",
        goal="programming",
        steps=[
            _step("step_01", "values"),
            _step("step_02", "control_flow", ["step_01"]),
            _step("step_03", "functions", ["step_02"]),
        ],
    )
    modules = [
        _module("m1", "values", "step_01"),
        _module("m2", "control flow", "step_02"),
    ]

    result = validate_modules_against_roadmap(
        modules, roadmap, CourseScopeAnalysis(), {}
    )

    assert result["passed"] is False
    assert "covered" in result["issues"][0].lower() or "step" in result["issues"][0].lower()


def test_module_roadmap_step_id_default():
    module = Module(
        id="m1",
        title="Values",
        concept="values",
        domain_framing="values in Python",
        prerequisites=[],
        estimated_minutes=15,
        depth_level="standard",
    )

    assert module.roadmap_step_id == ""


@pytest.mark.asyncio
async def test_decision_log_flushed_after_course_creation(monkeypatch):
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
                roadmap_step_id="step_01",
            )
        ],
    )

    class FakeState:
        student_id = "s1"
        domain = "machine learning"
        goal = "understand embeddings"
        pace = "medium"
        curriculum = None
        concept_mastery = {}
        concept_depth = {}
        session_decisions = [
            AdaptationDecision(
                action="BUILD_CURRICULUM",
                reason="roadmap-first plan created",
                agent="curriculum_architect",
            )
        ]

    class FakeArchitect:
        def __init__(self, state):
            self.state = state
            self._master_roadmap = MasterRoadmap(
                topic="Vectors",
                goal="understand embeddings",
                steps=[_step("step_01", "vectors"), _step("step_02", "similarity", ["step_01"])],
            )

        async def build_curriculum(self, topic):
            return plan, 123

    async def fake_noop(*args, **kwargs):
        return None

    async def fake_latest(student_id):
        return {"id": 42}

    async def fake_create_record(**kwargs):
        return {"id": "course-42", "topic": "Vectors", "goal": "understand embeddings", "pace": "medium"}

    async def fake_modules(course_id):
        return []

    async def fake_save_roadmap(course_id, roadmap):
        return roadmap

    bulk_write = AsyncMock()
    save_master = AsyncMock(return_value={})

    monkeypatch.setattr(course_service, "upsert_student", fake_noop)
    monkeypatch.setattr(course_service.StudentState, "load", AsyncMock(return_value=FakeState()))
    monkeypatch.setattr(course_service, "CurriculumArchitectAgent", FakeArchitect)
    monkeypatch.setattr(course_service, "latest_curriculum_for_student", fake_latest)
    monkeypatch.setattr(course_service, "create_course_from_plan", fake_create_record)
    monkeypatch.setattr(course_service, "list_course_modules", fake_modules)
    monkeypatch.setattr(course_service, "get_student_history_snapshot", AsyncMock(return_value={}))
    monkeypatch.setattr(course_service, "save_course_roadmap", fake_save_roadmap)
    monkeypatch.setattr(course_service, "save_master_roadmap", save_master)
    monkeypatch.setattr(course_service, "bulk_write_decisions", bulk_write)

    course = await course_service.create_course(
        student_id="s1",
        topic="Vectors",
        goal="understand embeddings",
        pace="medium",
        name="Dana",
    )

    assert course["id"] == "course-42"
    save_master.assert_awaited_once()
    bulk_write.assert_awaited_once()
    records = bulk_write.await_args.args[0]
    assert records
    assert records[0]["course_id"] == "course-42"
    assert records[0]["session_id"] == "course:course-42"
