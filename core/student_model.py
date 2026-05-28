"""
core/student_model.py
Single source of truth for all EduMind data models.
Never import or redefine these elsewhere.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, PrivateAttr

from config import settings


# ── Metacognition Profile ─────────────────────────────────────────────────────

class MetacognitionProfile(BaseModel):
    # Style evidence — {style_name: [depth_scores per session]}
    style_depth_scores: dict[str, list[float]] = Field(default_factory=dict)
    preferred_style: str = "formal"  # derived from style_depth_scores

    # Lesson length — learned from fatigue data
    optimal_lesson_minutes: int = 10
    fatigue_threshold_minutes: int = 25

    # Calibration — updated after every evaluation
    calibration_history: list[float] = Field(default_factory=list)
    calibration_pattern: Literal[
        "unknown", "overconfident", "underconfident", "calibrated"
    ] = "unknown"

    # Concept type performance
    weak_concept_types: list[str] = Field(default_factory=list)
    strong_concept_types: list[str] = Field(default_factory=list)

    # Reteach tracking
    consecutive_reteach_count: int = 0
    depth_concern_flag: bool = False

    def record_style_depth(self, style: str, depth_score: float) -> None:
        """Append a depth score for the given style after each session."""
        if style not in self.style_depth_scores:
            self.style_depth_scores[style] = []
        self.style_depth_scores[style].append(round(depth_score, 4))
        self._update_preferred_style()

    def _update_preferred_style(self) -> None:
        """Derive preferred_style from average depth scores (min 2 data points)."""
        best_style = self.preferred_style
        best_avg = -1.0
        for style, scores in self.style_depth_scores.items():
            if len(scores) >= 2:
                avg = sum(scores) / len(scores)
                if avg > best_avg:
                    best_avg = avg
                    best_style = style
        self.preferred_style = best_style

    def update_calibration(self, delta: float) -> None:
        """
        delta = stated_confidence/5 - mastery_score
        positive → overconfident, negative → underconfident
        """
        self.calibration_history.append(round(delta, 4))
        if len(self.calibration_history) < 3:
            return  # not enough evidence yet
        recent = self.calibration_history[-5:]
        avg = sum(recent) / len(recent)
        if avg > 0.15:
            self.calibration_pattern = "overconfident"
        elif avg < -0.15:
            self.calibration_pattern = "underconfident"
        else:
            self.calibration_pattern = "calibrated"


# ── Curriculum ────────────────────────────────────────────────────────────────

class Module(BaseModel):
    id: str
    title: str
    concept: str
    domain_framing: str       # e.g. "matrix as linear transformation for ML"
    prerequisites: list[str]
    estimated_minutes: int
    depth_level: Literal["surface", "standard", "deep"]
    purpose: str = ""
    why_it_matters_for_goal: str = ""
    difficulty: str = ""
    must_teach: list[str] = Field(default_factory=list)
    examples_to_include: list[str] = Field(default_factory=list)
    practice_type: str = ""
    concepts_taught: list[str] = Field(default_factory=list)
    depends_on_concepts: list[str] = Field(default_factory=list)
    unlocks_concepts: list[str] = Field(default_factory=list)
    module_goal: str = ""
    why_now: str = ""
    what_this_module_will_not_cover: list[str] = Field(default_factory=list)
    lesson_requirements: list[str] = Field(default_factory=list)
    practice_requirements: list[str] = Field(default_factory=list)
    question_scope: list[str] = Field(default_factory=list)
    roadmap_step_id: str = ""


class IntentAnalysis(BaseModel):
    subject: str
    exact_subject: str
    target_context: str
    course_type: str
    learner_stage: str
    must_include: list[str] = Field(default_factory=list)
    should_avoid: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    confidence: float = 0.7
    reasoning: str = ""
    is_ambiguous: bool = False


class ResearchQuery(BaseModel):
    query: str
    category: str
    priority: int


class ResearchSummary(BaseModel):
    queries_run: list[ResearchQuery] = Field(default_factory=list)
    raw_results: dict[str, str] = Field(default_factory=dict)
    summary_by_category: dict[str, str] = Field(default_factory=dict)
    coverage_confidence: float = 0.5
    full_text: str = ""


class RoadmapStep(BaseModel):
    step_id: str
    title: str
    concept_cluster: str
    subtopics: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    estimated_minutes: int = 30
    why_this_step_exists: str = ""
    goal_alignment: str = ""
    depth_level: str = "standard"
    module_generation_hint: str = ""
    must_teach: list[str] = Field(default_factory=list)
    examples_to_include: list[str] = Field(default_factory=list)
    practice_requirements: list[str] = Field(default_factory=list)
    lesson_requirements: list[str] = Field(default_factory=list)
    question_scope: list[str] = Field(default_factory=list)
    check_question_targets: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    teaching_sequence: list[str] = Field(default_factory=list)
    mini_task: str = ""
    learning_objective: str = ""
    do_not_cover: list[str] = Field(default_factory=list)
    module_split_hint: str = ""
    is_optional: bool = False


class MasterRoadmap(BaseModel):
    course_id: str = ""
    topic: str
    goal: str
    steps: list[RoadmapStep]
    total_estimated_minutes: int = 0
    rationale: str = ""
    created_at: str = ""
    research_summary: dict[str, Any] = Field(default_factory=dict)


class CurriculumDecisionLog(BaseModel):
    course_id: str
    intent_analysis: dict[str, Any] = Field(default_factory=dict)
    research_summary: dict[str, Any] = Field(default_factory=dict)
    scope_decision: dict[str, Any] = Field(default_factory=dict)
    roadmap_step_count: int = 0
    module_count: int = 0
    module_count_vs_scope_estimate: str = ""
    decisions: list[dict[str, Any]] = Field(default_factory=list)


class CurriculumPlan(BaseModel):
    topic: str
    domain: str
    goal: str
    modules: list[Module]
    current_index: int = 0
    version: int = 1
    scope_analysis: dict[str, Any] = Field(default_factory=dict)
    concept_inventory: dict[str, Any] = Field(default_factory=dict)
    prerequisite_graph: dict[str, list[str]] = Field(default_factory=dict)
    learning_path: list[dict[str, Any]] = Field(default_factory=list)
    roadmap_steps: list[str] = Field(default_factory=list)
    schedule_plan: list[dict[str, Any]] = Field(default_factory=list)
    repair_history: list[dict[str, Any]] = Field(default_factory=list)
    validation_result: dict[str, Any] = Field(default_factory=dict)


# ── Evaluation ────────────────────────────────────────────────────────────────

class EvaluationReport(BaseModel):
    concept: str
    session_id: str
    correctness_score: float      # 0.0–1.0
    depth_score: float            # 0.0–1.0
    mastery_score: float          # 0.6*correctness + 0.4*depth
    misconception_type: Optional[
        Literal["conceptual", "formula_misuse", "application_error"]
    ] = None
    misconception_detail: str = ""
    confidence_stated: int        # 1–5 from student
    calibration_delta: float      # stated/5 - mastery_score
    questions_asked: int
    recommended_action: Literal[
        "MOVE_FORWARD", "MOVE_FORWARD_WITH_FLAG",
        "RETEACH", "DETOUR", "ESCALATE", "COMPRESS", "HOLD"
    ]


# ── Adaptation ────────────────────────────────────────────────────────────────

class AdaptationDecision(BaseModel):
    action: str
    reason: str
    agent: str = "adaptation_engine"        # which agent made this decision
    style_for_reteach: Optional[str] = None
    missing_concept: Optional[str] = None   # for DETOUR
    metacognition_updates: dict[str, Any] = Field(default_factory=dict)


# ── Student State ─────────────────────────────────────────────────────────────

class StudentState(BaseModel):
    # Identity
    student_id: str
    name: str = ""
    domain: str
    goal: str
    pace: Literal["fast", "medium", "deep"] = "medium"

    # Knowledge
    concept_mastery: dict[str, float] = Field(default_factory=dict)
    concept_depth: dict[str, float] = Field(default_factory=dict)

    # Curriculum
    curriculum: Optional[CurriculumPlan] = None

    # Metacognition — the long-term adaptation memory
    metacognition: MetacognitionProfile = Field(
        default_factory=MetacognitionProfile
    )

    # Session-scoped (reset each session; NOT written mid-session)
    session_id: str = ""
    session_doubt_counts: dict[str, int] = Field(default_factory=dict)
    session_doubt_types: dict[str, dict[str, int]] = Field(default_factory=dict)
    session_decisions: list[AdaptationDecision] = Field(default_factory=list)
    session_response_times: list[float] = Field(default_factory=list)
    # Lesson text delivered during this session, keyed by module id.
    # EvaluatorAgent uses this as the authoritative source for check questions.
    session_module_content: dict[str, str] = Field(default_factory=dict)
    # In-memory log of EvaluationReports for this session.
    # Populated by EvaluatorAgent after each evaluation.
    # Used by AdaptationEngine.run_gap_analysis() — never persisted here
    # (persisted separately to evaluation_history table by write_evaluation()).
    evaluation_history: list["EvaluationReport"] = Field(default_factory=list)
    evaluation_cycle_count: int = 0
    session_started_at: Optional[datetime] = None

    # Private dirty-tracking set (excluded from serialisation)
    _dirty: set[str] = PrivateAttr(default_factory=set)

    # ── Pace threshold ────────────────────────────────────────────────────────
    @property
    def advance_threshold(self) -> float:
        return {
            "fast": settings.mastery_threshold_fast,
            "medium": settings.mastery_threshold_medium,
            "deep": settings.mastery_threshold_deep
        }[self.pace]
        
    # ── Dirty tracking ────────────────────────────────────────────────────────

    def mark_dirty(self, field: str) -> None:
        self._dirty.add(field)

    def clear_dirty(self) -> None:
        self._dirty.clear()

    # ── Knowledge updates ────────────────────────────────────────────────────

    def update_mastery(self, concept: str, correctness: float, depth: float) -> None:
        # Safety floor on depth prevents zero-division in derived calculations
        # and ensures mastery never collapses to correctness-only silently
        safe_depth = max(0.001, depth)
        mastery = round(0.6 * correctness + 0.4 * safe_depth, 3)
        self.concept_mastery[concept] = mastery
        self.concept_depth[concept] = round(safe_depth, 3)
        self.mark_dirty("concept_mastery")

    def get_mastery(self, concept: str) -> float:
        return self.concept_mastery.get(concept, 0.0)

    def ready_to_advance(self, concept: str) -> bool:
        return self.get_mastery(concept) >= self.advance_threshold

    # ── Doubt tracking ───────────────────────────────────────────────────────

    def log_doubt(self, concept: str, doubt_type: str = "general") -> None:
        doubt_type = (doubt_type or "general").strip() or "general"
        self.session_doubt_counts[concept] = (
            self.session_doubt_counts.get(concept, 0) + 1
        )
        if concept not in self.session_doubt_types:
            self.session_doubt_types[concept] = {}
        self.session_doubt_types[concept][doubt_type] = (
            self.session_doubt_types[concept].get(doubt_type, 0) + 1
        )
        self.mark_dirty("session_doubt_counts")

    def get_doubt_count(self, concept: str) -> int:
        return self.session_doubt_counts.get(concept, 0)

    # ── Module content tracking ──────────────────────────────────────────────

    def record_module_content(self, module_id: str, text: str) -> None:
        """Append delivered lesson/module text for grounded evaluation."""
        clean_text = text.strip()
        if not module_id or not clean_text:
            return

        existing = self.session_module_content.get(module_id, "")
        if existing:
            self.session_module_content[module_id] = existing + "\n\n" + clean_text
        else:
            self.session_module_content[module_id] = clean_text
        self.mark_dirty("session_module_content")

    def get_module_content(self, module_id: str) -> str:
        """Return lesson/module text recorded for the current session."""
        return self.session_module_content.get(module_id, "")

    # ── Session management ───────────────────────────────────────────────────

    def start_session(self) -> str:
        """Generate a fresh session_id and reset all session-scoped fields."""
        from clients.tavily_client import clear_cache
        clear_cache()  # prevent stale Tavily results leaking across sessions
        self.session_id = str(uuid.uuid4())
        self.session_doubt_counts = {}
        self.session_doubt_types = {}
        self.session_decisions = []
        self.session_response_times = []
        self.session_module_content = {}
        self.evaluation_history = []
        self.evaluation_cycle_count = 0
        self.session_started_at = datetime.now(timezone.utc)
        return self.session_id

    def add_decision(self, decision: AdaptationDecision) -> None:
        self.session_decisions.append(decision)

    # ── Persistence ──────────────────────────────────────────────────────────

    @classmethod
    async def load(cls, student_id: str) -> "StudentState":
        """
        Load a StudentState from PostgreSQL.
        Returns a new default state if the student doesn't exist yet.
        """
        from db.postgres import get_conn

        async with get_conn() as conn:
            # Basic identity
            row = await conn.fetchrow(
                "SELECT * FROM students WHERE student_id=$1", student_id
            )
            if row is None:
                raise ValueError(f"Student '{student_id}' not found in DB.")

            state = cls(
                student_id=student_id,
                name=row["name"],
                domain=row["domain"],
                goal=row["goal"],
                pace=row["pace"],
            )

            import json

            # Concept mastery
            mastery_rows = await conn.fetch(
                "SELECT concept, mastery_score, depth FROM concept_mastery "
                "WHERE student_id=$1", student_id
            )
            for mr in mastery_rows:
                state.concept_mastery[mr["concept"]] = mr["mastery_score"]
                state.concept_depth[mr["concept"]] = mr["depth"]

            # Metacognition
            meta_row = await conn.fetchrow(
                "SELECT profile_json FROM metacognition WHERE student_id=$1",
                student_id,
            )
            if meta_row:
                profile_data = json.loads(meta_row["profile_json"])
                state.metacognition = MetacognitionProfile(**profile_data)

            # Active curriculum
            cur_row = await conn.fetchrow(
                "SELECT plan_json, current_index, version FROM curricula "
                "WHERE student_id=$1 AND is_active=TRUE ORDER BY id DESC LIMIT 1",
                student_id,
            )
            if cur_row:
                plan_data = json.loads(cur_row["plan_json"])
                plan_data["current_index"] = cur_row["current_index"]
                plan_data["version"] = cur_row["version"]
                state.curriculum = CurriculumPlan(**plan_data)

        state.clear_dirty()
        return state

    async def save(self) -> None:
        """
        Persist dirty fields to PostgreSQL.
        Called at session-end (Layer 3) or explicitly after mid-session eval.
        """
        from db.postgres import (
            get_conn,
            upsert_student,
            save_metacognition,
            upsert_concept_mastery,
            bulk_write_decisions,
        )

        # Always keep students row current
        await upsert_student(
            self.student_id, self.name, self.domain, self.goal, self.pace
        )

        if "concept_mastery" in self._dirty:
            for concept, mastery in self.concept_mastery.items():
                depth = self.concept_depth.get(concept, 0.0)
                correctness = (mastery - 0.4 * depth) / 0.6 if depth else mastery
                await upsert_concept_mastery(
                    self.student_id, concept, correctness, depth
                )

        # Always save metacognition at session-end
        await save_metacognition(
            self.student_id, self.metacognition.model_dump()
        )

        # Session decisions
        if self.session_decisions:
            decision_records = [
                {
                    "student_id": self.student_id,
                    "session_id": self.session_id,
                    "agent": getattr(d, "agent", "adaptation_engine"),
                    "action": d.action,
                    "rationale": d.reason,
                    "payload": d.model_dump(),
                }
                for d in self.session_decisions
            ]
            await bulk_write_decisions(decision_records)

        # Persist session doubt counts to doubt_log
        if self.session_doubt_counts:
            async with get_conn() as conn:
                for concept, count in self.session_doubt_counts.items():
                    type_counts = self.session_doubt_types.get(concept) or {"general": count}
                    for doubt_type, type_count in type_counts.items():
                        await conn.execute(
                            """
                            INSERT INTO doubt_log
                              (student_id, session_id, concept, count, doubt_type)
                            VALUES ($1, $2, $3, $4, $5)
                            """,
                            self.student_id,
                            self.session_id,
                            concept,
                            type_count,
                            doubt_type,
                        )

        # Curriculum current_index
        if self.curriculum is not None and "curriculum" in self._dirty:
            async with get_conn() as conn:
                await conn.execute(
                    """
                    UPDATE curricula SET current_index=$1, updated_at=NOW()
                    WHERE student_id=$2 AND is_active=TRUE
                    """,
                    self.curriculum.current_index, self.student_id,
                )
        # ===== METACOGNITION SAVE  =====
        if "metacognition" in self._dirty:
            await save_metacognition(
                self.student_id,
                self.metacognition.model_dump()
            )
        # ==============================================

        self.clear_dirty()

    # ── Context summary for LLM prompts ──────────────────────────────────────

    def as_prompt_context(self) -> str:
        """Return a compact string summary for injection into agent prompts."""
        meta = self.metacognition
        current_concept = ""
        if self.curriculum and self.curriculum.current_index < len(
            self.curriculum.modules
        ):
            current_concept = self.curriculum.modules[
                self.curriculum.current_index
            ].concept

        # Mastery breadth — how many concepts the student has solidly learned
        mastered_count = sum(1 for m in self.concept_mastery.values() if m >= 0.7)
        partial_count = sum(
            1 for m in self.concept_mastery.values() if 0.3 <= m < 0.7
        )
        mastered_concepts = [c for c, m in self.concept_mastery.items() if m >= 0.7]

        # Prior curriculum summary — helps architect size new courses correctly
        prior_topic = ""
        prior_modules_total = 0
        prior_modules_done = 0
        if self.curriculum:
            prior_topic = self.curriculum.topic
            prior_modules_total = len(self.curriculum.modules)
            prior_modules_done = self.curriculum.current_index

        return (
            f"Student: {self.name or self.student_id}\n"
            f"Domain: {self.domain} | Goal: {self.goal} | Pace: {self.pace}\n"
            f"Advance threshold: {self.advance_threshold}\n"
            f"Current concept: {current_concept}\n"
            f"Preferred style: {meta.preferred_style}\n"
            f"Style evidence: {dict(meta.style_depth_scores)}\n"
            f"Calibration: {meta.calibration_pattern}\n"
            f"Calibration history (last 5): {meta.calibration_history[-5:]}\n"
            f"Weak concept types: {meta.weak_concept_types}\n"
            f"Reteach count: {meta.consecutive_reteach_count}\n"
            f"Depth concern: {meta.depth_concern_flag}\n"
            f"Session doubts: {dict(self.session_doubt_counts)}\n"
            f"Eval cycle: {self.evaluation_cycle_count}\n"
            f"\n--- KNOWLEDGE PROFILE ---\n"
            f"Total concepts mastered (>=0.7): {mastered_count}\n"
            f"Partial concepts (0.3-0.7): {partial_count}\n"
            f"Mastered concepts: {', '.join(mastered_concepts) if mastered_concepts else 'none'}\n"
            f"\n--- COURSE HISTORY ---\n"
            f"Previous curriculum topic: {prior_topic if prior_topic else 'none (first course)'}\n"
            f"Previous course size: {prior_modules_total} modules total, "
            f"{prior_modules_done} completed\n"
            f"Student experience level: "
            f"{'expert' if mastered_count >= 10 else 'intermediate' if mastered_count >= 4 else 'beginner'}\n"
        )
