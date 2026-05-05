"""
core/student_model.py
Single source of truth for all EduMind data models.
Never import or redefine these elsewhere.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
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


class CurriculumPlan(BaseModel):
    topic: str
    domain: str
    goal: str
    modules: list[Module]
    current_index: int = 0
    version: int = 1


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
    session_decisions: list[AdaptationDecision] = Field(default_factory=list)
    session_response_times: list[float] = Field(default_factory=list)
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
        mastery = round(0.6 * correctness + 0.4 * depth, 3)
        self.concept_mastery[concept] = mastery
        self.concept_depth[concept] = round(depth, 3)
        self.mark_dirty("concept_mastery")

    def get_mastery(self, concept: str) -> float:
        return self.concept_mastery.get(concept, 0.0)

    def ready_to_advance(self, concept: str) -> bool:
        return self.get_mastery(concept) >= self.advance_threshold

    # ── Doubt tracking ───────────────────────────────────────────────────────

    def log_doubt(self, concept: str, doubt_type: str = "general") -> None:
        self.session_doubt_counts[concept] = (
            self.session_doubt_counts.get(concept, 0) + 1
        )
        self.mark_dirty("session_doubt_counts")

    def get_doubt_count(self, concept: str) -> int:
        return self.session_doubt_counts.get(concept, 0)

    # ── Session management ───────────────────────────────────────────────────

    def start_session(self) -> str:
        """Generate a fresh session_id and reset all session-scoped fields."""
        from clients.tavily_client import clear_cache
        clear_cache()  # prevent stale Tavily results leaking across sessions
        self.session_id = str(uuid.uuid4())
        self.session_doubt_counts = {}
        self.session_decisions = []
        self.session_response_times = []
        self.evaluation_cycle_count = 0
        self.session_started_at = datetime.utcnow()
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
            # ===== METACOGNITION LOAD =====
            import json

            try:
                meta_row = await conn.fetchrow(
                    "SELECT profile_json FROM metacognition WHERE student_id=$1",
                    student_id
                )

                if meta_row and meta_row["profile_json"]:
                    meta_data = json.loads(meta_row["profile_json"])
                    state.metacognition = MetacognitionProfile(**meta_data)

            except Exception as e:
                print(f"[WARN] Failed to load metacognition: {e}")
                state.metacognition = MetacognitionProfile()
            # =============================================

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
                    "agent": "adaptation_engine",
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
                    await conn.execute(
                        """
                        INSERT INTO doubt_log
                          (student_id, session_id, concept, count, doubt_type)
                        VALUES ($1, $2, $3, $4, 'general')
                        ON CONFLICT DO NOTHING
                        """,
                        self.student_id, self.session_id, concept, count,
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
        )
