"""
agents/adaptation_engine.py
AdaptationEngine — reads EvaluationReport + StudentState, decides next action.

Terminal tool: submit_decision
Non-terminal tools: analyse_metacognition, check_prerequisites
"""

from __future__ import annotations

from loguru import logger

from agents.base_agent import BaseAgent
from core.student_model import StudentState, EvaluationReport, AdaptationDecision
from config import settings


class AdaptationEngine(BaseAgent):
    NAME = "adaptation_engine"
    TERMINAL_TOOL = "submit_decision"

    def __init__(self, state: StudentState):
        super().__init__(state)

        self.TOOLS = [
            self.build_tool(
                name="analyse_metacognition",
                description=(
                    "Analyse the student's metacognition profile to inform the adaptation decision. "
                    "Call this to check calibration pattern, reteach count, and style preferences."
                ),
                properties={
                    "focus": {
                        "type": "string",
                        "enum": ["calibration", "style", "fatigue", "reteach_risk"],
                        "description": "Which aspect of metacognition to analyse",
                    },
                },
                required=["focus"],
            ),
            self.build_tool(
                name="check_prerequisites",
                description=(
                    "Check if the student has sufficient mastery of prerequisites "
                    "for the current concept. Use when DETOUR is being considered."
                ),
                properties={
                    "concept": {
                        "type": "string",
                        "description": "The concept whose prerequisites to check",
                    },
                },
                required=["concept"],
            ),
            self.build_tool(
                name="submit_decision",
                description=(
                    "Submit the final adaptation decision. "
                    "Call this after analysis is complete."
                ),
                properties={
                    "action": {
                        "type": "string",
                        "enum": [
                            "MOVE_FORWARD",
                            "MOVE_FORWARD_WITH_FLAG",
                            "RETEACH",
                            "DETOUR",
                            "ESCALATE",
                            "COMPRESS",
                            "HOLD",
                        ],
                        "description": "The adaptation action to take",
                    },
                    "reason": {
                        "type": "string",
                        "description": "One sentence explanation of the decision",
                    },
                    "style_for_reteach": {
                        "type": "string",
                        "description": "If RETEACH, which style to use. One of: formal, analogy, example_first, visual, story. Leave empty if not RETEACH.",
                    },
                    "missing_concept": {
                        "type": "string",
                        "description": "If DETOUR, the prerequisite concept to teach first",
                    },
                    "metacognition_updates": {
                        "type": "object",
                        "description": (
                            "JSON object of metacognition fields to update. "
                            "E.g. {'consecutive_reteach_count': 2, 'depth_concern_flag': true}"
                        ),
                    },
                },
                required=["action", "reason"],
            ),
        ]
    # ── Gap Analysis ──────────────────────────────────────────────────────────

    async def run_gap_analysis(self) -> str | None:
        """
        Every 3 evaluation cycles, analyse the session's evaluation history
        for a pattern of weakness indicating a missing prerequisite.

        This is a genuine agentic call: the LLM receives the full eval history
        and uses the check_prerequisites tool before deciding whether a gap
        exists and what concept is missing.

        Returns:
            The missing concept name if a gap is found, else None.
        """
        # Only run every 3 completed evaluations
        if self.state.evaluation_cycle_count == 0:
            return None
        if self.state.evaluation_cycle_count % 3 != 0:
            return None

        # Need at least 3 evaluation records in memory this session
        history = self.state.evaluation_history
        if len(history) < 3:
            return None

        recent = history[-3:]
        weak = [r for r in recent if r.mastery_score < 0.5]

        # Only act if at least 2 of the last 3 evaluations show weakness
        if len(weak) < 2:
            return None

        weak_concepts = [r.concept for r in weak]
        weak_details = "\n".join(
            f"  - {r.concept}: mastery={r.mastery_score:.2f}, "
            f"correctness={r.correctness_score:.2f}, depth={r.depth_score:.2f}, "
            f"misconception={r.misconception_type or 'none'}"
            for r in weak
        )

        system = f"""You are a prerequisite gap analyst for an adaptive learning system.
A student is consistently failing multiple concepts. Your job is to identify
the single most likely missing prerequisite concept causing these failures.

STUDENT CONTEXT:
{self._student_context()}

ANALYSIS RULES:
1. Call check_prerequisites on the concept with the lowest mastery score
2. Then call submit_decision with:
   - action = "DETOUR"
   - missing_concept = the prerequisite the student is most likely missing
   - reason = one sentence explaining the pattern
   
   OR if no clear prerequisite gap exists:
   - action = "RETEACH"
   - reason = "No single prerequisite gap identified; recommend style change"

The missing_concept field is the KEY OUTPUT — it must be a specific, teachable
concept name (e.g. "function closures", "matrix multiplication", "gradient descent").
"""

        result = await self.run(
            system=system,
            user_message=(
                f"Gap analysis: student failed {len(weak)}/3 recent concepts.\n\n"
                f"Weak concepts:\n{weak_details}\n\n"
                f"Domain: {self.state.domain}\n"
                f"Identify the most likely missing prerequisite."
            ),
            model=settings.reasoning_model,
        )

        missing = result.get("missing_concept")
        action = result.get("action", "")

        if action == "DETOUR" and missing:
            self._log_decision(
                action="GAP_DETECTED",
                reason=result.get("reason", f"Gap analysis found missing: {missing}"),
                payload={"missing_concept": missing, "weak_concepts": weak_concepts},
            )
            logger.info("Gap analysis found missing prerequisite: '{}'", missing)
            return missing

        logger.info("Gap analysis: no clear prerequisite gap in {}", weak_concepts)
        return None


    # ── Tool executor ─────────────────────────────────────────────────────────

    async def _execute_tool(self, tool_name: str, args: dict) -> str:
        meta = self.state.metacognition

        if tool_name == "analyse_metacognition":
            focus = args["focus"]

            if focus == "calibration":
                return (
                    f"Calibration pattern: {meta.calibration_pattern}. "
                    f"History (last 5): {meta.calibration_history[-5:]}. "
                    f"Overconfident students need harder questions and deeper probing."
                )

            if focus == "style":
                scores_summary = {
                    s: round(sum(v)/len(v), 2)
                    for s, v in meta.style_depth_scores.items() if v
                }
                return (
                    f"Preferred style: {meta.preferred_style}. "
                    f"Depth scores by style: {scores_summary}. "
                    f"For reteach, choose a style different from '{meta.preferred_style}'."
                )

            if focus == "fatigue":
                session_doubts = sum(self.state.session_doubt_counts.values())
                return (
                    f"Session doubts so far: {session_doubts}. "
                    f"Optimal lesson length: {meta.optimal_lesson_minutes} min. "
                    f"Fatigue threshold: {meta.fatigue_threshold_minutes} min."
                )

            if focus == "reteach_risk":
                return (
                    f"Consecutive reteach count: {meta.consecutive_reteach_count}. "
                    f"Depth concern flag: {meta.depth_concern_flag}. "
                    f"If reteach >= 2 and mastery remains below threshold, recommend ESCALATE."
                )

            return "Unknown focus."

        if tool_name == "check_prerequisites":
            concept = args["concept"]
            module = self._current_module()
            prereqs = module.prerequisites if module else []

            if not prereqs:
                return f"No prerequisites defined for '{concept}'."

            results = []
            for prereq in prereqs:
                mastery = self.state.get_mastery(prereq)
                status = "✅" if mastery >= self.state.advance_threshold else "❌"
                results.append(f"{status} {prereq}: mastery={mastery:.2f}")

            weak = [p for p in prereqs
                    if self.state.get_mastery(p) < self.state.advance_threshold]
            if weak:
                return (
                    f"Prerequisite check for '{concept}':\n" +
                    "\n".join(results) +
                    f"\n\nWeak prerequisites: {weak}. Consider DETOUR to: {weak[0]}"
                )
            return f"All prerequisites met for '{concept}':\n" + "\n".join(results)

        return await super()._execute_tool(tool_name, args)

    def _alternate_style(self) -> str:
        order = ["formal", "analogy", "example_first", "visual", "story"]
        current = self.state.metacognition.preferred_style
        for style in order:
            if style != current:
                return style
        return "example_first"

    def _first_weak_prerequisite(self, concept: str) -> str | None:
        module = self._current_module()
        prereqs = module.prerequisites if module else []
        for prereq in prereqs:
            if self.state.get_mastery(prereq) < self.state.advance_threshold:
                return prereq
        return None

    def _explicit_decision(
        self,
        report: EvaluationReport,
        llm_result: dict | None = None,
    ) -> tuple[str, str, str | None, str | None, dict]:
        """
        Apply the product rules directly. The LLM may provide missing_concept
        wording, but it cannot override the score/threshold decision boundary.
        """
        llm_result = llm_result or {}
        meta = self.state.metacognition
        threshold = self.state.advance_threshold
        mastery = report.mastery_score
        misconception = report.misconception_type
        style_for_reteach: str | None = None
        missing_concept: str | None = None
        updates: dict = {}

        if report.recommended_action == "HOLD":
            return (
                "HOLD",
                "Student requested a pause or stop, so the session should hold.",
                None,
                None,
                {},
            )

        if report.recommended_action == "COMPRESS" and mastery >= 0.85:
            return (
                "COMPRESS",
                f"Mastery {mastery:.2f} shows the student is clearly ahead of this module.",
                None,
                None,
                {"consecutive_reteach_count": 0},
            )

        if mastery >= threshold and not misconception:
            return (
                "MOVE_FORWARD",
                f"Mastery {mastery:.2f} cleared the {threshold:.2f} threshold with no misconception.",
                None,
                None,
                {"consecutive_reteach_count": 0},
            )

        if mastery >= threshold and misconception:
            return (
                "MOVE_FORWARD_WITH_FLAG",
                f"Mastery cleared threshold, but {misconception} needs targeted follow-up.",
                None,
                None,
                {"consecutive_reteach_count": 0, "flagged_misconception": misconception},
            )

        if meta.consecutive_reteach_count >= 2 and mastery < threshold:
            return (
                "ESCALATE",
                f"Mastery {mastery:.2f} is still below threshold after {meta.consecutive_reteach_count} reteach cycle(s).",
                None,
                None,
                {"depth_concern_flag": True},
            )

        if mastery < 0.4 or report.recommended_action == "DETOUR":
            missing_concept = (
                self._first_weak_prerequisite(report.concept)
                or llm_result.get("missing_concept")
                or f"prerequisite for {report.concept}"
            )
            return (
                "DETOUR",
                f"Mastery {mastery:.2f} or the evaluator's report suggests a missing prerequisite before continuing.",
                None,
                missing_concept,
                {"depth_concern_flag": True},
            )

        style_for_reteach = (
            llm_result.get("style_for_reteach")
            if llm_result.get("style_for_reteach") not in (None, "", "none")
            else self._alternate_style()
        )
        updates = {"consecutive_reteach_count": meta.consecutive_reteach_count + 1}
        if report.depth_score < 0.5:
            updates["depth_concern_flag"] = True
        return (
            "RETEACH",
            f"Mastery {mastery:.2f} is below the {threshold:.2f} threshold; reteach with a different style.",
            style_for_reteach,
            None,
            updates,
        )

    # ── Public run method ─────────────────────────────────────────────────────

    async def decide(self, report: EvaluationReport) -> AdaptationDecision:
        """
        Analyse an EvaluationReport and decide the next action.

        Args:
            report: EvaluationReport from EvaluatorAgent

        Returns:
            AdaptationDecision (also logged to state.session_decisions)
        """
        meta = self.state.metacognition
        module = self._current_module()
        concept = module.concept if module else report.concept

        system = f"""You are an adaptation engine for an adaptive learning system.
Analyse the student's evaluation result and decide the optimal next action.

STUDENT CONTEXT:
{self._student_context()}

EVALUATION RESULT:
- Concept: {report.concept}
- Correctness: {report.correctness_score:.2f}
- Depth: {report.depth_score:.2f}
- Mastery: {report.mastery_score:.2f}
- Advance threshold: {self.state.advance_threshold}
- Misconception: {report.misconception_type} — {report.misconception_detail}
- Evaluator recommended: {report.recommended_action}
- Confidence stated: {report.confidence_stated}/5
- Calibration delta: {report.calibration_delta:+.2f}

METACOGNITION SUMMARY:
- Calibration: {meta.calibration_pattern}
- Consecutive reteach count: {meta.consecutive_reteach_count}
- Depth concern: {meta.depth_concern_flag}
- Preferred style: {meta.preferred_style}

DECISION RULES:
1. Call analyse_metacognition to check relevant patterns
2. If considering DETOUR, call check_prerequisites
3. Then call submit_decision with your final choice

ACTION GUIDE:
- MOVE_FORWARD: mastery >= {self.state.advance_threshold}
- MOVE_FORWARD_WITH_FLAG: mastery >= {self.state.advance_threshold} with a non-blocking misconception flag
- RETEACH: 0.4 <= mastery < {self.state.advance_threshold}, reteach_count < 2 → pick DIFFERENT style
- DETOUR: mastery < 0.4 or prerequisite gap → teach missing concept first
- ESCALATE: reteach_count >= 2 and mastery remains below threshold → rebuild the course sequence
- COMPRESS: evaluator explicitly recommends compression and mastery >= 0.85
- HOLD: student requested break

METACOGNITION UPDATE RULES:
- RETEACH → increment consecutive_reteach_count
- MOVE_FORWARD → reset consecutive_reteach_count to 0
- depth_score < 0.5 for 2nd time → set depth_concern_flag=true
"""

        try:
            result = await self.run(
                system=system,
                user_message=(
                    f"Decide the next action after evaluating concept '{report.concept}'. "
                    f"Mastery={report.mastery_score:.2f}, threshold={self.state.advance_threshold}"
                ),
                model=settings.reasoning_model,
            )
        except Exception as exc:
            logger.warning(
                "Adaptation LLM failed for concept='{}': {}. Using explicit rules.",
                report.concept, exc,
            )
            result = {}

        action, reason, style_for_reteach, missing_concept, meta_updates = (
            self._explicit_decision(report, result)
        )

        # ── Apply metacognition updates ───────────────────────────────────────
        if isinstance(meta_updates, dict):
            if "consecutive_reteach_count" in meta_updates:
                meta.consecutive_reteach_count = int(meta_updates["consecutive_reteach_count"])
            if "depth_concern_flag" in meta_updates:
                meta.depth_concern_flag = bool(meta_updates["depth_concern_flag"])
            self.state.mark_dirty("metacognition")

        decision = AdaptationDecision(
            action=action,
            reason=reason,
            style_for_reteach=style_for_reteach if style_for_reteach != "none" else None,
            missing_concept=missing_concept,
            metacognition_updates=meta_updates if isinstance(meta_updates, dict) else {},
        )

        self._log_decision(action, reason, decision.model_dump())
        logger.info("Adaptation decision: {} — {}", action, reason)
        return decision
