"""
agents/evaluator.py
EvaluatorAgent — asks 3-5 Socratic questions, scores correctness + depth,
detects misconceptions, computes calibration delta, writes to DB immediately.

Terminal tool: submit_evaluation
Non-terminal tools: ask_question, request_clarification
"""

from __future__ import annotations

from loguru import logger

from agents.base_agent import BaseAgent
from core.student_model import StudentState, EvaluationReport
from db.postgres import write_evaluation
from config import settings


class EvaluatorAgent(BaseAgent):
    NAME = "evaluator_agent"
    TERMINAL_TOOL = "submit_evaluation"

    def __init__(self, state: StudentState):
        super().__init__(state)
        self.TOOLS = [
            self.build_tool(
                name="ask_question",
                description=(
                    "Ask the student one Socratic question to probe understanding. "
                    "Use this 3-5 times before submitting evaluation. "
                    "Questions should progress from recall → application → edge cases."
                ),
                properties={
                    "question": {
                        "type": "string",
                        "description": "The question to ask the student",
                    },
                    "question_type": {
                        "type": "string",
                        "enum": ["recall", "application", "edge_case", "misconception_probe"],
                        "description": "Type of question being asked",
                    },
                },
                required=["question", "question_type"],
            ),
            self.build_tool(
                name="request_clarification",
                description="Ask the student to clarify or expand on a vague answer.",
                properties={
                    "prompt": {
                        "type": "string",
                        "description": "Clarification prompt shown to student",
                    },
                },
                required=["prompt"],
            ),
            self.build_tool(
                name="submit_evaluation",
                description=(
                    "Submit final evaluation after asking 3-5 questions. "
                    "This ends the evaluation. Call this exactly once."
                ),
                properties={
                    "correctness_score": {
                        "type": "number",
                        "description": "0.0-1.0. Factual accuracy across all answers.",
                    },
                    "depth_score": {
                        "type": "number",
                        "description": "0.0-1.0. Conceptual depth, not just recall.",
                    },
                    "misconception_type": {
                        "type": "string",
                        "enum": ["conceptual", "formula_misuse", "application_error", "none"],
                        "description": "Primary misconception type if any.",
                    },
                    "misconception_detail": {
                        "type": "string",
                        "description": "One sentence describing the misconception, or empty string.",
                    },
                    "recommended_action": {
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
                        "description": (
                            "MOVE_FORWARD: mastery clear. "
                            "MOVE_FORWARD_WITH_FLAG: borderline, note weakness. "
                            "RETEACH: same concept, different style. "
                            "DETOUR: prerequisite gap detected. "
                            "ESCALATE: 3+ reteach cycles, needs human. "
                            "COMPRESS: overconfident, accelerate. "
                            "HOLD: student requested pause."
                        ),
                    },
                    "questions_asked": {
                        "type": "integer",
                        "description": "Total number of questions asked (3-5).",
                    },
                },
                required=[
                    "correctness_score",
                    "depth_score",
                    "misconception_type",
                    "misconception_detail",
                    "recommended_action",
                    "questions_asked",
                ],
            ),
        ]

        # Conversation log — questions and student answers collected during eval
        self._qa_log: list[dict] = []

    # ── Tool executor ─────────────────────────────────────────────────────────

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        if tool_name == "ask_question":
            question = args["question"]
            q_type = args.get("question_type", "recall")
            print(f"\n🔍 [{q_type.upper()}] {question}")
            answer = input("Your answer: ").strip()
            self._qa_log.append({
                "question": question,
                "type": q_type,
                "answer": answer,
            })
            return f"Student answered: {answer}"

        if tool_name == "request_clarification":
            prompt = args["prompt"]
            print(f"\n💬 {prompt}")
            answer = input("Clarification: ").strip()
            self._qa_log.append({
                "question": prompt,
                "type": "clarification",
                "answer": answer,
            })
            return f"Student clarified: {answer}"

        return super()._execute_tool(tool_name, args)

    # ── Public run method ─────────────────────────────────────────────────────

    async def evaluate(self, concept: str, confidence_stated: int) -> EvaluationReport:
        """
        Run a full evaluation for the given concept.

        Args:
            concept:           the concept being evaluated
            confidence_stated: student's self-reported confidence (1-5)

        Returns:
            EvaluationReport (also written to DB immediately)
        """
        module = self._current_module()
        domain_framing = module.domain_framing if module else ""
        prior_mastery = self.state.get_mastery(concept)
        meta = self.state.metacognition
        reteach_count = meta.consecutive_reteach_count

        # Pace-bound question count — strictly enforced in prompt
        pace_questions = {"fast": 2, "medium": 3, "deep": 5}.get(self.state.pace, 3)
        pace_style = {
            "fast": "direct application questions only — no deep probing",
            "medium": "mix of application and conceptual connection questions",
            "deep": "full Socratic sequence: recall → application → edge cases → cross-concept",
        }.get(self.state.pace, "balanced")

        system = f"""You are a Socratic evaluator for an adaptive learning system.
Your job: assess the student's understanding of '{concept}'.

PACE CONSTRAINT (STRICTLY ENFORCED):
- Student pace: {self.state.pace.upper()}
- You MUST ask EXACTLY {pace_questions} question(s) — no more, no fewer
- Style for this pace: {pace_style}
- After exactly {pace_questions} question(s), call submit_evaluation immediately

STUDENT CONTEXT:
{self._student_context()}

EVALUATION RULES:
- Domain framing: {domain_framing}
- Prior mastery score: {prior_mastery:.2f}
- Reteach cycles so far: {reteach_count}
- If calibration={meta.calibration_pattern}, probe deeper on overconfident students

SCORING GUIDE:
- correctness_score: factual accuracy (0=wrong, 0.5=partial, 1=correct)
- depth_score: conceptual depth (0=surface recall, 0.5=can apply, 1=can explain why)
- mastery = 0.6*correctness + 0.4*depth

RECOMMENDED ACTION GUIDE:
- mastery >= {self.state.advance_threshold}: MOVE_FORWARD
- mastery >= {self.state.advance_threshold - 0.1} but weak depth: MOVE_FORWARD_WITH_FLAG
- mastery < {self.state.advance_threshold} and reteach < 3: RETEACH
- prerequisite gap detected: DETOUR
- reteach >= 3: ESCALATE
- student clearly ahead: COMPRESS
"""

        result = self.run(
            system=system,
            user_message=f"Evaluate the student on concept: '{concept}'",
            model=settings.reasoning_model,
        )

        # ── Build EvaluationReport ────────────────────────────────────────────
        correctness = float(result.get("correctness_score", 0.0))
        depth = float(result.get("depth_score", 0.0))
        mastery = round(0.6 * correctness + 0.4 * depth, 3)
        calibration_delta = round(confidence_stated / 5 - mastery, 4)

        misconception_type = result.get("misconception_type", "none")
        if misconception_type == "none":
            misconception_type = None

        report = EvaluationReport(
            concept=concept,
            session_id=self.state.session_id,
            correctness_score=correctness,
            depth_score=depth,
            mastery_score=mastery,
            misconception_type=misconception_type,
            misconception_detail=result.get("misconception_detail", ""),
            confidence_stated=confidence_stated,
            calibration_delta=calibration_delta,
            questions_asked=int(result.get("questions_asked", len(self._qa_log))),
            recommended_action=result.get("recommended_action", "RETEACH"),
        )

        # ── Update StudentState ───────────────────────────────────────────────
        self.state.update_mastery(concept, correctness, depth)
        self.state.metacognition.update_calibration(calibration_delta)
        self.state.evaluation_cycle_count += 1
        # Append to in-memory history so AdaptationEngine.run_gap_analysis() can
        # read it without a DB round-trip during the same session.
        self.state.evaluation_history.append(report)

        # ── Write to DB immediately (mid-session) ─────────────────────────────
        await write_evaluation({
            "student_id": self.state.student_id,
            "session_id": self.state.session_id,
            "concept": concept,
            "correctness_score": correctness,
            "depth_score": depth,
            "mastery_score": mastery,
            "misconception_type": misconception_type,
            "misconception_detail": report.misconception_detail,
            "confidence_stated": confidence_stated,
            "calibration_delta": calibration_delta,
            "questions_asked": report.questions_asked,
            "recommended_action": report.recommended_action,
        })

        logger.info(
            "Evaluation done: concept={} mastery={} action={}",
            concept, mastery, report.recommended_action
        )
        return report