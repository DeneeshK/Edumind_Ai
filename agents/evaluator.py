"""
LEGACY — interactive CLI/SSE session flow. Not used by the deployed frontend,
which uses the /api/courses flow (see docs/ARCHITECTURE.md). Kept as a working
reference implementation of the queue-based interactive session pattern.

agents/evaluator.py
EvaluatorAgent — asks 3-5 Socratic questions, scores correctness + depth,
detects misconceptions, computes calibration delta, writes to DB immediately.

Terminal tool: submit_evaluation
Non-terminal tools: ask_question, request_clarification
"""

from __future__ import annotations

import asyncio
import re

from loguru import logger

from agents.base_agent import BaseAgent
from core.student_model import StudentState, EvaluationReport
from db.postgres import write_evaluation
from config import settings


class EvaluatorAgent(BaseAgent):
    """
    Run the legacy interactive evaluation flow for one module.

    The evaluator asks grounded questions, records the question/answer log,
    computes scores, persists the EvaluationReport, and feeds the adaptation
    engine with enough evidence to decide whether the student should advance,
    reteach, detour, or escalate.
    """

    NAME = "evaluator_agent"
    TERMINAL_TOOL = "submit_evaluation"
    MAX_GROUNDING_CHARS = 6000

    def __init__(self, state: StudentState, emit_fn=None, ask_fn=None):
        super().__init__(state)
        async def _default_emit(text: str) -> None:
            """Emit evaluator text in legacy CLI mode."""
            print(text, flush=True)

        async def _default_ask(question: str, **kw) -> str:
            """Ask an evaluation question in legacy CLI mode."""
            print(question, flush=True)
            return input("Your answer: ").strip()

        self._emit = emit_fn or _default_emit
        self._ask = ask_fn or _default_ask

        self.TOOLS = [
            self.build_tool(
                name="ask_question",
                description=(
                    "Ask the student one open-ended evaluation question grounded "
                    "only in the provided module content. The question type must "
                    "match the current pace constraints."
                ),
                properties={
                    "question": {
                        "type": "string",
                        "description": "The question to ask the student",
                    },
                    "question_type": {
                        "type": "string",
                        "enum": [
                            "recall",
                            "conceptual",
                            "application",
                            "edge_case",
                            "misconception_probe",
                        ],
                        "description": "Type of question being asked",
                    },
                    "source_quote": {
                        "type": "string",
                        "description": (
                            "A short exact phrase copied from MODULE CONTENT that "
                            "contains the answer or directly supports this question."
                        ),
                    },
                },
                required=["question", "question_type", "source_quote"],
            ),
            self.build_tool(
                name="request_clarification",
                description=(
                    "Ask the student to clarify or expand on a vague answer. "
                    "Do not introduce any new concept, example, or rule that was "
                    "not already in the original question or MODULE CONTENT."
                ),
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
                    "Submit final evaluation after asking the exact pace-bound "
                    "number of questions specified in the prompt. "
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
                            "ESCALATE: repeated failure, rebuild curriculum sequence. "
                            "COMPRESS: mastery is clearly ahead of the current plan. "
                            "HOLD: student requested pause."
                        ),
                    },
                    "questions_asked": {
                        "type": "integer",
                        "description": "Total number of questions asked.",
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

    def _compact_grounding_text(self, text: str) -> str:
        """Keep module content within the evaluator prompt budget."""
        text = text.strip()
        if len(text) <= self.MAX_GROUNDING_CHARS:
            return text

        head_len = self.MAX_GROUNDING_CHARS // 2
        tail_len = self.MAX_GROUNDING_CHARS - head_len
        return (
            text[:head_len].rstrip()
            + "\n\n[...module content shortened for evaluation prompt...]\n\n"
            + text[-tail_len:].lstrip()
        )

    def _module_grounding_text(self, concept: str = "") -> str:
        """
        Return the authoritative text allowed for evaluation questions.

        Delivered lesson content is preferred. When the lesson text is missing,
        the evaluator falls back to module metadata so it can still ask basic
        concept questions without inventing content outside the module.
        """
        module = self._current_module()
        content = ""
        if module is not None:
            content = self.state.get_module_content(module.id).strip()

        if not content and module is not None:
            prereqs = ", ".join(module.prerequisites) if module.prerequisites else "none"
            content = (
                "Module title: " + module.title + "\n"
                "Module concept: " + module.concept + "\n"
                "Domain framing: " + module.domain_framing + "\n"
                "Prerequisites explicitly listed: " + prereqs + "\n"
                "Depth level: " + module.depth_level
            )
        elif not content:
            content = "Module concept: " + concept

        return self._compact_grounding_text(content)

    def _source_quote_supported(self, source_quote: str, grounding_text: str) -> bool:
        """Return whether a proposed source quote appears verbatim in grounding text."""
        quote = " ".join(source_quote.strip().lower().split())
        haystack = " ".join(grounding_text.strip().lower().split())
        return bool(quote and len(quote) >= 8 and quote in haystack)

    def _unsupported_question_terms(self, question: str, grounding_text: str) -> list[str]:
        """Return important question terms not present in the module grounding text."""
        ignored_terms = {
            "about",
            "according",
            "answer",
            "before",
            "careful",
            "check",
            "concept",
            "connect",
            "describe",
            "detail",
            "does",
            "example",
            "explain",
            "framing",
            "from",
            "idea",
            "important",
            "main",
            "mean",
            "means",
            "mentioned",
            "module",
            "relationship",
            "relate",
            "related",
            "review",
            "student",
            "takeaway",
            "term",
            "using",
            "what",
            "when",
            "where",
            "which",
            "would",
            "your",
        }
        question_text = question.lower().replace("_", " ")
        grounding_text = grounding_text.lower().replace("_", " ")
        question_tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]*", question_text)
        content_tokens = set(
            re.findall(r"[a-zA-Z][a-zA-Z0-9'-]*", grounding_text)
        )

        unsupported: list[str] = []
        for token in question_tokens:
            if len(token) < 4 or token in ignored_terms:
                continue
            variants = {token, token.rstrip("s"), token + "s"}
            if not variants.intersection(content_tokens):
                unsupported.append(token)

        return sorted(set(unsupported))

    def _fallback_questions(self, concept: str) -> list[tuple[str, str]]:
        """Return safe evaluation questions when the tool loop cannot produce them."""
        module = self._current_module()
        module_name = module.title if module else concept

        return [
            (
                "recall",
                f"What was the main idea of the '{module_name}' module in your own words?",
            ),
            (
                "conceptual",
                "Name one important term, rule, or relationship the module explicitly mentioned.",
            ),
            (
                "application",
                "What example or domain framing from the module helped explain the idea?",
            ),
            (
                "misconception_probe",
                "What detail from the module would you be careful not to forget?",
            ),
            (
                "edge_case",
                "What key takeaway from the module would you check before moving on?",
            ),
        ]

    def _question_out_of_scope_reason(self, question: str) -> str | None:
        """Return a reason for rejecting a question, or None when it is allowed."""
        # Generic scope control is enforced by source_quote support and
        # unsupported-term checks. No topic-specific curriculum rules live here.
        return None

    def _module_boundary_rules(self, module) -> str:
        """Return the evaluator prompt rule that keeps assessment inside one module."""
        return (
            "Assess only the current module concept, its explicitly listed "
            "prerequisites, and content already delivered in this module."
        )

    async def _run_fallback_evaluation(self, concept: str, target_count: int) -> dict:
        """
        Ask deterministic fallback questions and compute approximate scores.

        This path keeps the session usable if the LLM tool loop fails. It uses
        answer presence and length as a conservative signal rather than claiming
        deep semantic grading.
        """
        for q_type, question in self._fallback_questions(concept):
            if len(self._qa_log) >= target_count:
                break
            answer = (await self._ask(f"🔍 [{q_type.upper()}] {question}")).strip()
            self._qa_log.append({
                "question": question,
                "type": q_type,
                "answer": answer,
            })

        answers = [qa.get("answer", "").strip() for qa in self._qa_log]
        non_empty = [a for a in answers if a]
        avg_words = (
            sum(len(a.split()) for a in non_empty) / len(non_empty)
            if non_empty else 0.0
        )
        answer_ratio = len(non_empty) / max(1, target_count)
        correctness = round(min(1.0, 0.15 + 0.65 * answer_ratio), 2) if non_empty else 0.0
        depth = round(min(1.0, avg_words / 28), 2) if non_empty else 0.0
        mastery = 0.6 * correctness + 0.4 * depth

        return {
            "correctness_score": correctness,
            "depth_score": depth,
            "misconception_type": "none",
            "misconception_detail": "",
            "recommended_action": (
                "MOVE_FORWARD"
                if mastery >= self.state.advance_threshold
                else "RETEACH"
            ),
            "questions_asked": len(self._qa_log),
        }

    async def _execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute evaluator tool calls for asking questions and submitting scores."""
        if tool_name == "ask_question":
            question = args["question"]
            q_type = args.get("question_type", "recall")
            source_quote = args.get("source_quote", "")
            grounding_text = self._module_grounding_text()

            # Only enforce strict source-quote grounding when we have real
            # delivered lesson content. When grounding_text is only module
            # metadata (title, concept, prerequisites) the vocabulary is too
            # sparse and the check incorrectly rejects valid questions, causing
            # the system to silently fall back to heuristic scoring.
            module = self._current_module()
            has_real_content = (
                module is not None
                and bool(self.state.get_module_content(module.id).strip())
            )

            if has_real_content:
                if not self._source_quote_supported(source_quote, grounding_text):
                    logger.warning(
                        "Rejected evaluator question without grounded source quote: {}",
                        question,
                    )
                    return (
                        "Question rejected: source_quote must be an exact phrase from "
                        "MODULE CONTENT and must directly support the question. Ask a "
                        "new, easier question using only MODULE CONTENT."
                    )
                unsupported_terms = self._unsupported_question_terms(question, grounding_text)
                if unsupported_terms:
                    logger.warning(
                        "Rejected evaluator question with terms outside module content: {}",
                        unsupported_terms,
                    )
                    return (
                        "Question rejected: these terms are not in MODULE CONTENT: "
                        + ", ".join(unsupported_terms[:5])
                        + ". Ask a new question using only terms from MODULE CONTENT."
                    )
            else:
                logger.debug(
                    "Skipping source-quote grounding check — no delivered lesson "
                    "content yet for concept='{}'; using metadata fallback.",
                    args.get("concept", "unknown"),
                )

            out_of_scope = self._question_out_of_scope_reason(question)
            if out_of_scope:
                logger.warning("Rejected out-of-scope evaluator question: {}", question)
                return (
                    "Question rejected as out of scope: " + out_of_scope
                    + " Ask a new question from the current module only."
                )
            answer = (await self._ask(f"🔍 [{q_type.upper()}] {question}")).strip()
            self._qa_log.append({
                "question": question,
                "type": q_type,
                "source_quote": source_quote,
                "answer": answer,
            })
            return f"Student answered: {answer}"

        if tool_name == "request_clarification":
            prompt = args["prompt"]
            answer = (await self._ask(f"💬 {prompt}")).strip()
            self._qa_log.append({
                "question": prompt,
                "type": "clarification",
                "answer": answer,
            })
            return f"Student clarified: {answer}"

        return await super()._execute_tool(tool_name, args)

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
        module_title = module.title if module else concept
        module_prereqs = ", ".join(module.prerequisites) if module and module.prerequisites else "none"
        prior_mastery = self.state.get_mastery(concept)
        meta = self.state.metacognition
        reteach_count = meta.consecutive_reteach_count

        # Pace-bound question count — strict practical caps matching the design:
        # fast = 2 targeted, medium = 3-4, deep = complete model within a cap.
        pace_questions = {"fast": 2, "medium": 4, "deep": 5}.get(self.state.pace, 4)
        pace_style = {
            "fast": "Type 1 direct application only: solve or explain one concrete use of the current concept",
            "medium": "Type 1 direct application plus Type 2 conceptual connection when supported by the module",
            "deep": "Type 1, Type 2, and Type 3 misconception probe when supported by the module",
        }.get(self.state.pace, "balanced")

        module_content = self._module_grounding_text(concept)
        module_content_source = (
            "delivered lesson text"
            if module and self.state.get_module_content(module.id).strip()
            else "module metadata fallback"
        )

        system = f"""You are a Socratic evaluator for an adaptive learning system.
Your job: assess the student's understanding of '{concept}'.

PACE CONSTRAINT (STRICTLY ENFORCED):
- Student pace: {self.state.pace.upper()}
- You MUST ask EXACTLY {pace_questions} question(s) — no more, no fewer
- Style for this pace: {pace_style}
- After exactly {pace_questions} question(s), call submit_evaluation immediately
- Fast mode removes breadth, not correctness: all answers are open-ended, never multiple choice

STUDENT CONTEXT:
{self._student_context()}

EVALUATION RULES:
- Current module title: {module_title}
- Current module concept: {concept}
- Domain framing: {domain_framing}
- Current module prerequisites: {module_prereqs}
- Scope boundary: {self._module_boundary_rules(module) if module else "Assess only the current concept."}
- Never ask about future modules or formulas/rules not explicitly present in MODULE CONTENT.
- The questions must be easy enough to answer from the module content alone.
- Prior mastery score: {prior_mastery:.2f}
- Reteach cycles so far: {reteach_count}
- If calibration={meta.calibration_pattern}, probe a little deeper only when MODULE CONTENT supports it

MODULE CONTENT - ONLY SOURCE FOR QUESTIONS ({module_content_source}):
\"\"\"
{module_content}
\"\"\"

ABSOLUTE GROUNDING RULES:
- Ask ONLY about facts, terms, examples, rules, relationships, and takeaways explicitly stated in MODULE CONTENT.
- Do NOT use outside knowledge, related topic knowledge, future/later modules, or examples absent from MODULE CONTENT.
- Before each ask_question call, silently verify that the answer can be found in MODULE CONTENT.
- Every ask_question call MUST include source_quote: a short exact phrase copied from MODULE CONTENT that directly supports the question.
- If MODULE CONTENT does not support an application or edge-case question, ask a simpler recall/conceptual question instead.
- Prefer plain, direct wording over tricky or broad questions.

SCORING GUIDE:
- correctness_score: factual accuracy (0=wrong, 0.5=partial, 1=correct)
- depth_score rubric:
  0.0-0.3 = recalled phrase with no explanation
  0.3-0.6 = correct claim with partial reasoning
  0.6-0.8 = correct claim with full reasoning
  0.8-1.0 = full reasoning plus a supported connection
- mastery = 0.6*correctness + 0.4*depth
- misconception_type must be:
  conceptual when the student misunderstands the idea,
  formula_misuse when the idea is understood but the rule/calculation is misapplied,
  application_error when direct recall is fine but transfer to a new context fails,
  none when no misconception is present

RECOMMENDED ACTION GUIDE:
- mastery >= {self.state.advance_threshold}: MOVE_FORWARD
- mastery >= {self.state.advance_threshold - 0.1} but weak depth: MOVE_FORWARD_WITH_FLAG
- 0.4 <= mastery < {self.state.advance_threshold}: RETEACH
- mastery < 0.4 or prerequisite gap detected: DETOUR
- reteach >= 2 and mastery < {self.state.advance_threshold}: ESCALATE
- student clearly ahead: COMPRESS
"""

        try:
            result = await self.run(
                system=system,
                user_message=f"Evaluate the student on concept: '{concept}'",
                model=settings.reasoning_model,
            )
        except Exception as exc:
            logger.error("Evaluator tool loop failed for '{}': {}", concept, exc)
            result = {}

        if not {
            "correctness_score",
            "depth_score",
            "recommended_action",
        }.issubset(result.keys()):
            await self._emit(
                f"⚠️ Evaluation fallback activated. I will ask {pace_questions} grounded check questions."
            )
            result = await self._run_fallback_evaluation(concept, pace_questions)

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

        if hasattr(self, "_eval_runner") and self._eval_runner is not None:
            mastery_history = [
                r.mastery_score for r in self.state.evaluation_history
                if r.concept == concept
            ]
            asyncio.create_task(
                self._eval_runner.on_evaluation_complete(
                    concept=concept,
                    qa_log=self._qa_log,
                    questions_asked=[
                        {
                            "question": q.get("question", ""),
                            "source_quote": q.get("source_quote", ""),
                            "question_type": q.get("type", ""),
                        }
                        for q in self._qa_log
                    ],
                    lesson_text=self._module_grounding_text(concept),
                    mastery_score=report.mastery_score,
                    advance_threshold=self.state.advance_threshold,
                    actual_action=report.recommended_action,
                    misconception_type=report.misconception_type,
                    reteach_count=self.state.metacognition.consecutive_reteach_count,
                    calibration_delta=report.calibration_delta,
                    mastery_history=mastery_history,
                    modules_attempted=self.state.evaluation_cycle_count,
                    modules_mastered=len([
                        r for r in self.state.evaluation_history
                        if r.mastery_score >= self.state.advance_threshold
                    ]),
                )
            )

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
