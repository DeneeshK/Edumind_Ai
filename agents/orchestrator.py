"""
agents/orchestrator.py
OrchestratorAgent — controls the full learning session loop.

Layer 1: Onboarding (first session only)
Layer 2: Per-module teach → evaluate → adapt loop
Layer 3: Session end — flush all to DB

The Orchestrator does NOT teach or evaluate directly.
It delegates to: CurriculumArchitectAgent, TutorAgent, EvaluatorAgent, AdaptationEngine.
"""

from __future__ import annotations

import sys

from loguru import logger

from agents.base_agent import BaseAgent
from agents.curriculum_architect import CurriculumArchitectAgent
from agents.tutor import TutorAgent
from agents.evaluator import EvaluatorAgent
from agents.adaptation_engine import AdaptationEngine
from core.student_model import StudentState
from db.postgres import upsert_student
from config import settings


class OrchestratorAgent(BaseAgent):
    NAME = "orchestrator"
    TERMINAL_TOOL = "end_session"

    def __init__(self, state: StudentState):
        super().__init__(state)

        self.TOOLS = [
            self.build_tool(
                name="plan_session",
                description="Plan the session: decide which modules to cover and in what order.",
                properties={
                    "modules_to_cover": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of module IDs to cover this session",
                    },
                    "session_goal": {
                        "type": "string",
                        "description": "One sentence goal for this session",
                    },
                },
                required=["modules_to_cover", "session_goal"],
            ),
            self.build_tool(
                name="end_session",
                description="End the session and trigger DB flush.",
                properties={
                    "session_summary": {
                        "type": "string",
                        "description": "2-3 sentence summary of what was learned this session",
                    },
                    "modules_completed": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of module IDs completed",
                    },
                    "next_session_hint": {
                        "type": "string",
                        "description": "One sentence hint for what to focus on next session",
                    },
                },
                required=[
                    "session_summary",
                    "modules_completed",
                    "next_session_hint"],
            ),
            self.build_tool(
                name="route_after_evaluation",
                description=(
                    "Called after each module evaluation. "
                    "Reason over the evaluation report and student state, "
                    "then decide the next action. "
                    "You MUST call this after every module evaluation — "
                    "never skip it."
                ),
                properties={
                    "action": {
                        "type": "string",
                        "enum": [
                            "MOVE_FORWARD",
                            "RETEACH",
                            "DETOUR",
                            "ESCALATE",
                            "COMPRESS",
                            "HOLD",
                        ],
                        "description": (
                            "MOVE_FORWARD: mastery sufficient, go to next module. "
                            "RETEACH: mastery insufficient, retry with different style. "
                            "DETOUR: prerequisite gap detected, insert a prereq module. "
                            "ESCALATE: repeated failures, rebuild curriculum. "
                            "COMPRESS: student far ahead, accelerate. "
                            "HOLD: student requests pause."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": "One sentence explaining why you chose this action",
                    },
                    "style_for_reteach": {
                        "type": "string",
                        "description": "Only required when action=RETEACH. Which style to switch to. Must be one of: formal, socratic, example_first, visual, story. Leave empty otherwise.",
                    },
                    "missing_concept": {
                        "type": "string",
                        "description": "Only required when action=DETOUR. The prerequisite concept name.",
                    },
                },
                required=["action", "reason"],
            ),
        ]

    # ── Tool executor ───────────────────────────────────────────────────────

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        if tool_name == "plan_session":
            modules_to_cover = args.get("modules_to_cover", [])
            session_goal = args.get("session_goal", "")
            logger.info(
                "Session plan: {} modules, goal='{}'",
                len(modules_to_cover),
                session_goal)
            return "Session planned: covering " + \
                str(modules_to_cover) + ". Goal: " + session_goal

        if tool_name == "route_after_evaluation":
            # Store routing decision so _run_module_loop can read it
            self._last_route = args
            action = args.get("action", "")
            reason = args.get("reason", "")
            logger.info(
                "LLM routing decision: action={} reason='{}'",
                action,
                reason)
            return "Routing decision recorded: " + action

        return super()._execute_tool(tool_name, args)

    # ── LLM routing ─────────────────────────────────────────────────────────

    async def _llm_route(self, module_concept: str,
                         report_summary: str) -> dict:
        """
        Ask the Orchestrator LLM to reason over the evaluation result
        and current student state, then call route_after_evaluation.

        This replaces the Python if/else chain — the LLM is now the
        decision-maker for every post-evaluation routing step.

        Returns the args dict from the route_after_evaluation tool call.
        """
        self._last_route = {}

        mastery_threshold = {
            "fast": 0.60, "medium": 0.72, "deep": 0.85
        }.get(self.state.pace, 0.72)

        consecutive = self.state.metacognition.consecutive_reteach_count
        eval_count = self.state.evaluation_cycle_count

        system = (
            "You are the Orchestrator of an adaptive learning system. "
            "Your ONLY job right now is to call route_after_evaluation "
            "with the correct action based on the evidence below. "
            "Do not explain. Do not ask questions. Call the tool immediately.\n\n"
            "ROUTING RULES:\n"
            "- mastery >= " + str(mastery_threshold) +
            " AND no critical misconception -> MOVE_FORWARD\n"
            "- mastery < " + str(mastery_threshold) +
            " AND consecutive_reteach < 3 -> RETEACH\n"
            "- misconception_type = prerequisite_gap -> DETOUR (name the missing concept)\n"
            "- consecutive_reteach >= 3 -> ESCALATE\n"
            "- mastery >= 0.90 and student answered quickly -> COMPRESS\n"
            "- student literally typed 'stop', 'quit', or 'exit' -> HOLD\n"
            "CRITICAL: Never output HOLD unless the student's text literally says 'stop', 'quit', 'pause' or 'exit'.\n\n"
            "STUDENT STATE:\n"
            + self._student_context() + "\n"
            "consecutive_reteach_count: " + str(consecutive) + "\n"
            "mastery_threshold_for_pace: " + str(mastery_threshold) + "\n"
            "evaluation_cycle: " + str(eval_count) + "\n"
        )

        user_msg = (
            "Evaluation result for concept: '" + module_concept + "'\n"
            + report_summary
            + "\n\nCall route_after_evaluation now."
        )

        await self.arun(
            system=system,
            user_message=user_msg,
            model=settings.reasoning_model,
        )

        if not self._last_route:
            # LLM failed to call the tool — apply safe fallback
            logger.warning(
                "LLM did not call route_after_evaluation — defaulting to RETEACH")
            self._last_route = {
                "action": "RETEACH",
                "reason": "Routing fallback: LLM did not return a tool call.",
            }

        return self._last_route

    # ── Layer 1: Onboarding ─────────────────────────────────────────────────

    async def _onboard(self) -> str:
        """Collect student name, domain, goal, pace via CLI. Returns topic."""
        if not sys.stdin.isatty():
            name = self.state.name or "Student"
            domain = self.state.domain or "general"
            goal = self.state.goal or "learn the requested topic"
            topic = self.state.goal or self.state.domain or "requested topic"
            pace = self.state.pace if self.state.pace in (
                "fast", "medium", "deep") else "medium"

            self.state.name = name
            self.state.domain = domain
            self.state.goal = goal
            self.state.pace = pace

            await upsert_student(
                self.state.student_id,
                name, domain, goal, pace,
            )
            logger.info(
                "Non-interactive onboarding used state defaults: domain='{}' topic='{}'",
                domain,
                topic,
            )
            return topic

        print("\n" + "=" * 60)
        print("🎓 Welcome to EduMind — Adaptive Learning System")
        print("=" * 60 + "\n")

        name = input("Your name: ").strip() or "Student"
        domain = input(
            "Your domain/field (e.g. 'machine learning', 'physics'): ").strip()
        goal = input(
            "Your learning goal (e.g. 'understand transformers for NLP'): ").strip()
        topic = input("Topic to learn today: ").strip()

        print("\nLearning pace:")
        print("  fast   — quick overview, advance at 60% mastery")
        print("  medium — balanced, advance at 72% mastery")
        print("  deep   — thorough, advance at 85% mastery")
        pace = input("Choose pace [fast/medium/deep]: ").strip().lower()
        if pace not in ("fast", "medium", "deep"):
            pace = "medium"

        self.state.name = name
        self.state.domain = domain
        self.state.goal = goal
        self.state.pace = pace

        await upsert_student(
            self.state.student_id,
            name, domain, goal, pace,
        )
        print(f"\n✅ Welcome {name}! Let's learn '{topic}'.\n")
        return topic

    # ── Layer 2: Module loop ────────────────────────────────────────────────

    async def _run_module_loop(self) -> list[str]:
        """
        Teach → Evaluate → Adapt loop for all curriculum modules.

        Each agent call is wrapped in a try/except so a single failure
        (Groq timeout, DB error) cannot silently discard session state.
        Returns list of completed module IDs.
        """
        completed = []
        curriculum = self.state.curriculum

        if curriculum is None:
            logger.error("No curriculum set — cannot run module loop")
            return completed

        max_modules_per_session = 5
        modules_done = 0

        while (
            curriculum.current_index < len(curriculum.modules)
            and modules_done < max_modules_per_session
        ):
            module = curriculum.modules[curriculum.current_index]
            print(f"\n{'─' * 60}")
            print(
                f" Module {curriculum.current_index + 1}/{len(curriculum.modules)}: {module.title}")
            print(f"{'─' * 60}")

            # ── Teach ────────────────────────────────────────────────────────
            try:
                tutor = TutorAgent(self.state)
                lesson_result = await tutor.teach()
            except Exception as exc:
                logger.error(
                    "TutorAgent.teach() failed for '{}': {}",
                    module.concept,
                    exc)
                print(
                    f"\n⚠️  Lesson delivery failed ({exc}). Skipping to evaluation with partial context.")
                lesson_result = {
                    "style_used": "formal",
                    "fatigue_detected": "no",
                    "doubt_count": 0,
                    "lesson_text": "",
                }

            # ── Doubt count trigger (micro-example injection) ────────────────
            doubt_count = self.state.get_doubt_count(module.concept)
            if doubt_count >= 2:
                print(
                    f"\n💡 You raised {doubt_count} doubts on '{module.concept}'. Injecting a worked example…\n")
                try:
                    self._inject_micro_example(module.concept)
                except Exception as exc:
                    logger.warning("Micro-example injection failed: {}", exc)

            style_used = lesson_result.get("style_used", "formal")

            # ── Confidence rating ────────────────────────────────────────────
            print(
                f"\n📊 How confident are you about '{module.concept}'? (1–5): ",
                end="")
            try:
                confidence = int(input().strip()) if sys.stdin.isatty() else 3
                confidence = max(1, min(5, confidence))
            except (ValueError, EOFError):
                confidence = 3

            # ── Evaluate ─────────────────────────────────────────────────────
            try:
                evaluator = EvaluatorAgent(self.state)
                report = await evaluator.evaluate(module.concept, confidence)
            except Exception as exc:
                logger.error(
                    "EvaluatorAgent.evaluate() failed for '{}': {}",
                    module.concept,
                    exc)
                print(
                    f"\n⚠️  Evaluation failed ({exc}). Recording zero mastery and reteaching.")
                # Safe fallback: build a zero-score report so adaptation can
                # still run
                from core.student_model import EvaluationReport
                report = EvaluationReport(
                    concept=module.concept,
                    session_id=self.state.session_id,
                    correctness_score=0.0,
                    depth_score=0.0,
                    mastery_score=0.0,
                    misconception_type=None,
                    misconception_detail="Evaluation failed due to system error.",
                    confidence_stated=confidence,
                    calibration_delta=confidence / 5,
                    questions_asked=0,
                    recommended_action="RETEACH",
                )

            # Update metacognition style score with post-eval depth
            self.state.metacognition.record_style_depth(
                style_used, report.depth_score)

            # ── AdaptationEngine: deep metacognitive decision ────────────────
            # AdaptationEngine runs its own agentic tool loop to decide the
            # recommended action. Its output feeds the Orchestrator LLM as
            # evidence — the Orchestrator makes the FINAL routing call.
            try:
                engine = AdaptationEngine(self.state)
                engine_decision = await engine.decide(report)
                engine_summary = (
                    "AdaptationEngine recommended: " + engine_decision.action
                    + " | reason: " + engine_decision.reason
                )
            except Exception as exc:
                logger.error("AdaptationEngine.decide() failed: {}", exc)
                engine_summary = "AdaptationEngine failed — use evaluation scores only."

            # ── Gap analysis (every 3 evaluation cycles) ─────────────────────
            try:
                gap_concept = await engine.run_gap_analysis()
                if gap_concept:
                    print(
                        f"\n🔍 Gap analysis: missing prerequisite '{gap_concept}'")
                    engine_summary += " | Gap detected: " + gap_concept
            except Exception as exc:
                logger.warning(
                    "run_gap_analysis() failed (non-critical): {}", exc)
                gap_concept = None

            # ── Orchestrator LLM routing (TRULY AGENTIC) ─────────────────────
            # The LLM reasons over: evaluation report + engine recommendation
            # + student metacognition + mastery threshold for pace.
            # It calls route_after_evaluation to emit its decision.
            # This is the key architectural fix: routing is emergent from
            # LLM reasoning, not hardcoded Python if/else.
            report_summary = (
                "mastery_score: " + str(round(report.mastery_score, 2)) + "\n"
                "correctness: " +
                str(round(report.correctness_score, 2)) + "\n"
                "depth: " + str(round(report.depth_score, 2)) + "\n"
                "misconception_type: " +
                str(report.misconception_type or "none") + "\n"
                "misconception_detail: " +
                str(report.misconception_detail or "none") + "\n"
                "confidence_stated: " + str(report.confidence_stated) + "\n"
                "calibration_delta: " +
                str(round(report.calibration_delta, 2)) + "\n"
                + engine_summary
            )
            route = await self._llm_route(module.concept, report_summary)
            action = route.get("action", "RETEACH")
            reason = route.get("reason", "")

            print("\n" + "=" * 60)
            print("📝 EVALUATION FEEDBACK")
            print("=" * 60)
            print(f"Mastery Score: {round(report.mastery_score * 100)}%")
            print(f"Correctness:   {round(report.correctness_score * 100)}%")
            print(f"Depth/Nuance:  {round(report.depth_score * 100)}%")
            if report.misconception_type and report.misconception_type != "none":
                print(f"Misconception: {report.misconception_detail}")
            print("=" * 60)

            print(f"\n⚙️  Orchestrator decision: {action} — {reason}")

            # ── Apply routing decision ───────────────────────────────────────
            if action in ("MOVE_FORWARD", "MOVE_FORWARD_WITH_FLAG"):
                completed.append(module.id)
                curriculum.current_index += 1
                self.state.mark_dirty("curriculum")
                self.state.metacognition.consecutive_reteach_count = 0
                modules_done += 1
                print(f"✅ '{module.concept}' mastered! Moving forward.\n")

            elif action == "RETEACH":
                self.state.metacognition.consecutive_reteach_count += 1
                new_style = route.get("style_for_reteach")
                if new_style:
                    self.state.metacognition.preferred_style = new_style
                    print(
                        f"🔄 Reteaching '{module.concept}' using '{new_style}' style.\n")
                else:
                    print(f"🔄 Reteaching '{module.concept}'.\n")

            elif action == "DETOUR":
                missing_concept = route.get("missing_concept")
                if missing_concept:
                    print(
                        f"↩️  Detour — must learn '{missing_concept}' first.\n")
                    from core.student_model import Module as CurrModule
                    detour = CurrModule(
                        id=f"detour_{missing_concept.replace(' ', '_')}",
                        title=f"Prerequisite: {missing_concept}",
                        concept=missing_concept,
                        domain_framing=f"{missing_concept} in {self.state.domain}",
                        prerequisites=[],
                        estimated_minutes=10,
                        depth_level="surface",
                    )
                    curriculum.modules.insert(curriculum.current_index, detour)
                    self.state.mark_dirty("curriculum")
                else:
                    logger.warning(
                        "DETOUR decision has no missing_concept — treating as RETEACH")
                    self.state.metacognition.consecutive_reteach_count += 1
                    print(
                        "🔄 Detour requested but no concept specified — reteaching.\n")

            elif action == "ESCALATE":
                print(
                    f"\n🚨 '{module.concept}' could not be mastered after repeated attempts.")
                print("   Rebuilding curriculum from this point...\n")
                try:
                    architect = CurriculumArchitectAgent(self.state)
                    escalate_topic = (
                        f"Prerequisite foundation for: {module.concept} "
                        f"in {self.state.domain}"
                    )
                    new_plan = await architect.build_curriculum(escalate_topic)
                    # Replace remaining modules from current index onward
                    # with the newly built remedial curriculum
                    curriculum.modules = (
                        curriculum.modules[:curriculum.current_index]
                        + new_plan.modules
                    )
                    self.state.mark_dirty("curriculum")
                    logger.info(
                        "ESCALATE: rebuilt curriculum from index {} "
                        "with {} remedial modules for concept='{}'",
                        curriculum.current_index,
                        len(new_plan.modules),
                        module.concept,
                    )
                    print(
                        f"   New remedial path: {len(new_plan.modules)} modules built.\n")
                except Exception as exc:
                    # Fallback: advance past the blocking concept so the
                    # session is not permanently stuck
                    logger.error(
                        "ESCALATE curriculum rebuild failed: {} — advancing", exc)
                    print(
                        "   Rebuild failed — skipping concept to avoid session lock.\n")
                    completed.append(module.id)
                    curriculum.current_index += 1
                    self.state.mark_dirty("curriculum")
                    modules_done += 1

            elif action == "COMPRESS":
                print("⚡ Compressing — student is ahead. Accelerating.\n")
                completed.append(module.id)
                curriculum.current_index += 1
                self.state.mark_dirty("curriculum")
                modules_done += 1

            elif action == "HOLD":
                print("\n⏸️  Session paused at student request.")
                break

            # Fatigue check
            if lesson_result.get("fatigue_detected") == "yes":
                print("\n😴 Fatigue detected — ending session early.")
                break

        return completed

    # ── Layer 3: Session end ────────────────────────────────────────────────

    async def _end_session(self, completed_modules: list[str]) -> None:
        """Flush all session data to DB atomically."""

        # Use plain generate() with NO tools — no tool definitions means
        # the LLM cannot accidentally trigger a new routing cycle.
        # It can only return a plain text summary.
        from clients.groq_client import generate as groq_generate

        decision_records = []
        for decision in self.state.session_decisions:
            payload = (
                decision
                if isinstance(decision, dict)
                else decision.model_dump()
            )
            decision_records.append({
                "student_id": self.state.student_id,
                "session_id": self.state.session_id,
                "agent": payload.get("agent", "adaptation_engine"),
                "action": payload.get("action", "UNKNOWN"),
                "rationale": payload.get("reason", ""),
                "payload": payload,
            })

        decisions_taken = [d["action"] for d in decision_records]

        summary = await groq_generate(
            messages=[{
                "role": "user",
                "content": (
                    f"Write a 2-3 sentence plain-English learning session summary.\n"
                    f"Student: {self.state.name}\n"
                    f"Domain: {self.state.domain}\n"
                    f"Modules completed: {', '.join(completed_modules) or 'none'}\n"
                    f"Decisions this session: {decisions_taken}\n"
                    f"Write only the summary text. No headings. No bullet points."
                )
            }],
            model=settings.generation_model,
            system="You are a helpful tutor summarising a learning session. Be concise and encouraging.",
        )

        completed_ids = set(completed_modules)
        completed_concepts = set(completed_modules)
        if self.state.curriculum:
            completed_concepts.update({
                module.concept
                for module in self.state.curriculum.modules
                if module.id in completed_ids or module.concept in completed_ids
            })

        # Build mastery updates for atomic write
        mastery_updates = [
            {
                "concept": concept,
                "mastery_score": score,
                "correctness": self.state.concept_mastery.get(concept, score),
                "depth": self.state.concept_depth.get(concept, 0.0),
            }
            for concept, score in self.state.concept_mastery.items()
            if concept in completed_concepts
        ]

        # Single atomic transaction
        from db.postgres import flush_session_to_db
        await flush_session_to_db(
            student_id=self.state.student_id,
            session_id=self.state.session_id,
            summary=summary,
            modules_covered=completed_modules,
            started_at=self.state.session_started_at,
            decisions=decision_records,
            metacognition_json=self.state.metacognition.model_dump(),
            mastery_updates=mastery_updates,
        )

        print(f"\n{'=' * 60}")
        print("✅ Session complete!")
        print(f"   {summary}")
        print(f"{'=' * 60}\n")

    # ── Micro-example injection ─────────────────────────────────────────────

    def _inject_micro_example(self, concept: str) -> None:
        """
        Stream a short, focused worked example directly to stdout.

        This is a GENERATION task, not an agentic task — the LLM produces
        a single coherent worked example. We use stream() directly so:
          1. Output reaches the student token-by-token with no buffering delay.
          2. We avoid the TutorAgent's tool-call loop, which expects a
             finish_lesson tool call that a raw micro-example prompt will
             never trigger correctly.
          3. The example length and structure is driven purely by the prompt,
             not by tool schema constraints.

        Raises nothing — caller already wraps this in try/except.
        """
        from clients.groq_client import stream, GroqTimeoutError, GroqRateLimitError

        preferred_style = self.state.metacognition.preferred_style or "example_first"
        domain = self.state.domain

        system = (
            f"You are a precise, concise tutor for a student learning {domain}.\n"
            f"The student's preferred learning style is: {preferred_style}.\n"
            f"Write ONE clear, self-contained worked example for the concept below.\n\n"
            f"FORMAT RULES:\n"
            f"- Start directly with the example — no preamble\n"
            f"- Show the problem, step-by-step reasoning, and final answer\n"
            f"- Keep it under 200 words\n"
            f"- Use the student's domain framing ({domain})\n"
            f"- End with one sentence explaining WHY each step works"
        )

        messages = [
            {"role": "user", "content": f"Worked example for: {concept}"}]

        print(f"\n{'─' * 50}")
        print(f"📌 Worked Example: {concept}")
        print(f"{'─' * 50}\n")

        try:
            full_text = []
            for chunk in stream(messages=messages, system=system,
                                model=settings.generation_model):
                print(chunk, end="", flush=True)
                full_text.append(chunk)
            print("\n")

            logger.info(
                "Micro-example injected for concept='{}' ({} chars)",
                concept, sum(len(c) for c in full_text)
            )

        except GroqTimeoutError:
            logger.warning(
                "Micro-example stream timed out for concept='{}'",
                concept)
            print("\n[Worked example unavailable — Groq timeout]\n")

        except GroqRateLimitError:
            logger.warning(
                "Micro-example rate-limited for concept='{}'",
                concept)
            print("\n[Worked example unavailable — rate limit]\n")

    # ── Main entry point ────────────────────────────────────────────────────

    async def _check_cross_session_doubts(self) -> None:
        """
        Query doubt_log for concepts doubted across >= 2 distinct sessions.
        If a concept has 3+ cross-session doubts, prepend a prerequisite
        detour module at the front of the current curriculum so it is
        addressed immediately at session start.
        """
        from db.postgres import get_conn
        from core.student_model import Module as CurrModule

        try:
            async with get_conn() as conn:
                rows = await conn.fetch(
                    """
                    SELECT concept, SUM(count) as total_doubts,
                           COUNT(DISTINCT session_id) as session_count
                    FROM doubt_log
                    WHERE student_id = $1
                    GROUP BY concept
                    HAVING COUNT(DISTINCT session_id) >= 2
                      AND SUM(count) >= 3
                    ORDER BY total_doubts DESC
                    LIMIT 3
                    """,
                    self.state.student_id,
                )
        except Exception as e:
            logger.warning(
                "Cross-session doubt query failed: {} — skipping", e)
            return

        if not rows:
            return

        curriculum = self.state.curriculum
        inserted = 0
        for row in rows:
            concept = row["concept"]
            # Don't insert if this concept is already the current or next
            # module
            current_concepts = {
                m.concept for m in curriculum.modules[curriculum.current_index:]
            }
            if concept in current_concepts:
                continue

            detour = CurrModule(
                id="cross_session_detour_" + concept.replace(" ", "_")[:40],
                title="Revisit: " + concept,
                concept=concept,
                domain_framing=concept + " in " + self.state.domain,
                prerequisites=[],
                estimated_minutes=10,
                depth_level="surface",
            )
            curriculum.modules.insert(curriculum.current_index, detour)
            self.state.mark_dirty("curriculum")
            inserted += 1
            logger.info(
                "Cross-session gap: prepended revisit module for concept='{}'",
                concept
            )
            print(
                "\n🔁 Persistent gap detected: '" + concept + "' "
                "will be revisited this session.\n"
            )

        if inserted > 0:
            logger.info(
                "{} cross-session gap module(s) prepended for student='{}'",
                inserted, self.state.student_id
            )

    async def run_session(self, student_id: str, is_new: bool = False) -> None:
        """
        Run a complete learning session.

        Args:
            student_id: student identifier
            is_new:     True for first session (triggers onboarding)
        """
        self.state.start_session()
        topic = ""

        # ── Cross-session doubt detection ────────────────────────────────────
        # Reads doubt_log to find concepts the student has doubted across
        # multiple sessions — a signal of a persistent knowledge gap.
        # If found, a prerequisite detour module is prepended before the loop.
        if not is_new and self.state.curriculum:
            await self._check_cross_session_doubts()

        # ── Layer 1: Onboarding (first session only) ─────────────────────────
        if is_new or not self.state.curriculum:
            topic = await self._onboard()
            architect = CurriculumArchitectAgent(self.state)
            await architect.build_curriculum(topic)
        else:
            print(f"\n👋 Welcome back, {self.state.name}!")
            print(f"   Resuming: {self.state.curriculum.topic}")
            print(f"   Progress: module {self.state.curriculum.current_index + 1}"
                  f"/{len(self.state.curriculum.modules)}\n")

        # ── Layer 2: Module loop ─────────────────────────────────────────────
        completed = await self._run_module_loop()

        # Check if curriculum is complete
        if (self.state.curriculum and
                self.state.curriculum.current_index >= len(self.state.curriculum.modules)):
            print(
                f"\n🎉 Curriculum complete! You've mastered all modules in '{self.state.curriculum.topic}'!")

        # ── Layer 3: Session end ─────────────────────────────────────────────
        await self._end_session(completed)
