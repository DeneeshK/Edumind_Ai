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

import asyncio
from datetime import datetime

from loguru import logger

from agents.base_agent import BaseAgent
from agents.curriculum_architect import CurriculumArchitectAgent
from agents.tutor import TutorAgent
from agents.evaluator import EvaluatorAgent
from agents.adaptation_engine import AdaptationEngine
from core.student_model import StudentState
from db.postgres import (
    init_db, upsert_student, write_session_memory, bulk_write_decisions
)
from clients.tavily_client import clear_cache
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
                required=["session_summary", "modules_completed", "next_session_hint"],
            ),
        ]

    # ── Tool executor ─────────────────────────────────────────────────────────

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        if tool_name == "plan_session":
            modules_to_cover = args.get("modules_to_cover", [])
            session_goal = args.get("session_goal", "")
            logger.info("Session plan: {} modules, goal='{}'", len(modules_to_cover), session_goal)
            return f"Session planned: covering {modules_to_cover}. Goal: {session_goal}"
        return super()._execute_tool(tool_name, args)

    # ── Layer 1: Onboarding ───────────────────────────────────────────────────

    async def _onboard(self) -> str:
        """Collect student name, domain, goal, pace via CLI. Returns topic."""
        print("\n" + "="*60)
        print("🎓 Welcome to EduMind — Adaptive Learning System")
        print("="*60 + "\n")

        name = input("Your name: ").strip() or "Student"
        domain = input("Your domain/field (e.g. 'machine learning', 'physics'): ").strip()
        goal = input("Your learning goal (e.g. 'understand transformers for NLP'): ").strip()
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

    # ── Layer 2: Module loop ──────────────────────────────────────────────────

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

        max_modules_per_session = 3
        modules_done = 0

        while (
            curriculum.current_index < len(curriculum.modules)
            and modules_done < max_modules_per_session
        ):
            module = curriculum.modules[curriculum.current_index]
            print(f"\n{'─'*60}")
            print(f" Module {curriculum.current_index + 1}/{len(curriculum.modules)}: {module.title}")
            print(f"{'─'*60}")

            # ── Teach ─────────────────────────────────────────────────────────
            try:
                tutor = TutorAgent(self.state)
                lesson_result = tutor.teach()
            except Exception as exc:
                logger.error("TutorAgent.teach() failed for '{}': {}", module.concept, exc)
                print(f"\n⚠️  Lesson delivery failed ({exc}). Skipping to evaluation with partial context.")
                lesson_result = {"style_used": "formal", "fatigue_detected": "no", "doubt_count": 0}

            # ── Doubt count trigger (micro-example injection) ─────────────────
            doubt_count = self.state.get_doubt_count(module.concept)
            if doubt_count >= 2:
                print(f"\n💡 You raised {doubt_count} doubts on '{module.concept}'. Injecting a worked example…\n")
                try:
                    self._inject_micro_example(module.concept)
                except Exception as exc:
                    logger.warning("Micro-example injection failed: {}", exc)

            style_used = lesson_result.get("style_used", "formal")

            # ── Confidence rating ─────────────────────────────────────────────
            print(f"\n📊 How confident are you about '{module.concept}'? (1–5): ", end="")
            try:
                confidence = int(input().strip())
                confidence = max(1, min(5, confidence))
            except (ValueError, EOFError):
                confidence = 3

            # ── Evaluate ──────────────────────────────────────────────────────
            try:
                evaluator = EvaluatorAgent(self.state)
                report = await evaluator.evaluate(module.concept, confidence)
            except Exception as exc:
                logger.error("EvaluatorAgent.evaluate() failed for '{}': {}", module.concept, exc)
                print(f"\n⚠️  Evaluation failed ({exc}). Recording zero mastery and reteaching.")
                # Safe fallback: build a zero-score report so adaptation can still run
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
            self.state.metacognition.record_style_depth(style_used, report.depth_score)

            # ── Adapt ──────────────────────────────────────────────────────────
            try:
                engine = AdaptationEngine(self.state)
                decision = engine.decide(report)
            except Exception as exc:
                logger.error("AdaptationEngine.decide() failed: {}", exc)
                print(f"\n⚠️  Adaptation decision failed ({exc}). Defaulting to RETEACH.")
                from core.student_model import AdaptationDecision
                decision = AdaptationDecision(
                    action="RETEACH",
                    reason="Adaptation engine error — defaulting to reteach.",
                )

            # ── Gap analysis (every 3 evaluation cycles) ───────────────────────
            # Called on the SAME engine instance so it has the correct self.state
            try:
                gap_concept = engine.run_gap_analysis()
                if gap_concept:
                    print(f"\n🔍 Gap analysis detected missing prerequisite: '{gap_concept}'")
                    print(f"   A micro-session on '{gap_concept}' will run at your next session start.")
                    # Record for next-session injection (stored in decision_log via _log_decision
                    # inside run_gap_analysis itself; Orchestrator reads it at session start)
            except Exception as exc:
                logger.warning("run_gap_analysis() failed (non-critical): {}", exc)

            print(f"\n⚙️  Decision: {decision.action} — {decision.reason}")

            # ── Apply decision ─────────────────────────────────────────────────
            if decision.action in ("MOVE_FORWARD", "MOVE_FORWARD_WITH_FLAG"):
                completed.append(module.id)
                curriculum.current_index += 1
                self.state.mark_dirty("curriculum")
                self.state.metacognition.consecutive_reteach_count = 0
                modules_done += 1
                print(f"✅ '{module.concept}' mastered! Moving forward.\n")

            elif decision.action == "RETEACH":
                self.state.metacognition.consecutive_reteach_count += 1
                new_style = decision.style_for_reteach
                if new_style:
                    self.state.metacognition.preferred_style = new_style
                    print(f"🔄 Reteaching '{module.concept}' using '{new_style}' style.\n")
                else:
                    print(f"🔄 Reteaching '{module.concept}'.\n")

            elif decision.action == "DETOUR":
                missing_concept = decision.missing_concept
                if missing_concept:
                    print(f"↩️  Detour — must learn '{missing_concept}' first.\n")
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
                    logger.warning("DETOUR decision has no missing_concept — treating as RETEACH")
                    self.state.metacognition.consecutive_reteach_count += 1
                    print(f"🔄 Detour requested but no concept specified — reteaching.\n")

            elif decision.action == "ESCALATE":
                print(f"\n🚨 '{module.concept}' could not be mastered after repeated attempts.")
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
                    print(f"   New remedial path: {len(new_plan.modules)} modules built.\n")
                except Exception as exc:
                    # Fallback: advance past the blocking concept so the
                    # session is not permanently stuck
                    logger.error("ESCALATE curriculum rebuild failed: {} — advancing", exc)
                    print("   Rebuild failed — skipping concept to avoid session lock.\n")
                    completed.append(module.id)
                    curriculum.current_index += 1
                    self.state.mark_dirty("curriculum")
                    modules_done += 1

            elif decision.action == "COMPRESS":
                print(f"⚡ Compressing — student is ahead. Accelerating.\n")
                completed.append(module.id)
                curriculum.current_index += 1
                self.state.mark_dirty("curriculum")
                modules_done += 1

            elif decision.action == "HOLD":
                print(f"\n⏸️  Session paused at student request.")
                break

            # Fatigue check
            if lesson_result.get("fatigue_detected") == "yes":
                print("\n😴 Fatigue detected — ending session early.")
                break

        return completed


    # ── Layer 3: Session end ──────────────────────────────────────────────────

    async def _end_session(self, completed_modules: list[str]) -> None:
        """Flush all session data to DB."""

        # Orchestrator LLM generates summary
        result = self.run(
            system=(
                "You are ending a learning session. "
                "Generate a session summary and end_session tool call."
            ),
            user_message=(
                f"Session complete. Student: {self.state.name}. "
                f"Modules completed: {completed_modules}. "
                f"Domain: {self.state.domain}. "
                f"Generate a session summary."
            ),
            model=settings.generation_model,
        )

        summary = result.get("session_summary", f"Covered {len(completed_modules)} modules.")

        # Write session memory
        await write_session_memory(
            student_id=self.state.student_id,
            session_id=self.state.session_id,
            summary=summary,
            modules_covered=completed_modules,
            started_at=self.state.session_started_at,
        )

        # Flush state to DB
        await self.state.save()

        # Clear Tavily cache
        clear_cache()

        print(f"\n{'='*60}")
        print(f"✅ Session complete!")
        print(f"   Summary: {summary}")
        hint = result.get("next_session_hint", "")
        if hint:
            print(f"   Next: {hint}")
        print(f"{'='*60}\n")
    
    # ── Micro-example injection ───────────────────────────────────────────────

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

        messages = [{"role": "user", "content": f"Worked example for: {concept}"}]

        print(f"\n{'─'*50}")
        print(f"📌 Worked Example: {concept}")
        print(f"{'─'*50}\n")

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
            logger.warning("Micro-example stream timed out for concept='{}'", concept)
            print(f"\n[Worked example unavailable — Groq timeout]\n")

        except GroqRateLimitError:
            logger.warning("Micro-example rate-limited for concept='{}'", concept)
            print(f"\n[Worked example unavailable — rate limit]\n")


    # ── Main entry point ──────────────────────────────────────────────────────

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
            logger.warning("Cross-session doubt query failed: {} — skipping", e)
            return

        if not rows:
            return

        curriculum = self.state.curriculum
        inserted = 0
        for row in rows:
            concept = row["concept"]
            # Don't insert if this concept is already the current or next module
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

        # ── Cross-session doubt detection ─────────────────────────────────────
        # Reads doubt_log to find concepts the student has doubted across
        # multiple sessions — a signal of a persistent knowledge gap.
        # If found, a prerequisite detour module is prepended before the loop.
        if not is_new and self.state.curriculum:
            await self._check_cross_session_doubts()

        # ── Layer 1: Onboarding (first session only) ──────────────────────────
        if is_new or not self.state.curriculum:
            topic = await self._onboard()
            architect = CurriculumArchitectAgent(self.state)
            await architect.build_curriculum(topic)
        else:
            print(f"\n👋 Welcome back, {self.state.name}!")
            print(f"   Resuming: {self.state.curriculum.topic}")
            print(f"   Progress: module {self.state.curriculum.current_index + 1}"
                  f"/{len(self.state.curriculum.modules)}\n")

        # ── Layer 2: Module loop ──────────────────────────────────────────────
        completed = await self._run_module_loop()

        # Check if curriculum is complete
        if (self.state.curriculum and
                self.state.curriculum.current_index >= len(self.state.curriculum.modules)):
            print(f"\n🎉 Curriculum complete! You've mastered all modules in '{self.state.curriculum.topic}'!")

        # ── Layer 3: Session end ──────────────────────────────────────────────
        await self._end_session(completed)