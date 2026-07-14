"""
LEGACY — interactive CLI/SSE session flow. Not used by the deployed frontend,
which uses the /api/courses flow (see docs/ARCHITECTURE.md). Kept as a working
reference implementation of the queue-based interactive session pattern.

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

from loguru import logger

from agents.base_agent import BaseAgent
from agents.curriculum_architect import CurriculumArchitectAgent
from agents.tutor import TutorAgent
from agents.evaluator import EvaluatorAgent
from agents.adaptation_engine import AdaptationEngine
from core.student_model import StudentState
from config import settings


class OrchestratorAgent(BaseAgent):
    """Coordinates onboarding, tutoring, evaluation, adaptation, and session close."""

    NAME = "orchestrator"
    TERMINAL_TOOL = "end_session"

    def __init__(
        self,
        state: StudentState,
        emit_fn=None,
        ask_fn=None,
    ):
        super().__init__(state)
        # API mode: use injected async emit/ask functions
        # CLI mode (fallback): use print/input
        self._emit = emit_fn or self._cli_emit
        self._ask = ask_fn or self._cli_ask
        self._eval_runner = None
        self.TOOLS = self._build_tools()

    def _ensure_eval_runner(self, topic: str = ""):
        """Create and register the optional evaluation runner for the session."""
        if not settings.eval_enabled:
            return None
        if self._eval_runner is None:
            from evaluation.runner import EvaluationRunner

            curriculum_topic = (
                self.state.curriculum.topic
                if self.state.curriculum else ""
            )
            self._eval_runner = EvaluationRunner(
                session_id=self.state.session_id,
                student_id=self.state.student_id,
                topic=topic or curriculum_topic,
                pace=self.state.pace,
            )
        # The runner is handed to each agent directly (tutor/evaluator/architect
        # set `_eval_runner`). The old core/rag_pipeline registration hook was
        # removed together with the disabled retrieval path.
        return self._eval_runner

    def _clear_eval_runner(self) -> None:
        """Clear the optional evaluation runner for the session."""
        self._eval_runner = None

    async def _cli_emit(self, text: str) -> None:
        """Emit session text in legacy CLI mode."""
        print(text)

    async def _cli_ask(self, question: str, is_confidence: bool = False) -> str:
        """Ask for input in legacy CLI mode, matching the API ask contract."""
        label = "Confidence (1-5)" if is_confidence else "Your answer"
        print(f"\n{question}")
        return input(f"{label}: ").strip()

    def _build_tools(self) -> list[dict]:
        """Build tool schemas that let the LLM plan, route, and close sessions."""
        return [
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
                        "description": (
                            "Only required when action=RETEACH. "
                            "Which style to switch to. "
                            "Must be one of: formal, socratic, example_first, visual, story. "
                            "Leave empty otherwise."
                        ),
                    },
                    "missing_concept": {
                        "type": "string",
                        "description": (
                            "Only required when action=DETOUR. "
                            "The prerequisite concept name."
                        ),
                    },
                },
                required=["action", "reason"],
            ),
        ]

    # ── Tool executor ─────────────────────────────────────────────────────────

    async def _execute_tool(self, tool_name: str, args: dict) -> str:
        """Handle Orchestrator-owned tool calls and delegate shared tools."""
        if tool_name == "plan_session":
            modules_to_cover = args.get("modules_to_cover", [])
            session_goal = args.get("session_goal", "")
            logger.info("Session plan: {} modules, goal='{}'", len(modules_to_cover), session_goal)
            return "Session planned: covering " + str(modules_to_cover) + ". Goal: " + session_goal

        if tool_name == "route_after_evaluation":
            # Store routing decision so _run_module_loop can read it
            self._last_route = args
            action = args.get("action", "")
            reason = args.get("reason", "")
            logger.info(
                "LLM routing decision: action={} reason='{}'",
                action,
                reason,
            )
            return "Routing decision recorded: " + action

        return await super()._execute_tool(tool_name, args)

    # ── LLM routing ───────────────────────────────────────────────────────────

    async def _llm_route(self, module_concept: str, report_summary: str) -> dict:
        """
        Ask the Orchestrator LLM to reason over the evaluation result
        and current student state, then call route_after_evaluation.

        This replaces a bare Python if/else chain — the LLM is the
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
            "CRITICAL: Never output HOLD unless the student's text literally says "
            "'stop', 'quit', 'pause' or 'exit'.\n\n"
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

        # Use a dedicated tool list containing only route_after_evaluation
        # so the LLM cannot accidentally call plan_session or end_session here.
        routing_tools = [
            t for t in self.TOOLS
            if t["function"]["name"] == "route_after_evaluation"
        ]

        from clients.groq_client import tool_call_loop
        await tool_call_loop(
            system=system,
            user_message=user_msg,
            tools=routing_tools,
            terminal_tool_name="route_after_evaluation",
            model=settings.reasoning_model,
            tool_executor=self._tool_executor_wrapper,
            _caller=self.NAME,
        )

        if not self._last_route:
            logger.warning(
                "LLM did not call route_after_evaluation — defaulting to RETEACH"
            )
            self._last_route = {
                "action": "RETEACH",
                "reason": "Routing fallback: LLM did not return a tool call.",
            }

        return self._last_route

    # ── Layer 1: Initial Session Setup ────────────────────────────────────────

    async def _emit_curriculum_overview(self) -> None:
        """Show the student the module path as soon as it exists."""
        curriculum = self.state.curriculum
        if not curriculum or not curriculum.modules:
            return

        await self._emit(
            f"🧭 Curriculum built: {len(curriculum.modules)} modules for "
            f"'{curriculum.topic}'."
        )
        for idx, module in enumerate(curriculum.modules, start=1):
            await self._emit(
                f"{idx}. {module.title} — {module.concept} "
                f"({module.estimated_minutes} min)"
            )

    async def _run_initial_setup_api(self, topic: str) -> None:
        """Initial setup for API mode; state is set by /session/start."""
        from db.postgres import upsert_student
        await upsert_student(
            self.state.student_id,
            self.state.name,
            self.state.domain,
            self.state.goal,
            self.state.pace,
        )
        await self._emit(
            f"✅ Welcome {self.state.name}! Let's learn '{topic}'. "
            f"Domain: {self.state.domain} | Pace: {self.state.pace}"
        )
        await self._emit(
            f"🔍 Building your personalised curriculum for '{topic}'... "
            f"This takes about 30-60 seconds."
        )
        if self._eval_runner is not None and not self._eval_runner.topic:
            self._eval_runner.topic = topic
        architect = CurriculumArchitectAgent(self.state)
        architect._eval_runner = self._eval_runner
        profile = getattr(architect, "personalization_profile", None) or {}
        # Emit periodic progress pings so the frontend connection stays alive
        # and the student knows the system is working.
        import asyncio as _asyncio
        async def _progress_ping(stop_event: _asyncio.Event) -> None:
            """Emit periodic curriculum-build progress messages until stopped."""
            messages = [
                "📚 Researching the topic...",
                "🗺️ Mapping concept dependencies...",
                "✍️ Designing your learning path...",
                "🔧 Finalising module sequence...",
            ]
            for msg in messages:
                await _asyncio.sleep(20)
                if stop_event.is_set():
                    return
                await self._emit(msg)

        stop_event = _asyncio.Event()
        ping_task = _asyncio.create_task(_progress_ping(stop_event))
        try:
            plan = await architect.build_curriculum(topic, profile)
        finally:
            stop_event.set()
            ping_task.cancel()
            try:
                await ping_task
            except _asyncio.CancelledError:
                pass
        await self._emit_curriculum_overview()

    async def _collect_cli_session_setup(self) -> str:
        """Collect the legacy CLI session setup interactively."""
        await self._emit("=" * 60)
        await self._emit("🎓 Welcome to EduMind — Adaptive Learning System")
        await self._emit("=" * 60)

        name = await self._ask("Your name:")
        name = name or "Student"
        domain = await self._ask("Your domain/field (e.g. 'machine learning', 'web development'):")
        goal = await self._ask("Your learning goal:")
        topic = await self._ask("Topic to learn today:")
        pace_raw = await self._ask("Choose pace [fast/medium/deep]:")
        pace = pace_raw.strip().lower() if pace_raw.strip().lower() in ("fast", "medium", "deep") else "medium"

        self.state.name = name
        self.state.domain = domain
        self.state.goal = goal
        self.state.pace = pace

        from db.postgres import upsert_student
        await upsert_student(self.state.student_id, name, domain, goal, pace)
        await self._emit(f"✅ Welcome {name}! Let's learn '{topic}'.")
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

        max_modules_per_session = 5
        modules_done = 0

        while (
            curriculum.current_index < len(curriculum.modules)
            and modules_done < max_modules_per_session
        ):
            module = curriculum.modules[curriculum.current_index]
            await self._emit(f"📖 MODULE {curriculum.current_index + 1} of {len(curriculum.modules)}: {module.title}")
            await self._emit(f"Concept: {module.concept} | Estimated: {module.estimated_minutes} min")

            # ── Teach ─────────────────────────────────────────────────────────
            await self._emit(f"💡 Preparing lesson on '{module.concept}'... (may take 20-40 seconds)")
            try:
                tutor = TutorAgent(self.state, emit_fn=self._emit, ask_fn=self._ask)
                tutor._eval_runner = self._eval_runner
                lesson_result = await tutor.teach()
            except Exception as exc:
                logger.error("TutorAgent.teach() failed for '{}': {}", module.concept, exc)
                await self._emit(f"⚠️ Lesson delivery failed: {exc}")
                lesson_result = {"style_used": "formal", "fatigue_detected": "no", "doubt_count": 0}

            # ── Doubt count trigger (micro-example injection) ─────────────────
            doubt_count = self.state.get_doubt_count(module.concept)
            if doubt_count >= 2:
                await self._emit(f"💡 You raised {doubt_count} doubts on '{module.concept}'. Injecting a worked example…")
                try:
                    await self._inject_micro_example(module.concept)
                except Exception as exc:
                    logger.warning("Micro-example injection failed: {}", exc)

            style_used = lesson_result.get("style_used", "formal")

            # ── Confidence rating ─────────────────────────────────────────────
            try:
                conf_raw = await self._ask(f"How confident are you about '{module.concept}'? (1=low 5=high)", is_confidence=True)
                confidence = max(1, min(5, int(conf_raw)))
            except (ValueError, EOFError):
                confidence = 3

            # ── Evaluate ──────────────────────────────────────────────────────
            try:
                evaluator = EvaluatorAgent(self.state, emit_fn=self._emit, ask_fn=self._ask)
                evaluator._eval_runner = self._eval_runner
                report = await evaluator.evaluate(module.concept, confidence)
            except Exception as exc:
                logger.error("EvaluatorAgent.evaluate() failed for '{}': {}", module.concept, exc)
                await self._emit(f"⚠️ Evaluation failed: {exc}")
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

            # ── AdaptationEngine: deep metacognitive decision ────────────────
            # AdaptationEngine runs its own agentic tool loop to decide the
            # recommended action. Its output feeds the Orchestrator LLM as
            # evidence — the Orchestrator makes the FINAL routing call.
            engine = AdaptationEngine(self.state)
            try:
                engine_decision = await engine.decide(report)
                engine_summary = (
                    "AdaptationEngine recommended: " + engine_decision.action
                    + " | reason: " + engine_decision.reason
                )
            except Exception as exc:
                logger.error("AdaptationEngine.decide() failed: {}", exc)
                engine_summary = "AdaptationEngine failed — use evaluation scores only."

            # ── Gap analysis (every 3 evaluation cycles) ───────────────────────
            try:
                gap_concept = await engine.run_gap_analysis()
                if gap_concept:
                    await self._emit(f"🔍 Gap analysis: missing prerequisite '{gap_concept}'")
                    engine_summary += " | Gap detected: " + gap_concept
            except Exception as exc:
                logger.warning("run_gap_analysis() failed (non-critical): {}", exc)
                gap_concept = None

            # ── Orchestrator LLM routing (TRULY AGENTIC) ─────────────────────
            # The LLM reasons over: evaluation report + engine recommendation
            # + student metacognition + mastery threshold for pace.
            # It calls route_after_evaluation to emit its decision.
            report_summary = (
                "mastery_score: " + str(round(report.mastery_score, 2)) + "\n"
                "correctness: " + str(round(report.correctness_score, 2)) + "\n"
                "depth: " + str(round(report.depth_score, 2)) + "\n"
                "misconception_type: " + str(report.misconception_type or "none") + "\n"
                "misconception_detail: " + str(report.misconception_detail or "none") + "\n"
                "confidence_stated: " + str(report.confidence_stated) + "\n"
                "calibration_delta: " + str(round(report.calibration_delta, 2)) + "\n"
                + engine_summary
            )
            route = await self._llm_route(module.concept, report_summary)
            action = route.get("action", "RETEACH")
            reason = route.get("reason", "")

            await self._emit(f"⚙️ Decision: {action} — {reason}")

            # ── Apply routing decision ─────────────────────────────────────────
            if action in ("MOVE_FORWARD", "MOVE_FORWARD_WITH_FLAG", "COMPRESS"):
                completed.append(module.id)
                curriculum.current_index += 1
                self.state.mark_dirty("curriculum")
                self.state.metacognition.consecutive_reteach_count = 0
                self.state.mark_dirty("metacognition")
                modules_done += 1
                if action == "COMPRESS":
                    await self._emit(f"⚡ Compressing — student is ahead. '{module.concept}' marked complete.")
                else:
                    await self._emit(f"✅ '{module.concept}' mastered!")
                if action == "MOVE_FORWARD_WITH_FLAG":
                    await self._emit(
                        "🎯 I will carry this misconception forward as a short "
                        "targeted correction in the next lesson."
                    )

            elif action == "RETEACH":
                new_style = route.get("style_for_reteach")
                if new_style:
                    self.state.metacognition.preferred_style = new_style
                    self.state.mark_dirty("metacognition")
                    await self._emit(f"🔄 Let's go through '{module.concept}' again using a {new_style} approach. This is attempt {self.state.metacognition.consecutive_reteach_count}.")
                else:
                    await self._emit(f"🔄 Let's revisit '{module.concept}' from a different angle. Attempt {self.state.metacognition.consecutive_reteach_count}.")
                # Small pause so frontend can display the reteach message before
                # the next lesson starts
                await asyncio.sleep(0.5)

            elif action == "DETOUR":
                missing_concept = route.get("missing_concept")
                if missing_concept:
                    await self._emit(f"↩️ Detour — must learn '{missing_concept}' first.")
                    from core.student_model import Module as CurrModule
                    detour = CurrModule(
                        id=f"detour_{missing_concept.replace(' ', '_')}",
                        title=f"Prerequisite: {missing_concept}",
                        concept=missing_concept,
                        domain_framing=f"{missing_concept} in {self.state.domain}",
                        prerequisites=[],
                        estimated_minutes=10,
                        depth_level={
                            "fast": "surface",
                            "medium": "standard",
                            "deep": "deep",
                        }.get(self.state.pace, "standard"),
                    )
                    curriculum.modules.insert(curriculum.current_index, detour)
                    self.state.mark_dirty("curriculum")
                else:
                    logger.warning("DETOUR decision has no missing_concept — treating as RETEACH")
                    await self._emit("🔄 Detour: no concept specified — reteaching.")

            elif action == "ESCALATE":
                await self._emit(f"🚨 '{module.concept}' could not be mastered. Rebuilding curriculum…")
                try:
                    architect = CurriculumArchitectAgent(self.state)
                    architect._eval_runner = self._eval_runner
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
                    await self._emit(f"New remedial path: {len(new_plan.modules)} modules built.")
                except Exception as exc:
                    # Fallback: advance past the blocking concept so the
                    # session is not permanently stuck
                    logger.error("ESCALATE curriculum rebuild failed: {} — advancing", exc)
                    await self._emit("Rebuild failed — skipping concept.")
                    completed.append(module.id)
                    curriculum.current_index += 1
                    self.state.mark_dirty("curriculum")
                    modules_done += 1

            elif action == "HOLD":
                await self._emit("⏸️ Session paused at your request.")
                break

            # Fatigue check
            if lesson_result.get("fatigue_detected") == "yes":
                await self._emit("😴 Fatigue detected — ending session early.")
                break

        return completed


    # ── Layer 3: Session end ──────────────────────────────────────────────────

    async def _end_session(self, completed_modules: list[str]) -> None:
        """Flush all session data to DB atomically."""

        # Use plain generate() with NO tools — no tool definitions means
        # the LLM cannot accidentally trigger a new routing cycle.
        # It can only return a plain text summary.
        from clients.groq_client import generate as groq_generate
        decision_records = []
        for d in self.state.session_decisions:
            if isinstance(d, dict):
                action = d.get("action", "?")
                reason = d.get("reason", d.get("rationale", ""))
                agent = d.get("agent", "adaptation_engine")
                payload = d.get("payload", d)
            else:
                action = getattr(d, "action", "?")
                reason = getattr(d, "reason", "")
                agent = getattr(d, "agent", "adaptation_engine")
                payload = d.model_dump() if hasattr(d, "model_dump") else {}

            decision_records.append({
                "student_id": self.state.student_id,
                "session_id": self.state.session_id,
                "agent": agent,
                "action": action,
                "rationale": reason,
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

        # Build mastery updates for atomic write. completed_modules contains
        # module IDs, while concept_mastery is keyed by concept name.
        completed_concepts = set(completed_modules)
        if self.state.curriculum:
            completed_ids = set(completed_modules)
            completed_concepts.update(
                m.concept for m in self.state.curriculum.modules
                if m.id in completed_ids
            )

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

        doubt_records = []
        for concept, count in self.state.session_doubt_counts.items():
            type_counts = self.state.session_doubt_types.get(concept) or {"general": count}
            for doubt_type, type_count in type_counts.items():
                doubt_records.append({
                    "concept": concept,
                    "doubt_type": doubt_type,
                    "count": type_count,
                })

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
            doubt_records=doubt_records,
        )

        await self._emit(f"✅ Session complete!\n{summary}")

        eval_runner = self._eval_runner
        if eval_runner is not None:
            curriculum = self.state.curriculum
            reteach_event_count = sum(
                1 for d in decision_records
                if str(d.get("action", "")).upper() == "RETEACH"
            )
            asyncio.create_task(
                eval_runner.on_session_end(
                    modules_mastered=len(completed_modules),
                    total_curriculum_modules=(
                        len(curriculum.modules) if curriculum else 0
                    ),
                    reteach_events=reteach_event_count,
                    calibration_deltas=[
                        r.calibration_delta
                        for r in self.state.evaluation_history
                    ],
                )
            )
            self._clear_eval_runner()
    
    # ── Micro-example injection ───────────────────────────────────────────────

    async def _inject_micro_example(self, concept: str) -> None:
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

        await self._emit(f"📌 Worked Example: {concept}")

        try:
            full_text = []
            async for chunk in stream(messages=messages, system=system,
                                      model=settings.generation_model,
                                      _caller="worked_example"):
                await self._emit(chunk)
                full_text.append(chunk)

            module = self._current_module()
            example_text = "".join(full_text).strip()
            if module is not None and example_text:
                self.state.record_module_content(
                    module.id,
                    "Worked example: " + concept + "\n" + example_text,
                )

            logger.info(
                "Micro-example injected for concept='{}' ({} chars)",
                concept, sum(len(c) for c in full_text)
            )

        except GroqTimeoutError:
            logger.warning("Micro-example stream timed out for concept='{}'", concept)
            await self._emit("[Worked example unavailable — Groq timeout]")

        except GroqRateLimitError:
            logger.warning("Micro-example rate-limited for concept='{}'", concept)
            await self._emit("[Worked example unavailable — rate limit]")


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
            await self._emit(
                "🔁 Persistent gap detected: '" + concept + "' "
                "will be revisited this session."
            )

        if inserted > 0:
            logger.info(
                "{} cross-session gap module(s) prepended for student='{}'",
                inserted, self.state.student_id
            )

    async def run_session(self, student_id: str, is_new: bool = False, topic: str = "") -> None:
        """
        Run a complete learning session.

        Args:
            student_id: student identifier
            is_new:     True for first session (triggers initial setup)
            topic:      topic string — required in API mode for new students
                        (already collected by /session/start before this runs)
        """
        self.state.start_session()
        self._ensure_eval_runner(topic)

        # Detect API mode: emit_fn was injected (not the default _cli_emit).
        # In API mode we must never call the CLI setup method, which uses input() and
        # would block the event loop under uvicorn.
        _api_mode = self._emit is not self._cli_emit

        # ── Cross-session doubt detection ─────────────────────────────────────
        if not is_new and self.state.curriculum:
            await self._check_cross_session_doubts()

        # ── Layer 1: Initial setup (first session only) ───────────────────────
        if is_new or not self.state.curriculum:
            if _api_mode:
                # In API mode state is already populated by /session/start;
                # just build the curriculum and show the overview.
                await self._run_initial_setup_api(topic)
            else:
                resolved_topic = await self._collect_cli_session_setup()
                if self._eval_runner is not None and not self._eval_runner.topic:
                    self._eval_runner.topic = resolved_topic
                architect = CurriculumArchitectAgent(self.state)
                architect._eval_runner = self._eval_runner
                await architect.build_curriculum(resolved_topic)
                await self._emit_curriculum_overview()
        else:
            await self._emit(
                f"👋 Welcome back, {self.state.name}! "
                f"Resuming: {self.state.curriculum.topic} — "
                f"module {self.state.curriculum.current_index + 1}"
                f"/{len(self.state.curriculum.modules)}"
            )
            await self._emit_curriculum_overview()

        # ── Layer 2: Module loop ──────────────────────────────────────────────
        completed = await self._run_module_loop()

        # Check if curriculum is complete
        if (self.state.curriculum and
                self.state.curriculum.current_index >= len(self.state.curriculum.modules)):
            await self._emit(
                f"🎉 Curriculum complete! You've mastered all modules "
                f"in '{self.state.curriculum.topic}'!"
            )

        # ── Layer 3: Session end ──────────────────────────────────────────────
        await self._end_session(completed)
