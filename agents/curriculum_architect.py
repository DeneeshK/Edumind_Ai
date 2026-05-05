"""
agents/curriculum_architect.py
CurriculumArchitectAgent — builds a CurriculumPlan from scratch or repairs it.

Uses Tavily to gather real domain content, then structures it into modules.
Terminal tool: submit_curriculum
Non-terminal tools: search_domain, add_module
"""

from __future__ import annotations

import json

from loguru import logger

from agents.base_agent import BaseAgent
from core.student_model import StudentState, CurriculumPlan, Module
from clients.tavily_client import search as tavily_search
from db.postgres import get_conn
from config import settings


class CurriculumArchitectAgent(BaseAgent):
    NAME = "curriculum_architect"
    TERMINAL_TOOL = "submit_curriculum"

    def __init__(self, state: StudentState):
        super().__init__(state)
        self._modules_buffer: list[dict] = []

        self.TOOLS = [
            self.build_tool(
                name="search_domain",
                description=(
                    "Search the web for content about a topic in the student's domain. "
                    "Use 2-3 targeted queries to gather prerequisite structure and key concepts."
                ),
                properties={
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g. 'linear algebra prerequisites for ML')",
                    },
                },
                required=["query"],
            ),
            self.build_tool(
                name="add_module",
                description=(
                    "Add one module to the curriculum plan. "
                    "Call this once per concept in learning order. "
                    "Build 4-8 modules total depending on topic complexity."
                ),
                properties={
                    "id": {
                        "type": "string",
                        "description": "Unique module ID (e.g. 'm1', 'm2')",
                    },
                    "title": {
                        "type": "string",
                        "description": "Short module title",
                    },
                    "concept": {
                        "type": "string",
                        "description": "Core concept this module teaches",
                    },
                    "domain_framing": {
                        "type": "string",
                        "description": (
                            "How to frame this concept in the student's domain. "
                            "E.g. 'dot product as similarity measure for embeddings in NLP'"
                        ),
                    },
                    "prerequisites": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of concept names that must be learned first",
                    },
                    "estimated_minutes": {
                        "type": "integer",
                        "description": "Estimated learning time in minutes (5-30)",
                    },
                    "depth_level": {
                        "type": "string",
                        "enum": ["surface", "standard", "deep"],
                        "description": (
                            "surface=fast pace overview, "
                            "standard=medium pace full coverage, "
                            "deep=deep pace with proofs and edge cases"
                        ),
                    },
                },
                required=[
                    "id", "title", "concept", "domain_framing",
                    "prerequisites", "estimated_minutes", "depth_level",
                ],
            ),
            self.build_tool(
                name="submit_curriculum",
                description=(
                    "Submit the completed curriculum plan. "
                    "Call this after adding all modules (minimum 4)."
                ),
                properties={
                    "topic": {
                        "type": "string",
                        "description": "Overall topic of the curriculum",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "One paragraph explaining the module ordering and depth choices",
                    },
                },
                required=["topic", "rationale"],
            ),
        ]

    # ── Tool executor ─────────────────────────────────────────────────────────

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        if tool_name == "search_domain":
            query = args["query"]
            results = tavily_search(query, max_results=5)
            if not results:
                return "No results found. Try a different query."
            summaries = []
            for r in results[:3]:
                title = r.get("title", "")
                content = r.get("content", "")[:300]
                summaries.append(f"• {title}: {content}")
            return "\n".join(summaries)

        if tool_name == "add_module":
            self._modules_buffer.append(args)
            logger.debug("Module added: {}", args.get("id"))
            return f"Module '{args.get('id')}' ({args.get('concept')}) added. Total: {len(self._modules_buffer)}"

        return super()._execute_tool(tool_name, args)

    # ── Public run method ─────────────────────────────────────────────────────

    async def build_curriculum(self, topic: str) -> CurriculumPlan:
        """
        Build a full CurriculumPlan for the given topic.
        Saves to DB and updates state.curriculum.

        Args:
            topic: the subject to build a curriculum for

        Returns:
            CurriculumPlan
        """
        pace = self.state.pace
        depth_map = {"fast": "surface", "medium": "standard", "deep": "deep"}
        depth_level = depth_map[pace]

        known = [c for c, m in self.state.concept_mastery.items() if m >= 0.7]
        known_str = ", ".join(known) if known else "none"

        system = f"""You are a curriculum architect for an adaptive learning system.
Build a structured curriculum plan for a student learning '{topic}'.

STUDENT CONTEXT:
{self._student_context()}

CURRICULUM RULES:
- Use search_domain 2-3 times to understand prerequisites and concept ordering
- Add modules in strict learning order (prerequisites before dependents)
- Build 4-8 modules total
- All modules should use depth_level='{depth_level}' matching the student's pace
- Domain: {self.state.domain} — frame every concept in this domain context
- Goal: {self.state.goal}
- Concepts already mastered (skip these): {known_str}
- estimated_minutes per module: 8-15 for fast, 10-20 for medium, 15-30 for deep
- After adding all modules, call submit_curriculum
"""

        result = self.run(
            system=system,
            user_message=f"Build a curriculum for topic: '{topic}' in domain: '{self.state.domain}'",
            model=settings.generation_model,
        )

        # ── Build CurriculumPlan from buffered modules ─────────────────────────
        modules = []
        for i, m in enumerate(self._modules_buffer):
            modules.append(Module(
                id=m.get("id", f"m{i+1}"),
                title=m.get("title", f"Module {i+1}"),
                concept=m.get("concept", ""),
                domain_framing=m.get("domain_framing", ""),
                prerequisites=m.get("prerequisites", []),
                estimated_minutes=int(m.get("estimated_minutes", 10)),
                depth_level=m.get("depth_level", depth_level),
            ))

        if not modules:
            logger.warning("No modules buffered — creating fallback single-module plan")
            modules = [Module(
                id="m1", title=topic, concept=topic,
                domain_framing=f"{topic} in {self.state.domain}",
                prerequisites=[], estimated_minutes=15,
                depth_level=depth_level,
            )]

        plan = CurriculumPlan(
            topic=result.get("topic", topic),
            domain=self.state.domain,
            goal=self.state.goal,
            modules=modules,
            current_index=0,
            version=1,
        )

        # ── Save to DB ────────────────────────────────────────────────────────
        async with get_conn() as conn:
            # Deactivate old plans
            await conn.execute(
                "UPDATE curricula SET is_active=FALSE WHERE student_id=$1",
                self.state.student_id,
            )
            # Insert new plan
            row = await conn.fetchrow(
                """
                INSERT INTO curricula
                  (student_id, topic, plan_json, current_index, version, is_active)
                VALUES ($1, $2, $3, 0, 1, TRUE)
                RETURNING id
                """,
                self.state.student_id,
                plan.topic,
                plan.model_dump_json(),
            )
            logger.info("Curriculum saved to DB id={}", row["id"])

        # ── Update state ──────────────────────────────────────────────────────
        self.state.curriculum = plan
        self.state.mark_dirty("curriculum")

        self._log_decision(
            action="BUILD_CURRICULUM",
            reason=result.get("rationale", "New curriculum built"),
            payload={"topic": plan.topic, "module_count": len(modules)},
        )

        logger.info("Curriculum built: {} modules for topic='{}'", len(modules), plan.topic)
        return plan
