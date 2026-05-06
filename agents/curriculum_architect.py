"""
agents/curriculum_architect.py
CurriculumArchitectAgent — builds a CurriculumPlan from scratch or repairs it.

Uses Tavily to gather real domain content, then structures it into modules.
After building the plan it embeds each module into ChromaDB so the RAG
pipeline has a real knowledge base to retrieve from during lesson delivery.

Terminal tool: submit_curriculum
Non-terminal tools: search_domain, add_module
"""

from __future__ import annotations

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from agents.base_agent import BaseAgent
from core.student_model import StudentState, CurriculumPlan, Module
from clients.tavily_client import search as tavily_search
from db.chromadb_client import insert as chroma_insert
from db.postgres import get_conn
from config import settings

# Shared executor for CPU-bound embedding — offloads sentence-transformer
# inference off the async event loop.
_EMBED_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="chroma_embed")


class CurriculumArchitectAgent(BaseAgent):
    NAME = "curriculum_architect"
    TERMINAL_TOOL = "submit_curriculum"

    def __init__(self, state: StudentState):
        super().__init__(state)
        self._modules_buffer: list[dict] = []
        # Keyed by query string -> list of content strings from Tavily.
        # Populated during search_domain tool calls; consumed during embedding.
        self._tavily_cache: dict[str, list[str]] = {}

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

    # -- Tool executor ---------------------------------------------------------

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        if tool_name == "search_domain":
            query = args["query"]
            results = tavily_search(query, max_results=5)
            if not results:
                return "No results found. Try a different query."

            summaries = []
            content_texts = []
            for r in results[:3]:
                title = r.get("title", "")
                content = r.get("content", "")[:300]
                summaries.append("- " + title + ": " + content)
                full_content = r.get("content", "").strip()
                if full_content:
                    content_texts.append(full_content[:500])

            # Cache Tavily content keyed by query for embedding phase
            self._tavily_cache[query] = content_texts
            return "\n".join(summaries)

        if tool_name == "add_module":
            self._modules_buffer.append(args)
            logger.debug("Module added: {}", args.get("id"))
            return (
                "Module '" + str(args.get("id")) + "' ("
                + str(args.get("concept")) + ") added. "
                "Total: " + str(len(self._modules_buffer))
            )

        return super()._execute_tool(tool_name, args)

    # -- Public build method ---------------------------------------------------

    async def build_curriculum(self, topic: str) -> CurriculumPlan:
        """
        Build a full CurriculumPlan for the given topic.
        Saves to DB, updates state.curriculum, and populates ChromaDB
        so the RAG pipeline has a knowledge base from the very first lesson.
        """
        pace = self.state.pace
        depth_map = {"fast": "surface", "medium": "standard", "deep": "deep"}
        depth_level = depth_map[pace]

        known = [c for c, m in self.state.concept_mastery.items() if m >= 0.7]
        known_str = ", ".join(known) if known else "none"

        system = (
            "You are a curriculum architect for an adaptive learning system.\n"
            "Build a structured curriculum plan for a student learning '" + topic + "'.\n\n"
            "STUDENT CONTEXT:\n"
            + self._student_context() + "\n\n"
            "CURRICULUM RULES:\n"
            "- Use search_domain 2-3 times to understand prerequisites and concept ordering\n"
            "- Add modules in strict learning order (prerequisites before dependents)\n"
            "- Build 4-8 modules total\n"
            "- All modules should use depth_level='" + depth_level + "' matching the student's pace\n"
            "- Domain: " + self.state.domain + " -- frame every concept in this domain context\n"
            "- Goal: " + self.state.goal + "\n"
            "- Concepts already mastered (skip these): " + known_str + "\n"
            "- estimated_minutes per module: 8-15 for fast, 10-20 for medium, 15-30 for deep\n"
            "- After adding all modules, call submit_curriculum\n"
        )

        result = self.run(
            system=system,
            user_message=(
                "Build a curriculum for topic: '" + topic
                + "' in domain: '" + self.state.domain + "'"
            ),
            model=settings.generation_model,
        )

        # Build CurriculumPlan from buffered modules
        modules = []
        for i, m in enumerate(self._modules_buffer):
            modules.append(Module(
                id=m.get("id", "m" + str(i + 1)),
                title=m.get("title", "Module " + str(i + 1)),
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
                domain_framing=topic + " in " + self.state.domain,
                prerequisites=[], estimated_minutes=15,
                depth_level=depth_level,
            )]

        # Validate minimum module count
        if len(modules) < 4:
            logger.warning(
                "Only {} modules buffered (minimum is 4). "
                "Curriculum may be incomplete.", len(modules)
            )

        plan = CurriculumPlan(
            topic=result.get("topic", topic),
            domain=self.state.domain,
            goal=self.state.goal,
            modules=modules,
            current_index=0,
            version=1,
        )

        # Save to PostgreSQL
        async with get_conn() as conn:
            await conn.execute(
                "UPDATE curricula SET is_active=FALSE WHERE student_id=$1",
                self.state.student_id,
            )
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

        # Update state
        self.state.curriculum = plan
        self.state.mark_dirty("curriculum")

        self._log_decision(
            action="BUILD_CURRICULUM",
            reason=result.get("rationale", "New curriculum built"),
            payload={"topic": plan.topic, "module_count": len(modules)},
        )

        logger.info(
            "Curriculum built: {} modules for topic='{}'", len(modules), plan.topic
        )

        # Populate ChromaDB — error-safe, never blocks curriculum delivery
        await self._embed_modules_to_chromadb(plan)

        return plan

    # -- ChromaDB population ---------------------------------------------------

    async def _embed_modules_to_chromadb(self, plan: CurriculumPlan) -> None:
        """
        Embed each module into ChromaDB so the RAG pipeline has a real
        knowledge base for lesson delivery.

        Without this, ChromaDB is always empty for new curricula.
        The RAG pipeline falls back to Tavily-only — which works, but is
        slower, costs more API calls, and loses domain-specific context
        accumulated over multiple sessions.

        Three-layer embedding per module:
          1. Structured concept card  (title, framing, depth, prerequisites)
          2. Tavily content from search_domain calls  (real domain knowledge)
          3. LLM-generated retrieval summary (80-100 words, once at build time)

        CPU-bound encoding is offloaded to _EMBED_EXECUTOR so the async
        event loop is never blocked during embedding.
        """
        loop = asyncio.get_event_loop()
        domain = plan.domain
        embedded_count = 0
        failed_count = 0

        # Flatten all cached Tavily texts from the curriculum build phase
        all_tavily_texts: list[str] = []
        for texts in self._tavily_cache.values():
            all_tavily_texts.extend(texts)

        for module in plan.modules:
            concept = module.concept
            framing = module.domain_framing
            prereqs = ", ".join(module.prerequisites) if module.prerequisites else "none"
            safe_concept = re.sub(r"[^a-zA-Z0-9_]", "_", concept)[:60]

            # Layer 1: Structured concept card
            concept_card_lines = [
                "Concept: " + concept,
                "Domain: " + domain,
                "Domain framing: " + framing,
                "Module title: " + module.title,
                "Depth level: " + module.depth_level,
                "Prerequisites: " + prereqs,
            ]
            concept_card = "\n".join(concept_card_lines)

            # Layer 2: Matching Tavily content (keyword overlap with concept)
            concept_keywords = concept.lower().split()[:3]
            matching_tavily = [
                t for t in all_tavily_texts
                if any(kw in t.lower() for kw in concept_keywords)
            ][:2]

            # Layer 3: LLM retrieval summary generated once at build time
            retrieval_summary = ""
            try:
                from clients.groq_client import generate
                summary_prompt = (
                    "Write an 80-100 word concept summary for vector retrieval.\n"
                    "Concept: " + concept + "\n"
                    "Domain framing: " + framing + "\n"
                    "Depth: " + module.depth_level + "\n"
                    "Cover: definition, key properties, one concrete example. "
                    "Be factual. No preamble."
                )
                retrieval_summary = generate(
                    messages=[{"role": "user", "content": summary_prompt}],
                    model=settings.generation_model,
                )
            except Exception as e:
                logger.warning(
                    "LLM summary failed for concept='{}': {} -- "
                    "embedding concept card only", concept, e
                )

            # Assemble full embedding document
            parts = [concept_card]
            if retrieval_summary:
                parts.append("Summary:\n" + retrieval_summary)
            if matching_tavily:
                parts.append("Related content:\n" + "\n---\n".join(matching_tavily))

            full_text = "\n\n".join(parts)

            # Insert into ChromaDB via thread pool (CPU-bound)
            chunk_id = (plan.topic + "_" + safe_concept).replace(" ", "_")[:180]
            try:
                await loop.run_in_executor(
                    _EMBED_EXECUTOR,
                    chroma_insert,
                    chunk_id,
                    domain,
                    full_text,
                )
                embedded_count += 1
                logger.info(
                    "ChromaDB: embedded '{}' -> chunk_id='{}'", concept, chunk_id
                )
            except Exception as e:
                failed_count += 1
                logger.warning(
                    "ChromaDB insert failed for concept='{}': {} -- "
                    "RAG will use Tavily fallback for this concept", concept, e
                )

        if embedded_count > 0:
            logger.info(
                "ChromaDB population complete: {}/{} modules embedded for domain='{}'",
                embedded_count, len(plan.modules), domain,
            )
        if failed_count > 0:
            logger.warning(
                "{} modules failed ChromaDB insertion -- "
                "RAG pipeline will use Tavily fallback for those concepts",
                failed_count,
            )