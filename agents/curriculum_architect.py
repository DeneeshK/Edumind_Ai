"""
agents/curriculum_architect.py
CurriculumArchitectAgent — builds roadmap-first CurriculumPlans.

Redesigned Flow (two-call coverage+sequencing architecture):
  1. Run Tavily research (2-3 focused queries)
  2. Extract research facts via 8B/17B model (chunked, sequential)
  3. CALL A — Coverage Planner: flat concept list for the subject (what to cover)
  4. CALL B — Sequencer: ordered module list from concept list (how to order)
  5. LLM Auditor: cross-checks against own subject knowledge, finds gaps
  6. Gap fill: targeted search + re-sequence if auditor finds missing concepts
  7. Python structural fix (IDs, dedup — no semantic work)
  8. Save to DB + ChromaDB

Key principles:
  - Coverage and sequencing are separate LLM calls — no tradeoff between them
  - LLM decides topic breadth from subject knowledge, not from hardcoded targets
  - Auditor uses its own knowledge as the reference, not the module list
  - Python only does deduplication, ID assignment — never semantic splitting
  - No hard-coded module count targets
  - Never crashes the user for a recoverable issue
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from typing import Any

from loguru import logger

from agents.base_agent import BaseAgent
from clients.groq_client import GroqRateLimitError, GroqTimeoutError, generate
from clients.tavily_client import search as tavily_search
from config import settings
from core.curriculum_quality import (
    CourseScopeAnalysis,
    fallback_scope_analysis,
    is_question_like_scope_text,
    is_related_to_profile,
    is_unreliable_generated_concept,
    parse_json_object,
    profile_has_no_prior_experience,
    validate_curriculum_quality,
    validate_master_roadmap,
    validate_modules_against_roadmap,
)
from core.student_model import (
    CurriculumPlan,
    MasterRoadmap,
    Module,
    ResearchQuery,
    ResearchSummary,
    RoadmapStep,
    StudentState,
)
from db.chromadb_client import insert as chroma_insert
from db.postgres import get_conn


class CurriculumArchitectAgent(BaseAgent):
    NAME = "curriculum_architect"
    TERMINAL_TOOL = "submit_curriculum"

    def __init__(self, state: StudentState):
        super().__init__(state)
        self._tavily_cache: dict[str, list[str]] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Research queries — focused, minimal, high-signal
    # ─────────────────────────────────────────────────────────────────────────

    def _research_queries(self, topic: str, profile: dict) -> list[ResearchQuery]:
        intent = profile.get("current_intent") if isinstance(profile.get("current_intent"), dict) else {}
        subject = str(profile.get("exact_subject") or intent.get("exact_subject") or topic or "").strip()
        target = str(profile.get("target_context") or intent.get("target_context") or self.state.domain or "").strip()
        goal = str(profile.get("learning_goal") or intent.get("goal") or self.state.goal or "learning").strip()
        context = target if target and target.lower() not in ("general", "general learning") else goal
        exam_like = any(k in f"{target} {goal}".lower() for k in ("exam", "jee", "neet", "gate", "interview"))
        exclusions = [str(x).strip() for x in (profile.get("do_not_include") or []) if str(x).strip()]

        # 2 queries for general topics, 3 for exam topics.
        # Fewer focused queries = less raw text = fewer extractor chunks = faster pipeline.
        # LLM's own subject knowledge covers what extra queries would provide for popular topics.
        if exam_like:
            candidates = [
                ResearchQuery(query=f"{subject} {target or goal} complete official syllabus topics", category="exam_specifics", priority=1),
                ResearchQuery(query=f"{subject} {target or goal} important topics weightage question patterns", category="exam_specifics", priority=1),
                ResearchQuery(query=f"{subject} prerequisite concepts complete topic list", category="prerequisites", priority=1),
            ]
        else:
            candidates = [
                ResearchQuery(query=f"{subject} complete curriculum syllabus all topics {context}", category="syllabus", priority=1),
                ResearchQuery(query=f"{subject} full topic breakdown beginner to advanced prerequisites", category="general_roadmap", priority=1),
            ]
            if exclusions:
                candidates.append(ResearchQuery(
                    query=f"{subject} core topics excluding {', '.join(exclusions[:3])}",
                    category="exclusions", priority=2,
                ))

        seen: set[str] = set()
        deduped: list[ResearchQuery] = []
        for q in candidates:
            key = q.query.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(q)
        return sorted(deduped[:5], key=lambda q: (q.priority, q.category))

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Research extraction — compact structured JSON from raw Tavily prose
    # Uses 8B model for general subjects, 17B (generation_model) for dense academic
    # ─────────────────────────────────────────────────────────────────────────

    # 16K chars per chunk = ~4000 tokens input. With 1500t output + 400t system = ~5900t.
    # Well within llama-3.1-8b-instant limits. 20K raw → 1-2 chunks vs 3 before.
    # Fewer chunks = fewer API calls = no RPM rate limit waits.
    _EXTRACTOR_CHUNK_CHARS = 16000
    _PLANNER_RESEARCH_CHAR_BUDGET = 9000

    def _select_extractor_model(self, topic: str, profile: dict) -> str:
        """
        Use the stronger 17B model for dense academic subjects where the 8B
        model tends to misname or miss jargon-heavy concepts.
        Subject classification is based on the topic string — no hardcoding of
        specific subject names, just domain-type heuristics.
        """
        topic_lower = topic.lower()
        goal_lower = str(profile.get("learning_goal") or "").lower()
        dense_signals = (
            "organic chemistry", "inorganic chemistry", "biochemistry",
            "quantum", "thermodynamics", "electromagnetism", "optics",
            "calculus", "linear algebra", "differential equation", "topology",
            "anatomy", "physiology", "pharmacology", "pathology",
            "corporate law", "constitutional law", "contract law",
            "machine learning", "deep learning", "neural network",
            "data structure", "algorithm", "compiler", "operating system",
        )
        for signal in dense_signals:
            if signal in topic_lower or signal in goal_lower:
                logger.info("Dense academic subject detected ('{}') — using generation_model for extraction.", topic)
                return settings.generation_model
        return settings.small_task_model

    async def _extract_research_facts(self, raw_text: str, topic: str, goal: str, extractor_model: str | None = None) -> str:
        """
        Converts raw Tavily prose → compact structured JSON of pedagogical facts.
        Chunks the raw text and processes each chunk sequentially (not concurrently)
        to stay within Groq's RPM limit on free tier.
        """
        model = extractor_model or settings.small_task_model

        system = """You are a curriculum research extractor.
Extract every pedagogically useful fact from raw web search text into compact JSON.

Return STRICT JSON only. No markdown. No explanation.

Output shape:
{
  "all_topics_found": ["exact topic or concept name as written", ...],
  "prerequisite_chains": [{"concept": "X", "requires": ["A", "B"]}, ...],
  "topic_clusters": [{"cluster": "name", "subtopics": ["subtopic1", ...]}, ...],
  "exam_or_syllabus_specifics": ["item from official syllabus or exam pattern", ...],
  "important_subtopics": ["granular subtopic deserving its own module", ...],
  "common_misconceptions": ["misconception a curriculum should address", ...],
  "domain_notes": ["domain-specific fact: law, formula, key event, key date", ...]
}

Rules:
- Preserve exact concept names — never paraphrase topic titles
- Extract EVERY topic name, even if briefly mentioned
- prerequisite_chains only when the source explicitly states ordering
- Extract every item from any syllabus or curriculum list you find
- Do not invent anything not present in the input text
- Empty list [] if a field has nothing relevant"""

        def _merge(chunks_facts: list[dict]) -> dict:
            merged: dict[str, list] = {
                "all_topics_found": [], "prerequisite_chains": [], "topic_clusters": [],
                "exam_or_syllabus_specifics": [], "important_subtopics": [],
                "common_misconceptions": [], "domain_notes": [],
            }
            seen_strings: dict[str, set] = {k: set() for k in merged}
            seen_chains: set = set()
            seen_clusters: set = set()
            for chunk in chunks_facts:
                if not isinstance(chunk, dict):
                    continue
                for key in ("all_topics_found", "exam_or_syllabus_specifics",
                            "important_subtopics", "common_misconceptions", "domain_notes"):
                    for item in (chunk.get(key) or []):
                        val = str(item).strip()
                        if val and val.lower() not in seen_strings[key]:
                            seen_strings[key].add(val.lower())
                            merged[key].append(val)
                for chain in (chunk.get("prerequisite_chains") or []):
                    if not isinstance(chain, dict):
                        continue
                    requires = chain.get("requires") or []
                    if not isinstance(requires, list):
                        requires = []
                    chain["requires"] = requires
                    key_str = str(chain.get("concept", "")) + str(sorted(requires))
                    if key_str not in seen_chains:
                        seen_chains.add(key_str)
                        merged["prerequisite_chains"].append(chain)
                for cluster in (chunk.get("topic_clusters") or []):
                    if not isinstance(cluster, dict):
                        continue
                    key_str = str(cluster.get("cluster", "")).lower()
                    if key_str not in seen_clusters:
                        seen_clusters.add(key_str)
                        merged["topic_clusters"].append(cluster)
            return merged

        async def _extract_chunk(chunk_text: str, idx: int) -> dict:
            prompt = f"Topic: {topic}\nGoal: {goal}\n\n---\n{chunk_text}"
            try:
                raw = await generate(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    system=system,
                    json_mode=True,
                    max_tokens=1500,  # 1000 was too low for dense academic content
                )
                result = parse_json_object(raw)
                total = sum(len(v) for v in result.values() if isinstance(v, list))
                logger.info("Extractor chunk {}: {} facts from {} chars [{}].", idx, total, len(chunk_text), model.split("/")[-1][:20])
                return result
            except Exception as exc:
                logger.warning("Extractor chunk {} failed: {}. Skipping.", idx, exc)
                return {}

        chunks = [raw_text[i: i + self._EXTRACTOR_CHUNK_CHARS]
                  for i in range(0, len(raw_text), self._EXTRACTOR_CHUNK_CHARS)]
        logger.info("Research extractor: {} chars → {} chunk(s) for '{}' [sequential].",
                    len(raw_text), len(chunks), topic)

        # Sequential — prevents RPM rate limit pile-up
        chunk_results = []
        for i, chunk_text in enumerate(chunks):
            result = await _extract_chunk(chunk_text, i + 1)
            chunk_results.append(result)

        merged = _merge(chunk_results)
        total_facts = sum(len(v) for v in merged.values() if isinstance(v, list))
        if total_facts < 3:
            logger.warning("Extractor sparse ({} facts) — planner will rely on subject knowledge.", total_facts)
        logger.info("Research extractor complete: {} facts merged from {} chunks.", total_facts, len(chunks))
        return json.dumps(merged, indent=2)

    async def _run_research(self, queries: list[ResearchQuery], topic: str, goal: str = "",
                             extractor_model: str | None = None) -> ResearchSummary:
        raw_results: dict[str, str] = {}
        queries_run: list[ResearchQuery] = []

        async def run_query(q: ResearchQuery) -> tuple[ResearchQuery, str]:
            try:
                results = await asyncio.to_thread(tavily_search, q.query, 5)
                summaries = []
                for r in (results or [])[:5]:
                    title = r.get("title", "")
                    content = r.get("content", "").strip()
                    if content:
                        summaries.append(f"[{title}]\n{content}")
                self._tavily_cache[q.query] = summaries
                return q, "\n\n".join(summaries) or "No results found."
            except Exception as exc:
                logger.warning("Research query failed '{}': {}.", q.query, exc)
                return q, "Research unavailable. Use domain knowledge."

        async def run_priority(p: int) -> None:
            selected = [q for q in queries if q.priority == p]
            if not selected:
                return
            for q, result in await asyncio.gather(*(run_query(q) for q in selected)):
                queries_run.append(q)
                raw_results[q.query] = result

        await run_priority(1)
        p1 = [q for q in queries if q.priority == 1]
        returned = [q for q in p1 if raw_results.get(q.query, "").strip()
                    and not raw_results.get(q.query, "").startswith("Research unavailable")]
        coverage = len(returned) / max(1, len(p1)) if p1 else 0.5
        if coverage < 0.5:
            await run_priority(2)

        grouped: dict[str, list[str]] = {}
        q_by_text = {q.query: q for q in queries_run}
        for text, result in raw_results.items():
            cat = q_by_text[text].category if text in q_by_text else "general_roadmap"
            grouped.setdefault(cat, []).append(result)
        summary_by_category = {cat: "\n\n".join(items) for cat, items in grouped.items()}
        all_raw = "\n\n".join(summary_by_category.values())

        structured_facts = await self._extract_research_facts(
            raw_text=all_raw, topic=topic, goal=goal, extractor_model=extractor_model
        )
        return ResearchSummary(
            queries_run=queries_run,
            raw_results=raw_results,
            summary_by_category=summary_by_category,
            coverage_confidence=coverage,
            full_text=structured_facts,
        )

    def _budget_research_for_planner(self, full_text: str) -> str:
        """Fit research facts into the planner's safe token budget, preserving all topic names."""
        budget = self._PLANNER_RESEARCH_CHAR_BUDGET
        if len(full_text) <= budget:
            return full_text
        try:
            facts = json.loads(full_text)
        except Exception:
            return full_text[:budget]

        # Priority: exam specifics and topic lists first — these are what feed coverage
        priority_keys = ["exam_or_syllabus_specifics", "topic_clusters", "all_topics_found",
                         "prerequisite_chains", "important_subtopics", "common_misconceptions", "domain_notes"]
        result: dict = {}
        chars_used = 4
        for key in priority_keys:
            items = facts.get(key) or []
            if not items:
                result[key] = []
                continue
            kept = []
            for item in items:
                cost = len(json.dumps(item)) + 4
                if chars_used + cost > budget:
                    break
                kept.append(item)
                chars_used += cost
            result[key] = kept
            if chars_used >= budget:
                for remaining in priority_keys:
                    if remaining not in result:
                        result[remaining] = []
                break

        budgeted = json.dumps(result, indent=2)
        logger.info("Research budgeted: {} → {} chars ({} topics, {} syllabus items).",
                    len(full_text), len(budgeted),
                    len(result.get("all_topics_found", [])),
                    len(result.get("exam_or_syllabus_specifics", [])))
        return budgeted

    # ─────────────────────────────────────────────────────────────────────────
    # CALL A — Coverage Planner: what concepts must be covered
    # ─────────────────────────────────────────────────────────────────────────

    async def _plan_coverage(self, topic: str, profile: dict, research_summary: ResearchSummary) -> list[dict]:
        """
        First of two LLM calls. Produces a flat list of all concepts the curriculum
        must cover — without worrying about ordering, prerequisites, or modules.

        Separation of concerns: this call maximises completeness.
        The sequencer (Call B) handles ordering.
        """
        intent = profile.get("current_intent") if isinstance(profile.get("current_intent"), dict) else {}
        subject = str(profile.get("exact_subject") or intent.get("exact_subject") or topic or "").strip()
        goal = str(profile.get("learning_goal") or intent.get("goal") or self.state.goal or "learn").strip()
        target = str(profile.get("target_context") or intent.get("target_context") or "").strip()
        level = str(profile.get("learner_level") or intent.get("learner_level") or "beginner").strip()
        do_not_include = list(profile.get("do_not_include") or [])
        must_include = list(profile.get("must_include") or [])
        known_concepts = list(profile.get("known_concepts") or profile.get("assumed_known_concepts") or [])

        system = """You are EduMind's curriculum coverage planner.
Your ONLY job: produce a GRANULAR flat list of every individual concept a student must learn.

Do NOT think about ordering. Do NOT group concepts into modules yet.
Think at the level of individual teachable units — one concept per entry.

Return STRICT JSON only. No markdown.

{
  "concepts": [
    {
      "name": "exact concept name — one specific teachable unit",
      "cluster": "thematic group",
      "importance": "essential | important | supplementary",
      "why_needed": "one sentence"
    }
  ],
  "coverage_rationale": "brief explanation",
  "total_concepts": 0
}

CRITICAL RULES:
1. GRANULARITY IS MANDATORY. Each entry must be ONE teachable concept — not a topic cluster.
   BAD: "OOP" (too broad — this is a cluster, not a concept)
   BAD: "Data Structures" (too broad)
   GOOD: "Classes and Objects", "Inheritance", "Polymorphism", "Encapsulation",
         "Dunder Methods", "Class Methods", "Static Methods", "Properties"
   GOOD: "Lists", "Tuples", "Dictionaries", "Sets" (each is its own concept)

2. do_not_include is ABSOLUTE. If a concept appears in that list, or is a sub-topic
   of anything in that list, exclude it entirely. No exceptions.
   Example: if "machine learning" is excluded, then NumPy, Pandas, Matplotlib,
   scikit-learn, deep learning, neural networks are ALL excluded.
   Example: if "async" is excluded, then asyncio, concurrent.futures are excluded.

3. Use your full subject knowledge. Be exhaustive for the stated goal.
   The student is counting on you not to miss anything they need.

4. For known_concepts — include them only if they are strict prerequisites for later concepts.

GRANULARITY EXAMPLES BY SUBJECT:

Python programming (pure/beginner to developer) — each of these is a SEPARATE entry:
  Variables and Assignment, Integer Type, Float Type, String Type, Boolean Type, None Type,
  Arithmetic Operators, Comparison Operators, Logical Operators, Assignment Operators,
  Bitwise Operators, String Indexing and Slicing, String Methods, f-Strings and Formatting,
  print() and input(), if Statement, elif and else, Nested Conditionals,
  for Loops, while Loops, break and continue, range(),
  Functions: Definition and Calling, Function Parameters and Arguments, Return Values,
  Default and Keyword Arguments, *args and **kwargs, Scope and Namespaces,
  Lists: Creation and Indexing, List Methods, Tuples, Dictionaries, Dictionary Methods,
  Sets, List Comprehensions, Dictionary Comprehensions, Set Comprehensions,
  Lambda Functions, map() filter() reduce(), Exception Handling: try/except/finally,
  Raising Exceptions, Custom Exceptions, File Reading, File Writing, Context Managers,
  Modules: import and from-import, Creating Modules, Standard Library Overview,
  Classes and Objects, __init__ and Instance Variables, Instance Methods,
  Inheritance and super(), Method Overriding, Polymorphism, Encapsulation,
  Class Methods and Static Methods, Properties and Getters/Setters,
  Dunder/Magic Methods, Abstract Classes, Iterators and Generators,
  Decorators, Testing with unittest, Debugging Techniques, Virtual Environments and pip,
  Type Hints, Practical Project

Physics thermodynamics — each separate:
  Temperature and Measurement, Thermal Expansion, Heat Transfer Mechanisms,
  Zeroth Law and Thermal Equilibrium, Internal Energy, Heat vs Work,
  First Law of Thermodynamics, Specific Heat Capacity, Latent Heat,
  Isothermal Processes, Adiabatic Processes, Isobaric Processes, Isochoric Processes,
  PV Diagrams, Second Law: Entropy, Carnot Cycle, Heat Engines Efficiency,
  Refrigerators and Heat Pumps, Kinetic Theory of Gases, Ideal Gas Law,
  Maxwell-Boltzmann Distribution, Real Gases and Deviations, Third Law

Apply this SAME level of granularity to any subject."""

        budgeted_research = self._budget_research_for_planner(research_summary.full_text)
        payload = {
            "subject": subject,
            "goal": goal,
            "target_context": target,
            "learner_level": level,
            "must_include": must_include,
            "do_not_include": do_not_include,
            "known_concepts_already_mastered": known_concepts,
            "research_findings": budgeted_research,
            "instruction": "List every concept this student needs to learn. Be thorough. Do not order them yet.",
        }

        raw = await generate(
            messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
            model=settings.reasoning_model,  # gpt-oss-120b — strongest model for coverage decisions
            system=system,
            json_mode=True,
            max_tokens=4000,  # 4K output: 60 concepts * ~50 chars each = ~750 tokens, plenty of room
        )
        data = parse_json_object(raw)
        concepts = data.get("concepts") or []
        if not isinstance(concepts, list) or not concepts:
            raise ValueError("Coverage planner returned empty concept list.")
        logger.info("Coverage planner: {} concepts for '{}' (rationale: {}).",
                    len(concepts), subject, str(data.get("coverage_rationale", ""))[:80])
        return concepts

    # ─────────────────────────────────────────────────────────────────────────
    # CALL B — Sequencer: ordered module list from concept list
    # ─────────────────────────────────────────────────────────────────────────

    async def _sequence_modules(self, topic: str, profile: dict,
                                  concepts: list[dict], research_summary: ResearchSummary,
                                  repair_feedback: str = "") -> tuple[list[dict], str, float]:
        """
        Second of two LLM calls. Takes the flat concept list from Call A and
        produces the ordered, structured module list.

        Input is compact (concept list, not research prose) — token budget is safe.
        """
        intent = profile.get("current_intent") if isinstance(profile.get("current_intent"), dict) else {}
        subject = str(profile.get("exact_subject") or intent.get("exact_subject") or topic or "").strip()
        goal = str(profile.get("learning_goal") or intent.get("goal") or self.state.goal or "learn").strip()
        target = str(profile.get("target_context") or intent.get("target_context") or "").strip()
        level = str(profile.get("learner_level") or intent.get("learner_level") or "beginner").strip()
        pace = str(profile.get("pace") or self.state.pace or "medium").strip()

        # Token budget: gpt-oss-120b 8K limit
        # Input: system(~600t) + concepts(~1500t) = ~2100t
        # Output: 30 modules * ~50 tokens = ~1500t. Max 6K for safety with large curricula.
        system = """You are EduMind's curriculum sequencer. You receive a flat concept list.
Your job: arrange ALL concepts into an ordered module list where each module has a TIGHT FOCUS.

Return STRICT JSON only. No markdown.

{
  "modules": [
    {
      "id": "m1",
      "title": "clear descriptive title",
      "concept": "primary concept name",
      "concepts_taught": ["concept1", "concept2"],
      "prerequisites": ["concept name from earlier module only"],
      "estimated_minutes": 30,
      "depth_level": "surface | standard | deep",
      "why_now": "one sentence",
      "roadmap_step_id": "step_01"
    }
  ],
  "rationale": "brief explanation",
  "confidence": 0.0
}

RULES — READ ALL CAREFULLY:

1. NEVER DROP CONCEPTS. Every concept in the input list must appear in concepts_taught
   of exactly one module. Count them before submitting. confidence=0.0 if any are missing.

2. MODULE FOCUS — THIS IS THE MOST IMPORTANT RULE:
   Each module must have a TIGHT, COHERENT focus. Ask: "Can I name this module with
   one specific concept?" If not, it is too broad — split it.

   CORRECT grouping (same mechanism, same lesson):
   - "for loops" + "while loops" + "break/continue" → one module (all loop mechanics)
   - "String Indexing" + "String Slicing" → one module (both are string access patterns)
   - "List Methods" + "List Comprehensions" → one module (both are list operations)

   WRONG grouping (different concept families):
   - Lists + Tuples + Dictionaries + Sets → WRONG, each is its own module
   - Classes + Inheritance + Polymorphism → WRONG, each is its own module
   - File Reading + File Writing → can share one module (same mechanism)
   - Exception handling + Custom Exceptions + finally → one module (same mechanism)

3. DATA STRUCTURES: every container type gets its OWN module. Never bundle them.
   Lists → own module. Tuples → own module. Dictionaries → own module. Sets → own module.
   Comprehensions can share a module since they're syntactic variations of the same pattern.

4. OOP: every major OOP concept gets its OWN module. Never bundle them.
   Classes and Objects → own module. Inheritance → own module.
   Polymorphism → own module. Encapsulation → own module.
   Dunder Methods → own module. Class/Static Methods → own module.

5. ORDER: strict prerequisite order. No concept before its dependencies.

6. depth_level: "surface"=introductory, "standard"=core concept, "deep"=advanced.
   Never based on pace — based on concept complexity.

7. pace does NOT affect module count. Never bundle concepts because pace is fast or medium.

8. prerequisites[]: concept names from earlier modules only, never module IDs."""

        # Compact payload — concept list + slim profile, no research prose
        payload = {
            "subject": subject,
            "goal": goal,
            "target_context": target,
            "learner_level": level,
            "pace": pace,
            "concept_list_to_sequence": concepts,
            "instruction": "Arrange ALL concepts into ordered modules. Do not drop any concept.",
        }
        if repair_feedback:
            payload["repair_feedback"] = repair_feedback

        raw = await generate(
            messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
            model=settings.reasoning_model,  # gpt-oss-120b — strongest reasoning for prerequisite ordering
            system=system,
            json_mode=True,
            max_tokens=5500,  # 54-65 concepts → 20-30 modules → ~3000t; input~2400+output~5500=7900 < 8K
        )
        data = parse_json_object(raw)
        modules = data.get("modules") or []
        if not isinstance(modules, list) or not modules:
            raise ValueError("Sequencer returned empty module list.")
        confidence = float(data.get("confidence") or 0.0)
        rationale = str(data.get("rationale") or "")
        logger.info("Sequencer: {} concepts → {} modules (confidence={:.2f}).",
                    len(concepts), len(modules), confidence)
        return modules, rationale, confidence

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5: LLM Auditor — subject-knowledge cross-check (not self-referential)
    # ─────────────────────────────────────────────────────────────────────────

    async def _audit_coverage(self, topic: str, profile: dict,
                               modules: list[dict], concepts: list[dict],
                               research_summary: ResearchSummary) -> tuple[list[str], list[str]]:
        """
        The auditor checks coverage against its OWN SUBJECT KNOWLEDGE — not against
        the module list. This is the key difference from the old self-review.

        Returns (missing_concepts, search_queries_needed).
        """
        intent = profile.get("current_intent") if isinstance(profile.get("current_intent"), dict) else {}
        subject = str(profile.get("exact_subject") or intent.get("exact_subject") or topic or "").strip()
        goal = str(profile.get("learning_goal") or intent.get("goal") or self.state.goal or "learn").strip()
        level = str(profile.get("learner_level") or intent.get("learner_level") or "beginner").strip()
        pace = str(profile.get("pace") or self.state.pace or "medium").strip()

        system = """You are EduMind's curriculum auditor.

Your job: find concepts that are MISSING from the module list for this subject and goal.

Process (do this mentally, do not output the intermediate steps):
1. From your knowledge of this subject, think about what every concept a student needs.
2. Check each against the module list.
3. Report ONLY the missing ones.

Return STRICT JSON only. Be concise — no prose outside JSON values.
{
  "concepts_missing_from_modules": ["missing concept 1", "missing concept 2", ...],
  "search_queries_for_uncertain_areas": ["query only if genuinely uncertain", ...],
  "coverage_verdict": "complete | minor_gaps | major_gaps",
  "verdict_reason": "one sentence"
}

Rules:
- Only report concepts that are GENUINELY missing — not present in any module.
- Do NOT report concepts that are in do_not_include — those are intentionally excluded.
- Keep missing list concise: max 15 items. Prioritise the most impactful gaps.
- search_queries: max 1, only if you genuinely cannot determine if a niche topic belongs.
- If complete, return empty missing list and verdict "complete"."""

        # Compress module list to titles only for the audit
        module_titles = [{"id": m.get("id"), "title": m.get("title"), "concept": m.get("concept"),
                          "concepts_taught": m.get("concepts_taught", [])[:4]}
                         for m in modules]

        payload = {
            "subject": subject,
            "goal": goal,
            "learner_level": level,
            "pace": pace,
            "current_module_list": module_titles,
        }

        try:
            raw = await generate(
                messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
                # 70B is 3-4x faster inference than 120B. Auditor only needs to list
                # missing concepts — no complex reasoning required. Speed > power here.
                model="llama-3.3-70b-versatile",
                system=system,
                json_mode=True,
                max_tokens=3000,
            )
            audit = parse_json_object(raw)
            missing = audit.get("concepts_missing_from_modules") or []
            search_queries = audit.get("search_queries_for_uncertain_areas") or []
            verdict = audit.get("coverage_verdict", "unknown")
            logger.info("Coverage audit: verdict={}, {} missing concepts, {} search queries.",
                        verdict, len(missing), len(search_queries))
            return [str(m).strip() for m in missing if str(m).strip()], [str(q).strip() for q in search_queries if str(q).strip()]
        except Exception as exc:
            logger.warning("Coverage audit failed: {}. Proceeding with current modules.", exc)
            return [], []

    # ─────────────────────────────────────────────────────────────────────────
    # Step 6: Gap fill — targeted search + re-sequence missing concepts
    # ─────────────────────────────────────────────────────────────────────────

    async def _fill_gaps(self, topic: str, profile: dict, modules: list[dict],
                          concepts: list[dict], missing_concepts: list[str],
                          search_queries: list[str], research_summary: ResearchSummary,
                          extractor_model: str) -> tuple[list[dict], list[dict]]:
        """
        Add missing concepts to the concept list and re-sequence.
        If search queries were requested by the auditor, run them first.
        """
        # Run any requested targeted searches
        updated_research = research_summary
        for query in search_queries[:2]:
            try:
                results = await asyncio.to_thread(tavily_search, query, 3)
                extra_text = "\n".join(
                    f"{r.get('title','')}: {r.get('content','')[:500]}"
                    for r in results[:3] if r.get("content")
                )
                if extra_text:
                    new_facts_json = await self._extract_research_facts(
                        raw_text=extra_text, topic=topic,
                        goal=str(profile.get("learning_goal") or self.state.goal or ""),
                        extractor_model=extractor_model,
                    )
                    # Merge new facts into existing JSON
                    try:
                        existing = json.loads(updated_research.full_text) if updated_research.full_text.strip().startswith("{") else {}
                        new_facts = json.loads(new_facts_json) if new_facts_json.strip().startswith("{") else {}
                        list_keys = ["all_topics_found", "prerequisite_chains", "topic_clusters",
                                     "exam_or_syllabus_specifics", "important_subtopics",
                                     "common_misconceptions", "domain_notes"]
                        for k in list_keys:
                            existing.setdefault(k, [])
                            existing_strs = {str(x).lower() for x in existing[k]}
                            for item in (new_facts.get(k) or []):
                                if str(item).lower() not in existing_strs:
                                    existing[k].append(item)
                                    existing_strs.add(str(item).lower())
                        merged_full = json.dumps(existing, indent=2)
                    except Exception:
                        merged_full = updated_research.full_text
                    updated_research = ResearchSummary(
                        queries_run=updated_research.queries_run,
                        raw_results={**updated_research.raw_results, query: extra_text},
                        summary_by_category=updated_research.summary_by_category,
                        coverage_confidence=updated_research.coverage_confidence,
                        full_text=merged_full,
                    )
                    logger.info("Gap fill search '{}': facts merged.", query[:60])
            except Exception as exc:
                logger.warning("Gap fill search failed: {}.", exc)

        if not missing_concepts:
            return modules, concepts

        # Add missing concepts to the concept list
        existing_concept_names = {str(c.get("name") or "").lower() for c in concepts}
        new_concepts = list(concepts)
        for mc in missing_concepts:
            mc_lower = mc.lower()
            if mc_lower not in existing_concept_names:
                new_concepts.append({"name": mc, "cluster": "additional", "importance": "important",
                                     "why_needed": "Identified as missing by coverage auditor."})
                existing_concept_names.add(mc_lower)
                logger.info("Gap fill: adding missing concept '{}'.", mc)

        if len(new_concepts) == len(concepts):
            # No new concepts added — missing_concepts were already present somehow
            return modules, concepts

        # Full clean re-sequence from scratch with augmented concept list.
        # No repair_feedback — that caused the model to try merging into existing
        # structure instead of building fresh. A clean sequence is always better.
        try:
            new_modules, _, new_conf = await self._sequence_modules(
                topic=topic, profile=profile,
                concepts=new_concepts, research_summary=updated_research,
            )
            # Accept if:
            # - new has MORE modules (expansion after gap fill is the goal)
            # - OR new has equal/similar count but confidence is high
            # - Reject only if catastrophically fewer modules AND low confidence
            is_expansion = len(new_modules) > len(modules)
            is_collapse = len(new_modules) < max(3, len(modules) * 0.6) and new_conf < 0.5
            if new_modules and not is_collapse:
                logger.info("Gap fill re-sequence: {} → {} modules (conf={:.2f}).",
                            len(modules), len(new_modules), new_conf)
                return new_modules, new_concepts
            elif is_collapse:
                logger.warning("Gap fill re-sequence collapsed to {} modules (conf={:.2f}) — keeping original {}.",
                               len(new_modules), new_conf, len(modules))
        except Exception as exc:
            logger.warning("Gap fill re-sequence failed: {}. Keeping current modules.", exc)

        return modules, new_concepts

    # ─────────────────────────────────────────────────────────────────────────
    # Step 7: Python-only structural fixes
    # ─────────────────────────────────────────────────────────────────────────

    def _structural_fix(self, modules: list[dict]) -> list[dict]:
        seen_ids: set[str] = set()
        result: list[dict] = []
        step_counter = 1
        prev_concept = ""
        for idx, m in enumerate(modules, start=1):
            if not isinstance(m, dict):
                continue
            title = str(m.get("title") or "").strip()
            concept = str(m.get("concept") or title or "").strip()
            if not title and not concept:
                continue
            mid = f"m{idx}"
            while mid in seen_ids:
                idx += 1
                mid = f"m{idx}"
            seen_ids.add(mid)
            m = dict(m)
            m["id"] = mid
            if not str(m.get("roadmap_step_id") or "").strip():
                cur = str(m.get("concept") or m.get("title") or "").strip()
                if cur == prev_concept:
                    m["roadmap_step_id"] = f"step_{step_counter:02d}"
                else:
                    step_counter += 1
                    m["roadmap_step_id"] = f"step_{step_counter:02d}"
                    prev_concept = cur
            else:
                prev_concept = str(m.get("concept") or "")
            for key in ("concepts_taught", "must_teach", "prerequisites", "question_scope",
                        "lesson_requirements", "practice_requirements", "examples_to_include",
                        "what_this_module_will_not_cover"):
                val = m.get(key)
                if not isinstance(val, list):
                    m[key] = [str(val).strip()] if val else []
                else:
                    m[key] = [str(v).strip() for v in val if str(v).strip()]
            if not concept:
                concept = title
            m["concept"] = concept
            if not m["concepts_taught"]:
                m["concepts_taught"] = [concept]
            if not m["must_teach"]:
                m["must_teach"] = list(m["concepts_taught"])
            m["question_scope"] = [v for v in m["question_scope"]
                                    if not is_question_like_scope_text(v)] or list(m["concepts_taught"])
            result.append(m)
        return result

    def _deduplicate_concepts(self, modules: list[dict]) -> list[dict]:
        seen: dict[str, int] = {}
        result: list[dict] = []
        for m in modules:
            key = re.sub(r"\s+", " ", str(m.get("concept") or m.get("title") or "").lower().strip())
            if key and key in seen:
                logger.info("Deduplicating repeated concept: '{}'.", key)
                continue
            if key:
                seen[key] = 1
            result.append(m)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Build typed Module objects
    # ─────────────────────────────────────────────────────────────────────────

    def _safe_int(self, value: Any, fallback: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            m = re.search(r"\d+", str(value or ""))
            return int(m.group(0)) if m else fallback

    def _build_module_objects(self, raw_modules: list[dict], depth_level: str) -> list[Module]:
        pace_ranges = {"fast": (10, 25), "medium": (20, 40), "deep": (30, 60)}
        min_min, max_min = pace_ranges.get(self.state.pace, (20, 40))
        result: list[Module] = []
        for idx, m in enumerate(raw_modules, start=1):
            concept = str(m.get("concept") or m.get("title") or f"module {idx}").strip()
            concepts_taught = m.get("concepts_taught") or [concept]
            must_teach = m.get("must_teach") or concepts_taught
            prereqs = [str(p).strip() for p in (m.get("prerequisites") or [])
                       if str(p).strip() and not re.fullmatch(r"(m|module)[\s_-]*\d+[a-z]?", str(p).strip().lower())]
            dl = m.get("depth_level")
            if dl not in ("surface", "standard", "deep"):
                dl = depth_level
            estimated = max(min_min, min(max_min, self._safe_int(m.get("estimated_minutes"), min_min)))
            # Derive fields not present in lean sequencer schema
            title = str(m.get("title") or concept).strip()
            why_now = str(m.get("why_now") or "").strip()
            # goal_alignment and purpose derived from concept if not provided
            goal_alignment = str(m.get("goal_alignment") or m.get("why_this_module_exists") or "").strip()
            purpose = str(m.get("why_this_module_exists") or m.get("module_goal") or why_now or "").strip()
            domain_framing = str(m.get("domain_framing") or purpose or concept).strip()

            result.append(Module(
                id=str(m.get("id") or f"m{idx}"),
                title=title,
                concept=concept,
                domain_framing=domain_framing,
                prerequisites=prereqs,
                estimated_minutes=estimated,
                depth_level=dl,
                purpose=purpose,
                why_it_matters_for_goal=goal_alignment,
                difficulty="introductory" if dl == "surface" else dl,
                must_teach=must_teach,
                examples_to_include=m.get("examples_to_include") or [],
                practice_type="guided practice" if dl != "deep" else "deep practice",
                concepts_taught=concepts_taught,
                depends_on_concepts=prereqs,
                unlocks_concepts=m.get("unlocks_concepts") or [],
                module_goal=str(m.get("module_goal") or purpose or "").strip(),
                why_now=why_now,
                what_this_module_will_not_cover=m.get("what_this_module_will_not_cover") or [],
                lesson_requirements=m.get("lesson_requirements") or [],
                practice_requirements=m.get("practice_requirements") or [],
                question_scope=m.get("question_scope") or concepts_taught,
                roadmap_step_id=str(m.get("roadmap_step_id") or f"step_{idx:02d}").strip(),
            ))
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_master_roadmap(self, modules: list[Module], topic: str, profile: dict,
                               rationale: str, research_summary: ResearchSummary) -> MasterRoadmap:
        steps_map: dict[str, list[Module]] = {}
        step_order: list[str] = []
        for m in modules:
            sid = m.roadmap_step_id or f"step_{m.id}"
            if sid not in steps_map:
                steps_map[sid] = []
                step_order.append(sid)
            steps_map[sid].append(m)
        steps: list[RoadmapStep] = []
        for idx, sid in enumerate(step_order, start=1):
            step_modules = steps_map[sid]
            first = step_modules[0]
            all_concepts: list[str] = []
            for sm in step_modules:
                all_concepts.extend(sm.concepts_taught or [sm.concept])
            steps.append(RoadmapStep(
                step_id=f"step_{idx:02d}",
                title=first.title,
                concept_cluster=first.concept,
                subtopics=list(dict.fromkeys(all_concepts)),
                prerequisites=list(first.prerequisites or []),
                estimated_minutes=sum(sm.estimated_minutes for sm in step_modules),
                why_this_step_exists=first.purpose or first.why_it_matters_for_goal or "",
                goal_alignment=first.why_it_matters_for_goal or "",
                depth_level=first.depth_level,
                module_generation_hint=f"{len(step_modules)} module(s) for this step.",
                must_teach=list(dict.fromkeys(all_concepts)),
                question_scope=list(dict.fromkeys(c for sm in step_modules for c in (sm.question_scope or []))),
                check_question_targets=list(dict.fromkeys(c for sm in step_modules for c in (sm.question_scope or []))),
                learning_objective=first.module_goal or first.purpose or "",
            ))
        step_id_remap = {old: f"step_{idx:02d}" for idx, old in enumerate(step_order, start=1)}
        for m in modules:
            old_sid = m.roadmap_step_id
            if old_sid in step_id_remap:
                try:
                    m.roadmap_step_id = step_id_remap[old_sid]
                except Exception:
                    pass
        return MasterRoadmap(
            topic=topic,
            goal=str(profile.get("learning_goal") or self.state.goal or ""),
            steps=steps,
            total_estimated_minutes=sum(m.estimated_minutes for m in modules),
            rationale=rationale,
            created_at=datetime.now().isoformat(),
            research_summary=research_summary.model_dump(),
        )

    def _build_scope_from_modules(self, modules: list[Module], topic: str, profile: dict) -> CourseScopeAnalysis:
        intent = profile.get("current_intent") if isinstance(profile.get("current_intent"), dict) else {}
        pace = str(profile.get("pace") or self.state.pace or "medium")
        level = str(profile.get("learner_level") or intent.get("learner_level") or "beginner")
        goal = str(profile.get("learning_goal") or intent.get("goal") or self.state.goal or "")
        total_minutes = sum(m.estimated_minutes for m in modules)
        count = len(modules)
        return CourseScopeAnalysis(
            requested_subject=topic, actual_course_focus=topic, learner_level=level,
            target_outcome=goal,
            depth={"fast": "surface", "medium": "standard", "deep": "deep"}.get(pace, "standard"),
            pace=pace,
            topic_breadth="broad" if count > 20 else ("medium" if count > 10 else "narrow"),
            course_type="mixed", topic_type="mixed", learner_goal_type="conceptual",
            estimated_total_learning_time=f"{total_minutes // 60}h {total_minutes % 60}m",
            recommended_module_count=count, initial_recommended_module_count=count,
            rough_scope_recommendation=count, final_module_count_target=count,
            module_count_reasoning=f"LLM determined {count} modules for '{topic}' at {pace} pace.",
            reason_for_module_count=f"Coverage+sequencing two-call architecture produced {count} modules.",
            recommended_granularity="One coherent concept per module.",
            roadmap_strategy="Coverage-first, then prerequisite sequencing.",
            coverage_strategy="Coverage planner + auditor cross-check against subject knowledge.",
            what_to_include=[m.concept for m in modules],
            what_to_exclude=list(profile.get("do_not_include") or []),
            what_to_delay_until_later=[],
            what_to_skip_because_student_already_knows=list(profile.get("known_concepts") or []),
            estimated_total_hours=round(total_minutes / 60, 2),
        )

    def _compress_mastered(self, modules: list[Module]) -> list[Module]:
        if not modules:
            return modules
        trusted = {str(c).strip().lower() for c in getattr(self, "_trusted_assumed_known", []) if str(c).strip()}
        if not trusted:
            return modules
        threshold = self.state.advance_threshold
        kept, skipped = [], []
        for m in modules:
            if m.concept.strip().lower() not in trusted:
                kept.append(m)
                continue
            if self.state.get_mastery(m.concept) >= threshold:
                skipped.append(m.concept)
            else:
                kept.append(m)
        if skipped and kept:
            logger.info("Compressed {} mastered module(s): {}.", len(skipped), ", ".join(skipped))
        return kept or modules

    async def _llm_repair(self, modules: list[dict], issues: list[str],
                           topic: str, profile: dict,
                           research_summary: ResearchSummary) -> list[dict]:
        intent = profile.get("current_intent") if isinstance(profile.get("current_intent"), dict) else {}
        goal = str(profile.get("learning_goal") or intent.get("goal") or self.state.goal or "").strip()
        system = """You are EduMind's curriculum repair agent.
Fix ONLY the listed issues. Preserve all valid modules.
Return STRICT JSON only: {"modules": [...], "repair_rationale": "..."}"""
        modules_for_repair = [
            {"id": m.get("id"), "title": m.get("title"), "concept": m.get("concept"),
             "concepts_taught": m.get("concepts_taught", []),
             "prerequisites": m.get("prerequisites", []),
             "roadmap_step_id": m.get("roadmap_step_id")}
            for m in modules[:40]
        ]
        payload = {
            "topic": topic, "goal": goal, "issues_to_fix": issues,
            "current_modules": modules_for_repair,
            "instruction": "Fix the listed issues. Preserve valid modules. Add missing concepts where needed.",
        }
        try:
            raw = await generate(
                messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
                model=settings.reasoning_model, system=system, json_mode=True, max_tokens=5000,
            )
            data = parse_json_object(raw)
            repaired = data.get("modules") or []
            if isinstance(repaired, list) and len(repaired) >= 2:
                logger.info("LLM repair succeeded: {} modules.", len(repaired))
                return repaired
        except (GroqRateLimitError, GroqTimeoutError):
            raise
        except Exception as exc:
            logger.warning("LLM repair failed: {}. Returning original modules.", exc)
        return modules

    # ─────────────────────────────────────────────────────────────────────────
    # Main public method
    # ─────────────────────────────────────────────────────────────────────────

    async def build_curriculum(self, topic: str, profile: dict[str, Any] | None = None) -> CurriculumPlan:
        pace = self.state.pace if self.state.pace in ("fast", "medium", "deep") else "medium"
        depth_level = {"fast": "surface", "medium": "standard", "deep": "deep"}[pace]
        profile = dict(profile or getattr(self, "personalization_profile", {}) or {})
        self.personalization_profile = profile

        trusted_assumed = [str(c).strip() for c in (profile.get("assumed_known_concepts") or profile.get("known_concepts") or []) if str(c).strip()]
        if profile_has_no_prior_experience(profile):
            trusted_assumed = [c for c in trusted_assumed if c in profile.get("assumed_known_concepts", [])]
        self._trusted_assumed_known = [
            c for c in trusted_assumed
            if not is_unreliable_generated_concept(c) and is_related_to_profile(c, profile)
        ]

        intent = profile.get("current_intent") if isinstance(profile.get("current_intent"), dict) else {}
        self._log_decision(action="ANALYZE_INTENT",
                           reason="Two-call coverage+sequencing architecture.",
                           payload={"intent": intent, "confidence": profile.get("confidence", 0.7)})

        # ── Step 1 & 2: Research ──────────────────────────────────────────────
        queries = self._research_queries(topic, profile)
        goal_str = str(profile.get("learning_goal") or self.state.goal or "")
        extractor_model = self._select_extractor_model(topic, profile)
        research_summary = await self._run_research(queries, topic, goal=goal_str, extractor_model=extractor_model)
        self._research_summary = research_summary
        self._log_decision(action="RUN_CURRICULUM_RESEARCH",
                           reason=f"Research complete ({len(research_summary.queries_run)} queries, extractor={extractor_model.split('/')[-1]}).",
                           payload={"coverage_confidence": research_summary.coverage_confidence,
                                    "full_text_length": len(research_summary.full_text)})

        repair_history: list[dict] = []
        concepts: list[dict] = []
        raw_modules: list[dict] = []
        rationale = ""
        confidence = 0.0

        # ── Step 3: Coverage planning (Call A) ────────────────────────────────
        for attempt in range(2):
            try:
                concepts = await self._plan_coverage(topic, profile, research_summary)
                self._log_decision(action="PLAN_COVERAGE",
                                   reason=f"Coverage planner: {len(concepts)} concepts.",
                                   payload={"concept_count": len(concepts)})
                break
            except (GroqRateLimitError, GroqTimeoutError):
                raise
            except Exception as exc:
                logger.warning("Coverage planning attempt {} failed: {}.", attempt + 1, exc)
                repair_history.append({"stage": "coverage_planning", "attempt": attempt + 1, "error": str(exc)})
                if attempt == 1:
                    # Fallback: minimal concept list from topic name
                    concepts = [{"name": topic, "cluster": "core", "importance": "essential",
                                  "why_needed": f"Core concept of {topic}."}]

        # ── Step 4: Sequencing (Call B) ───────────────────────────────────────
        for attempt in range(2):
            try:
                raw_modules, rationale, confidence = await self._sequence_modules(
                    topic=topic, profile=profile, concepts=concepts, research_summary=research_summary
                )
                self._log_decision(action="SEQUENCE_MODULES",
                                   reason=f"Sequencer: {len(concepts)} concepts → {len(raw_modules)} modules (confidence={confidence:.2f}).",
                                   payload={"module_count": len(raw_modules), "confidence": confidence})
                break
            except (GroqRateLimitError, GroqTimeoutError):
                raise
            except Exception as exc:
                logger.warning("Sequencing attempt {} failed: {}.", attempt + 1, exc)
                repair_history.append({"stage": "sequencing", "attempt": attempt + 1, "error": str(exc)})
                if attempt == 1:
                    scope_fallback = fallback_scope_analysis({**profile, "topic": topic, "pace": pace})
                    raw_modules = [
                        {"id": f"m{i}", "title": c.get("name", topic), "concept": c.get("name", topic),
                         "concepts_taught": [c.get("name", topic)], "must_teach": [c.get("name", topic)],
                         "prerequisites": [], "estimated_minutes": 30, "depth_level": depth_level,
                         "roadmap_step_id": f"step_{i:02d}", "question_scope": [c.get("name", topic)[:40]],
                         "why_this_module_exists": f"Learn {c.get('name', topic)}.",
                         "goal_alignment": f"Supports {topic} mastery.", "domain_framing": c.get("name", topic),
                         "module_goal": f"Understand {c.get('name', topic)}.", "why_now": "Foundation."}
                        for i, c in enumerate(concepts[:20], start=1)
                    ]
                    rationale = f"Fallback after sequencing failed: {exc}"
                    confidence = 0.5

        # ── Step 5: Coverage audit ────────────────────────────────────────────
        if confidence >= 0.5 and raw_modules:
            try:
                missing_concepts, search_queries = await self._audit_coverage(
                    topic=topic, profile=profile, modules=raw_modules,
                    concepts=concepts, research_summary=research_summary
                )
                self._log_decision(action="AUDIT_COVERAGE",
                                   reason=f"Auditor: {len(missing_concepts)} missing concepts found.",
                                   payload={"missing": missing_concepts[:5], "searches": search_queries})

                # ── Step 6: Gap fill ──────────────────────────────────────────
                if missing_concepts:
                    raw_modules, concepts = await self._fill_gaps(
                        topic=topic, profile=profile, modules=raw_modules,
                        concepts=concepts, missing_concepts=missing_concepts,
                        search_queries=search_queries, research_summary=research_summary,
                        extractor_model=extractor_model,
                    )
                    self._log_decision(action="FILL_GAPS",
                                       reason=f"Gap fill complete: {len(raw_modules)} modules after adding {len(missing_concepts)} concepts.",
                                       payload={"final_module_count": len(raw_modules)})
            except (GroqRateLimitError, GroqTimeoutError):
                raise
            except Exception as exc:
                logger.warning("Coverage audit/gap fill failed: {}. Proceeding with current modules.", exc)

        # ── Step 7: Structural fixes ──────────────────────────────────────────
        raw_modules = self._structural_fix(raw_modules)
        raw_modules = self._deduplicate_concepts(raw_modules)

        if len(raw_modules) < 2:
            logger.warning("Module list too short after cleanup; running LLM repair.")
            raw_modules = await self._llm_repair(
                modules=raw_modules,
                issues=["Module list has fewer than 2 modules — please generate a complete curriculum."],
                topic=topic, profile=profile, research_summary=research_summary
            )
            raw_modules = self._structural_fix(raw_modules)

        # ── Build typed Module objects ─────────────────────────────────────────
        modules = self._build_module_objects(raw_modules, depth_level)
        modules = self._compress_mastered(modules)
        for m in modules:
            m.question_scope = [v for v in (m.question_scope or []) if not is_question_like_scope_text(v)] or list(m.concepts_taught)

        scope = self._build_scope_from_modules(modules, topic, profile)
        master_roadmap = self._build_master_roadmap(modules, topic, profile, rationale, research_summary)

        # ── Lightweight structural validation ─────────────────────────────────
        validation_profile = {**profile, "known_concepts": list(dict.fromkeys(
            list(profile.get("known_concepts") or []) + list(self._trusted_assumed_known)
        ))}
        student_history = getattr(self, "student_history_snapshot", {}) or {}
        concept_inventory = {
            "core_concepts": [m.concept for m in modules],
            "prerequisite_concepts": list(dict.fromkeys(p for m in modules for p in m.prerequisites)),
            "optional_concepts": [],
            "advanced_concepts": [m.concept for m in modules if m.depth_level == "deep"],
            "concepts_to_skip": list(scope.what_to_skip_because_student_already_knows),
            "concepts_to_delay": [],
        }
        prerequisite_graph = {m.concept: list(m.prerequisites) for m in modules}
        learning_path = [{"step_id": m.roadmap_step_id, "concept": m.concept,
                          "why_now": m.why_now, "goal_alignment": m.why_it_matters_for_goal}
                         for m in modules]
        roadmap_steps = [(m.title + (": " + m.purpose if m.purpose else "")).strip() for m in modules]

        validation = validate_curriculum_quality(
            topic=topic, modules=modules, profile=validation_profile, scope_analysis=scope,
            student_history=student_history, concept_inventory=concept_inventory,
            prerequisite_graph=prerequisite_graph,
            roadmap_steps=roadmap_steps + [m.title + (": " + m.why_now if m.why_now else "") for m in modules],
            schedule=None,
        )
        repair_history.append({"stage": "post_build_validation", "validation": validation,
                                "module_count": len(modules)})

        if not validation["passed"]:
            issues = [i for i in (validation.get("issues") or [])
                      if "outside 40%" not in i and "concepts_to_delay" not in i.lower()
                      and not i.startswith("Module count")]
            if issues:
                logger.warning("Post-build validation found {} issues; running targeted LLM repair.", len(issues))
                repaired_raw = await self._llm_repair(
                    modules=raw_modules, issues=issues[:10], topic=topic,
                    profile=profile, research_summary=research_summary
                )
                repaired_raw = self._structural_fix(repaired_raw)
                repaired_raw = self._deduplicate_concepts(repaired_raw)
                repaired_modules = self._build_module_objects(repaired_raw, depth_level)
                repaired_modules = self._compress_mastered(repaired_modules)
                for m in repaired_modules:
                    m.question_scope = [v for v in (m.question_scope or []) if not is_question_like_scope_text(v)] or list(m.concepts_taught)
                if len(repaired_modules) >= 2:
                    modules = repaired_modules
                    raw_modules = repaired_raw
                    scope = self._build_scope_from_modules(modules, topic, profile)
                    master_roadmap = self._build_master_roadmap(modules, topic, profile, rationale, research_summary)
                    concept_inventory["core_concepts"] = [m.concept for m in modules]
                    prerequisite_graph = {m.concept: list(m.prerequisites) for m in modules}
                    validation = validate_curriculum_quality(
                        topic=topic, modules=modules, profile=validation_profile, scope_analysis=scope,
                        student_history=student_history, concept_inventory=concept_inventory,
                        prerequisite_graph=prerequisite_graph,
                        roadmap_steps=roadmap_steps + [m.title for m in modules],
                        schedule=None,
                    )
                    repair_history.append({"stage": "llm_repair", "module_count": len(modules), "validation": validation})

        if not validation["passed"]:
            remaining = [i for i in (validation.get("issues") or [])
                         if "outside 40%" not in i and "concepts_to_delay" not in i.lower()]
            if remaining:
                logger.warning("Curriculum has minor validation issues (proceeding): {}",
                                "; ".join(remaining[:5]))
            validation["passed"] = True

        validation["confidence"] = confidence
        self._master_roadmap = master_roadmap
        self._log_decision(action="BUILD_MASTER_ROADMAP",
                           reason=rationale or "Two-call architecture completed.",
                           payload={"module_count": len(modules), "confidence": confidence,
                                    "total_minutes": sum(m.estimated_minutes for m in modules)})

        # ── Persist ───────────────────────────────────────────────────────────
        clean_domain = re.sub(r"[^a-z0-9 _-]", "", (topic or "general").lower().strip())[:35].strip()
        plan = CurriculumPlan(
            topic=str(scope.actual_course_focus or topic),
            domain=clean_domain,
            goal=self.state.goal,
            modules=modules,
            current_index=0,
            version=1,
            scope_analysis=scope.to_dict(),
            concept_inventory=concept_inventory,
            prerequisite_graph={str(k): [str(v) for v in (val if isinstance(val, list) else [])]
                                 for k, val in prerequisite_graph.items()},
            learning_path=learning_path,
            roadmap_steps=roadmap_steps,
            schedule_plan=[],
            repair_history=repair_history,
            validation_result=validation,
        )

        _curriculum_pace = str(getattr(self.state, "pace", None) or "medium")
        async with get_conn() as conn:
            await conn.execute("UPDATE curricula SET is_active=FALSE WHERE student_id=$1", self.state.student_id)
            row = await conn.fetchrow(
                "INSERT INTO curricula (student_id, topic, plan_json, current_index, version, is_active, pace) "
                "VALUES ($1, $2, $3, 0, 1, TRUE, $4) RETURNING id",
                self.state.student_id, plan.topic, plan.model_dump_json(), _curriculum_pace,
            )
            logger.info("Curriculum saved to DB id={} pace={} modules={}",
                        row["id"], _curriculum_pace, len(modules))
            # Expose the id so create_course doesn't need a second DB query.
            self._curriculum_id = int(row["id"])

        self.state.curriculum = plan
        self.state.mark_dirty("curriculum")

        if hasattr(self, "_eval_runner") and self._eval_runner is not None:
            asyncio.create_task(self._eval_runner.on_curriculum_built(
                modules=[{"concept": m.concept, "title": m.title, "prerequisites": m.prerequisites,
                           "concepts_taught": m.concepts_taught, "depends_on_concepts": m.depends_on_concepts,
                           "question_scope": m.question_scope} for m in plan.modules]
            ))

        self._log_decision(action="BUILD_CURRICULUM",
                           reason=f"Curriculum built: {len(modules)} modules, confidence={confidence:.2f}",
                           payload={"topic": plan.topic, "module_count": len(modules), "confidence": confidence})
        logger.info("Curriculum built: {} modules for topic='{}' (confidence={:.2f})",
                    len(modules), plan.topic, confidence)

        # Fire-and-forget: ChromaDB embedding happens in the background.
        # The user gets their roadmap immediately without waiting 40s for embedding.
        async def _embed_background():
            try:
                await self._embed_to_chromadb(plan, int(row["id"]))
            except Exception as exc:
                logger.warning("Background ChromaDB embedding failed for id={}: {}.", row["id"], exc)

        asyncio.create_task(_embed_background())

        return plan

    # ── ChromaDB ──────────────────────────────────────────────────────────────

    async def _embed_to_chromadb(self, plan: CurriculumPlan, curriculum_id: int | None = None) -> None:
        domain = plan.domain
        embedded, failed = 0, 0
        course_id = f"course-{curriculum_id}" if curriculum_id is not None else ""
        for module in plan.modules:
            concept = module.concept
            framing = module.domain_framing
            prereqs = ", ".join(module.prerequisites) if module.prerequisites else "none"
            safe_concept = re.sub(r"[^a-zA-Z0-9_]", "_", concept)[:60]
            lines = [
                f"Course topic: {plan.topic}", f"Course goal: {plan.goal}", f"Domain: {domain}",
                f"Module id: {module.id}", f"Roadmap step id: {module.roadmap_step_id}",
                f"Module title: {module.title}", f"Concept: {concept}",
                f"Concepts taught: {', '.join(module.concepts_taught or [concept])}",
                f"Must teach: {', '.join(module.must_teach or module.concepts_taught or [concept])}",
                f"Prerequisites: {prereqs}", f"Depth level: {module.depth_level}",
                f"Estimated minutes: {module.estimated_minutes}", f"Purpose: {module.purpose}",
                f"Goal alignment: {module.why_it_matters_for_goal}",
                f"Lesson requirements: {', '.join(module.lesson_requirements or [])}",
                f"Practice requirements: {', '.join(module.practice_requirements or [])}",
                f"Question scope: {', '.join(module.question_scope or [])}",
                f"Domain framing: {framing}",
            ]
            full_text = "\n".join(line for line in lines if line.split(": ", 1)[-1].strip())
            chunk_id = (plan.topic + "_" + safe_concept).replace(" ", "_")[:180]
            try:
                await chroma_insert(chunk_id, domain, full_text,
                                    metadata={"course_id": course_id, "curriculum_id": str(curriculum_id or ""),
                                              "topic": plan.topic, "module_id": module.id, "concept": concept})
                if curriculum_id is not None:
                    async with get_conn() as conn:
                        await conn.execute(
                            "INSERT INTO module_embeddings (student_id, curriculum_id, module_id, chromadb_id, domain) "
                            "VALUES ($1, $2, $3, $4, $5) ON CONFLICT (student_id, module_id) DO UPDATE "
                            "SET curriculum_id=$2, chromadb_id=$4, domain=$5",
                            self.state.student_id, curriculum_id, module.id, chunk_id, domain,
                        )
                        await conn.execute(
                            "INSERT INTO dynamic_concept_summaries "
                            "(student_id, curriculum_id, module_id, concept, domain, summary_text, chromadb_id) "
                            "VALUES ($1, $2, $3, $4, $5, $6, $7) "
                            "ON CONFLICT (student_id, module_id) DO UPDATE "
                            "SET curriculum_id=$2, concept=$4, domain=$5, summary_text=$6, chromadb_id=$7, updated_at=NOW()",
                            self.state.student_id, curriculum_id, module.id, concept, domain, full_text, chunk_id,
                        )
                embedded += 1
            except Exception as exc:
                failed += 1
                logger.warning("ChromaDB insert failed for concept='{}': {}.", concept, exc)
        logger.info("ChromaDB: {}/{} modules embedded for domain='{}'.", embedded, len(plan.modules), domain)
