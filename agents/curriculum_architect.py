"""
agents/curriculum_architect.py
CurriculumArchitectAgent — builds roadmap-first CurriculumPlans.

Redesigned Flow (two-call, no Tavily):
  1. CALL A — Coverage Planner: flat concept list driven entirely by the
     reasoning model's subject knowledge + full student profile injection.
  2. CALL B — Sequencer: ordered module list from concept list.
  3. LLM Auditor: cross-checks against subject knowledge, finds gaps.
  4. Gap fill: re-sequence if auditor finds missing concepts.
  5. Python structural fix (IDs, dedup — no semantic work).
  6. Save to DB + ChromaDB.

Why no Tavily:
  - gpt-oss-120b has vastly better subject knowledge than random SEO pages.
  - 45 s sequential extraction was adding latency and then getting truncated
    to 9000 chars anyway, making the whole thing a wasted round-trip.
  - Removing it cuts ~50 s of latency and eliminates the 400-JSON-validation
    failure that was the root cause of 10-module Python roadmaps.

Key principles:
  - Student profile (prior knowledge, weak concepts, target context, pace)
    is injected RICHLY into every LLM call — not just as metadata.
  - Coverage and sequencing are separate LLM calls.
  - No hardcoded module count targets.
  - Never crashes the user for a recoverable issue.
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
    ResearchSummary,
    RoadmapStep,
    StudentState,
)
from db.postgres import get_conn


def _empty_research(topic: str) -> ResearchSummary:
    """Return a no-op ResearchSummary — we no longer run Tavily searches."""
    return ResearchSummary(
        queries_run=[],
        raw_results={},
        summary_by_category={},
        coverage_confidence=1.0,   # full confidence: reasoning model knows the subject
        full_text="{}",
    )


def _build_student_context_block(topic: str, profile: dict) -> str:
    """
    Build a richly-worded natural-language block that describes the student's
    situation.  This is injected into every LLM call so the model personalises
    the curriculum — not just records metadata.

    Example output:
      Subject: Python programming
      Goal: Build web apps with FastAPI
      Target context: Backend web development
      Learner level: Some basic knowledge
      Pace: deep
      Prior knowledge: The student already knows basic HTML and has done one
        JavaScript tutorial but has never used Python.
      Concepts already mastered (skip or treat as prerequisite only):
        - HTML basics, CSS basics
      Concepts the student finds difficult (reinforce and explain carefully):
        - Functions, Scope
      Must include: FastAPI, Pydantic
      Do not include: machine learning, pandas, numpy
    """
    intent = profile.get("current_intent") if isinstance(profile.get("current_intent"), dict) else {}

    subject       = str(profile.get("exact_subject") or intent.get("exact_subject") or topic or "").strip()
    goal          = str(profile.get("learning_goal") or intent.get("goal") or "").strip()
    target        = str(profile.get("target_context") or intent.get("target_context") or "").strip()
    level         = str(profile.get("learner_level") or intent.get("learner_level") or "not specified").strip()
    pace          = str(profile.get("pace") or "medium").strip()
    prior_summary = str(profile.get("prior_knowledge_summary") or profile.get("prior_knowledge") or "").strip()
    prior_exp     = str(profile.get("prior_experience") or "").strip()
    known         = list(profile.get("known_concepts") or profile.get("assumed_known_concepts") or [])
    weak          = list(profile.get("weak_concepts") or [])
    must_inc      = list(profile.get("must_include") or [])
    do_not_inc    = list(profile.get("do_not_include") or [])
    goal_desc     = str(profile.get("goal_description") or intent.get("goal_description") or "").strip()
    time_con      = str(profile.get("time_constraint") or "").strip()
    depth_pref    = str(profile.get("depth_preference") or "").strip()

    lines = [f"Subject: {subject}"]
    if goal:
        lines.append(f"Goal: {goal}")
    if goal_desc and goal_desc.lower() != goal.lower():
        lines.append(f"Goal description: {goal_desc}")
    if target:
        lines.append(f"Target context: {target}")
    lines.append(f"Learner level: {level}")
    lines.append(f"Pace: {pace}" + (f" ({depth_pref})" if depth_pref else ""))
    if time_con:
        lines.append(f"Time constraint: {time_con}")

    # Prior knowledge — combine summary + prior experience into one human paragraph
    pk_parts = []
    if prior_summary:
        pk_parts.append(prior_summary)
    if prior_exp and prior_exp.lower() not in prior_summary.lower():
        pk_parts.append(prior_exp)
    if pk_parts:
        lines.append(f"Prior knowledge: {' '.join(pk_parts)}")
    elif level in ("complete beginner", "beginner"):
        lines.append("Prior knowledge: No prior knowledge of the subject.")

    if known:
        lines.append("Concepts already mastered (skip or treat as prerequisite only):")
        for c in known[:10]:
            lines.append(f"  - {c}")
    if weak:
        lines.append("Concepts the student finds difficult (reinforce and explain carefully):")
        for c in weak[:10]:
            lines.append(f"  - {c}")
    if must_inc:
        lines.append(f"Must include: {', '.join(must_inc)}")
    if do_not_inc:
        lines.append(f"Do NOT include (absolute exclusion): {', '.join(do_not_inc)}")

    return "\n".join(lines)


class CurriculumArchitectAgent(BaseAgent):
    """Builds curriculum coverage, sequencing, roadmap, and validation artifacts."""

    NAME = "curriculum_architect"
    TERMINAL_TOOL = "submit_curriculum"

    def __init__(self, state: StudentState):
        super().__init__(state)

    # ─────────────────────────────────────────────────────────────────────────
    # CALL A — Coverage Planner: what concepts must be covered
    # ─────────────────────────────────────────────────────────────────────────

    async def _plan_coverage(self, topic: str, profile: dict) -> list[dict]:
        """
        Produces a flat list of all concepts the curriculum must cover.
        Drives entirely from the reasoning model's subject knowledge + full
        student profile — no Tavily search needed.

        Separation of concerns: this call maximises completeness.
        The sequencer (Call B) handles ordering.
        """
        student_ctx = _build_student_context_block(topic, profile)

        system = """You are EduMind's curriculum coverage planner.
Your ONLY job: produce a GRANULAR flat list of every individual concept a student must learn,
personalised to the student's context below.

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
  "coverage_rationale": "brief explanation of why this set covers the goal",
  "total_concepts": 0
}

CRITICAL RULES:

1. PERSONALISATION IS MANDATORY.
   - Read the student context carefully. The goal and target context define WHAT version
     of the subject to teach. Two students learning "matrices" for different goals need
     completely different curricula.
   - If known_concepts lists things the student already knows, SKIP those concepts
     unless they are strict prerequisites for what comes next. Do not re-teach them.
   - If weak_concepts lists things the student struggles with, ADD extra foundational
     and bridging concepts around those areas.
   - do_not_include is ABSOLUTE. If a concept appears in that list, or is a subtopic
     of anything in that list, exclude it entirely.

2. GRANULARITY IS MANDATORY. Each entry must be ONE teachable concept.
   BAD: "OOP" (too broad — cluster, not a concept)
   BAD: "Data Structures" (too broad)
   GOOD: "Classes and Objects", "Inheritance", "Polymorphism", "Encapsulation"
   GOOD: "Lists", "Tuples", "Dictionaries", "Sets" (each is its own concept)

3. USE YOUR FULL SUBJECT KNOWLEDGE. Be exhaustive for the stated goal and level.
   The student is counting on you not to miss anything they need.

GRANULARITY EXAMPLES BY SUBJECT:

Python programming (pure/beginner to developer) — each is a SEPARATE entry:
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

        payload = {
            "student_context": student_ctx,
            "instruction": (
                "List every concept this student needs to learn based on their context above. "
                "Be thorough and granular. Skip anything they already know. "
                "Tailor the concept set to their stated goal and target context. "
                "Do not order them yet."
            ),
        }

        raw = await generate(
            messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
            model=settings.reasoning_model,
            system=system,
            json_mode=True,
            max_tokens=4000,
        )
        data = parse_json_object(raw)
        concepts = data.get("concepts") or []
        if not isinstance(concepts, list) or not concepts:
            raise ValueError("Coverage planner returned empty concept list.")
        logger.info(
            "Coverage planner: {} concepts for '{}' (rationale: {}).",
            len(concepts), topic, str(data.get("coverage_rationale", ""))[:80]
        )
        return concepts

    # ─────────────────────────────────────────────────────────────────────────
    # CALL B — Sequencer: ordered module list from concept list
    # ─────────────────────────────────────────────────────────────────────────

    async def _sequence_modules(
        self,
        topic: str,
        profile: dict,
        concepts: list[dict],
        repair_feedback: str = "",
    ) -> tuple[list[dict], str, float]:
        """
        Takes the flat concept list from Call A and produces the ordered,
        structured module list.  Input is compact — token budget is safe.
        """
        student_ctx = _build_student_context_block(topic, profile)
        pace = str(profile.get("pace") or self.state.pace or "medium").strip()

        # Pace-aware grouping rules injected into the system prompt.
        pace_hint = {
            "fast": (
                "FAST PACE — Aggressively merge related small concepts into one module.\n"
                "   - All operator types → one module: 'Python Operators'\n"
                "   - All primitive types → one module: 'Python Data Types'\n"
                "   - String indexing + slicing + methods → one module\n"
                "   - All comprehensions → one module\n"
                "   - All OOP except Classes+Objects can share 1-2 modules\n"
                "   Target: 20-30 modules total for a full language curriculum."
            ),
            "medium": (
                "MEDIUM PACE — Group by natural thematic cluster. Each module = one coherent topic.\n"
                "   - Arithmetic + comparison + logical + assignment operators → ONE 'Operators' module\n"
                "     (they are all operators, learned together, same lesson)\n"
                "   - Membership + identity operators → can join the operators module\n"
                "   - Primitive types (int, float, bool, None) → can be ONE 'Numeric & Boolean Types' module\n"
                "   - String type + string basics → own module; string methods → own module\n"
                "   - if/elif/else → own module; for/while loops → own module\n"
                "   - Each data structure (list, tuple, dict, set) → own module\n"
                "   - Each major OOP concept → own module\n"
                "   Target: 30-45 modules total for a full language curriculum."
            ),
            "deep": (
                "DEEP PACE — One concept per module. Maximum granularity.\n"
                "   - Every operator TYPE gets its own module (arithmetic, comparison, logical, etc.)\n"
                "   - Every data type gets its own module\n"
                "   - Every OOP concept gets its own module\n"
                "   - Err on the side of splitting, never merging\n"
                "   Target: 50-70 modules total for a full language curriculum."
            ),
        }.get(pace, "Group concepts by natural thematic clusters. Each module = one coherent topic.")

        system = f"""You are EduMind's curriculum sequencer. You receive a flat concept list.
Your job: arrange ALL concepts into an ordered module list following the PACE RULES below.

Return STRICT JSON only. No markdown.

{{
  "modules": [
    {{
      "id": "m1",
      "title": "clear descriptive title",
      "concept": "primary concept name",
      "concepts_taught": ["concept1", "concept2"],
      "prerequisites": ["concept name from earlier module only"],
      "estimated_minutes": 30,
      "depth_level": "surface | standard | deep",
      "why_now": "one sentence",
      "roadmap_step_id": "step_01"
    }}
  ],
  "rationale": "brief explanation",
  "confidence": 0.0
}}

━━━ PACE RULES (HIGHEST PRIORITY) ━━━
{pace_hint}

━━━ UNIVERSAL RULES ━━━

1. NEVER DROP CONCEPTS. Every concept in the input list must appear in concepts_taught
   of exactly one module. Count before submitting. Set confidence=0.0 if any are missing.

2. ORDER: strict prerequisite order. No concept appears before its dependencies.
   Setup/install → types → operators → control flow → functions → data structures
   → comprehensions → exceptions → files → modules → OOP → advanced topics → project.

3. depth_level: based on concept complexity relative to student level.
   "surface"=introductory, "standard"=core, "deep"=advanced. Never based on pace.

4. prerequisites[]: concept names from earlier modules only. Never module IDs.

5. PERSONALISATION: If the student has weak concepts, give those modules more
   estimated_minutes and depth_level="deep"."""

        # Strip extra fields before sending — sequencer only needs concept names.
        # Full dicts (cluster, importance, why_needed) add ~1,350 tokens of noise
        # that gpt-oss-120b doesn't need and that caused the 413 token-limit errors.
        slim_concepts = [
            {"name": str(c.get("name") or c)} for c in concepts if c.get("name") or isinstance(c, str)
        ]

        payload = {
            "student_context": student_ctx,
            "concept_list_to_sequence": slim_concepts,
            "instruction": "Arrange ALL concepts into ordered modules. Do not drop any concept.",
        }
        if repair_feedback:
            payload["repair_feedback"] = repair_feedback

        # Use generation_model (llama-4-scout, 30K TPM) for sequencing.
        # Sequencing is an ordering task, not deep reasoning — llama-4-scout
        # handles it well and has 30K TPM vs gpt-oss-120b's 8K limit.
        # gpt-oss-120b is reserved for coverage planning (the "what to teach" decision).
        raw = await generate(
            messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
            model=settings.generation_model,
            system=system,
            json_mode=True,
            max_tokens=6000,
        )
        data = parse_json_object(raw)
        modules = data.get("modules") or []
        if not isinstance(modules, list) or not modules:
            raise ValueError("Sequencer returned empty module list.")
        confidence = float(data.get("confidence") or 0.0)
        rationale = str(data.get("rationale") or "")
        logger.info(
            "Sequencer: {} concepts → {} modules (confidence={:.2f}).",
            len(concepts), len(modules), confidence
        )
        return modules, rationale, confidence

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: LLM Auditor — subject-knowledge cross-check
    # ─────────────────────────────────────────────────────────────────────────

    async def _audit_coverage(
        self,
        topic: str,
        profile: dict,
        modules: list[dict],
        concepts: list[dict],
    ) -> tuple[list[str], list[str]]:
        """
        The auditor checks coverage against its OWN SUBJECT KNOWLEDGE — not
        against the module list.  Returns (missing_concepts, []).

        The 15-item cap from the old design is removed.  If the planner
        produced a broken 1-module roadmap, the auditor will now report ALL
        genuinely missing concepts — not just 15 high-level umbrella terms.
        """
        student_ctx = _build_student_context_block(topic, profile)

        system = """You are EduMind's curriculum auditor. You perform TWO checks:

CHECK 1 — COVERAGE: Find concepts genuinely missing from the module list.
CHECK 2 — STRUCTURE: Find structural problems in the roadmap.

Return STRICT JSON only.
{
  "concepts_missing_from_modules": ["missing concept 1", ...],
  "structural_issues": ["issue description", ...],
  "coverage_verdict": "complete | minor_gaps | major_gaps",
  "verdict_reason": "one sentence"
}

CHECK 1 RULES:
- Only report concepts GENUINELY missing — not present in any module's concepts_taught.
- Do NOT report concepts in do_not_include — those are intentionally excluded.
- Do NOT report concepts the student already knows unless they are strict prerequisites.
- Report as many missing concepts as needed — do NOT cap the list.
- If complete, return empty list and verdict "complete".

CHECK 2 RULES — flag these structural issues:
- A module with 0 concepts_taught
- OOP concepts (classes, inheritance, polymorphism, encapsulation) bundled into one module
  when there are 4+ of them — each major OOP concept should be its own module on deep/medium pace
- Prerequisites referencing concepts that appear LATER in the list (ordering violation)
- Duplicate concept names across modules
- If no structural issues found, return empty structural_issues list."""

        module_titles = [
            {
                "id": m.get("id"),
                "title": m.get("title"),
                "concept": m.get("concept"),
                "concepts_taught": m.get("concepts_taught", [])[:4],
            }
            for m in modules
        ]

        payload = {
            "student_context": student_ctx,
            "current_module_list": module_titles,
        }

        try:
            raw = await generate(
                messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
                # generation_model (llama-4-scout, 30K TPM) — auditing is a knowledge
                # check + list comparison, not deep reasoning. Fast and high-capacity.
                model=settings.generation_model,
                system=system,
                json_mode=True,
                max_tokens=3000,
            )
            audit = parse_json_object(raw)
            missing = audit.get("concepts_missing_from_modules") or []
            structural = audit.get("structural_issues") or []
            verdict = audit.get("coverage_verdict", "unknown")
            logger.info(
                "Coverage audit: verdict={}, {} missing concepts, {} structural issues.",
                verdict, len(missing), len(structural)
            )
            if structural:
                for issue in structural[:5]:
                    logger.warning("Roadmap structural issue: {}", issue)
            return [str(m).strip() for m in missing if str(m).strip()], structural
        except Exception as exc:
            logger.warning("Coverage audit failed: {}. Proceeding with current modules.", exc)
            return [], []

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Gap fill — add missing concepts + re-sequence
    # ─────────────────────────────────────────────────────────────────────────

    async def _fill_gaps(
        self,
        topic: str,
        profile: dict,
        modules: list[dict],
        concepts: list[dict],
        missing_concepts: list[str],
        structural_issues: list[str] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """Add missing concepts to the concept list and re-sequence."""
        if not missing_concepts and not structural_issues:
            return modules, concepts

        existing_names = {str(c.get("name") or "").lower() for c in concepts}
        new_concepts = list(concepts)
        for mc in missing_concepts:
            if mc.lower() not in existing_names:
                new_concepts.append({
                    "name": mc,
                    "cluster": "additional",
                    "importance": "important",
                    "why_needed": "Identified as missing by coverage auditor.",
                })
                existing_names.add(mc.lower())
                logger.info("Gap fill: adding missing concept '{}'.", mc)

        repair_fb = ""
        if structural_issues:
            repair_fb = "Fix these structural issues: " + "; ".join(structural_issues[:5])

        # If no new concepts were added AND no structural issues, nothing to do.
        if len(new_concepts) == len(concepts) and not structural_issues:
            return modules, concepts

        try:
            new_modules, _, new_conf = await self._sequence_modules(
                topic=topic, profile=profile, concepts=new_concepts,
                repair_feedback=repair_fb,
            )
            is_collapse = (
                len(new_modules) < max(3, len(modules) * 0.6) and new_conf < 0.5
            )
            if new_modules and not is_collapse:
                logger.info(
                    "Gap fill re-sequence: {} → {} modules (conf={:.2f}).",
                    len(modules), len(new_modules), new_conf
                )
                return new_modules, new_concepts
            else:
                logger.warning(
                    "Gap fill re-sequence collapsed ({} modules, conf={:.2f}) — keeping original {}.",
                    len(new_modules), new_conf, len(modules)
                )
        except Exception as exc:
            logger.warning("Gap fill re-sequence failed: {}. Keeping current modules.", exc)

        return modules, new_concepts

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5: Python-only structural fixes
    # ─────────────────────────────────────────────────────────────────────────

    def _structural_fix(self, modules: list[dict]) -> list[dict]:
        """
        Normalize raw module dictionaries before typed model construction.

        The LLM may omit ids, mix scalar/list fields, or emit question-like
        scope entries. This pass gives downstream validators a consistent shape
        without changing the intended concept order.
        """
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
            for key in (
                "concepts_taught", "must_teach", "prerequisites", "question_scope",
                "lesson_requirements", "practice_requirements", "examples_to_include",
                "what_this_module_will_not_cover",
            ):
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
            m["question_scope"] = [
                v for v in m["question_scope"]
                if not is_question_like_scope_text(v)
            ] or list(m["concepts_taught"])
            result.append(m)
        return result

    def _deduplicate_concepts(self, modules: list[dict]) -> list[dict]:
        """Remove exact repeated concepts while preserving first occurrence order."""
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
        """Parse an integer estimate from raw LLM output with a fallback."""
        try:
            return int(value)
        except (TypeError, ValueError):
            m = re.search(r"\d+", str(value or ""))
            return int(m.group(0)) if m else fallback

    def _build_module_objects(self, raw_modules: list[dict], depth_level: str) -> list[Module]:
        """Convert normalized module dictionaries into typed Module objects."""
        pace_ranges = {"fast": (10, 25), "medium": (20, 40), "deep": (30, 60)}
        min_min, max_min = pace_ranges.get(self.state.pace, (20, 40))
        result: list[Module] = []
        for idx, m in enumerate(raw_modules, start=1):
            concept = str(m.get("concept") or m.get("title") or f"module {idx}").strip()
            concepts_taught = m.get("concepts_taught") or [concept]
            must_teach = m.get("must_teach") or concepts_taught
            prereqs = [
                str(p).strip() for p in (m.get("prerequisites") or [])
                if str(p).strip()
                and not re.fullmatch(r"(m|module)[\s_-]*\d+[a-z]?", str(p).strip().lower())
            ]
            dl = m.get("depth_level")
            if dl not in ("surface", "standard", "deep"):
                dl = depth_level
            estimated = max(min_min, min(max_min, self._safe_int(m.get("estimated_minutes"), min_min)))
            title = str(m.get("title") or concept).strip()
            why_now = str(m.get("why_now") or "").strip()
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

    def _build_master_roadmap(
        self,
        modules: list[Module],
        topic: str,
        profile: dict,
        rationale: str,
    ) -> MasterRoadmap:
        """Build the persisted master roadmap from finalized module objects."""
        # Use a stub ResearchSummary — no Tavily data.
        research_summary = _empty_research(topic)

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
        """Derive scope metadata from the module plan when modules drive the roadmap."""
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
        """Drop modules for trusted known concepts only when mastery clears threshold."""
        if not modules:
            return modules
        trusted = {
            str(c).strip().lower()
            for c in getattr(self, "_trusted_assumed_known", [])
            if str(c).strip()
        }
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

    async def _llm_repair(
        self,
        modules: list[dict],
        issues: list[str],
        topic: str,
        profile: dict,
    ) -> list[dict]:
        """Ask the reasoning model for a targeted repair of validator issues."""
        student_ctx = _build_student_context_block(topic, profile)
        system = """You are EduMind's curriculum repair agent.
Fix ONLY the listed issues. Preserve all valid modules.
Return STRICT JSON only: {"modules": [...], "repair_rationale": "..."}"""
        modules_for_repair = [
            {
                "id": m.get("id"), "title": m.get("title"), "concept": m.get("concept"),
                "concepts_taught": m.get("concepts_taught", []),
                "prerequisites": m.get("prerequisites", []),
                "roadmap_step_id": m.get("roadmap_step_id"),
            }
            for m in modules[:40]
        ]
        payload = {
            "student_context": student_ctx,
            "issues_to_fix": issues,
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
        """
        Build, validate, repair, persist, and return a CurriculumPlan.

        The planner uses a coverage-first LLM call, a sequencing LLM call,
        deterministic cleanup, validation, optional repair, and final typed
        conversion before saving the curriculum for the student.
        """
        pace = self.state.pace if self.state.pace in ("fast", "medium", "deep") else "medium"
        depth_level = {"fast": "surface", "medium": "standard", "deep": "deep"}[pace]
        profile = dict(profile or getattr(self, "personalization_profile", {}) or {})
        self.personalization_profile = profile

        trusted_assumed = [
            str(c).strip()
            for c in (profile.get("assumed_known_concepts") or profile.get("known_concepts") or [])
            if str(c).strip()
        ]
        if profile_has_no_prior_experience(profile):
            trusted_assumed = [c for c in trusted_assumed if c in profile.get("assumed_known_concepts", [])]
        self._trusted_assumed_known = [
            c for c in trusted_assumed
            if not is_unreliable_generated_concept(c) and is_related_to_profile(c, profile)
        ]

        intent = profile.get("current_intent") if isinstance(profile.get("current_intent"), dict) else {}
        self._log_decision(
            action="ANALYZE_INTENT",
            reason="Two-call coverage+sequencing architecture (no Tavily).",
            payload={"intent": intent, "confidence": profile.get("confidence", 0.7)},
        )

        repair_history: list[dict] = []
        concepts: list[dict] = []
        raw_modules: list[dict] = []
        rationale = ""
        confidence = 0.0

        # ── Call A: Coverage planning ─────────────────────────────────────────
        for attempt in range(2):
            try:
                concepts = await self._plan_coverage(topic, profile)
                self._log_decision(
                    action="PLAN_COVERAGE",
                    reason=f"Coverage planner: {len(concepts)} concepts.",
                    payload={"concept_count": len(concepts)},
                )
                break
            except (GroqRateLimitError, GroqTimeoutError):
                raise
            except Exception as exc:
                logger.warning("Coverage planning attempt {} failed: {}.", attempt + 1, exc)
                repair_history.append({"stage": "coverage_planning", "attempt": attempt + 1, "error": str(exc)})
                if attempt == 1:
                    # Hard fail — don't silently degrade to a 1-concept list.
                    # Surface the error so the caller can retry or inform the user.
                    raise ValueError(
                        f"Coverage planning failed after 2 attempts for '{topic}'. "
                        f"Last error: {exc}"
                    ) from exc

        # ── Call B: Sequencing ────────────────────────────────────────────────
        for attempt in range(2):
            try:
                raw_modules, rationale, confidence = await self._sequence_modules(
                    topic=topic, profile=profile, concepts=concepts
                )
                self._log_decision(
                    action="SEQUENCE_MODULES",
                    reason=f"Sequencer: {len(concepts)} concepts → {len(raw_modules)} modules (confidence={confidence:.2f}).",
                    payload={"module_count": len(raw_modules), "confidence": confidence},
                )
                break
            except (GroqRateLimitError, GroqTimeoutError):
                raise
            except Exception as exc:
                logger.warning("Sequencing attempt {} failed: {}.", attempt + 1, exc)
                repair_history.append({"stage": "sequencing", "attempt": attempt + 1, "error": str(exc)})
                if attempt == 1:
                    raw_modules = [
                        {
                            "id": f"m{i}", "title": c.get("name", topic),
                            "concept": c.get("name", topic),
                            "concepts_taught": [c.get("name", topic)],
                            "must_teach": [c.get("name", topic)],
                            "prerequisites": [], "estimated_minutes": 30,
                            "depth_level": depth_level,
                            "roadmap_step_id": f"step_{i:02d}",
                            "question_scope": [c.get("name", topic)[:40]],
                            "why_this_module_exists": f"Learn {c.get('name', topic)}.",
                            "goal_alignment": f"Supports {topic} mastery.",
                            "domain_framing": c.get("name", topic),
                            "module_goal": f"Understand {c.get('name', topic)}.",
                            "why_now": "Foundation.",
                        }
                        for i, c in enumerate(concepts[:20], start=1)
                    ]
                    rationale = f"Fallback after sequencing failed: {exc}"
                    confidence = 0.5

        # ── Coverage audit ────────────────────────────────────────────────────
        if confidence >= 0.5 and raw_modules:
            try:
                missing_concepts, structural_issues = await self._audit_coverage(
                    topic=topic, profile=profile,
                    modules=raw_modules, concepts=concepts,
                )
                self._log_decision(
                    action="AUDIT_COVERAGE",
                    reason=f"Auditor: {len(missing_concepts)} missing concepts, {len(structural_issues)} structural issues.",
                    payload={"missing": missing_concepts[:5], "structural": structural_issues[:3]},
                )

                # ── Gap fill ──────────────────────────────────────────────────
                if missing_concepts or structural_issues:
                    raw_modules, concepts = await self._fill_gaps(
                        topic=topic, profile=profile,
                        modules=raw_modules, concepts=concepts,
                        missing_concepts=missing_concepts,
                        structural_issues=structural_issues,
                    )
                    self._log_decision(
                        action="FILL_GAPS",
                        reason=f"Gap fill complete: {len(raw_modules)} modules after adding {len(missing_concepts)} concepts, fixing {len(structural_issues)} structural issues.",
                        payload={"final_module_count": len(raw_modules)},
                    )
            except (GroqRateLimitError, GroqTimeoutError):
                raise
            except Exception as exc:
                logger.warning("Coverage audit/gap fill failed: {}. Proceeding with current modules.", exc)

        # ── Structural fixes ──────────────────────────────────────────────────
        raw_modules = self._structural_fix(raw_modules)
        raw_modules = self._deduplicate_concepts(raw_modules)

        if len(raw_modules) < 2:
            logger.warning("Module list too short after cleanup; running LLM repair.")
            raw_modules = await self._llm_repair(
                modules=raw_modules,
                issues=["Module list has fewer than 2 modules — please generate a complete curriculum."],
                topic=topic, profile=profile,
            )
            raw_modules = self._structural_fix(raw_modules)

        # ── Build typed Module objects ─────────────────────────────────────────
        modules = self._build_module_objects(raw_modules, depth_level)
        modules = self._compress_mastered(modules)
        for m in modules:
            m.question_scope = [
                v for v in (m.question_scope or []) if not is_question_like_scope_text(v)
            ] or list(m.concepts_taught)

        scope = self._build_scope_from_modules(modules, topic, profile)
        master_roadmap = self._build_master_roadmap(modules, topic, profile, rationale)

        # ── Lightweight structural validation ─────────────────────────────────
        validation_profile = {
            **profile,
            "known_concepts": list(dict.fromkeys(
                list(profile.get("known_concepts") or []) + list(self._trusted_assumed_known)
            )),
        }
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
        learning_path = [
            {"step_id": m.roadmap_step_id, "concept": m.concept,
             "why_now": m.why_now, "goal_alignment": m.why_it_matters_for_goal}
            for m in modules
        ]
        roadmap_steps = [(m.title + (": " + m.purpose if m.purpose else "")).strip() for m in modules]

        validation = validate_curriculum_quality(
            topic=topic, modules=modules, profile=validation_profile, scope_analysis=scope,
            student_history=student_history, concept_inventory=concept_inventory,
            prerequisite_graph=prerequisite_graph,
            roadmap_steps=roadmap_steps + [m.title + (": " + m.why_now if m.why_now else "") for m in modules],
            schedule=None,
        )
        repair_history.append({
            "stage": "post_build_validation",
            "validation": validation,
            "module_count": len(modules),
        })

        if not validation["passed"]:
            issues = [
                i for i in (validation.get("issues") or [])
                if "outside 40%" not in i
                and "concepts_to_delay" not in i.lower()
                and not i.startswith("Module count")
            ]
            if issues:
                logger.warning("Post-build validation found {} issues; running targeted LLM repair.", len(issues))
                repaired_raw = await self._llm_repair(
                    modules=raw_modules, issues=issues[:10], topic=topic, profile=profile,
                )
                repaired_raw = self._structural_fix(repaired_raw)
                repaired_raw = self._deduplicate_concepts(repaired_raw)
                repaired_modules = self._build_module_objects(repaired_raw, depth_level)
                repaired_modules = self._compress_mastered(repaired_modules)
                for m in repaired_modules:
                    m.question_scope = [
                        v for v in (m.question_scope or []) if not is_question_like_scope_text(v)
                    ] or list(m.concepts_taught)
                if len(repaired_modules) >= 2:
                    modules = repaired_modules
                    raw_modules = repaired_raw
                    scope = self._build_scope_from_modules(modules, topic, profile)
                    master_roadmap = self._build_master_roadmap(modules, topic, profile, rationale)
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
            remaining = [
                i for i in (validation.get("issues") or [])
                if "outside 40%" not in i and "concepts_to_delay" not in i.lower()
            ]
            if remaining:
                logger.warning("Curriculum has minor validation issues (proceeding): {}",
                               "; ".join(remaining[:5]))
            validation["passed"] = True

        validation["confidence"] = confidence
        self._master_roadmap = master_roadmap
        self._log_decision(
            action="BUILD_MASTER_ROADMAP",
            reason=rationale or "Two-call architecture completed.",
            payload={"module_count": len(modules), "confidence": confidence,
                     "total_minutes": sum(m.estimated_minutes for m in modules)},
        )

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
            prerequisite_graph={
                str(k): [str(v) for v in (val if isinstance(val, list) else [])]
                for k, val in prerequisite_graph.items()
            },
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
            self._curriculum_id = int(row["id"])

        self.state.curriculum = plan
        self.state.mark_dirty("curriculum")

        if hasattr(self, "_eval_runner") and self._eval_runner is not None:
            asyncio.create_task(self._eval_runner.on_curriculum_built(
                modules=[{
                    "concept": m.concept, "title": m.title,
                    "prerequisites": m.prerequisites,
                    "concepts_taught": m.concepts_taught,
                    "depends_on_concepts": m.depends_on_concepts,
                    "question_scope": m.question_scope,
                } for m in plan.modules]
            ))

        self._log_decision(
            action="BUILD_CURRICULUM",
            reason=f"Curriculum built: {len(modules)} modules, confidence={confidence:.2f}",
            payload={"topic": plan.topic, "module_count": len(modules), "confidence": confidence},
        )
        logger.info("Curriculum built: {} modules for topic='{}' (confidence={:.2f})",
                    len(modules), plan.topic, confidence)

        # V1: ChromaDB background embedding disabled (BGE-M3 ~2.2GB RAM, crashes EC2).
        # V2: uncomment below to re-enable after moving to memory-optimised instance.
        # async def _embed_background():                       # V2
        #     try:                                             # V2
        #         await self._embed_to_chromadb(plan, int(row["id"]))  # V2
        #     except Exception as exc:                         # V2
        #         logger.warning("Background ChromaDB embedding failed for id={}: {}.", row["id"], exc)  # V2
        # asyncio.create_task(_embed_background())             # V2
        return plan

    # ── ChromaDB ──────────────────────────────────────────────────────────────

    async def _embed_to_chromadb(self, plan: CurriculumPlan, curriculum_id: int | None = None) -> None:
        """V1: disabled — full body hashed out below. V2: uncomment to restore."""
        return
        # domain = plan.domain                                 # V2
        # embedded, failed = 0, 0                             # V2
        # course_id = f"course-{curriculum_id}" if curriculum_id is not None else ""  # V2
        # for module in plan.modules:                          # V2
        #     concept = module.concept                         # V2
        #     framing = module.domain_framing                  # V2
        #     prereqs = ", ".join(module.prerequisites) if module.prerequisites else "none"  # V2
        #     safe_concept = re.sub(r"[^a-zA-Z0-9_]", "_", concept)[:60]  # V2
        #     lines = [                                        # V2
        #         f"Course topic: {plan.topic}", f"Course goal: {plan.goal}", f"Domain: {domain}",  # V2
        #         f"Module id: {module.id}", f"Roadmap step id: {module.roadmap_step_id}",  # V2
        #         f"Module title: {module.title}", f"Concept: {concept}",  # V2
        #         f"Concepts taught: {', '.join(module.concepts_taught or [concept])}",  # V2
        #         f"Must teach: {', '.join(module.must_teach or module.concepts_taught or [concept])}",  # V2
        #         f"Prerequisites: {prereqs}", f"Depth level: {module.depth_level}",  # V2
        #         f"Estimated minutes: {module.estimated_minutes}", f"Purpose: {module.purpose}",  # V2
        #         f"Goal alignment: {module.why_it_matters_for_goal}",  # V2
        #         f"Lesson requirements: {', '.join(module.lesson_requirements or [])}",  # V2
        #         f"Practice requirements: {', '.join(module.practice_requirements or [])}",  # V2
        #         f"Question scope: {', '.join(module.question_scope or [])}",  # V2
        #         f"Domain framing: {framing}",                # V2
        #     ]                                                # V2
        #     full_text = "\n".join(line for line in lines if line.split(": ", 1)[-1].strip())  # V2
        #     chunk_id = (plan.topic + "_" + safe_concept).replace(" ", "_")[:180]  # V2
        #     try:                                             # V2
        #         await chroma_insert(chunk_id, domain, full_text,  # V2
        #                             metadata={               # V2
        #                                 "course_id": course_id,  # V2
        #                                 "curriculum_id": str(curriculum_id or ""),  # V2
        #                                 "topic": plan.topic, "module_id": module.id, "concept": concept,  # V2
        #                             })                       # V2
        #         if curriculum_id is not None:                # V2
        #             async with get_conn() as conn:           # V2
        #                 await conn.execute(                  # V2
        #                     "INSERT INTO module_embeddings (student_id, curriculum_id, module_id, chromadb_id, domain) "  # V2
        #                     "VALUES ($1, $2, $3, $4, $5) ON CONFLICT (student_id, module_id) DO UPDATE "  # V2
        #                     "SET curriculum_id=$2, chromadb_id=$4, domain=$5",  # V2
        #                     self.state.student_id, curriculum_id, module.id, chunk_id, domain,  # V2
        #                 )                                    # V2
        #                 await conn.execute(                  # V2
        #                     "INSERT INTO dynamic_concept_summaries "  # V2
        #                     "(student_id, curriculum_id, module_id, concept, domain, summary_text, chromadb_id) "  # V2
        #                     "VALUES ($1, $2, $3, $4, $5, $6, $7) "  # V2
        #                     "ON CONFLICT (student_id, module_id) DO UPDATE "  # V2
        #                     "SET curriculum_id=$2, concept=$4, domain=$5, summary_text=$6, chromadb_id=$7, updated_at=NOW()",  # V2
        #                     self.state.student_id, curriculum_id, module.id, concept, domain, full_text, chunk_id,  # V2
        #                 )                                    # V2
        #         embedded += 1                                # V2
        #     except Exception as exc:                         # V2
        #         failed += 1                                  # V2
        #         logger.warning("ChromaDB insert failed for concept='{}': {}.", concept, exc)  # V2
        # logger.info("ChromaDB: {}/{} modules embedded for domain='{}'.", embedded, len(plan.modules), domain)  # V2
