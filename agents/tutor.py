"""
agents/tutor.py
TutorAgent — delivers a streamed lesson for one module, handles doubts inline.

Uses RAG to ground content. Streams output token by token.
Terminal tool: finish_lesson
Non-terminal tools: retrieve_content, inject_micro_lesson, handle_doubt
"""

from __future__ import annotations

from loguru import logger

from agents.base_agent import BaseAgent
from core.student_model import StudentState
from core.rag_pipeline import retrieve
from clients.groq_client import stream
from db.postgres import get_conn
from config import settings


class TutorAgent(BaseAgent):
    NAME = "tutor_agent"
    TERMINAL_TOOL = "finish_lesson"

    def __init__(self, state: StudentState):
        super().__init__(state)
        self._retrieved_chunks: list[str] = []

        self.TOOLS = [
            self.build_tool(
                name="retrieve_content",
                description=(
                    "Retrieve relevant content chunks from the knowledge base for the concept. "
                    "Call this first before delivering the lesson."
                ),
                properties={
                    "concept": {
                        "type": "string",
                        "description": "The concept to retrieve content for",
                    },
                    "query": {
                        "type": "string",
                        "description": "Specific retrieval query (more specific than concept name)",
                    },
                },
                required=["concept", "query"],
            ),
            self.build_tool(
                name="deliver_lesson",
                description=(
                    "Deliver the main lesson content to the student. "
                    "You MUST include the full lesson_text field with the complete lesson. "
                    "lesson_text is required and must contain the full explanation."
                ),
                properties={
                    "lesson_text": {
                        "type": "string",
                        "description": (
                            "REQUIRED. Full lesson content in markdown. "
                            "Must include: concept explanation, domain example, key takeaway. "
                            "Minimum 3 paragraphs."
                        ),
                    },
                    "style_used": {
                        "type": "string",
                        "enum": ["formal", "analogy", "example_first", "visual", "story"],
                        "description": "Teaching style used for this lesson",
                    },
                },
                required=["lesson_text", "style_used"],
            ),
            self.build_tool(
                name="handle_doubt",
                description=(
                    "Handle a student doubt or question inline during the lesson. "
                    "Call this when the student expresses confusion or asks a question."
                ),
                properties={
                    "doubt_text": {
                        "type": "string",
                        "description": "The student's doubt or question",
                    },
                    "doubt_type": {
                        "type": "string",
                        "enum": ["general", "prerequisite", "application", "none"],
                        "description": "Type of doubt. Use 'none' if student has no real question.",
                    },
                    "response": {
                        "type": "string",
                        "description": "Clear, concise response to the doubt",
                    },
                },
                required=["doubt_text", "doubt_type", "response"],
            ),
            self.build_tool(
                name="finish_lesson",
                description=(
                    "Mark the lesson as complete. "
                    "Call after delivering content and handling any doubts. "
                    "Only provide the 4 required fields, nothing else."
                ),
                properties={
                    "summary": {
                        "type": "string",
                        "description": "One sentence summary of what was taught",
                    },
                    "style_used": {
                        "type": "string",
                        "enum": ["formal", "analogy", "example_first", "visual", "story"],
                        "description": "Primary style used",
                    },
                    "doubt_count": {
                        "type": "integer",
                        "description": "Number of doubts raised during lesson",
                    },
                    "fatigue_detected": {
                        "type": "string",
                        "enum": ["yes", "no"],
                        "description": "Did student show signs of fatigue? Answer 'yes' or 'no'.",
                    },
                },
                required=["summary", "style_used", "doubt_count", "fatigue_detected"],
            ),
        ]

    # ── Tool executor ─────────────────────────────────────────────────────────

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        if tool_name == "retrieve_content":
            concept = args["concept"]
            query = args.get("query", concept)
            chunks = retrieve(
                query=query,
                domain=self.state.domain,
                top_k=5,
            )
            self._retrieved_chunks = chunks
            if not chunks:
                return "No content found in knowledge base. Use your training knowledge."
            summary = "\n\n".join(chunks[:3])
            return f"Retrieved {len(chunks)} chunks:\n\n{summary}"

        if tool_name == "deliver_lesson":
            lesson_text = args["lesson_text"]
            style_used = args.get("style_used", "formal")

            print(f"\n{'='*60}")
            print(f"📚 LESSON")
            print(f"{'='*60}\n")

            # Stream the lesson
            messages = [{
                "role": "user",
                "content": (
                    f"Present this lesson content clearly and engagingly:\n\n{lesson_text}"
                )
            }]
            full_text = ""
            for chunk in stream(messages=messages, model=settings.generation_model):
                print(chunk, end="", flush=True)
                full_text += chunk
            print(f"\n\n{'='*60}\n")

            # Check for doubt after lesson
            print("❓ Any questions before we continue? (press Enter to skip): ", end="")
            doubt = input().strip()
            if doubt:
                self.state.log_doubt(
                    self._current_module().concept if self._current_module() else "unknown",
                    "general"
                )
                return f"Lesson delivered ({style_used}). Student has a question: {doubt}"

            return f"Lesson delivered ({style_used}). Student understood."

        if tool_name == "handle_doubt":
            doubt_text = args["doubt_text"]
            doubt_type = args.get("doubt_type", "general")
            response = args["response"]

            module = self._current_module()
            concept = module.concept if module else "unknown"
            self.state.log_doubt(concept, doubt_type)

            print(f"\n💡 {response}\n")
            return f"Doubt handled: '{doubt_text[:50]}' → response delivered"

        return super()._execute_tool(tool_name, args)

    # ── Public run method ─────────────────────────────────────────────────────

    def teach(self) -> dict:
        """
        Deliver a lesson for the current curriculum module.

        Returns:
            dict with keys: summary, style_used, doubt_count, fatigue_detected
        """
        module = self._current_module()
        if module is None:
            logger.warning("TutorAgent.teach() called with no current module")
            return {
                "summary": "No module to teach",
                "style_used": "formal",
                "doubt_count": 0,
                "fatigue_detected": False,
            }

        meta = self.state.metacognition
        style = meta.preferred_style
        prior_doubts = self.state.get_doubt_count(module.concept)

        system = f"""You are an expert adaptive tutor delivering a lesson.

STUDENT CONTEXT:
{self._student_context()}

CURRENT MODULE:
- Title: {module.title}
- Concept: {module.concept}
- Domain framing: {module.domain_framing}
- Depth level: {module.depth_level}
- Estimated time: {module.estimated_minutes} minutes
- Prerequisites: {', '.join(module.prerequisites) or 'none'}

LESSON RULES:
1. Call retrieve_content first with a specific query about '{module.concept}'
2. Deliver lesson using preferred style='{style}'
3. Frame everything in domain context: {self.state.domain}
4. If student has a doubt, call handle_doubt immediately
5. Prior doubts on this concept: {prior_doubts} (if > 0, pre-empt common confusion)
6. Calibration={meta.calibration_pattern} — if overconfident, add harder examples
7. Call finish_lesson when done — provide ONLY summary, style_used, doubt_count, fatigue_detected. No other fields.

STYLE GUIDE:
- formal: structured definitions → theory → examples
- analogy: relatable real-world comparison first
- example_first: concrete example before theory
- visual: describe diagrams/visualisations
- story: narrative arc connecting concepts
"""

        result = self.run(
            system=system,
            user_message=f"Teach the module: '{module.title}' (concept: '{module.concept}')",
            model=settings.reasoning_model,  # 70b is more reliable with tool schemas
        )

        # Update metacognition with style performance (will be scored after eval)
        self._log_decision(
            action="LESSON_DELIVERED",
            reason=f"Taught {module.concept} using {result.get('style_used', style)} style",
            payload={
                "module_id": module.id,
                "concept": module.concept,
                "style": result.get("style_used", style),
                "doubt_count": result.get("doubt_count", 0),
            },
        )

        return result
