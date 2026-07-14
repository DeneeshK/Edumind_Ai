"""
LEGACY — interactive CLI/SSE session flow. Not used by the deployed frontend,
which uses the /api/courses flow (see docs/ARCHITECTURE.md). Kept as a working
reference implementation of the queue-based interactive session pattern.

agents/tutor.py
TutorAgent — delivers a world-class, interactive lesson for one module.

Lesson structure (10 sections):
  1. Lesson Header    — title, objectives, prereq reminder
  2. Opening Hook     — curiosity-creating question or scenario
  3. Intuition First  — plain-English mental model before any formalism
  4. Core Explanation — formal definition + mechanics, built layer by layer
  5. Think Moment     — inline "pause and predict" before the example
  6. Worked Example   — step-by-step with WHY at each step
  7. Common Mistakes  — 2-3 real pitfalls students hit
  8. Practice Task    — one problem the student solves immediately
  9. Final Summary    — 3 bullet recap
  10. What Comes Next — bridge to next module

Terminal tool: finish_lesson
Non-terminal tools: retrieve_content, deliver_lesson, handle_doubt
"""

from __future__ import annotations

import asyncio

from loguru import logger

from agents.base_agent import BaseAgent
from core.student_model import StudentState
from config import settings

# ── Section marker the runtime uses to split lesson at checkpoint ─────────────
CHECKPOINT_MARKER = "### ✏️ Think Moment"


class TutorAgent(BaseAgent):
    """
    Run the legacy interactive tutoring flow for one curriculum module.

    The tutor retrieves optional context, delivers a structured lesson, captures
    checkpoint/practice responses, records delivered content for later grounded
    evaluation, and logs student doubts for adaptation.
    """

    NAME = "tutor_agent"
    TERMINAL_TOOL = "finish_lesson"

    def __init__(self, state: StudentState, emit_fn=None, ask_fn=None):
        super().__init__(state)
        self._retrieved_chunks: list[str] = []
        self._delivered_lesson = False

        async def _default_emit(text: str) -> None:
            """Emit tutor text in legacy CLI mode."""
            print(text, flush=True)

        async def _default_ask(question: str, **kw) -> str:
            """Ask a tutor prompt in legacy CLI mode."""
            print(question, flush=True)
            return input("> ").strip()

        self._emit = emit_fn or _default_emit
        self._ask = ask_fn or _default_ask
        self.TOOLS = self._build_tools()

    # ── Tool definitions ───────────────────────────────────────────────────────

    def _build_tools(self) -> list[dict]:
        """Build Groq tool schemas used by the tutor agent loop."""
        return [
            self.build_tool(
                name="retrieve_content",
                description=(
                    "Retrieve relevant content from the knowledge base. "
                    "Call this FIRST with a specific query — not just the concept name. "
                    "E.g. for 'gradient descent' use 'gradient descent intuition learning rate convergence'."
                ),
                properties={
                    "concept": {"type": "string", "description": "Concept to retrieve for"},
                    "query": {"type": "string", "description": "Specific targeted retrieval query"},
                },
                required=["concept", "query"],
            ),
            self.build_tool(
                name="deliver_lesson",
                description=(
                    "Deliver the complete lesson. This is the most important call — write rich, substantive content. "
                    "lesson_text must be a full, flowing lesson following the pace-appropriate structure. "
                    "CRITICAL: checkpoint_question must be copied VERBATIM from the '### ✏️ Think Moment' "
                    "section you wrote in lesson_text. Do NOT invent a new question here — find the "
                    "question you already wrote and paste it exactly."
                ),
                properties={
                    "lesson_text": {
                        "type": "string",
                        "description": (
                            "Full lesson in markdown. Must have substantive content in every section — "
                            "no 2-sentence stubs. Each section must contain real teaching: specific facts, "
                            "concrete examples, domain-specific context. "
                            "Must include '### ✏️ Think Moment' heading with one thinking question "
                            "derived from what was just explained in THIS lesson."
                        ),
                    },
                    "checkpoint_question": {
                        "type": "string",
                        "description": (
                            "VERBATIM copy of the question from the '### ✏️ Think Moment' section. "
                            "Must match the lesson_text exactly. This is NOT a new question — "
                            "it is the question you already wrote in lesson_text."
                        ),
                    },
                    "practice_task": {
                        "type": "string",
                        "description": "The practice problem from the lesson. Student attempts this right now.",
                    },
                    "style_used": {
                        "type": "string",
                        "enum": ["formal", "analogy", "example_first", "visual", "story"],
                        "description": "Teaching style used",
                    },
                },
                required=["lesson_text", "checkpoint_question", "practice_task", "style_used"],
            ),
            self.build_tool(
                name="handle_doubt",
                description="Handle a student doubt inline. Call when student expresses confusion.",
                properties={
                    "doubt_text": {"type": "string", "description": "Student's doubt"},
                    "doubt_type": {
                        "type": "string",
                        "enum": ["general", "prerequisite", "application", "none"],
                    },
                    "response": {"type": "string", "description": "Tutor's response"},
                },
                required=["doubt_text", "doubt_type", "response"],
            ),
            self.build_tool(
                name="finish_lesson",
                description="Mark lesson complete. Call only after deliver_lesson.",
                properties={
                    "summary": {"type": "string", "description": "One sentence: what was taught"},
                    "style_used": {
                        "type": "string",
                        "enum": ["formal", "analogy", "example_first", "visual", "story"],
                    },
                    "doubt_count": {"type": "integer"},
                    "fatigue_detected": {"type": "string", "enum": ["yes", "no"]},
                },
                required=["summary", "style_used", "doubt_count", "fatigue_detected"],
            ),
        ]

    # ── Helpers ────────────────────────────────────────────────────────────────

    async def _emit_block(self, text: str) -> None:
        """Emit a visually separated block through the active output channel."""
        await self._emit("\n" + "━" * 60)
        await self._emit(text)
        await self._emit("━" * 60 + "\n")

    def _classify_doubt_type(self, doubt_text: str, module) -> str:
        """Classify a student doubt so later adaptation can use the signal."""
        text = doubt_text.lower()
        prereqs = " ".join(module.prerequisites).lower() if module else ""
        if any(w in text for w in ("example", "apply", "use case", "how do i", "real world")):
            return "application"
        if any(p and p in text for p in prereqs.split()):
            return "prerequisite"
        if any(w in text for w in ("what is", "meaning", "definition", "prerequisite")):
            return "prerequisite"
        return "general"

    async def _answer_doubt(self, doubt_text: str, module, doubt_type: str) -> str:
        """Generate a short mid-lesson answer for a captured doubt."""
        concept = module.concept if module else "the concept"
        try:
            from clients.groq_client import generate
            return await generate(
                messages=[{"role": "user", "content": (
                    f"Student question mid-lesson: {doubt_text}\n"
                    f"Concept: {concept} | Domain: {self.state.domain} | Type: {doubt_type}\n\n"
                    "Answer in 3-5 sentences. Be warm, use one concrete example. "
                    "End with a sentence tying back to the lesson."
                )}],
                model=settings.generation_model,
                system="You are an encouraging expert tutor answering a mid-lesson question.",
            )
        except Exception as exc:
            logger.warning("Doubt answer failed: {}", exc)
            return (
                f"Great question. Anchor back to **{concept}**: what's the definition, "
                "the rule, and the example from the lesson? That triangle should resolve it."
            )

    async def _capture_doubt(self, module) -> int:
        """Ask for one optional doubt, answer it, and record it in session state."""
        doubt = await self._ask("❓ Any questions? (press Enter to skip)")
        if not doubt.strip():
            return 0
        dtype = self._classify_doubt_type(doubt, module)
        concept = module.concept if module else "unknown"
        self.state.log_doubt(concept, dtype)
        response = await self._answer_doubt(doubt, module, dtype)
        if module:
            self.state.record_module_content(
                module.id, f"Doubt: {doubt}\nResponse: {response}"
            )
        await self._emit(f"💡 {response}")
        return 1

    async def _run_checkpoint(self, question: str, module) -> None:
        """Pause mid-lesson, capture student prediction, give live feedback."""
        await self._emit(f"\n{CHECKPOINT_MARKER}\n\n> {question}")
        answer = await self._ask("✏️ Your answer (Enter to skip):")
        if not answer.strip():
            await self._emit("*(Skipped — we'll test this in the evaluation.)*")
            return
        concept = module.concept if module else "the concept"
        try:
            from clients.groq_client import generate
            feedback = await generate(
                messages=[{"role": "user", "content": (
                    f"Mid-lesson checkpoint.\nQuestion: {question}\n"
                    f"Student answer: {answer}\nConcept: {concept}\nDomain: {self.state.domain}\n\n"
                    "In 2-3 sentences: affirm what's right, correct what's wrong, "
                    "give the clearest version. Be encouraging."
                )}],
                model=settings.generation_model,
                system="Encouraging tutor giving instant checkpoint feedback.",
            )
        except Exception:
            feedback = f"Good thinking! Key idea: **{concept}** — we'll solidify this in the example below."
        await self._emit(f"\n💬 **Feedback:** {feedback}\n")
        if module:
            self.state.record_module_content(
                module.id, f"Checkpoint Q: {question}\nAnswer: {answer}\nFeedback: {feedback}"
            )

    async def _run_practice(self, task: str, module) -> None:
        """Let student attempt the practice task and get immediate feedback."""
        await self._emit(f"\n---\n### 📝 Your Turn\n\n{task}")
        answer = await self._ask("✏️ Your solution (Enter to skip):")
        if not answer.strip():
            await self._emit("*(Skipped — try this before the next session!)*")
            return
        concept = module.concept if module else "the concept"
        try:
            from clients.groq_client import generate
            feedback = await generate(
                messages=[{"role": "user", "content": (
                    f"Practice task: {task}\nStudent solution: {answer}\n"
                    f"Concept: {concept} | Domain: {self.state.domain}\n\n"
                    "Give feedback in 3-4 sentences: what's correct, what to fix, "
                    "and one tip to improve. Be specific and encouraging."
                )}],
                model=settings.generation_model,
                system="Expert tutor reviewing a student's practice solution.",
            )
        except Exception:
            feedback = f"Nice attempt! Compare your steps against the worked example — the logic of **{concept}** should mirror it."
        await self._emit(f"\n💬 **Practice Feedback:** {feedback}\n")
        if module:
            self.state.record_module_content(
                module.id, f"Practice task: {task}\nAnswer: {answer}\nFeedback: {feedback}"
            )

    # ── Fallback lesson when LLM fails ─────────────────────────────────────────

    def _fallback_lesson_text(self, module) -> str:
        """Return deterministic lesson markdown when LLM lesson delivery fails."""
        prereqs = ", ".join(module.prerequisites) if module.prerequisites else "none"
        domain = self.state.domain or "your field"
        goal = self.state.goal or "mastering this subject"
        pace = self.state.pace

        if pace == "fast":
            return self._fallback_fast(module, prereqs, domain, goal)
        if pace == "deep":
            return self._fallback_deep(module, prereqs, domain, goal)
        return self._fallback_medium(module, prereqs, domain, goal)

    def _fallback_fast(self, module, prereqs: str, domain: str, goal: str) -> str:
        """Fast pace: concise 5-section snapshot. No proofs, no edge cases."""
        return f"""# {module.title} ⚡ Fast Track

> **Pace:** Fast — you'll get the essential mental model + one example. No extras.
> **Prerequisites:** {prereqs}

---

### Why this matters

In {domain}, **{module.concept}** matters because it gives you a specific move
you can use while working toward {goal}.

### Core definition

**{module.concept}** — here's the simplest way to think about it in {domain} terms:
*{module.domain_framing}*

Everything below unpacks that one idea.

---

### Key rule or mechanism

1. **Step 1** → identify what you're working with in your {domain} problem
2. **Step 2** → apply the core rule of {module.concept}
3. **Step 3** → interpret the result in terms of {goal}

---

{CHECKPOINT_MARKER}

Quick prediction: before reading the example, what do you think happens at Step 2? Describe it in your own words.

---

### Worked example

In {domain}: start with a typical problem, apply {module.concept}, and the result directly
serves {goal}. The key move is Step 2 — that's exactly where this concept does its work.

---

### Connection

- **What it is:** {module.concept} = {module.domain_framing}
- **When to use it:** whenever {goal} requires this kind of operation
- **Coming up:** this feeds directly into the next module
"""

    def _fallback_medium(self, module, prereqs: str, domain: str, goal: str) -> str:
        """Medium pace: full 10-section course module."""
        return f"""# {module.title}

---

### 📋 Learning Objectives

By the end you will be able to:
- Explain **{module.concept}** in plain words
- Apply it to a real {domain} problem
- Connect it to {goal}

**Prerequisites:** {prereqs}

---

### Let's Start Here

Why would someone in {domain} need **{module.concept}**? Without it, certain problems
simply can't be solved. Your goal of *{goal}* makes this concept unavoidable — and once
you see it, you'll notice it everywhere.

---

### The Big Picture

Think of **{module.concept}** not as a rule to memorise, but as a *thinking tool* you reach for.
The clearest way to picture it: *{module.domain_framing}*

Every formal definition we add later is just a more precise version of that same idea.
Hold onto this framing — it's your anchor.

---

### Breaking It Down

1. **Identify** what you're given in the {domain} problem
2. **Apply** the core relationship of **{module.concept}**
3. **Interpret** — does the result serve {goal}?

---

{CHECKPOINT_MARKER}

Before we look at an example: in your own words, what do you think goes wrong if you skip step 2?

---

### Let's Walk Through an Example

A real {domain} problem using **{module.concept}**:

- **Setup:** starting from {prereqs}
- **Step 1:** identify the inputs clearly
- **Step 2:** apply {module.concept} exactly as described above
- **Step 3:** verify the result makes sense for {goal}

---

### Watch Out For These

1. **Skipping the intuition** — jumping straight to formulas without understanding the why leads to fragile knowledge
2. **Shaky prerequisites** — {prereqs} must be solid before {module.concept} clicks
3. **Over-generalising** — always check whether the conditions actually hold before applying

---

### Your Turn

Describe a {domain} situation where {module.concept} is the key tool.
Write: (a) what the problem is, (b) how {module.concept} helps, (c) what the result looks like.

---

### To Recap

- **{module.concept}** = {module.domain_framing}
- How it works: identify → apply → interpret
- Why it matters: direct path toward {goal}

---

### Up Next

Solid on this? Good. The next module builds directly on **{module.concept}** — you'll see it
reappear as a component inside something more powerful.
"""

    def _fallback_deep(self, module, prereqs: str, domain: str, goal: str) -> str:
        """Deep pace: rigorous 12-section treatment with edge cases and harder practice."""
        return f"""# {module.title} — Deep Dive

> **Pace:** Deep — expect formal definitions, edge cases, two examples, and a challenging practice problem.
> **Prerequisites (must be solid):** {prereqs}

---

### 📋 Learning Objectives

By the end you will be able to:
- State the formal definition of **{module.concept}** with precision
- Prove or derive the core rule from first principles
- Identify edge cases and failure modes
- Apply it to both standard and non-trivial {domain} problems
- Explain why it matters for {goal} at a technical level

---

### Let's Start Here

Most people treat **{module.concept}** as a black box — apply it, get an answer, move on.
This lesson breaks the box open. You'll understand *why* it works, *when* it breaks, and
how to use it with real precision in {domain} problems tied to {goal}.

---

### The Big Picture

Before any formalism, sit with this: *{module.domain_framing}*

The goal here is to make **{module.concept}** feel *inevitable* — like something you could
have figured out yourself with enough thought. That's the level of understanding we're after.

---

### The Formal Definition

Now that the intuition is in place, let's be precise. The formal definition specifies:
- **Inputs:** what it operates on
- **Operation:** the exact transformation or relationship it defines
- **Output / guarantee:** what you can conclude after applying it

Building from what you already know ({prereqs}), this is intuition made rigorous.

---

### Where Does This Come From?

Let's trace the rule back to its source:
1. State what you *want* to be true
2. Identify the assumptions that justify it
3. Follow the logic from assumptions to conclusion

At this pace, understanding the *why* is not optional — it's the difference between
fragile memorisation and durable, transferable knowledge.

---

{CHECKPOINT_MARKER}

Hard challenge before the examples: can you think of a situation where {module.concept} would *not* apply? What condition would have to break down for it to fail?

---

### Let's Walk Through an Example — Standard Case

A typical {domain} problem:
- **Setup:** given {prereqs}, apply {module.concept}
- **Step 1:** formalise the inputs precisely
- **Step 2:** apply the rule, showing every intermediate step
- **Step 3:** interpret — does the result satisfy all conditions? Why?
- **Key insight:** which step is the trickiest and why?

---

### Now a Harder One — Edge Case

What happens when one standard condition is weakened?
- Does {module.concept} still apply? In a modified form? Not at all?
- What does this reveal about the *boundaries* of the concept?

Edge cases are where real understanding shows. If you can handle this, you've got it.

---

### Where Students Go Wrong

1. **Condition blindness** — applying {module.concept} without checking the prerequisites hold
2. **Losing precision** — using informal intuition where formal precision is required
3. **Domain mismatch** — a technique valid in one {domain} setting may not transfer elsewhere
4. **Over-indexing on examples** — the rule is general; any one example is just one instance

---

### Your Turn (Challenging)

Design a {domain} problem where {module.concept} is required, but one standard condition
is weakened. Show:
1. Why the straightforward approach fails
2. How you adapt
3. What the result tells you about {goal}

---

### To Recap

- **Definition:** {module.concept} = {module.domain_framing} (built formally on {prereqs})
- **Mechanism:** formalise → apply → verify conditions → interpret
- **Depth:** know when it works AND when it fails
- **Why it matters:** this is a hard prerequisite for the next level of {goal}

---

### How It Connects

- **Builds on:** {prereqs}
- **Enables:** the concepts in the next module
- **Shows up in:** most non-trivial {domain} problems related to {goal}

---

### Up Next

You now have a rigorous handle on **{module.concept}**. The next module assumes this foundation
and uses it as a building block inside something more complex. If anything above felt shaky,
clear it up now — it will matter soon.
"""

        return f"""# {module.title}

---

### 📋 Learning Objectives

By the end of this lesson you will be able to:
- Explain **{module.concept}** in your own words
- Identify where it appears in {domain}
- Apply it to a simple problem related to {goal}

**Prerequisite check:** Make sure you're comfortable with: {prereqs}

---

### 🎯 Why Does This Matter?

Before we define anything, ask yourself: why would someone in {domain} need **{module.concept}**? \
The answer is directly tied to your goal of *{goal}*. Without this concept, certain problems \
in {domain} simply cannot be solved cleanly. That's why it's here.

---

### 🧠 Intuition First

Think of **{module.concept}** not as a rule to memorize but as a *thinking tool*. \
Here's the simplest possible way to picture it:

> *{module.domain_framing}*

That framing is your anchor. Every formal definition we add will just be a more precise \
version of that same idea.

---

### ⚙️ How It Works

Here is the mechanism, broken into steps:

1. **Identify** what you're given or observing in your {domain} problem.
2. **Apply** the core rule or relationship that **{module.concept}** defines.
3. **Interpret** the result — does it make sense given your goal of *{goal}*?

These three steps work for nearly every problem you'll encounter.

---

{CHECKPOINT_MARKER}

Before reading the example: In your own words, what do you think happens if you skip step 2 above? What would go wrong?

---

### 📖 Worked Example

Imagine a real {domain} problem requiring **{module.concept}**:

- **Setup:** You have a situation where {prereqs} has already been applied.
- **Step 1:** Identify the known inputs.
- **Step 2:** Apply the rule of {module.concept} exactly as defined above.
- **Step 3:** Check — does the output align with {goal}?

Walk through this logic with any practice problem and the intuition will build fast.

---

### ⚠️ Common Mistakes

1. **Skipping the intuition** — jumping straight to formulas without understanding the *why* leads to fragile knowledge.
2. **Misapplying prerequisites** — {prereqs} must be solid before {module.concept} makes sense.
3. **Over-generalizing** — {module.concept} applies in specific contexts; always check whether the conditions hold.

---

### 📝 Practice Task

Try this now:

*Describe a situation in {domain} where {module.concept} would be the key tool. Write 2-3 sentences explaining: (a) what the problem is, (b) how {module.concept} helps, and (c) what the result looks like.*

---

### ✅ Summary

- **{module.concept}** = {module.domain_framing}
- It works by: identify → apply → interpret
- It matters because it is a direct stepping stone toward {goal}

---

### 🔜 What Comes Next

You've built the foundation. The next module will take this further — you'll see **{module.concept}** appear as a building block in something more complex. Make sure you can state the core idea clearly before moving on.
"""

    # ── Tool executor ──────────────────────────────────────────────────────────

    async def _execute_tool(self, tool_name: str, args: dict) -> str:
        """Run tutor tool calls that retrieve context, deliver lessons, or handle doubts."""
        if tool_name == "retrieve_content":
            # Retrieval for the legacy session flow was removed together with the
            # old ChromaDB/Tavily pipeline (core/rag_pipeline.py). The live
            # /api/courses flow retrieves via clients/mcp_search_client.py instead.
            # The tool is kept so the tutor's agent loop is unchanged; it now
            # always reports no chunks, exactly as the disabled pipeline did.
            self._retrieved_chunks = []
            return "No content found. Use your training knowledge to write the lesson."

        if tool_name == "deliver_lesson":
            lesson_text = args.get("lesson_text", "")
            checkpoint_question = args.get("checkpoint_question", "")
            practice_task = args.get("practice_task", "")
            style_used = args.get("style_used", "formal")

            if not lesson_text.strip():
                logger.warning("deliver_lesson: empty lesson_text")
                return "Lesson delivery failed: empty lesson_text"

            module = self._current_module()

            async def _stream_text(text: str) -> None:
                """Emit lesson content paragraph-by-paragraph for smooth frontend rendering."""
                paragraphs = text.split("\n\n")
                for para in paragraphs:
                    if para.strip():
                        await self._emit(para)
                    else:
                        await self._emit("")   # preserve blank lines
                    await asyncio.sleep(0)     # yield to event loop between paragraphs

            # Split at checkpoint marker to create interactive pause
            if CHECKPOINT_MARKER in lesson_text and checkpoint_question:
                parts = lesson_text.split(CHECKPOINT_MARKER, 1)
                pre = parts[0].strip()
                post = (CHECKPOINT_MARKER + parts[1]).strip() if len(parts) > 1 else ""

                await self._emit("\n" + "━" * 60)
                await _stream_text(pre)
                if module:
                    self.state.record_module_content(module.id, pre)
                self._delivered_lesson = True

                await self._run_checkpoint(checkpoint_question, module)

                if post:
                    await _stream_text(post)
                    if module:
                        self.state.record_module_content(module.id, post)
                await self._emit("━" * 60 + "\n")
            else:
                await self._emit("\n" + "━" * 60)
                await _stream_text(lesson_text)
                await self._emit("━" * 60 + "\n")
                if module:
                    self.state.record_module_content(module.id, lesson_text)
                self._delivered_lesson = True

            if (
                module is not None
                and hasattr(self, "_eval_runner")
                and self._eval_runner is not None
            ):
                asyncio.create_task(
                    self._eval_runner.on_lesson_delivered(
                        lesson_text=lesson_text,
                        retrieved_chunks=self._retrieved_chunks,
                        module={
                            "title": module.title,
                            "concept": module.concept,
                            "depth_level": module.depth_level,
                            "estimated_minutes": module.estimated_minutes,
                        },
                    )
                )

            # Practice task
            if practice_task.strip():
                await self._run_practice(practice_task, module)

            # Post-lesson doubt capture
            doubt_count = await self._capture_doubt(module)
            return f"Lesson delivered ({style_used}). {'Doubt handled.' if doubt_count else 'No doubts.'}"

        if tool_name == "handle_doubt":
            doubt_text = args["doubt_text"]
            doubt_type = args.get("doubt_type", "general")
            response = args["response"]
            module = self._current_module()
            concept = module.concept if module else "unknown"
            self.state.log_doubt(concept, doubt_type)
            if module:
                self.state.record_module_content(
                    module.id, f"Doubt: {doubt_text}\nResponse: {response}"
                )
            await self._emit(f"💡 {response}")
            return f"Doubt handled: '{doubt_text[:50]}'"

        return await super()._execute_tool(tool_name, args)

    # ── System prompt builder ──────────────────────────────────────────────────

    def _build_system_prompt(self, module, style: str, prior_doubts: int) -> str:
        """Build the detailed teaching prompt for the current module and learner state."""
        prereqs = ", ".join(module.prerequisites) if module.prerequisites else "none"
        meta = self.state.metacognition

        previously_taught_concepts = []
        upcoming_concepts = []
        if self.state.curriculum:
            idx = self.state.curriculum.current_index
            for i, m in enumerate(self.state.curriculum.modules):
                if i < idx:
                    previously_taught_concepts.extend(m.concepts_taught or [m.concept])
                elif i > idx:
                    upcoming_concepts.extend(m.concepts_taught or [m.concept])

        calibration_note = {
            "overconfident": "Student tends to overestimate mastery — use trickier examples and edge cases.",
            "underconfident": "Student underestimates themselves — be extra reassuring, celebrate small wins.",
            "calibrated": "",
        }.get(meta.calibration_pattern, "")

        pace = self.state.pace
        prereqs_display = prereqs if prereqs else "No prerequisites — this is a starting point."
        not_cover_display = ", ".join(module.what_this_module_will_not_cover) if module.what_this_module_will_not_cover else "None specified"

        # ── Pace governs length, depth, sections, tone ────────────────────────
        # IMPORTANT: These are MINIMUM content requirements, not maximums.
        # Writing less than these minimums is a quality failure.
        pace_rules = {
            "fast": (
                "FAST PACE — crisp, direct, zero wasted words. The student is in a hurry but must learn everything.\n"
                "FORMAT: Use bullet points everywhere possible. Short sentences. Bold key terms inline.\n"
                "CONTENT REQUIREMENTS (MINIMUMS — do not write less):\n"
                "  • Total lesson: 400-550 words.\n"
                "  • Structure: Title → Objectives (2 bullet points) → Hook (2 sentences max) →\n"
                "    Core Concept (bullet-point breakdown, 4-6 bullets with bold terms) →\n"
                "    Think Moment (1 question) → Worked Example (numbered steps, no prose padding) →\n"
                "    Key Takeaway (3 bullet points maximum).\n"
                "  • Every bullet point teaches something concrete — no bullets like 'This is important because it helps you learn.'\n"
                "  • Core Concept bullets: each bullet = one fact/rule/definition, stated directly.\n"
                "    Example bullet: '**let** declares a block-scoped variable; **var** is function-scoped and hoists.'\n"
                "  • Worked Example: numbered steps only, each ≤ 2 lines. No narrative padding around the example.\n"
                "  • Think Moment: one sharp, specific question (reference a real value or step from the example).\n"
                "  • SKIP: long prose paragraphs, historical background, common mistakes section, practice task, derivations.\n"
                "  • Tone: direct, like a coach talking to someone with 10 minutes before a test. No 'great question!' filler."
            ),
            "medium": (
                "MEDIUM PACE — textbook quality. What a school topper reads. What a good blog post delivers.\n"
                "FORMAT: Flowing prose with occasional bullet points for lists/steps. Like a well-written textbook chapter.\n"
                "CONTENT REQUIREMENTS (MINIMUMS):\n"
                "  • Total lesson: 700-950 words of actual teaching content.\n"
                "  • All sections must be present and substantive — no one-line sections.\n"
                "  • Hook: 1 surprising fact or real-world connection that creates genuine curiosity (2-3 sentences).\n"
                "  • Intuition / Big Picture: 3 solid paragraphs. Real analogy. Build understanding before formal definition.\n"
                "  • Core Explanation: formal definition + mechanism, with at least 2 inline examples embedded in prose.\n"
                "  • Think Moment: reference something specific just taught (a value, step, or scenario).\n"
                "  • Worked Example: complete specific scenario, every step with 'because' reasoning.\n"
                "  • Common Mistakes: 2-3 specific real mistakes students make — not generic warnings.\n"
                "  • Practice Task: one concrete solvable problem using only what was just taught.\n"
                "  • Tone: conversational and encouraging — like a knowledgeable friend, not a textbook robot."
            ),
            "deep": (
                "DEEP PACE — advanced, exhaustive, researcher-grade. The student wants to know everything.\n"
                "FORMAT: Dense prose with structured sections. Use headers, subheaders where helpful. Tables where comparisons exist.\n"
                "CONTENT REQUIREMENTS (HARD MINIMUMS — writing less is a quality failure):\n"
                "  • Total lesson: MINIMUM 1200 words. Target 1400-1800 words. Do NOT stop at 1000.\n"
                "  • Sections required: Objectives → Hook → Big Picture (4 paragraphs) → Formal Definition →\n"
                "    Deep Explanation (mechanism, derivation/proof sketch, edge cases) →\n"
                "    Think Moment → Worked Example 1 (standard) → Worked Example 2 (edge case or harder variant) →\n"
                "    Common Mistakes (3-4, including one expert-level pitfall) →\n"
                "    Connections (how this concept links to past and future topics) →\n"
                "    Practice Task (challenging, requires synthesis) → Summary.\n"
                "  • Big Picture: 4 paragraphs. Build from first principles. Multiple analogies. WHY this concept matters deeply.\n"
                "  • Formal Definition: precise, complete, every term defined. If there is a formula, derive it step by step.\n"
                "  • Deep Explanation: go beyond the surface. Explain the mechanism at a level most teachers skip.\n"
                "    For programming: explain what the compiler/runtime does. For physics: explain the physics behind the formula.\n"
                "    For history: explain the underlying social/economic forces. For math: give the proof sketch.\n"
                "  • Worked Example 1: standard case, every step shown, full reasoning at each step.\n"
                "  • Worked Example 2: a harder variant that tests edge cases or pushes understanding further.\n"
                "  • Common Mistakes: 3-4 mistakes. At least one must be subtle — the kind experts debate.\n"
                "  • Connections: explicitly link this concept to 2-3 things already learned and 1-2 things coming later.\n"
                "  • Think Moment: a genuinely hard question — not answerable by rote recall; requires reasoning.\n"
                "  • Practice Task: requires combining what was just taught with something previously learned.\n"
                "  • Tone: precise and rigorous, but still human. Write like a researcher who genuinely loves explaining."
            ),
        }.get(pace, "")

        depth_note = pace_rules  # depth_level mirrors pace; use the richer pace_rules

        style_guide = {
            "formal": (
                "FORMAL: Start with a crisp definition. Build theory layer by layer. "
                "Use precise language. Ground every claim with a domain example. "
                "Numbered lists for steps, prose for explanations."
            ),
            "analogy": (
                "ANALOGY: Open with a memorable real-world comparison. "
                "Build the ENTIRE lesson around that analogy. Extend it, stress-test it. "
                "Only introduce formal terms AFTER the analogy has clicked."
            ),
            "example_first": (
                "EXAMPLE-FIRST: Start with a complete worked example — NO preamble, no definition. "
                "Let the student experience the concept before naming it. "
                "After the example: 'Notice what just happened? That is called [concept].'"
            ),
            "visual": (
                "VISUAL: Describe everything as if drawing on a whiteboard. "
                "'Imagine a graph where...', 'Draw an arrow from X to Y...', 'Picture a table...'. "
                "Student should be able to sketch the concept after reading."
            ),
            "story": (
                "STORY: Create a short narrative. A researcher/engineer/student faces a real problem, "
                "tries to solve it, hits a wall, discovers this concept, and succeeds. "
                "Weave ALL the theory into the story naturally."
            ),
        }.get(style, "")

        prior_note = f"Student had {prior_doubts} prior doubts on this concept — pre-empt those confusion points." if prior_doubts > 0 else ""

        return f"""You are an expert home tutor delivering a lesson to a student one-on-one.

Your job is to TEACH, not to describe what you will teach.
Write like you're sitting next to the student, explaining things clearly and enthusiastically.
Every paragraph must contain real information — never write a sentence that is just meta-commentary
like "This section explains thermodynamics" or "Understanding this is important."

IMAGINE: A great home tutor who knows this subject deeply is explaining it fresh.
They use examples. They build intuition before theory. They check understanding.
They connect everything to why the student is learning this.
That is the voice and quality standard for this lesson.

══════════════════════════════════════════════════════
STUDENT CONTEXT:
{self._student_context()}

CURRENT MODULE:
  Title        : {module.title}
  Concept      : {module.concept}
  Domain       : {self.state.domain}
  Goal         : {self.state.goal}
  Framing      : {module.domain_framing}
  Pace         : {pace}
  Duration     : {module.estimated_minutes} min
  Prerequisites: {prereqs_display}
  Do NOT Cover : {not_cover_display}

ROADMAP BOUNDARIES:
  Previously Taught: {", ".join(previously_taught_concepts) if previously_taught_concepts else "None (this is the first module)"}
  Coming Later     : {", ".join(upcoming_concepts) if upcoming_concepts else "None"}

ADAPTIVE NOTES:
  {calibration_note}
  {prior_note}
══════════════════════════════════════════════════════

PACE REQUIREMENTS FOR THIS LESSON:
{depth_note}

══════════════════════════════════════════════════════
STRICT CONTENT RULES — violations mean the lesson has failed:

1. NO STUB CONTENT: Every section must contain actual teaching.
   BAD: "Thermodynamics is the study of energy and its interactions with matter."
   GOOD: "Thermodynamics answers one question: where does energy go? Pour hot coffee into
   a cold cup — the coffee cools and the cup warms. Energy moved. Thermodynamics is the
   science that tracks exactly how, why, and how much energy moved, and what limits that process."

2. CHECKPOINT QUESTION MUST COME FROM YOUR LESSON:
   After writing your lesson, find the '### ✏️ Think Moment' section.
   The checkpoint_question field in your tool call = COPY of that exact question.
   The question must reference something SPECIFIC you taught (a value, a scenario, a step)
   — NOT a generic question about the topic that any student could answer without reading.

3. NO "null" OR "none" FOR PREREQUISITES:
   If there are no prerequisites, write: "No prerequisites — this is a starting point."
   If there are prerequisites, name them: "Make sure you're comfortable with: [list]"

4. EXAMPLE MUST BE CONCRETE:
   Use specific scenarios, real numbers, domain-specific situations.
   BAD: "Step 1: identify inputs. Step 2: apply the rule."
   GOOD: "A gas in a piston at 300K expands until its volume doubles. Step 1: we identify
   this as an isothermal process (temperature stays constant at 300K). Step 2: we use
   Boyle's Law: P₁V₁ = P₂V₂. If initial pressure was 2 atm and volume doubled, then
   P₂ = (2 × V₁) / (2V₁) = 1 atm."

5. DO NOT JUMP AHEAD:
   Look closely at the "Coming Later" and "Do NOT Cover" lists in the boundaries above.
   You are FORBIDDEN from explaining or using these concepts in this lesson. 
   If you absolutely must mention them to frame the current topic, label them clearly as "(coming later)".
   Teach ONLY the current concept using ONLY the previously taught concepts.
══════════════════════════════════════════════════════

LESSON STRUCTURE ({pace.upper()} PACE):

Here is the exact heading text to use for each section. Use these headings word-for-word
so the lesson reads like a real tutor wrote it — not like a template was filled in.

## [Lesson Title] [⚡ if fast pace | nothing if medium | 🔬 if deep pace]

### What You'll Take Away
2-3 specific, testable objectives written as "You'll be able to..."
One line for prerequisites: {prereqs_display}

### Let's Start Here
Open with something that creates genuine curiosity — a surprising fact, a real-world
failure, or a question the student can't yet answer. Do NOT start with "In this lesson..."
Write as if you're speaking directly to the student.

### The Big Picture
Explain the concept in plain English before any formalism.
Use the framing: "{module.domain_framing}"
A good analogy or vivid image. At least 2 solid paragraphs. Conversational tone.

### Breaking It Down (skip for fast pace)
Now introduce the formal definition and mechanism, layer by layer.
Explain the WHY at every step. Use inline examples. Flowing prose, not equations alone.

### ✏️ Think Moment
[THIS EXACT HEADING triggers the interactive pause — copy it exactly]
Ask ONE prediction question based on something SPECIFIC just taught above.
The question must name a specific value, scenario, or step from THIS lesson.
End the section with just the question. Do NOT answer it here.

### Let's Walk Through an Example
A specific, realistic {self.state.domain} scenario with real details.
Format each step as:
  **Step [N] — [short name]:** [what you do] → [result] *(because: [reason])*
Show every step. Explain the reasoning at each one.

### Watch Out For These (skip for fast pace)
Real mistakes students make — specific, not generic warnings.
Format: **The mistake:** [what goes wrong] → **The fix:** [what to do instead]

### Your Turn (skip for fast pace)
One concrete problem the student solves right now using only what was taught.
Clear setup. Clear ask. Specific constraints.

### To Recap
3 bullet points: what it is, how it works, why it matters for {self.state.goal}.

### Up Next
1-2 sentences bridging to the next module. Hint at it, don't teach it.

══════════════════════════════════════════════════════
STYLE FOR THIS LESSON:
{style_guide}

MODULE BOUNDARY: Teach ONLY this module. Label future topics as "(coming later)".

WORKFLOW:
  1. Call retrieve_content with a specific targeted query
  2. Write the full lesson in your head, then call deliver_lesson
     (checkpoint_question = verbatim copy from your '### ✏️ Think Moment' section)
  3. Call finish_lesson
"""

    # ── Public teach method ────────────────────────────────────────────────────

    async def teach(self) -> dict:
        """Deliver a lesson for the current curriculum module."""
        module = self._current_module()
        if module is None:
            logger.warning("TutorAgent.teach() called with no current module")
            return {"summary": "No module", "style_used": "formal", "doubt_count": 0, "fatigue_detected": "no"}

        meta = self.state.metacognition
        style = meta.preferred_style or "formal"
        prior_doubts = self.state.get_doubt_count(module.concept)

        system = self._build_system_prompt(module, style, prior_doubts)

        try:
            result = await self.run(
                system=system,
                user_message=(
                    f"Deliver a {module.depth_level}-depth lesson on '{module.title}' "
                    f"(concept: '{module.concept}') for a {self.state.pace}-pace student "
                    f"in {self.state.domain}."
                ),
                model=settings.reasoning_model,
            )
        except Exception as exc:
            logger.error("Tutor tool loop failed for '{}': {}", module.concept, exc)
            return {
                "error": "lesson_generation_failed",
                "summary": "",
                "style_used": style,
                "doubt_count": 0,
                "fatigue_detected": "no",
            }

        if not self._delivered_lesson:
            logger.error("Tutor did not deliver lesson content for '{}'", module.concept)
            return {
                "error": "lesson_not_delivered",
                "summary": "",
                "style_used": result.get("style_used", style),
                "doubt_count": 0,
                "fatigue_detected": result.get("fatigue_detected", "no"),
            }

        actual_doubts = max(0, self.state.get_doubt_count(module.concept) - prior_doubts)
        try:
            reported = int(result.get("doubt_count", 0) or 0)
        except (TypeError, ValueError):
            reported = 0
        result["doubt_count"] = max(reported, actual_doubts)

        self._log_decision(
            action="LESSON_DELIVERED",
            reason=f"Taught '{module.concept}' using '{result.get('style_used', style)}' style",
            payload={
                "module_id": module.id,
                "concept": module.concept,
                "style": result.get("style_used", style),
                "doubt_count": result.get("doubt_count", 0),
            },
        )
        return result
