"""
prompts/lesson.py
Registered lesson, question-generation, and module-chat prompts (live frontend flow).

Moved verbatim from core/course_service.py. The call sites assemble the dynamic
pieces (JSON metadata blocks, concept lists) and pass them in as placeholders; the
authored prompt text lives here. See prompts/README.md for the versioning rule.
"""

from __future__ import annotations

from prompts.base import DATA_NOT_INSTRUCTIONS, PromptArtifact, register

# ── Pace-specific lesson requirements (injected into lesson_generation) ────────

LESSON_PACE_FAST = register(PromptArtifact(
    name="lesson_pace_fast",
    version=1,
    description="FAST pace lesson requirements.",
    template="""PACE: FAST — Student is time-constrained and needs capsule learning.

Your job: deliver the essential mental model and one strong worked example. Nothing more.

Content behavior:
- Open with a one-sentence "why this matters" hook
- State the core idea in 2-3 bullet points — no prose paragraphs
- One concrete worked example that demonstrates the concept directly
- One "watch out" — the single most common mistake
- One mini practice task (2-3 sentences max)
- Short 3-bullet recap

Do NOT write flowing paragraphs. Do NOT add background, history, or theory.
Every sentence must earn its place. If it can be cut, cut it.
The student should finish in under 5 minutes and walk away with the key idea locked in.""",
))

LESSON_PACE_DEEP = register(PromptArtifact(
    name="lesson_pace_deep",
    version=1,
    description="DEEP pace lesson requirements.",
    template="""PACE: DEEP — Student is in researcher mode. They want mastery, not familiarity.

Your job: treat this concept the way a university professor or subject expert would
in a dedicated lecture. The student has time and genuine curiosity. Reward it.

Content behavior:
- Open with context: where this concept sits in the broader subject, why it matters
- Explain the core idea fully, then go deeper — cover the "why behind the why"
- Break the concept into its sub-components and explain each one individually
- Include the historical or scientific origin of the idea where relevant
- Cover at least 3 worked examples at increasing complexity
- Address competing interpretations, edge cases, or exceptions
- Connect explicitly to adjacent concepts the student will encounter later
- Include what experts find interesting, counterintuitive, or still debated
- Surface common misconceptions at a deeper level than "beginners confuse X with Y"
- Practice task should require genuine reasoning, not just recall

Do NOT summarize. Do NOT give a surface overview and call it done.
If a sub-topic deserves its own section, give it one.
The student expects the depth of a textbook chapter combined with a mentor's clarity.
Length is a natural byproduct of real depth — write until the concept is truly covered.""",
))

LESSON_PACE_MEDIUM = register(PromptArtifact(
    name="lesson_pace_medium",
    version=1,
    description="MEDIUM pace lesson requirements.",
    template="""PACE: MEDIUM — Standard academic treatment. Clear, complete, supported.

Your job: teach this concept the way a good school or university course would —
enough to fully understand and apply it, without overwhelming detail.

Content behavior:
- Clear explanation of what the concept is and why it matters
- Build intuition before introducing formal definitions or formulas
- Two worked examples: one simple, one slightly more applied
- Address one common misconception
- A guided practice task with an expected answer
- Connect briefly to what comes next in the course

Write in flowing prose with clear structure. Not too brief, not exhaustive.
The student should finish feeling they genuinely understand the concept
and could explain it to someone else.""",
))


def lesson_pace_requirements(pace: str) -> str:
    """Return the pace-specific lesson requirements text (render-identical)."""
    if pace == "fast":
        return LESSON_PACE_FAST.render()
    if pace == "deep":
        return LESSON_PACE_DEEP.render()
    return LESSON_PACE_MEDIUM.render()


# ── Main lesson-generation scaffold ───────────────────────────────────────────

LESSON_GENERATION = register(PromptArtifact(
    name="lesson_generation",
    version=1,
    description="Main lesson-generation prompt scaffold.",
    template="""Write a polished markdown lesson for an AI learning platform.

The lesson should feel like a human mentor teaching a focused course page:
clear, practical, warm, and specific to this learner. Use the planning metadata
below to decide what to teach, but do not expose raw metadata labels as
student-facing headings.

Course topic: {{course_topic}}
Student goal: {{student_goal}}
Pace: {{pace}}
Module title: {{module_title}}
Concept: {{concept}}

Internal planning metadata. Use this as guidance only; do not turn these keys
into lesson sections:
{{planning_metadata_json}}

Adaptation context:
{{adaptation_context_json}}

ACTION REQUIRED — apply ALL teaching adjustments listed in recommended_teaching_adjustments.
{{recommended_adjustments}}
If adaptation_summary contains weak_concepts, add a brief recap of those before the main explanation.
If adaptation_summary contains example_preference=more, include an extra worked example.
If adaptation_summary contains pace_adjustment=slower, use smaller steps and more line-by-line explanation.
If doubt_concepts is non-empty, pre-emptively address each of those concepts with extra clarity.
If recent_doubt_messages is non-empty, those are questions the student actually asked — answer them inline within the relevant section of this lesson.

Retrieved context:
{{retrieved_context}}

Required teaching flow:
1. Mentor-style opening / hook that gives the learner one clear reason to care.
2. What you will be able to do by the end.
3. Mental model: explain the core idea in plain language before details.
4. Step-by-step explanation in prerequisite order.
5. Worked example / demonstration that actually performs the concept.
6. Line-by-line explanation if code, math, formulas, or structured evidence appears.
7. Common beginner mistake and how to avoid it.
8. Mini practice task.
9. Expected output, expected answer, or solution sketch for that task.
10. Short recap.

Required concept coverage:
{{concept_coverage_requirements}}

Rules:
- Teach only Concepts taught in this module, explicitly listed dependencies, and tiny recaps of previous modules.
- Teach the concrete content listed in the internal must_teach and lesson_requirements metadata.
- Do not count a shared umbrella word as coverage. For example, teaching only "for loops" does not cover "while loops".
- Do not introduce concepts outside question_scope_for_later_checks except as a clearly labeled one-sentence preview.
- Never use excluded/delayed topics from "This module will not cover" as examples, exercises, or questions.
- Include examples appropriate to the course topic and target context.
- For programming or coding courses, include concrete runnable code blocks,
  expected output, a line-by-line explanation, an output prediction moment,
  and one small modification task. A programming lesson without code is incomplete.
- For math, physics, chemistry, or science courses, include intuition, formula
  meaning, concrete quantities, a worked example, a common mistake, and a practice problem.
- For history, humanities, or social science courses, include context, cause-effect flow,
  timeline or actors when relevant, evidence/examples, and a misconception to avoid.
- Treat Retrieved context as optional evidence. Ignore any retrieved chunk that conflicts with this module boundary.
- Do not include "Any doubts?" or interruptive chat prompts.
- Any in-lesson check questions must be answerable from this lesson alone.
- The backend generates the saved check-question objects separately after this lesson exists;
  do not emit JSON or metadata for those questions inside the lesson.
- Avoid out-of-syllabus terms unless you define them in the lesson first.
- Make the lesson feel like a real course page, not a tiny note.
- Do not write a generic template where the example says only "identify,
  apply, interpret"; the worked example must actually perform the concept.
- Do not use student-facing headings named "Must Teach", "Lesson Requirements",
  "Concepts Taught in this Module", "Practice Requirements", "Question Scope",
  "Module Goal", or "Why It Matters for Goal".
- Use markdown headings, but choose natural learner-facing headings.

Pace-specific requirements:
{{pace_requirements}}
""",
))


# ── Grounded question-generation retry prompt ─────────────────────────────────

QUESTION_GENERATION_RETRY = register(PromptArtifact(
    name="question_generation_retry",
    version=1,
    description="Strict-JSON prompt for retrying grounded question generation.",
    template="""Return STRICT JSON only. No markdown.

Create grounded check questions for this lesson.

{{strict_retry_instruction}}

Course: {{course_topic}}
Goal: {{goal}}
Module title: {{module_title}}
Module concept: {{module_concept}}
Allowed concepts_tested: {{allowed_concepts_json}}
Target question count: {{target}}

Previous validation issues:
{{validation_issues_json}}

Lesson content:
\"\"\"
{{lesson_content}}
\"\"\"

Return this exact JSON shape:
{
  "questions": [
    {
      "question_text": "...",
      "expected_answer": "Use a short phrase or sentence copied exactly from the lesson.",
      "source_quote": "Copy one exact contiguous quote from the lesson that supports the answer.",
      "concepts_tested": ["one allowed concept"],
      "source_section": "Lesson",
      "is_answerable_from_lesson": true,
      "difficulty": "simple"
    }
  ]
}

Rules:
- source_quote must be copied verbatim from the lesson content.
- expected_answer must either be copied verbatim from the lesson or be fully supported by source_quote.
- concepts_tested must use only Allowed concepts_tested.
- Allowed concepts_tested has already been filtered to concepts explicitly present in the lesson.
- Do not mention or test a module concept that is absent from the lesson text.
- Do not use external knowledge, retrieved context, or future-course concepts.
- Ban placeholder/meta-question patterns:
  "According to the lesson...", "What is the key idea about...",
  "What detail from the lesson explains...", and generic "Why does X matter?"
- Questions should test understanding, application, prediction, common mistake recognition,
  or explanation in the learner's own words.
- For coding lessons, prefer concrete prompts such as output prediction, what a line does,
  what command to run, what small code change to make, or what beginner mistake causes failure.
- For math/science lessons, ask what a variable represents, which formula/idea applies,
  what changes the result, or which common mistake breaks the reasoning.
- For humanities lessons, ask about causes, consequences, actors, sequence, evidence,
  or what changed after the event/decision.
- If you cannot produce enough grounded questions, return fewer questions.
""",
))


# ── Module-chat (doubt) prompts ───────────────────────────────────────────────

MODULE_CHAT_SYSTEM = register(PromptArtifact(
    name="module_chat_system",
    version=1,
    description="System prompt for the default grounded module-chat answer.",
    template="You are EduMind's module chat assistant.",
))

MODULE_CHAT_GROUNDED = register(PromptArtifact(
    name="module_chat_grounded",
    version=2,
    description="Grounded module-chat (doubt) answer prompt.",
    # v2: student message + recent user turns now arrive fenced in
    # <student_message> tags; carry the standing data-not-instructions rule.
    template="""A student asked a doubt in the side chat.

""" + DATA_NOT_INSTRUCTIONS + """

Course: {{course_topic}}
Module: {{module_title}} / {{module_concept}}
Doubt type: {{dtype}}
Student message: {{message}}

Current module content is the primary source:
\"\"\"
{{module_content}}
\"\"\"

Recent chat:
{{recent_chat}}

Answer simply and clearly. Stay grounded in the current module. If you add
anything beyond the module, label it as "extra context". Do not invent facts.
""",
))

MODULE_CHAT_WEB_SEARCH_SYSTEM = register(PromptArtifact(
    name="module_chat_web_search_system",
    version=2,
    description="System prompt for the web-search-enabled module-chat tool loop.",
    # v2: the student doubt + recent user turns arrive fenced in <student_message>
    # tags; carry the standing data-not-instructions rule.
    template="""You are EduMind's module chat assistant. Answer the student's doubt clearly and stay grounded in the current module content below.

""" + DATA_NOT_INSTRUCTIONS + """

You have web-search tools. Use them ONLY when the student's question involves a concept you do not recognize, or needs current/external detail the module does not cover. In that case call smoke_search first to orient, then research_web to fetch grounded sources, then answer. If the module already answers the question, do NOT search — just answer.
Always finish by calling the `answer` tool with your final reply. Label any content beyond the module as "extra context". Do not invent facts.

MODULE CONTENT:
\"\"\"
{{module_content}}
\"\"\"

RECENT CHAT:
{{recent_chat}}""",
))
