"""
agents/institution/test_generation_agent.py

Generates a difficulty-balanced classroom test (MCQ + short answer +
conceptual) grounded in the classroom course's module content and biased
toward the class's current weak concepts.

Follows the schedule_agent pattern: a single JSON-mode reasoning call with a
strict schema, followed by deterministic validation/coercion so a malformed
LLM response can never produce a broken test.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from loguru import logger

from clients.groq_client import generate
from config import settings

_SYSTEM = """
You are EduMind's expert test author for classroom teachers.
Write assessment questions that are precise, unambiguous, and grounded ONLY in
the provided topic and course material. Never invent facts outside the topic.

Rules:
- MCQ questions have exactly 4 options and exactly one correct option.
  "correct_answer" is the index of the correct option as a string: "0".."3".
  Wrong options must be plausible misconceptions, not jokes.
- short_answer questions expect 1-3 sentence answers; put an ideal answer in
  "correct_answer".
- conceptual questions probe understanding/why/what-if; put a model answer in
  "correct_answer".
- Balance difficulty across "easy", "medium", "hard" as requested.
- If weak concepts are listed, include extra questions targeting them.
- Every question lists the concept(s) it tests in "concepts_tested".

Return ONLY valid JSON — no markdown fences, no preamble.
Schema:
{
  "title": "short test title",
  "questions": [
    {
      "question_type": "mcq" | "short_answer" | "conceptual",
      "question_text": "...",
      "options": ["A", "B", "C", "D"],          // [] for non-MCQ
      "correct_answer": "0",                     // index string for MCQ, model answer otherwise
      "explanation": "why the answer is correct",
      "concepts_tested": ["concept"],
      "difficulty": "easy" | "medium" | "hard",
      "points": 1.0
    }
  ]
}
"""


def _coerce_question(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Validate one generated question; return None if unusable."""
    text = str(raw.get("question_text") or "").strip()
    if len(text) < 8:
        return None
    qtype = str(raw.get("question_type") or "mcq").strip().lower()
    if qtype not in ("mcq", "short_answer", "conceptual"):
        qtype = "conceptual"

    options = raw.get("options") or []
    options = [str(o).strip() for o in options if str(o).strip()]
    correct = str(raw.get("correct_answer") or "").strip()

    if qtype == "mcq":
        if len(options) < 2:
            return None
        options = options[:4]
        try:
            idx = int(correct)
        except (TypeError, ValueError):
            # Model sometimes returns the option text — recover by matching it.
            idx = next((i for i, o in enumerate(options) if o == correct), -1)
        if not (0 <= idx < len(options)):
            return None
        correct = str(idx)
    else:
        options = []
        if not correct:
            return None

    difficulty = str(raw.get("difficulty") or "medium").lower()
    if difficulty not in ("easy", "medium", "hard"):
        difficulty = "medium"

    concepts = raw.get("concepts_tested") or []
    if isinstance(concepts, str):
        concepts = [concepts]
    concepts = [str(c).strip() for c in concepts if str(c).strip()][:5]

    try:
        points = max(0.5, min(float(raw.get("points") or 1.0), 5.0))
    except (TypeError, ValueError):
        points = 1.0

    return {
        "id": f"q-{uuid.uuid4().hex[:10]}",
        "question_type": qtype,
        "question_text": text,
        "options": options,
        "correct_answer": correct,
        "explanation": str(raw.get("explanation") or "").strip(),
        "concepts_tested": concepts,
        "difficulty": difficulty,
        "points": points,
    }


async def generate_test_questions(
    *,
    topic: str,
    subject: str = "",
    grade_level: str = "",
    num_mcq: int = 5,
    num_short: int = 3,
    num_conceptual: int = 2,
    difficulty_mix: str = "balanced",   # easy_heavy | balanced | hard_heavy
    course_context: str = "",
    weak_concepts: list[str] | None = None,
    extra_instructions: str = "",
) -> dict[str, Any]:
    """
    Generate a validated test. Returns {"title": str, "questions": [...]}.
    Raises ValueError when the model cannot produce enough usable questions.
    """
    num_mcq = max(0, min(int(num_mcq), 20))
    num_short = max(0, min(int(num_short), 10))
    num_conceptual = max(0, min(int(num_conceptual), 10))
    total = num_mcq + num_short + num_conceptual
    if total == 0:
        raise ValueError("Test must request at least one question")

    weak = [str(c) for c in (weak_concepts or [])][:8]
    prompt_parts = [
        f"Topic: {topic}",
        f"Subject: {subject}" if subject else "",
        f"Grade level: {grade_level}" if grade_level else "",
        f"Generate exactly {num_mcq} mcq, {num_short} short_answer, "
        f"{num_conceptual} conceptual questions ({total} total).",
        f"Difficulty mix: {difficulty_mix}.",
        f"Class weak concepts to emphasise: {', '.join(weak)}" if weak else "",
        f"Additional teacher instructions: {extra_instructions}" if extra_instructions else "",
    ]
    if course_context:
        prompt_parts.append(
            "Ground questions in this course material:\n" + course_context[:6000]
        )
    prompt = "\n".join(p for p in prompt_parts if p)

    raw = await generate(
        messages=[{"role": "user", "content": prompt}],
        model=settings.reasoning_model,
        system=_SYSTEM,
        json_mode=True,
        max_tokens=6000,
        _caller="test_generation_agent",
    )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Test generation returned invalid JSON: {}", exc)
        raise ValueError("The AI returned an invalid test. Please regenerate.") from exc

    questions = []
    for item in data.get("questions") or []:
        if isinstance(item, dict):
            coerced = _coerce_question(item)
            if coerced:
                questions.append(coerced)

    if len(questions) < max(1, total // 2):
        raise ValueError(
            f"Only {len(questions)} usable questions were generated. Please regenerate."
        )

    logger.info("Test generation: {} usable questions for topic '{}'", len(questions), topic)
    return {
        "title": str(data.get("title") or f"{topic} Test").strip(),
        "questions": questions,
    }


async def regenerate_single_question(
    *,
    topic: str,
    question_type: str,
    difficulty: str,
    course_context: str = "",
    avoid_texts: list[str] | None = None,
) -> dict[str, Any]:
    """Regenerate one question of a given type/difficulty, avoiding duplicates."""
    avoid = "\n".join(f"- {t}" for t in (avoid_texts or [])[:15])
    prompt = (
        f"Topic: {topic}\n"
        f"Generate exactly 1 {question_type} question at {difficulty} difficulty.\n"
        + (f"Do NOT repeat any of these existing questions:\n{avoid}\n" if avoid else "")
        + (f"Course material:\n{course_context[:3000]}" if course_context else "")
    )
    raw = await generate(
        messages=[{"role": "user", "content": prompt}],
        model=settings.reasoning_model,
        system=_SYSTEM,
        json_mode=True,
        max_tokens=1200,
        _caller="test_generation_agent",
    )
    data = json.loads(raw)
    items = data.get("questions") or []
    for item in items:
        coerced = _coerce_question(item) if isinstance(item, dict) else None
        if coerced:
            return coerced
    raise ValueError("Could not regenerate a usable question. Please try again.")
