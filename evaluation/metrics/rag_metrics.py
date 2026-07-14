"""Lesson-grounding metric for EduMind evaluation reports.

Only `rag_faithfulness_score` remains here — it is invoked live from
`EvaluationRunner.on_lesson_delivered`. The HyDE / ChromaDB-precision /
Tavily-relevance / reranker-gain metrics were removed together with the
disabled ChromaDB+Tavily retrieval path (core/rag_pipeline.py), which was the
only thing that ever produced their inputs.
"""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

from clients.groq_client import generate as groq_generate
from config import settings
from evaluation.collector import record_metric


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a numeric metric value into the configured score range."""
    return max(low, min(high, float(value)))


def _tokenize(text: str) -> set[str]:
    """Tokenize text for lightweight lexical overlap scoring."""
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9'-]*", (text or "").lower())
        if len(token) > 2
    }


def _json_score(raw: str) -> dict[str, Any]:
    """Parse a compact JSON judge response, falling back to the first score token."""
    try:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if match:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
    except Exception:
        pass
    match = re.search(r"([01](?:\.\d+)?)", raw or "")
    return {"score": float(match.group(1)) if match else 0.0}


async def _llm_judge_score(prompt: str) -> dict[str, Any]:
    """Call the evaluation judge model and parse its compact score JSON."""
    response = await groq_generate(
        messages=[{"role": "user", "content": prompt}],
        model=settings.eval_judge_model,
        system=(
            "You are an evaluation judge. Return compact JSON only. "
            "All scores must be floats from 0.0 to 1.0."
        ),
    )
    return _json_score(response)


def _lexical_faithfulness(lesson_text: str, retrieved_chunks: list[str]) -> float:
    """Estimate lesson support using lexical overlap when judge scoring is unavailable."""
    source_tokens = _tokenize("\n".join(retrieved_chunks))
    if not source_tokens:
        return 0.0
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", lesson_text) if s.strip()]
    claim_like = [s for s in sentences if len(_tokenize(s)) >= 5]
    if not claim_like:
        return 0.0
    checked = claim_like[: settings.eval_faithfulness_claim_limit]
    supported = 0
    overlaps: list[float] = []
    for sentence in checked:
        tokens = _tokenize(sentence)
        overlap = len(tokens & source_tokens) / max(1, len(tokens))
        overlaps.append(overlap)
        if overlap >= 0.35:
            supported += 1
    return supported / max(1, len(checked))


async def rag_faithfulness_score(
    lesson_text: str,
    retrieved_chunks: list[str],
    session_id: str,
    student_id: str,
) -> dict:
    """Score whether lesson claims are supported by retrieved context."""
    metric_name = "rag_faithfulness"
    try:
        context = "\n\n".join(retrieved_chunks)[:6000]
        lesson = lesson_text[:6000]
        if not lesson.strip() or not context.strip():
            score = 0.0
            details = {"reason": "missing_lesson_or_context"}
        else:
            prompt = (
                "Score whether the LESSON is supported by the RETRIEVED CONTEXT.\n"
                f"Check at most {settings.eval_faithfulness_claim_limit} factual claims.\n"
                "Return JSON: {\"score\": float, \"supported_claims\": int, "
                "\"checked_claims\": int, \"notes\": string}.\n\n"
                "RETRIEVED CONTEXT:\n"
                f"{context}\n\nLESSON:\n{lesson}"
            )
            judged = await _llm_judge_score(prompt)
            score = _clamp(float(judged.get("score", 0.0)))
            details = {
                "judge": judged,
                "context_chunks": len(retrieved_chunks),
                "lesson_chars": len(lesson_text),
            }
            if score == 0.0:
                fallback = _lexical_faithfulness(lesson_text, retrieved_chunks)
                details["lexical_fallback"] = fallback
                score = fallback
    except Exception as exc:
        logger.warning("rag_faithfulness_score failed: {}", exc)
        score = 0.0
        details = {"error": str(exc)}

    await record_metric(metric_name, "rag", score, details, session_id, student_id)
    return {"score": score, "details": details}
