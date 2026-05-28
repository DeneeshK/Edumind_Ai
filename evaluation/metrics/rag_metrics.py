from __future__ import annotations

import asyncio
import json
import math
import re
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from clients.groq_client import generate as groq_generate
from config import settings
from evaluation.collector import record_metric

_embed_model = None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9'-]*", (text or "").lower())
        if len(token) > 2
    }


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading eval embed model '{}'...", settings.eval_embed_model)
        _embed_model = SentenceTransformer(settings.eval_embed_model, device="cpu")
    return _embed_model


async def _embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_embed_model()
    vectors = await asyncio.to_thread(
        lambda: model.encode(texts, normalize_embeddings=True)
    )
    return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return _clamp((dot / (norm_a * norm_b) + 1.0) / 2.0)


async def _semantic_similarity(left: str, right: str) -> float:
    if not left.strip() or not right.strip():
        return 0.0
    vectors = await _embed_texts([left, right])
    return _cosine(vectors[0], vectors[1])


async def _similarities(query: str, texts: list[str]) -> list[float]:
    clean = [text for text in texts if text and text.strip()]
    if not query.strip() or not clean:
        return []
    vectors = await _embed_texts([query] + clean)
    query_vec = vectors[0]
    return [_cosine(query_vec, vec) for vec in vectors[1:]]


def _json_score(raw: str) -> dict[str, Any]:
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
    response = await groq_generate(
        messages=[{"role": "user", "content": prompt}],
        model=settings.eval_judge_model,
        system=(
            "You are an evaluation judge. Return compact JSON only. "
            "All scores must be floats from 0.0 to 1.0."
        ),
    )
    return _json_score(response)


async def hyde_quality_score(
    original_query: str,
    hypothetical_answer: str,
    concept_card_text: str,
    session_id: str,
    student_id: str,
) -> dict:
    metric_name = "hyde_quality"
    try:
        query_alignment = await _semantic_similarity(original_query, hypothetical_answer)
        card_alignment = (
            await _semantic_similarity(hypothetical_answer, concept_card_text)
            if concept_card_text.strip()
            else query_alignment
        )
        length_words = len(hypothetical_answer.split())
        length_score = _clamp(length_words / 90.0) if length_words else 0.0
        score = _clamp(0.45 * query_alignment + 0.45 * card_alignment + 0.10 * length_score)
        details = {
            "query_alignment": query_alignment,
            "concept_card_alignment": card_alignment,
            "length_words": length_words,
            "length_score": length_score,
            "has_concept_card": bool(concept_card_text.strip()),
        }
    except Exception as exc:
        logger.warning("hyde_quality_score failed: {}", exc)
        score = 0.0
        details = {"error": str(exc)}

    await record_metric(metric_name, "rag", score, details, session_id, student_id)
    return {"score": score, "details": details}


async def chromadb_precision_at_k(
    concept: str,
    retrieved_chunks: list[str],
    session_id: str,
    student_id: str,
) -> dict:
    metric_name = "chromadb_precision_at_k"
    try:
        k = max(1, int(settings.eval_precision_k))
        chunks = [c for c in retrieved_chunks[:k] if c and c.strip()]
        sims = await _similarities(concept, chunks)
        relevant = [sim for sim in sims if sim >= 0.55]
        score = len(relevant) / max(1, min(k, len(chunks) or k))
        details = {
            "k": k,
            "chunks_seen": len(chunks),
            "relevance_threshold": 0.55,
            "similarities": [round(s, 6) for s in sims],
            "relevant_count": len(relevant),
        }
    except Exception as exc:
        logger.warning("chromadb_precision_at_k failed: {}", exc)
        score = 0.0
        details = {"error": str(exc)}

    await record_metric(metric_name, "rag", score, details, session_id, student_id)
    return {"score": score, "details": details}


def _parse_result_date(result: dict) -> datetime | None:
    for key in ("published_date", "published_at", "date", "created_at"):
        value = result.get(key)
        if not value:
            continue
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def _freshness(result: dict) -> float:
    parsed = _parse_result_date(result)
    if parsed is None:
        return 0.6
    age_days = max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds() / 86400.0)
    return _clamp(math.exp(-age_days / 365.0))


async def tavily_relevance_score(
    concept: str,
    tavily_results: list[dict],
    session_id: str,
    student_id: str,
) -> dict:
    metric_name = "tavily_relevance"
    try:
        texts = [
            ((r.get("title", "") + "\n" + r.get("content", "")).strip())
            for r in tavily_results
            if isinstance(r, dict)
        ]
        sims = await _similarities(concept, texts)
        relevance = sum(sims) / len(sims) if sims else 0.0
        freshness_scores = [_freshness(r) for r in tavily_results if isinstance(r, dict)]
        freshness = sum(freshness_scores) / len(freshness_scores) if freshness_scores else 0.0
        score = _clamp(0.7 * relevance + 0.3 * freshness)
        details = {
            "results_seen": len(tavily_results),
            "relevance": relevance,
            "freshness": freshness,
            "similarities": [round(s, 6) for s in sims],
            "freshness_scores": [round(s, 6) for s in freshness_scores],
        }
    except Exception as exc:
        logger.warning("tavily_relevance_score failed: {}", exc)
        score = 0.0
        details = {"error": str(exc)}

    await record_metric(metric_name, "rag", score, details, session_id, student_id)
    return {"score": score, "details": details}


async def reranker_gain_score(
    query: str,
    chunks_before: list[str],
    chunks_after: list[str],
    session_id: str,
    student_id: str,
) -> dict:
    metric_name = "reranker_gain"
    try:
        before_top = [c for c in chunks_before[: max(1, len(chunks_after))] if c.strip()]
        after_top = [c for c in chunks_after if c.strip()]
        before_sims = await _similarities(query, before_top)
        after_sims = await _similarities(query, after_top)
        before_avg = sum(before_sims) / len(before_sims) if before_sims else 0.0
        after_avg = sum(after_sims) / len(after_sims) if after_sims else 0.0
        raw_gain = after_avg - before_avg
        score = _clamp(0.5 + raw_gain)
        details = {
            "before_avg": before_avg,
            "after_avg": after_avg,
            "raw_gain": raw_gain,
            "before_count": len(before_top),
            "after_count": len(after_top),
        }
    except Exception as exc:
        logger.warning("reranker_gain_score failed: {}", exc)
        score = 0.0
        details = {"error": str(exc)}

    await record_metric(metric_name, "rag", score, details, session_id, student_id)
    return {"score": score, "details": details}


def _lexical_faithfulness(lesson_text: str, retrieved_chunks: list[str]) -> float:
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
