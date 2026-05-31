"""Agent and curriculum quality metrics used by the evaluation runner."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from loguru import logger

from evaluation.collector import record_metric
from evaluation.metrics.rag_metrics import _clamp, _json_score, _llm_judge_score, _tokenize


def _schedule_metric(
    metric_name: str,
    component: str,
    score: float,
    details: dict,
    session_id: str,
    student_id: str,
) -> None:
    """Schedule a metric write from synchronous metric code when a loop exists."""
    try:
        asyncio.get_running_loop().create_task(
            record_metric(metric_name, component, score, details, session_id, student_id)
        )
    except RuntimeError:
        logger.warning("No running loop; could not record sync metric '{}'", metric_name)


def _parse_judge(raw: dict[str, Any], fallback: float) -> tuple[float, dict]:
    """Extract a bounded score from an LLM judge payload."""
    score = raw.get("score", fallback)
    try:
        score = _clamp(float(score))
    except Exception:
        score = fallback
    return score, raw


async def curriculum_coverage_score(
    topic: str,
    domain: str,
    generated_modules: list[dict],
    session_id: str,
    student_id: str,
) -> dict:
    """Score whether a generated curriculum covers the requested topic."""
    metric_name = "curriculum_coverage"
    try:
        concepts = [str(m.get("concept", "")).strip() for m in generated_modules if m.get("concept")]
        unique_ratio = len(set(c.lower() for c in concepts)) / max(1, len(concepts))
        size_score = _clamp(len(concepts) / 5.0)
        fallback = _clamp(0.65 * size_score + 0.35 * unique_ratio)
        prompt = (
            "Evaluate whether this generated curriculum covers the main concepts "
            "needed for the requested topic and domain. Return JSON: "
            "{\"score\": float, \"missing_concepts\": [string], \"notes\": string}.\n\n"
            f"Topic: {topic}\nDomain: {domain}\nModules:\n"
            f"{json.dumps(generated_modules[:20], default=str)}"
        )
        judged = await _llm_judge_score(prompt)
        score, judge = _parse_judge(judged, fallback)
        details = {
            "topic": topic,
            "domain": domain,
            "module_count": len(generated_modules),
            "unique_ratio": unique_ratio,
            "size_score": size_score,
            "judge": judge,
        }
    except Exception as exc:
        logger.warning("curriculum_coverage_score failed: {}", exc)
        score = 0.0
        details = {"error": str(exc)}

    await record_metric(metric_name, "agent", score, details, session_id, student_id)
    return {"score": score, "details": details}


async def curriculum_ordering_score(
    modules: list[dict],
    session_id: str,
    student_id: str,
) -> dict:
    """Score prerequisite ordering across generated modules."""
    metric_name = "curriculum_ordering"
    try:
        seen: set[str] = set()
        total = 0
        satisfied = 0
        violations: list[dict] = []
        for idx, module in enumerate(modules):
            concept = str(module.get("concept", "")).strip()
            prereqs = [str(p).strip() for p in module.get("prerequisites", []) if str(p).strip()]
            for prereq in prereqs:
                total += 1
                if prereq.lower() in seen:
                    satisfied += 1
                else:
                    violations.append({"index": idx, "concept": concept, "missing_prereq": prereq})
            if concept:
                seen.add(concept.lower())
        score = satisfied / total if total else 1.0
        details = {
            "module_count": len(modules),
            "prerequisites_checked": total,
            "satisfied": satisfied,
            "violations": violations[:20],
        }
    except Exception as exc:
        logger.warning("curriculum_ordering_score failed: {}", exc)
        score = 0.0
        details = {"error": str(exc)}

    await record_metric(metric_name, "agent", score, details, session_id, student_id)
    return {"score": score, "details": details}


def _lesson_structure_score(lesson_text: str, module: dict, student_pace: str) -> dict:
    """Compute a deterministic fallback score for lesson structure."""
    word_count = len(lesson_text.split())
    expected_words = {"fast": 400, "medium": 700, "deep": 1100}.get(student_pace, 700)
    word_score = _clamp(word_count / expected_words)
    headings = len(re.findall(r"^#{1,4}\s+", lesson_text, flags=re.M))
    heading_score = _clamp(headings / (5 if student_pace == "fast" else 8))
    concept = str(module.get("concept", "")).lower()
    concept_score = 1.0 if concept and concept in lesson_text.lower() else 0.5
    practice_score = 1.0 if re.search(r"practice|your turn|example", lesson_text, flags=re.I) else 0.4
    score = _clamp(
        0.30 * word_score
        + 0.25 * heading_score
        + 0.25 * concept_score
        + 0.20 * practice_score
    )
    return {
        "score": score,
        "word_count": word_count,
        "expected_words": expected_words,
        "headings": headings,
        "word_score": word_score,
        "heading_score": heading_score,
        "concept_score": concept_score,
        "practice_score": practice_score,
    }


async def lesson_quality_score(
    lesson_text: str,
    module: dict,
    student_pace: str,
    session_id: str,
    student_id: str,
) -> dict:
    """Score lesson quality using deterministic structure and an LLM judge."""
    metric_name = "lesson_quality"
    try:
        structure = _lesson_structure_score(lesson_text, module, student_pace)
        prompt = (
            "Evaluate the teaching quality of this lesson for an adaptive learning session. "
            "Consider clarity, correctness, pace fit, examples, and whether it teaches the "
            "specified module. Return JSON: {\"score\": float, \"notes\": string}.\n\n"
            f"Student pace: {student_pace}\nModule: {json.dumps(module, default=str)}\n\n"
            f"Lesson:\n{lesson_text[:7000]}"
        )
        judged = await _llm_judge_score(prompt)
        judge_score, judge = _parse_judge(judged, structure["score"])
        score = _clamp(0.70 * judge_score + 0.30 * structure["score"])
        details = {"structure": structure, "judge": judge}
    except Exception as exc:
        logger.warning("lesson_quality_score failed: {}", exc)
        score = 0.0
        details = {"error": str(exc)}

    await record_metric(metric_name, "agent", score, details, session_id, student_id)
    return {"score": score, "details": details}


async def question_quality_score(
    questions: list[dict],
    lesson_text: str,
    session_id: str,
    student_id: str,
) -> dict:
    """Score generated question quality and grounding against lesson text."""
    metric_name = "question_quality"
    try:
        if not questions:
            score = 0.0
            details = {"reason": "no_questions"}
        else:
            lesson_norm = " ".join(lesson_text.lower().split())
            type_set = {str(q.get("question_type") or q.get("type") or "") for q in questions}
            per_question: list[dict] = []
            total = 0.0
            for q in questions:
                question = str(q.get("question", "")).strip()
                quote = str(q.get("source_quote", "")).strip()
                q_tokens = _tokenize(question)
                lesson_tokens = _tokenize(lesson_text)
                grounding = (
                    1.0
                    if quote and " ".join(quote.lower().split()) in lesson_norm
                    else len(q_tokens & lesson_tokens) / max(1, len(q_tokens))
                )
                open_ended = 1.0 if not re.search(r"\b(a|b|c|d)\)", question.lower()) else 0.5
                length_score = _clamp(len(question.split()) / 12.0)
                q_score = _clamp(0.50 * grounding + 0.25 * open_ended + 0.25 * length_score)
                total += q_score
                per_question.append({
                    "question": question[:160],
                    "grounding": grounding,
                    "open_ended": open_ended,
                    "length_score": length_score,
                    "score": q_score,
                })
            variety = _clamp(len([t for t in type_set if t]) / min(3, max(1, len(questions))))
            score = _clamp(0.85 * (total / len(questions)) + 0.15 * variety)
            details = {
                "question_count": len(questions),
                "type_variety": variety,
                "per_question": per_question,
            }
    except Exception as exc:
        logger.warning("question_quality_score failed: {}", exc)
        score = 0.0
        details = {"error": str(exc)}

    await record_metric(metric_name, "agent", score, details, session_id, student_id)
    return {"score": score, "details": details}


async def scoring_consistency_score(
    qa_log: list[dict],
    concept: str,
    session_id: str,
    student_id: str,
) -> dict:
    """Score whether evaluator scoring is internally consistent with a QA log."""
    metric_name = "scoring_consistency"
    try:
        if not qa_log:
            score = 0.0
            details = {"reason": "empty_qa_log"}
        else:
            prompt = (
                "Evaluate whether the final correctness/depth scoring implied by this QA log "
                "is internally consistent. Penalize missing answers, unsupported credit, and "
                "scores that ignore the answer quality. Return JSON: "
                "{\"score\": float, \"notes\": string}.\n\n"
                f"Concept: {concept}\nQA log:\n{json.dumps(qa_log[:8], default=str)}"
            )
            judged = await _llm_judge_score(prompt)
            answer_ratio = sum(1 for qa in qa_log if str(qa.get("answer", "")).strip()) / len(qa_log)
            fallback = _clamp(0.4 + 0.6 * answer_ratio)
            score, judge = _parse_judge(judged, fallback)
            details = {"qa_count": len(qa_log), "answer_ratio": answer_ratio, "judge": judge}
    except Exception as exc:
        logger.warning("scoring_consistency_score failed: {}", exc)
        score = 0.0
        details = {"error": str(exc)}

    await record_metric(metric_name, "agent", score, details, session_id, student_id)
    return {"score": score, "details": details}


def routing_accuracy_score(
    mastery_score: float,
    advance_threshold: float,
    actual_action: str,
    misconception_type: str | None,
    reteach_count: int,
    session_id: str,
    student_id: str,
) -> dict:
    """Score whether an adaptation action matches mastery and misconception signals."""
    metric_name = "routing_accuracy"
    try:
        normalized_action = (actual_action or "").upper()
        if (misconception_type or "").lower() in {"prerequisite_gap", "prerequisite"}:
            expected = "DETOUR"
        elif reteach_count >= 3 and mastery_score < advance_threshold:
            expected = "ESCALATE"
        elif mastery_score >= 0.90:
            expected = "COMPRESS"
        elif mastery_score >= advance_threshold:
            expected = "MOVE_FORWARD"
        else:
            expected = "RETEACH"

        acceptable = {expected}
        if expected == "MOVE_FORWARD":
            acceptable.add("MOVE_FORWARD_WITH_FLAG")
        if expected == "COMPRESS":
            acceptable.add("MOVE_FORWARD")

        score = 1.0 if normalized_action in acceptable else 0.0
        details = {
            "mastery_score": mastery_score,
            "advance_threshold": advance_threshold,
            "actual_action": normalized_action,
            "expected_action": expected,
            "acceptable_actions": sorted(acceptable),
            "misconception_type": misconception_type,
            "reteach_count": reteach_count,
        }
    except Exception as exc:
        logger.warning("routing_accuracy_score failed: {}", exc)
        score = 0.0
        details = {"error": str(exc)}

    _schedule_metric(metric_name, "agent", score, details, session_id, student_id)
    return {"score": score, "details": details}
