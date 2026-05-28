from __future__ import annotations

import asyncio

from loguru import logger

from evaluation.collector import record_metric
from evaluation.metrics.rag_metrics import _clamp


def _schedule_metric(
    metric_name: str,
    score: float,
    details: dict,
    session_id: str,
    student_id: str,
) -> None:
    try:
        asyncio.get_running_loop().create_task(
            record_metric(metric_name, "outcome", score, details, session_id, student_id)
        )
    except RuntimeError:
        logger.warning("No running loop; could not record sync metric '{}'", metric_name)


def mastery_progression_rate(
    mastery_history: list[float],
    concept: str,
    session_id: str,
    student_id: str,
) -> dict:
    metric_name = "mastery_progression_rate"
    try:
        history = [float(v) for v in mastery_history if v is not None]
        if not history:
            score = 0.0
            details = {"concept": concept, "reason": "empty_history"}
        elif len(history) == 1:
            score = _clamp(history[-1])
            details = {"concept": concept, "history": history, "single_observation": True}
        else:
            total_delta = history[-1] - history[0]
            positive_steps = [
                max(0.0, history[i] - history[i - 1])
                for i in range(1, len(history))
            ]
            avg_positive_step = sum(positive_steps) / max(1, len(positive_steps))
            score = _clamp(0.50 + total_delta + 0.50 * avg_positive_step)
            details = {
                "concept": concept,
                "history": history,
                "total_delta": total_delta,
                "avg_positive_step": avg_positive_step,
            }
    except Exception as exc:
        logger.warning("mastery_progression_rate failed: {}", exc)
        score = 0.0
        details = {"error": str(exc)}

    _schedule_metric(metric_name, score, details, session_id, student_id)
    return {"score": score, "details": details}


def calibration_quality_score(
    calibration_deltas: list[float],
    session_id: str,
    student_id: str,
) -> dict:
    metric_name = "calibration_quality"
    try:
        deltas = [abs(float(delta)) for delta in calibration_deltas if delta is not None]
        if not deltas:
            score = 0.0
            details = {"reason": "empty_deltas"}
        else:
            avg_abs_delta = sum(deltas) / len(deltas)
            max_abs_delta = max(deltas)
            score = _clamp(1.0 - avg_abs_delta)
            details = {
                "count": len(deltas),
                "avg_abs_delta": avg_abs_delta,
                "max_abs_delta": max_abs_delta,
            }
    except Exception as exc:
        logger.warning("calibration_quality_score failed: {}", exc)
        score = 0.0
        details = {"error": str(exc)}

    _schedule_metric(metric_name, score, details, session_id, student_id)
    return {"score": score, "details": details}


def session_efficiency_score(
    modules_attempted: int,
    modules_mastered: int,
    total_modules_in_curriculum: int,
    reteach_events: int,
    session_duration_minutes: float,
    pace: str,
    session_id: str,
    student_id: str,
) -> dict:
    metric_name = "session_efficiency"
    try:
        attempted = max(0, int(modules_attempted))
        mastered = max(0, int(modules_mastered))
        total = max(1, int(total_modules_in_curriculum))
        reteaches = max(0, int(reteach_events))
        duration = max(0.0, float(session_duration_minutes))

        mastery_ratio = mastered / max(1, attempted)
        curriculum_progress = mastered / total
        reteach_penalty = _clamp(1.0 - (reteaches / max(1, attempted + reteaches)))
        expected_minutes = {"fast": 12.0, "medium": 18.0, "deep": 28.0}.get(pace, 18.0)
        expected_duration = max(expected_minutes, expected_minutes * max(1, attempted))
        duration_score = _clamp(expected_duration / max(expected_minutes, duration or expected_minutes))

        score = _clamp(
            0.40 * mastery_ratio
            + 0.25 * curriculum_progress
            + 0.20 * reteach_penalty
            + 0.15 * duration_score
        )
        details = {
            "modules_attempted": attempted,
            "modules_mastered": mastered,
            "total_modules_in_curriculum": total,
            "reteach_events": reteaches,
            "session_duration_minutes": duration,
            "pace": pace,
            "mastery_ratio": mastery_ratio,
            "curriculum_progress": curriculum_progress,
            "reteach_penalty": reteach_penalty,
            "duration_score": duration_score,
        }
    except Exception as exc:
        logger.warning("session_efficiency_score failed: {}", exc)
        score = 0.0
        details = {"error": str(exc)}

    _schedule_metric(metric_name, score, details, session_id, student_id)
    return {"score": score, "details": details}
