from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable


SEPARATOR = "=" * 50
NOT_AVAILABLE = "not available"
NOT_AVAILABLE_SEPARATOR = " — "
INTERPRETATION_LIMIT = 280


def _as_entries(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _nested_get(value: Any, path: tuple[str, ...]) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _payload_sources(metrics: dict) -> list[dict]:
    sources = [metrics]
    for key in (
        "calculated_metrics",
        "metric_results",
        "metrics",
        "rag_metrics",
        "agent_metrics",
        "outcome_metrics",
        "system_metrics",
        "full_report_json",
    ):
        value = metrics.get(key)
        if isinstance(value, dict):
            sources.append(value)
    return sources


def _find_entries(metrics: dict, aliases: tuple[str, ...]) -> list[Any]:
    for source in _payload_sources(metrics):
        for alias in aliases:
            if alias in source:
                return _as_entries(source[alias])
    return []


def _score_value(entry: Any) -> Any:
    if isinstance(entry, dict):
        if entry.get("status") == "not_available":
            return None
        for key in ("score", "value", "metric_value"):
            if key in entry:
                return entry[key]
        return None
    return entry


def _format_value(value: Any) -> str:
    if value is None:
        return NOT_AVAILABLE
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _raw_interpretation(entry: Any) -> str | None:
    if not isinstance(entry, dict):
        return None
    details = entry.get("details")
    if not isinstance(details, dict):
        return None

    for key in ("interpretation", "notes", "reason", "error"):
        value = details.get(key)
        if value:
            return str(value)

    judge = details.get("judge")
    if isinstance(judge, dict):
        for key in ("interpretation", "notes", "reason", "error"):
            value = judge.get(key)
            if value:
                return str(value)
    return None


def _shorten(value: str, limit: int = INTERPRETATION_LIMIT) -> str:
    clean = " ".join(str(value).split())
    if len(clean) <= limit:
        return clean
    candidate = clean[: limit - 3].rstrip()
    last_space = candidate.rfind(" ")
    if last_space >= int(limit * 0.60):
        candidate = candidate[:last_space]
    candidate = candidate.rstrip(" ,;:-(")
    return candidate + "..."


def _score_is_zero(entry: dict) -> bool:
    for key in ("score", "value", "metric_value"):
        if key not in entry:
            continue
        try:
            return float(entry[key]) == 0.0
        except (TypeError, ValueError):
            return False
    return False


def _availability_reason_from_text(text: str | None) -> str | None:
    if not text:
        return None
    lowered = text.lower()
    if "correctness_score" in lowered or "depth_score" in lowered:
        return "missing correctness_score/depth_score in QA log"
    if "missing api key" in lowered:
        return "missing API key"
    if "skipped judge" in lowered:
        return "skipped judge"
    if "empty_qa_log" in lowered:
        return "missing input: evaluator.qa_log"
    if "no_questions" in lowered:
        return "missing input: evaluator.questions"
    if "missing_lesson_or_context" in lowered:
        return "missing input: lesson_text/retrieved_chunks"
    if "missing input" in lowered:
        return _shorten(text, 160)
    if "unsupported credit" in lowered or "not checked" in lowered:
        return "not checked due to incomplete scoring data"
    return None


def _not_available_reason(entry: Any) -> str | None:
    if not isinstance(entry, dict):
        return None
    if entry.get("status") == "ok":
        return None
    if entry.get("status") == "not_available":
        reason = entry.get("reason")
        if reason:
            return _shorten(str(reason))
        details = entry.get("details")
        if isinstance(details, dict) and details.get("reason"):
            return _shorten(str(details["reason"]))
        return NOT_AVAILABLE
    if _score_is_zero(entry):
        return _availability_reason_from_text(_raw_interpretation(entry))
    return None


def _interpretation(entry: Any) -> str | None:
    raw = _raw_interpretation(entry)
    return _shorten(raw) if raw else None


def _format_metric_entries(
    entries: list[Any],
    value_getter: Callable[[Any], Any],
) -> tuple[str, str | None]:
    values: list[str] = []
    interpretations: list[str] = []
    for index, entry in enumerate(entries, start=1):
        not_available_reason = _not_available_reason(entry)
        if not_available_reason:
            text = f"{NOT_AVAILABLE}{NOT_AVAILABLE_SEPARATOR}{not_available_reason}"
            if len(entries) == 1:
                values.append(text)
            else:
                values.append(f"run {index}: {text}")
            continue

        value = value_getter(entry)
        if value is None:
            continue
        if len(entries) == 1:
            values.append(_format_value(value))
        else:
            values.append(f"run {index}: {_format_value(value)}")

        note = _interpretation(entry)
        if note:
            if len(entries) == 1:
                interpretations.append(note)
            else:
                interpretations.append(f"run {index}: {note}")

    if not values:
        return NOT_AVAILABLE, None
    return "; ".join(values), "; ".join(interpretations) if interpretations else None


def _metric_line(
    metrics: dict,
    label: str,
    aliases: tuple[str, ...],
    missing: dict[str, str],
    detail_path: tuple[str, ...] | None = None,
) -> str:
    entries = _find_entries(metrics, aliases)
    value_getter = (
        (lambda entry: _nested_get(entry, detail_path))
        if detail_path
        else _score_value
    )
    value, note = _format_metric_entries(entries, value_getter)
    if NOT_AVAILABLE in value and label not in missing:
        missing[label] = _missing_reason_from_value(value)
    if note:
        return f"{label}: {value} | Interpretation: {note}"
    return f"{label}: {value}"


def _missing_reason_from_value(value: str) -> str:
    if NOT_AVAILABLE_SEPARATOR in value:
        reason = value.split(NOT_AVAILABLE_SEPARATOR, 1)[1]
        return reason.split(";", 1)[0].strip()
    return NOT_AVAILABLE


def _header_value(metrics: dict, *keys: str) -> str:
    for key in keys:
        value = metrics.get(key)
        if value not in (None, ""):
            return str(value)
    return NOT_AVAILABLE


def save_metrics_txt_report(
    metrics: dict,
    output_dir: str = "evaluation/reports",
) -> str:
    """
    Save already-calculated EduMind evaluation metrics to a timestamped TXT report.

    This writer only exports values present in the supplied metrics payload. Missing
    metrics are written as "not available".
    """
    metrics = metrics or {}
    generated_at = datetime.now()
    timestamp = generated_at.strftime("%Y-%m-%d_%H-%M-%S")

    reports_dir = Path(output_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"evaluation_report_{timestamp}.txt"

    missing: dict[str, str] = {}
    lines = [
        "EduMind Evaluation Metrics Report",
        f"Generated At: {generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Course ID: {_header_value(metrics, 'course_id', 'courseId')}",
        f"Session ID: {_header_value(metrics, 'session_id', 'sessionId')}",
        f"Student ID: {_header_value(metrics, 'student_id', 'studentId')}",
        f"Topic: {_header_value(metrics, 'topic', 'course_name', 'course_title', 'course')}",
        "",
        SEPARATOR,
        "1. RAG / Retrieval Metrics",
        SEPARATOR,
        _metric_line(metrics, "HyDE Quality Score", ("hyde_quality", "hyde_quality_score"), missing),
        _metric_line(
            metrics,
            "ChromaDB Precision@K",
            ("chromadb_precision_at_k", "chromadb_precision", "chromadb_precision_k"),
            missing,
        ),
        _metric_line(
            metrics,
            "Tavily/MCP Relevance Score",
            ("tavily_relevance", "tavily_mcp_relevance", "tavily_relevance_score"),
            missing,
        ),
        _metric_line(
            metrics,
            "Tavily/MCP Freshness Score",
            ("tavily_relevance", "tavily_mcp_freshness", "tavily_freshness"),
            missing,
            detail_path=("details", "freshness"),
        ),
        _metric_line(metrics, "Reranker Gain Score", ("reranker_gain", "reranker_gain_score"), missing),
        _metric_line(metrics, "RAG Faithfulness Score", ("rag_faithfulness", "rag_faithfulness_score"), missing),
        _metric_line(metrics, "Overall RAG Score", ("rag_score", "overall_rag_score"), missing),
        "",
        SEPARATOR,
        "2. Agent Metrics",
        SEPARATOR,
        _metric_line(
            metrics,
            "CurriculumArchitect Coverage Score",
            ("curriculum_coverage", "curriculum_architect_coverage"),
            missing,
        ),
        _metric_line(
            metrics,
            "CurriculumArchitect Ordering Score",
            ("curriculum_ordering", "curriculum_architect_ordering"),
            missing,
        ),
        _metric_line(metrics, "Tutor/Lesson Quality Score", ("lesson_quality", "tutor_lesson_quality"), missing),
        _metric_line(
            metrics,
            "Evaluator Question Quality Score",
            ("question_quality", "evaluator_question_quality"),
            missing,
        ),
        _metric_line(
            metrics,
            "Evaluator Scoring Consistency Score",
            ("scoring_consistency", "evaluator_scoring_consistency"),
            missing,
        ),
        _metric_line(
            metrics,
            "Adaptation/Orchestrator Routing Accuracy",
            ("routing_accuracy", "adaptation_routing_accuracy", "orchestrator_routing_accuracy"),
            missing,
        ),
        _metric_line(metrics, "Overall Agent Score", ("agent_score", "overall_agent_score"), missing),
        "",
        SEPARATOR,
        "3. Learning Outcome Metrics",
        SEPARATOR,
        _metric_line(
            metrics,
            "Mastery Progression Rate",
            ("mastery_progression_rate", "mastery_progression"),
            missing,
        ),
        _metric_line(metrics, "Calibration Quality Score", ("calibration_quality", "calibration_quality_score"), missing),
        _metric_line(metrics, "Session Efficiency Score", ("session_efficiency", "session_efficiency_score"), missing),
        _metric_line(metrics, "Overall Outcome Score", ("outcome_score", "overall_outcome_score"), missing),
        "",
        SEPARATOR,
        "4. System Metrics",
        SEPARATOR,
        _metric_line(metrics, "Overall RAG Score", ("rag_score", "overall_rag_score"), missing),
        _metric_line(metrics, "Overall Agent Score", ("agent_score", "overall_agent_score"), missing),
        _metric_line(metrics, "Final System Score", ("system_score", "final_system_score"), missing),
        "",
        SEPARATOR,
        "5. Missing / Not Available Metrics",
        SEPARATOR,
    ]

    if missing:
        lines.extend(f"- {name}: {reason}" for name, reason in missing.items())
    else:
        lines.append("No missing metrics.")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(report_path)
