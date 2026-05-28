import os
import sys
from datetime import datetime as real_datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation import report_writer


def test_save_metrics_txt_report_creates_readable_timestamped_txt(tmp_path, monkeypatch):
    class FixedDateTime:
        @classmethod
        def now(cls):
            return real_datetime(2026, 5, 24, 18, 35, 10)

    monkeypatch.setattr(report_writer, "datetime", FixedDateTime)

    metrics = {
        "course_id": "course-123",
        "session_id": "session-456",
        "student_id": "student-789",
        "topic": "Linear Algebra",
        "rag_score": 0.82,
        "agent_score": 0.76,
        "outcome_score": 0.7,
        "system_score": 0.77,
        "calculated_metrics": {
            "hyde_quality": [{"score": 0.91, "details": {"notes": "aligned to query"}}],
            "tavily_relevance": [{"score": 0.64, "details": {"freshness": 0.72}}],
            "lesson_quality": [{"score": 0.83, "details": {"judge": {"notes": "clear lesson"}}}],
            "calibration_quality": [{"score": 0.88, "details": {}}],
        },
    }

    saved_path = report_writer.save_metrics_txt_report(metrics, output_dir=str(tmp_path))

    report_path = Path(saved_path)
    assert report_path.exists()
    assert report_path.name == "evaluation_report_2026-05-24_18-35-10.txt"

    text = report_path.read_text(encoding="utf-8")
    assert "EduMind Evaluation Metrics Report" in text
    assert "Generated At: 2026-05-24 18:35:10" in text
    assert "Course ID: course-123" in text
    assert "1. RAG / Retrieval Metrics" in text
    assert "2. Agent Metrics" in text
    assert "3. Learning Outcome Metrics" in text
    assert "4. System Metrics" in text
    assert "5. Missing / Not Available Metrics" in text
    assert "HyDE Quality Score: 0.910000" in text
    assert "Tavily/MCP Relevance Score: 0.640000" in text
    assert "Tavily/MCP Freshness Score: 0.720000" in text
    assert "Tutor/Lesson Quality Score: 0.830000" in text
    assert "Calibration Quality Score: 0.880000" in text
    assert "Final System Score: 0.770000" in text
    assert "ChromaDB Precision@K: not available" in text


def test_missing_input_metric_not_rendered_as_zero(tmp_path):
    metrics = {
        "topic": "Python",
        "calculated_metrics": {
            "scoring_consistency": [{
                "score": 0.0,
                "details": {
                    "judge": {
                        "notes": "Missing correctness_score for answer, depth_score not checked, unsupported credit."
                    }
                },
            }],
        },
    }

    saved_path = report_writer.save_metrics_txt_report(metrics, output_dir=str(tmp_path))
    text = Path(saved_path).read_text(encoding="utf-8")

    assert (
        "Evaluator Scoring Consistency Score: not available — "
        "missing correctness_score/depth_score in QA log"
    ) in text
    assert "Evaluator Scoring Consistency Score: 0.000000" not in text


def test_missing_metrics_section_lists_unavailable_metric(tmp_path):
    metrics = {
        "topic": "Python",
        "calculated_metrics": {
            "scoring_consistency": [{
                "score": 0.0,
                "details": {"notes": "Missing correctness_score/depth_score in QA log."},
            }],
        },
    }

    saved_path = report_writer.save_metrics_txt_report(metrics, output_dir=str(tmp_path))
    text = Path(saved_path).read_text(encoding="utf-8")

    assert "No missing metrics." not in text
    assert (
        "- Evaluator Scoring Consistency Score: "
        "missing correctness_score/depth_score in QA log"
    ) in text


def test_real_zero_score_still_allowed_if_status_ok(tmp_path):
    metrics = {
        "topic": "Python",
        "calculated_metrics": {
            "scoring_consistency": [{
                "status": "ok",
                "score": 0.0,
                "details": {"notes": "Calculated valid zero score."},
            }],
        },
    }

    saved_path = report_writer.save_metrics_txt_report(metrics, output_dir=str(tmp_path))
    text = Path(saved_path).read_text(encoding="utf-8")

    assert "Evaluator Scoring Consistency Score: 0.000000" in text
    assert "- Evaluator Scoring Consistency Score:" not in text


def test_interpretation_truncates_cleanly(tmp_path):
    long_note = (
        "This interpretation is intentionally long and should be trimmed at a "
        "word boundary without leaving awkward partial tokens or an extra closing "
        "parenthesis after the ellipsis. " * 8
    )
    metrics = {
        "topic": "Python",
        "calculated_metrics": {
            "rag_faithfulness": [{"score": 0.82, "details": {"judge": {"notes": long_note}}}],
        },
    }

    saved_path = report_writer.save_metrics_txt_report(metrics, output_dir=str(tmp_path))
    text = Path(saved_path).read_text(encoding="utf-8")
    line = next(line for line in text.splitlines() if line.startswith("RAG Faithfulness Score:"))
    interpretation = line.split("| Interpretation: ", 1)[1]

    assert interpretation.endswith("...")
    assert not line.endswith("...)")
    assert len(interpretation) <= 300
