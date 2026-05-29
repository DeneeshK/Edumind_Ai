from __future__ import annotations

from datetime import datetime as real_datetime
from pathlib import Path

import pytest

from evaluation import report_writer


pytestmark = pytest.mark.unit


def test_save_metrics_txt_report_creates_readable_report(tmp_path, monkeypatch):
    class FixedDateTime:
        @classmethod
        def now(cls):
            return real_datetime(2026, 5, 24, 18, 35, 10)

    monkeypatch.setattr(report_writer, "datetime", FixedDateTime)

    saved_path = report_writer.save_metrics_txt_report(
        {
            "course_id": "course-123",
            "session_id": "session-456",
            "student_id": "student-789",
            "topic": "Linear Algebra",
            "rag_score": 0.82,
            "agent_score": 0.76,
            "outcome_score": 0.7,
            "system_score": 0.77,
            "calculated_metrics": {
                "hyde_quality": [{"score": 0.91, "details": {"notes": "aligned"}}],
                "lesson_quality": [{"score": 0.83, "details": {"judge": {"notes": "clear"}}}],
            },
        },
        output_dir=str(tmp_path),
    )

    text = Path(saved_path).read_text(encoding="utf-8")
    assert "EduMind Evaluation Metrics Report" in text
    assert "Generated At: 2026-05-24 18:35:10" in text
    assert "Course ID: course-123" in text
    assert "Final System Score: 0.770000" in text


def test_missing_metric_not_rendered_as_zero(tmp_path):
    saved_path = report_writer.save_metrics_txt_report(
        {
            "topic": "Python",
            "calculated_metrics": {
                "scoring_consistency": [
                    {
                        "score": 0.0,
                        "details": {"notes": "Missing correctness_score/depth_score in QA log."},
                    }
                ],
            },
        },
        output_dir=str(tmp_path),
    )

    text = Path(saved_path).read_text(encoding="utf-8")
    assert "Evaluator Scoring Consistency Score: not available" in text
    assert "Evaluator Scoring Consistency Score: 0.000000" not in text
