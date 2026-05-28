import builtins
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _write_minimal_fixture(tmp_path: Path) -> Path:
    fixture_path = tmp_path / "minimal_input.json"
    fixture_path.write_text(
        json.dumps({
            "session_id": "offline-test-session",
            "student_id": "offline-test-student",
            "course_id": "offline-test-course",
            "topic": "Python for beginners",
        }),
        encoding="utf-8",
    )
    return fixture_path


def test_offline_runner_module_exists():
    import evaluation.offline_metrics_runner as offline_metrics_runner

    assert offline_metrics_runner.DEFAULT_FIXTURE.name == "sample_full_evaluation_input.json"


def test_default_fixture_exists():
    from evaluation.offline_metrics_runner import DEFAULT_FIXTURE

    assert DEFAULT_FIXTURE.exists()


def test_offline_runner_creates_txt_report(tmp_path):
    from evaluation.offline_metrics_runner import run_offline_evaluation

    fixture_path = _write_minimal_fixture(tmp_path)
    report_path = Path(run_offline_evaluation(fixture_path, output_dir=str(tmp_path / "reports")))

    assert report_path.exists()
    assert report_path.name.startswith("evaluation_report_")
    assert report_path.suffix == ".txt"


def test_report_contains_all_sections(tmp_path):
    from evaluation.offline_metrics_runner import run_offline_evaluation

    fixture_path = _write_minimal_fixture(tmp_path)
    report_path = Path(run_offline_evaluation(fixture_path, output_dir=str(tmp_path / "reports")))
    text = report_path.read_text(encoding="utf-8")

    assert "1. RAG / Retrieval Metrics" in text
    assert "2. Agent Metrics" in text
    assert "3. Learning Outcome Metrics" in text
    assert "4. System Metrics" in text
    assert "5. Missing / Not Available Metrics" in text


def test_missing_metric_does_not_crash(tmp_path):
    from evaluation.offline_metrics_runner import run_offline_evaluation

    fixture_path = _write_minimal_fixture(tmp_path)
    report_path = Path(run_offline_evaluation(fixture_path, output_dir=str(tmp_path / "reports")))
    text = report_path.read_text(encoding="utf-8")

    assert "HyDE Quality Score: not available" in text
    assert "missing input: rag.original_query" in text


def test_runner_does_not_start_fastapi(tmp_path, monkeypatch):
    from evaluation.offline_metrics_runner import run_offline_evaluation

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "uvicorn" or name == "app.api" or name.startswith("app.api."):
            raise AssertionError(f"offline runner imported {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    fixture_path = _write_minimal_fixture(tmp_path)
    report_path = Path(run_offline_evaluation(fixture_path, output_dir=str(tmp_path / "reports")))

    assert report_path.exists()
