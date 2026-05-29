from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


def _write_minimal_fixture(tmp_path: Path) -> Path:
    fixture_path = tmp_path / "minimal_input.json"
    fixture_path.write_text(
        json.dumps(
            {
                "session_id": "offline-test-session",
                "student_id": "offline-test-student",
                "course_id": "offline-test-course",
                "topic": "Python for beginners",
            }
        ),
        encoding="utf-8",
    )
    return fixture_path


def test_offline_runner_creates_txt_report(tmp_path):
    from evaluation.offline_metrics_runner import run_offline_evaluation

    report_path = Path(
        run_offline_evaluation(
            _write_minimal_fixture(tmp_path),
            output_dir=str(tmp_path / "reports"),
        )
    )

    assert report_path.exists()
    assert report_path.name.startswith("evaluation_report_")
    assert report_path.suffix == ".txt"


def test_offline_runner_does_not_import_fastapi_app(tmp_path, monkeypatch):
    from evaluation.offline_metrics_runner import run_offline_evaluation

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "uvicorn" or name == "app.api" or name.startswith("app.api."):
            raise AssertionError(f"offline runner imported {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    report_path = Path(
        run_offline_evaluation(
            _write_minimal_fixture(tmp_path),
            output_dir=str(tmp_path / "reports"),
        )
    )

    assert report_path.exists()
