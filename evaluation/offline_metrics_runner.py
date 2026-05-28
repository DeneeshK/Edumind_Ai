from __future__ import annotations

import argparse
import asyncio
import inspect
import importlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Callable

from evaluation.report_writer import save_metrics_txt_report


DEFAULT_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample_full_evaluation_input.json"
DEFAULT_OUTPUT_DIR = "evaluation/reports"
MISSING_ENV_SENTINEL = "__edumind_offline_missing__"


class MissingInputError(Exception):
    pass


def _not_available(reason: str) -> dict:
    return {
        "status": "not_available",
        "reason": reason,
        "details": {"reason": reason},
    }


def _load_env_file_values() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    candidates = [Path.cwd() / ".env", backend_root / ".env"]
    for env_path in candidates:
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _prepare_env_for_metric_imports() -> None:
    _load_env_file_values()
    os.environ.setdefault("DATABASE_URL", "postgresql://offline:offline@localhost:5432/offline")
    os.environ.setdefault("GROQ_API_KEY", MISSING_ENV_SENTINEL)
    os.environ.setdefault("TAVILY_API_KEY", MISSING_ENV_SENTINEL)


def _has_real_api_key(env_name: str) -> bool:
    value = os.environ.get(env_name, "").strip()
    if not value or value == MISSING_ENV_SENTINEL:
        return False
    if value.lower() in {"...", "your_key_here", "replace_me", "changeme"}:
        return False
    return True


def _missing_dependency(dependencies: tuple[str, ...]) -> str | None:
    for dependency in dependencies:
        if importlib.util.find_spec(dependency) is None:
            return dependency
    return None


def _get(data: dict, dotted_path: str, default: Any = None) -> Any:
    current: Any = data
    for key in dotted_path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _require(data: dict, dotted_path: str) -> Any:
    value = _get(data, dotted_path)
    if value is None:
        raise MissingInputError(dotted_path)
    return value


def _require_any(data: dict, dotted_paths: tuple[str, ...], input_name: str) -> Any:
    for dotted_path in dotted_paths:
        value = _get(data, dotted_path)
        if value is not None:
            return value
    raise MissingInputError(input_name)


def _missing_input_reason(exc: MissingInputError) -> str:
    reason = str(exc)
    if reason.startswith("missing "):
        return reason
    return f"missing input: {reason}"


def _require_scoring_consistency_qa_log(data: dict) -> list[dict]:
    qa_log = _require(data, "evaluator.qa_log")
    if not isinstance(qa_log, list) or not qa_log:
        raise MissingInputError("evaluator.qa_log")
    for entry in qa_log:
        if not isinstance(entry, dict):
            raise MissingInputError("evaluator.qa_log")
        if "correctness_score" not in entry or "depth_score" not in entry:
            raise MissingInputError("missing correctness_score/depth_score in QA log")
    return qa_log


def _score_from_result(result: Any) -> float | None:
    if isinstance(result, dict):
        if result.get("status") == "not_available":
            return None
        if result.get("score") is not None:
            try:
                return float(result["score"])
            except (TypeError, ValueError):
                return None
    return None


def _normalize_metric_result(result: Any) -> dict:
    if not isinstance(result, dict):
        return {"score": result, "details": {}}

    details = result.get("details")
    if isinstance(details, dict) and details.get("error"):
        reason = str(details["error"])
        if "No module named" in reason:
            return _not_available(f"missing dependency: {reason}")
        return _not_available(reason)
    return result


async def safe_metric(
    name: str,
    fn: Callable,
    *args,
    api_key: str | None = None,
    dependencies: tuple[str, ...] = (),
    **kwargs,
) -> dict:
    try:
        if fn is None:
            return _not_available("function not implemented")
        if api_key and not _has_real_api_key(api_key):
            return _not_available("missing API key")
        missing_dependency = _missing_dependency(dependencies)
        if missing_dependency:
            return _not_available(f"missing dependency: {missing_dependency}")

        value = fn(*args, **kwargs)
        if inspect.isawaitable(value):
            value = await value
        return _normalize_metric_result(value)
    except MissingInputError as exc:
        return _not_available(_missing_input_reason(exc))
    except ImportError as exc:
        return _not_available(f"missing dependency: {exc}")
    except Exception as exc:
        return _not_available(str(exc))


async def _noop_record_metric(*args, **kwargs) -> None:
    return None


def _import_metric_modules() -> dict[str, Any]:
    _prepare_env_for_metric_imports()
    modules: dict[str, Any] = {}
    for module_name in (
        "evaluation.metrics.rag_metrics",
        "evaluation.metrics.agent_metrics",
        "evaluation.metrics.outcome_metrics",
        "evaluation.runner",
    ):
        short_name = module_name.rsplit(".", 1)[-1]
        try:
            modules[short_name] = importlib.import_module(module_name)
        except ImportError as exc:
            modules[short_name] = exc

    for short_name in ("rag_metrics", "agent_metrics", "outcome_metrics"):
        module = modules.get(short_name)
        if not isinstance(module, ImportError):
            setattr(module, "record_metric", _noop_record_metric)

    return modules


def _metric_fn(modules: dict[str, Any], module_name: str, function_name: str) -> Callable | None:
    module = modules.get(module_name)
    if isinstance(module, ImportError) or module is None:
        return None
    return getattr(module, function_name, None)


async def _store_metric(
    output: dict,
    key: str,
    build_call: Callable[[], tuple[Callable | None, tuple, dict]],
) -> None:
    try:
        fn, args, kwargs = build_call()
        result = await safe_metric(key, fn, *args, **kwargs)
    except MissingInputError as exc:
        result = _not_available(_missing_input_reason(exc))
    output.setdefault("calculated_metrics", {})[key] = [result]


def _metric_scores(output: dict, keys: tuple[str, ...]) -> list[float]:
    calculated = output.get("calculated_metrics", {})
    scores: list[float] = []
    for key in keys:
        for result in calculated.get(key, []):
            score = _score_from_result(result)
            if score is not None:
                scores.append(score)
    return scores


def _average_with_runner_formula(modules: dict[str, Any], scores: list[float]) -> float | None:
    runner_module = modules.get("runner")
    if isinstance(runner_module, ImportError) or runner_module is None:
        return sum(scores) / len(scores) if scores else None
    avg_fn = getattr(runner_module, "_avg", None)
    if avg_fn is None:
        return sum(scores) / len(scores) if scores else None
    return avg_fn(scores)


def _system_score_with_runner_formula(
    modules: dict[str, Any],
    rag_score: float | None,
    agent_score: float | None,
    outcome_score: float | None,
) -> float | None:
    runner_module = modules.get("runner")
    runner_cls = None if isinstance(runner_module, ImportError) else getattr(runner_module, "EvaluationRunner", None)
    if runner_cls is None:
        weighted = [
            (rag_score, 0.35),
            (agent_score, 0.40),
            (outcome_score, 0.25),
        ]
        present = [(score, weight) for score, weight in weighted if score is not None]
        if not present:
            return None
        total_weight = sum(weight for _, weight in present)
        return sum(float(score) * weight for score, weight in present) / total_weight

    runner = runner_cls(
        session_id="offline-system-score",
        student_id="offline",
        topic="offline",
        pace="medium",
    )
    return runner._compute_system_score(rag_score, agent_score, outcome_score)


def load_fixture(input_path: str | Path | None = None) -> dict:
    path = Path(input_path) if input_path else DEFAULT_FIXTURE
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


async def run_offline_evaluation_async(
    input_path: str | Path | None = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> str:
    fixture = load_fixture(input_path)
    modules = _import_metric_modules()

    session_id = fixture.get("session_id", "offline-demo-session")
    student_id = fixture.get("student_id", "offline-demo-student")
    topic = fixture.get("topic", "")
    domain = fixture.get("domain", "")
    student_pace = fixture.get("student_pace", "medium")

    output: dict[str, Any] = {
        "course_id": fixture.get("course_id"),
        "session_id": session_id,
        "student_id": student_id,
        "topic": topic,
        "calculated_metrics": {},
    }

    await _store_metric(
        output,
        "hyde_quality",
        lambda: (
            _metric_fn(modules, "rag_metrics", "hyde_quality_score"),
            (
                _require(fixture, "rag.original_query"),
                _require(fixture, "rag.hyde_answer"),
                _require(fixture, "rag.ground_truth_concept_card"),
                session_id,
                student_id,
            ),
            {"dependencies": ("sentence_transformers",)},
        ),
    )
    await _store_metric(
        output,
        "chromadb_precision_at_k",
        lambda: (
            _metric_fn(modules, "rag_metrics", "chromadb_precision_at_k"),
            (
                _require(fixture, "rag.original_query"),
                _require(fixture, "rag.chromadb_chunks"),
                session_id,
                student_id,
            ),
            {"dependencies": ("sentence_transformers",)},
        ),
    )
    await _store_metric(
        output,
        "tavily_relevance",
        lambda: (
            _metric_fn(modules, "rag_metrics", "tavily_relevance_score"),
            (
                _require(fixture, "rag.original_query"),
                _require(fixture, "rag.tavily_results"),
                session_id,
                student_id,
            ),
            {"dependencies": ("sentence_transformers",)},
        ),
    )
    await _store_metric(
        output,
        "reranker_gain",
        lambda: (
            _metric_fn(modules, "rag_metrics", "reranker_gain_score"),
            (
                _require(fixture, "rag.original_query"),
                _require(fixture, "rag.chunks_before_rerank"),
                _require(fixture, "rag.chunks_after_rerank"),
                session_id,
                student_id,
            ),
            {"dependencies": ("sentence_transformers",)},
        ),
    )
    await _store_metric(
        output,
        "rag_faithfulness",
        lambda: (
            _metric_fn(modules, "rag_metrics", "rag_faithfulness_score"),
            (
                _require(fixture, "rag.lesson_text"),
                _require_any(
                    fixture,
                    ("rag.chunks_after_rerank", "rag.chromadb_chunks"),
                    "rag.retrieved_chunks",
                ),
                session_id,
                student_id,
            ),
            {"api_key": "GROQ_API_KEY"},
        ),
    )

    await _store_metric(
        output,
        "curriculum_coverage",
        lambda: (
            _metric_fn(modules, "agent_metrics", "curriculum_coverage_score"),
            (
                topic,
                domain,
                _require(fixture, "curriculum.modules"),
                session_id,
                student_id,
            ),
            {"api_key": "GROQ_API_KEY"},
        ),
    )
    await _store_metric(
        output,
        "curriculum_ordering",
        lambda: (
            _metric_fn(modules, "agent_metrics", "curriculum_ordering_score"),
            (
                _require(fixture, "curriculum.modules"),
                session_id,
                student_id,
            ),
            {},
        ),
    )
    await _store_metric(
        output,
        "lesson_quality",
        lambda: (
            _metric_fn(modules, "agent_metrics", "lesson_quality_score"),
            (
                _require(fixture, "lesson.lesson_text"),
                _require(fixture, "lesson.module"),
                student_pace,
                session_id,
                student_id,
            ),
            {"api_key": "GROQ_API_KEY"},
        ),
    )
    await _store_metric(
        output,
        "question_quality",
        lambda: (
            _metric_fn(modules, "agent_metrics", "question_quality_score"),
            (
                _require(fixture, "evaluator.questions"),
                _require_any(
                    fixture,
                    ("lesson.lesson_text", "rag.lesson_text"),
                    "lesson.lesson_text",
                ),
                session_id,
                student_id,
            ),
            {},
        ),
    )
    await _store_metric(
        output,
        "scoring_consistency",
        lambda: (
            _metric_fn(modules, "agent_metrics", "scoring_consistency_score"),
            (
                _require_scoring_consistency_qa_log(fixture),
                _get(fixture, "lesson.module.concept", topic),
                session_id,
                student_id,
            ),
            {"api_key": "GROQ_API_KEY"},
        ),
    )
    await _store_metric(
        output,
        "routing_accuracy",
        lambda: (
            _metric_fn(modules, "agent_metrics", "routing_accuracy_score"),
            (
                _require(fixture, "adaptation.mastery_score"),
                _require(fixture, "adaptation.advance_threshold"),
                _require(fixture, "adaptation.actual_action"),
                _get(fixture, "adaptation.misconception_type"),
                _require(fixture, "adaptation.reteach_count"),
                session_id,
                student_id,
            ),
            {},
        ),
    )

    await _store_metric(
        output,
        "mastery_progression_rate",
        lambda: (
            _metric_fn(modules, "outcome_metrics", "mastery_progression_rate"),
            (
                _require(fixture, "learning_outcomes.mastery_history"),
                _get(fixture, "lesson.module.concept", topic),
                session_id,
                student_id,
            ),
            {},
        ),
    )
    await _store_metric(
        output,
        "calibration_quality",
        lambda: (
            _metric_fn(modules, "outcome_metrics", "calibration_quality_score"),
            (
                _require(fixture, "learning_outcomes.calibration_deltas"),
                session_id,
                student_id,
            ),
            {},
        ),
    )
    await _store_metric(
        output,
        "session_efficiency",
        lambda: (
            _metric_fn(modules, "outcome_metrics", "session_efficiency_score"),
            (
                _require(fixture, "learning_outcomes.session_summary.modules_attempted"),
                _require(fixture, "learning_outcomes.session_summary.modules_mastered"),
                _require(fixture, "learning_outcomes.session_summary.total_modules_in_curriculum"),
                _require(fixture, "learning_outcomes.session_summary.reteach_events"),
                _require(fixture, "learning_outcomes.session_summary.session_duration_minutes"),
                _get(fixture, "learning_outcomes.session_summary.pace", student_pace),
                session_id,
                student_id,
            ),
            {},
        ),
    )

    await asyncio.sleep(0)

    rag_keys = (
        "hyde_quality",
        "chromadb_precision_at_k",
        "tavily_relevance",
        "reranker_gain",
        "rag_faithfulness",
    )
    agent_keys = (
        "curriculum_coverage",
        "curriculum_ordering",
        "lesson_quality",
        "question_quality",
        "scoring_consistency",
        "routing_accuracy",
    )
    outcome_keys = (
        "mastery_progression_rate",
        "calibration_quality",
        "session_efficiency",
    )

    rag_score = _average_with_runner_formula(modules, _metric_scores(output, rag_keys))
    agent_score = _average_with_runner_formula(modules, _metric_scores(output, agent_keys))
    outcome_score = _average_with_runner_formula(modules, _metric_scores(output, outcome_keys))
    system_score = _system_score_with_runner_formula(modules, rag_score, agent_score, outcome_score)

    output["rag_score"] = (
        rag_score if rag_score is not None else _not_available("required component scores unavailable")
    )
    output["agent_score"] = (
        agent_score if agent_score is not None else _not_available("required component scores unavailable")
    )
    output["outcome_score"] = (
        outcome_score if outcome_score is not None else _not_available("required component scores unavailable")
    )
    output["system_score"] = (
        system_score if system_score is not None else _not_available("required component scores unavailable")
    )

    return save_metrics_txt_report(output, output_dir=output_dir)


def run_offline_evaluation(
    input_path: str | Path | None = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> str:
    return asyncio.run(run_offline_evaluation_async(input_path, output_dir))


def main(argv: list[str] | None = None) -> str:
    parser = argparse.ArgumentParser(description="Run EduMind offline evaluation metrics.")
    parser.add_argument(
        "input_path",
        nargs="?",
        default=str(DEFAULT_FIXTURE),
        help="Path to a JSON evaluation input fixture.",
    )
    args = parser.parse_args(argv)

    report_path = run_offline_evaluation(args.input_path)
    print(f"Saved report: {report_path}")
    return report_path


if __name__ == "__main__":
    main()
