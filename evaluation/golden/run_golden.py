"""
evaluation/golden/run_golden.py
Golden regression runner for EduMind's live prompt-driven flows.

Runs three suites of checked-in YAML cases through the real production code paths
(curriculum build, answer diagnosis, lesson generation) against the real Groq API,
and asserts deterministic + judge-based expectations. A prompt edit that degrades
curriculum quality or answer diagnosis fails a case here — the point of the suite.

Usage:
    python -m evaluation.golden.run_golden --suite all --report out.json
    python -m evaluation.golden.run_golden --suite diagnosis

Exit code is 1 if any case fails (or errors), 0 otherwise. A markdown summary is
written next to the JSON report.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from evaluation.golden import harness

CASES_DIR = Path(__file__).parent / "cases"
SUITES = ("curriculum", "diagnosis", "lesson")

# Practice-task detector — mirrors evaluation.metrics.agent_metrics practice check.
import re as _re

_PRACTICE_RE = _re.compile(r"practice|your turn|example", _re.I)


# ── small check helpers ───────────────────────────────────────────────────────

def _check(name: str, passed: bool, detail: str = "") -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _contains(haystack: str, needle: str) -> bool:
    return needle.strip().lower() in haystack.lower()


def _total_tokens() -> int:
    """Sum prompt+completion tokens recorded by the live token accounting so far."""
    try:
        from core.metrics import metrics
        total = 0.0
        for metric in metrics.llm_tokens.collect():
            for sample in metric.samples:
                if sample.name.endswith("_total"):
                    total += sample.value
        return int(total)
    except Exception:
        return 0


# ── suite runners ─────────────────────────────────────────────────────────────

async def _run_curriculum_case(case: dict[str, Any], judge_runs: int) -> dict[str, Any]:
    profile = case["profile"]
    a = case.get("assertions") or {}
    plan = await harness.build_curriculum_plan(profile)

    modules = harness.modules_as_dicts(plan)
    concepts_text = " || ".join(harness.module_concepts(plan))

    checks: list[dict[str, Any]] = []
    scores: dict[str, Any] = {"module_count": len(plan.modules)}

    # Deterministic: prerequisite ordering (existing curriculum_ordering_score).
    # NOTE: this metric is high-variance on real curricula — it exact-matches
    # LLM-generated prerequisite name strings against earlier module concepts, so the
    # same case swings 0.0–1.0 across runs (and heavy known-concept compression drives
    # it to 0.0 legitimately). It is therefore REPORTED but gated only as a per-case
    # floor (0.0 = report-only by default). The stable, meaningful curriculum gates are
    # the coverage judge + do_not_include/must_include checks below.
    from evaluation.metrics.agent_metrics import curriculum_ordering_score
    ordering = (await curriculum_ordering_score(modules, "golden", "golden"))["score"]
    ordering_min = float(a.get("ordering_min", 0.0))
    scores["ordering"] = round(ordering, 4)
    checks.append(_check(
        f"ordering>={ordering_min}", ordering >= ordering_min,
        f"ordering={ordering:.3f}",
    ))

    # Deterministic: module count band
    band = a.get("module_count_band") or [1, 999]
    lo, hi = int(band[0]), int(band[1])
    n = len(plan.modules)
    checks.append(_check(
        f"module_count in [{lo},{hi}]", lo <= n <= hi, f"count={n}",
    ))

    # Deterministic: do_not_include exclusion
    do_not = list(profile.get("do_not_include") or [])
    violations = [t for t in do_not if _contains(concepts_text, t)]
    checks.append(_check(
        "no do_not_include concept", not violations,
        f"violations={violations}" if violations else "clean",
    ))

    # Deterministic: must_include coverage
    must = list(profile.get("must_include") or [])
    missing = [t for t in must if not _contains(concepts_text, t)]
    checks.append(_check(
        "all must_include present", not missing,
        f"missing={missing}" if missing else "all present",
    ))

    # Judge: curriculum coverage (mean of N runs)
    from evaluation.metrics.agent_metrics import curriculum_coverage_score
    domain = str(profile.get("target_context") or plan.domain or profile.get("topic"))
    cov_min = float(a.get("coverage_min", 0.6))
    cov_mean, cov_runs = await harness.judge_mean(
        lambda: curriculum_coverage_score(
            str(profile["topic"]), domain, modules, "golden", "golden"
        ),
        judge_runs,
    )
    scores["coverage_judge"] = round(cov_mean, 4)
    scores["coverage_runs"] = [round(s, 4) for s in cov_runs]
    checks.append(_check(
        f"coverage_judge>={cov_min}", cov_mean >= cov_min, f"mean={cov_mean:.3f} runs={cov_runs}",
    ))

    return {"scores": scores, "checks": checks}


async def _run_diagnosis_case(case: dict[str, Any], judge_runs: int) -> dict[str, Any]:
    expect = case.get("expect") or {}
    diagnosis = await harness.run_diagnosis(case)

    signal = str(diagnosis.get("mastery_signal") or "").strip().lower()
    weak = [str(w).strip().lower() for w in (diagnosis.get("weak_concepts") or [])]
    weak_text = " || ".join(weak)

    checks: list[dict[str, Any]] = []
    scores = {"mastery_signal": signal, "weak_concepts": diagnosis.get("weak_concepts") or []}

    if "mastery_signal" in expect:
        allowed = [s.lower() for s in expect["mastery_signal"]]
        checks.append(_check(
            f"mastery_signal in {allowed}", signal in allowed, f"got={signal!r}",
        ))
    if "mastery_signal_not" in expect:
        banned = [s.lower() for s in expect["mastery_signal_not"]]
        checks.append(_check(
            f"mastery_signal not in {banned}", signal not in banned, f"got={signal!r}",
        ))
    for term in expect.get("weak_concepts_must") or []:
        checks.append(_check(
            f"weak_concept present: {term}", _contains(weak_text, term), f"weak={weak}",
        ))
    for term in expect.get("weak_concepts_must_not") or []:
        checks.append(_check(
            f"weak_concept absent: {term}", not _contains(weak_text, term), f"weak={weak}",
        ))

    return {"scores": scores, "checks": checks}


async def _run_lesson_case(case: dict[str, Any], judge_runs: int) -> dict[str, Any]:
    a = case.get("assertions") or {}
    lesson = await harness.run_lesson(case)
    if isinstance(lesson, dict):  # rate-limit fallback shape
        raise RuntimeError(f"lesson generation returned error: {lesson.get('message')}")

    word_count = len(lesson.split())
    checks: list[dict[str, Any]] = []
    scores: dict[str, Any] = {"word_count": word_count}

    # Deterministic: required concepts present by name
    required = list(a.get("required_concepts") or [])
    missing = [c for c in required if not _contains(lesson, c)]
    checks.append(_check(
        "required concepts present", not missing,
        f"missing={missing}" if missing else "all present",
    ))

    # Deterministic: length band (words)
    band = a.get("length_band_words") or [1, 100000]
    lo, hi = int(band[0]), int(band[1])
    checks.append(_check(
        f"length in [{lo},{hi}] words", lo <= word_count <= hi, f"words={word_count}",
    ))

    # Deterministic: has a practice/example section
    if a.get("has_practice", True):
        checks.append(_check(
            "has practice/example", bool(_PRACTICE_RE.search(lesson)),
        ))

    # Judge: lesson quality (mean of N runs)
    from evaluation.metrics.agent_metrics import lesson_quality_score
    q_min = float(a.get("quality_min", 0.6))
    pace = str(case["course"].get("pace") or "medium")
    q_mean, q_runs = await harness.judge_mean(
        lambda: lesson_quality_score(lesson, case["module"], pace, "golden", "golden"),
        judge_runs,
    )
    scores["quality_judge"] = round(q_mean, 4)
    scores["quality_runs"] = [round(s, 4) for s in q_runs]
    checks.append(_check(
        f"quality_judge>={q_min}", q_mean >= q_min, f"mean={q_mean:.3f} runs={q_runs}",
    ))

    return {"scores": scores, "checks": checks}


_SUITE_RUNNERS = {
    "curriculum": _run_curriculum_case,
    "diagnosis": _run_diagnosis_case,
    "lesson": _run_lesson_case,
}


# ── orchestration ─────────────────────────────────────────────────────────────

def _load_cases(suite: str) -> list[dict[str, Any]]:
    suite_dir = CASES_DIR / suite
    cases = []
    for path in sorted(suite_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        data.setdefault("name", path.stem)
        data["_file"] = str(path.relative_to(CASES_DIR.parent.parent))
        cases.append(data)
    return cases


async def run_suite(suite: str, judge_runs: int) -> list[dict[str, Any]]:
    runner = _SUITE_RUNNERS[suite]
    results = []
    for case in _load_cases(suite):
        name = case["name"]
        t0 = time.perf_counter()
        try:
            outcome = await runner(case, judge_runs)
            checks = outcome["checks"]
            passed = all(c["passed"] for c in checks)
            result = {
                "suite": suite, "name": name, "passed": passed,
                "checks": checks, "scores": outcome["scores"],
                "seconds": round(time.perf_counter() - t0, 1),
            }
        except Exception as exc:
            logger.exception("Golden case '{}/{}' errored", suite, name)
            result = {
                "suite": suite, "name": name, "passed": False,
                "checks": [_check("no-error", False, str(exc))],
                "scores": {}, "error": str(exc),
                "seconds": round(time.perf_counter() - t0, 1),
            }
        status = "PASS" if result["passed"] else "FAIL"
        logger.info("[{}] {}/{} ({}s)", status, suite, name, result["seconds"])
        results.append(result)
    return results


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# Golden eval report", ""]
    s = report["summary"]
    lines.append(f"- **Result:** {'✅ PASS' if s['passed'] else '❌ FAIL'}")
    lines.append(f"- **Cases:** {s['cases_passed']}/{s['cases_total']} passed")
    lines.append(f"- **Tokens (this run):** ~{s['tokens']:,}")
    lines.append(f"- **Duration:** {s['seconds']}s")
    lines.append("")
    for suite in SUITES:
        suite_results = [r for r in report["results"] if r["suite"] == suite]
        if not suite_results:
            continue
        lines.append(f"## {suite}")
        lines.append("")
        lines.append("| Case | Result | Key scores |")
        lines.append("| --- | --- | --- |")
        for r in suite_results:
            score_bits = ", ".join(
                f"{k}={v}" for k, v in r["scores"].items()
                if k in ("ordering", "module_count", "coverage_judge", "quality_judge",
                         "mastery_signal", "word_count")
            )
            lines.append(f"| {r['name']} | {'✅' if r['passed'] else '❌'} | {score_bits} |")
        # failing checks detail
        for r in suite_results:
            if not r["passed"]:
                failed = [c for c in r["checks"] if not c["passed"]]
                for c in failed:
                    lines.append(f"  - ❌ `{r['name']}` — {c['name']} ({c['detail']})")
        lines.append("")
    return "\n".join(lines)


async def main_async(args: argparse.Namespace) -> int:
    harness.disable_metric_persistence()
    suites = SUITES if args.suite == "all" else (args.suite,)

    tokens_before = _total_tokens()
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    for suite in suites:
        results.extend(await run_suite(suite, args.judge_runs))
    seconds = round(time.perf_counter() - t0, 1)
    tokens = _total_tokens() - tokens_before

    cases_passed = sum(1 for r in results if r["passed"])
    summary = {
        "passed": all(r["passed"] for r in results) and bool(results),
        "cases_total": len(results),
        "cases_passed": cases_passed,
        "tokens": tokens,
        "seconds": seconds,
        "judge_runs": args.judge_runs,
        "suites": list(suites),
    }
    report = {"summary": summary, "results": results}

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path = report_path.with_suffix(".md")
    md_path.write_text(_markdown(report), encoding="utf-8")

    logger.info(
        "Golden suite: {}/{} cases passed, ~{} tokens, {}s → {}",
        cases_passed, len(results), tokens, seconds, report_path,
    )
    print(_markdown(report))
    return 0 if summary["passed"] else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="EduMind golden eval runner")
    parser.add_argument("--suite", choices=("all", *SUITES), default="all")
    parser.add_argument("--report", default="evaluation/golden/reports/golden_report.json")
    parser.add_argument("--judge-runs", type=int, default=2,
                        help="times to run each judge-based score (mean is taken)")
    args = parser.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
