from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, Query

from app.auth import require_current_user
from db.postgres import get_conn

router = APIRouter(prefix="/eval", tags=["evaluation"])


def _decode_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _row_to_dict(row) -> dict:
    data = dict(row)
    for key in ("details_json", "full_report_json", "modules_covered", "raw_json"):
        if key in data:
            data[key] = _decode_json(data[key])
    return data


@router.get("/session/{session_id}")
async def get_session_eval(
    session_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """Return the aggregated evaluation report for a session."""
    async with get_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM eval_session_reports
             WHERE session_id=$1 AND student_id=$2
            """,
            session_id,
            current_user["student_id"],
        )
    if not row:
        return {"error": "No evaluation report found for this session."}
    return _row_to_dict(row)


@router.get("/metrics")
async def list_metrics(
    student_id: str | None = Query(None),
    metric_name: str | None = Query(None),
    component: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """List recent metric runs, filterable by student, metric name, or component."""
    conditions = ["student_id=$1"]
    params: list[Any] = [current_user["student_id"]]
    if metric_name:
        conditions.append(f"metric_name=${len(params) + 1}")
        params.append(metric_name)
    if component:
        conditions.append(f"component=${len(params) + 1}")
        params.append(component)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(limit)

    async with get_conn() as conn:
        rows = await conn.fetch(
            f"""
            SELECT * FROM eval_metric_runs
            {where}
            ORDER BY created_at DESC
            LIMIT ${len(params)}
            """,
            *params,
        )
    return [_row_to_dict(r) for r in rows]


@router.get("/aggregated")
async def get_aggregated_reports(period_type: str = Query("weekly")):
    """Return recent aggregated reports."""
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM eval_aggregated_reports
            WHERE period_type=$1
            ORDER BY created_at DESC
            LIMIT 10
            """,
            period_type,
        )
    return [_row_to_dict(r) for r in rows]


@router.post("/run-manual/{session_id}")
async def run_manual_eval(
    session_id: str,
    current_user: dict[str, Any] = Depends(require_current_user),
):
    """
    Manually trigger a report for a past session using stored evaluation data.
    Live RAG and lesson metrics are not reconstructed from storage.
    """
    async with get_conn() as conn:
        evals = await conn.fetch(
            """
            SELECT * FROM evaluation_history
             WHERE session_id=$1 AND student_id=$2
             ORDER BY created_at
            """,
            session_id,
            current_user["student_id"],
        )
        session = await conn.fetchrow(
            """
            SELECT * FROM session_memory
             WHERE session_id=$1 AND student_id=$2
            """,
            session_id,
            current_user["student_id"],
        )

    if not evals:
        return {"error": "No evaluation data found for this session."}

    from evaluation.collector import record_metric, write_session_report
    from evaluation.metrics.outcome_metrics import (
        calibration_quality_score,
        session_efficiency_score,
    )

    student_id = evals[0]["student_id"]
    calibration_deltas = [float(r["calibration_delta"]) for r in evals]
    mastered = sum(1 for r in evals if float(r["mastery_score"]) >= 0.72)
    modules_covered = []
    if session and session["modules_covered"]:
        modules_covered = _decode_json(session["modules_covered"]) or []

    cal_result = calibration_quality_score(calibration_deltas, session_id, student_id)
    eff_result = session_efficiency_score(
        modules_attempted=len(evals),
        modules_mastered=mastered,
        total_modules_in_curriculum=max(len(modules_covered), len(evals), 1),
        reteach_events=sum(1 for r in evals if r["recommended_action"] == "RETEACH"),
        session_duration_minutes=0.0,
        pace="medium",
        session_id=session_id,
        student_id=student_id,
    )
    outcome_score = (cal_result["score"] + eff_result["score"]) / 2.0

    report = {
        "session_id": session_id,
        "student_id": student_id,
        "evaluations_found": len(evals),
        "calibration_quality": cal_result,
        "session_efficiency": eff_result,
        "note": "Manual report reconstructed from stored evaluation_history and session_memory.",
        "trigger": "manual",
    }
    await record_metric(
        "manual_outcome_score",
        "outcome",
        outcome_score,
        report,
        session_id,
        student_id,
        trigger="manual",
    )
    await write_session_report(
        session_id,
        student_id,
        "",
        None,
        None,
        outcome_score,
        outcome_score,
        report,
    )
    return report
