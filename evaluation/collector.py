from __future__ import annotations

import json
from datetime import datetime

from loguru import logger

from db.postgres import get_conn
from evaluation.report_writer import save_metrics_txt_report


def _json_payload(details: dict) -> str:
    return json.dumps(details or {}, default=str)


async def record_metric(
    metric_name: str,
    component: str,
    score: float,
    details: dict,
    session_id: str | None = None,
    student_id: str | None = None,
    trigger: str = "auto",
) -> None:
    """
    Write one metric result to eval_metric_runs.
    All evaluation functions call this. Never raises; failures are logged only.
    """
    try:
        async with get_conn() as conn:
            await conn.execute(
                """
                INSERT INTO eval_metric_runs
                    (student_id, session_id, metric_name, component, score, details_json, trigger)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                student_id,
                session_id,
                metric_name,
                component,
                round(float(score), 6),
                _json_payload(details),
                trigger,
            )
    except Exception as exc:
        logger.warning("eval collector failed for '{}': {}", metric_name, exc)


async def write_session_report(
    session_id: str,
    student_id: str,
    topic: str,
    rag_score: float | None,
    agent_score: float | None,
    outcome_score: float | None,
    system_score: float | None,
    full_report: dict,
) -> None:
    """Write the aggregated session-level report."""
    report_payload = dict(full_report or {})
    report_payload["session_id"] = session_id
    report_payload["student_id"] = student_id
    if topic:
        report_payload["topic"] = topic
    else:
        report_payload.setdefault("topic", "")
    report_payload["rag_score"] = rag_score
    report_payload["agent_score"] = agent_score
    report_payload["outcome_score"] = outcome_score
    report_payload["system_score"] = system_score

    try:
        async with get_conn() as conn:
            await conn.execute(
                """
                INSERT INTO eval_session_reports
                    (session_id, student_id, topic,
                     rag_score, agent_score, outcome_score, system_score, full_report_json)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (session_id) DO UPDATE SET
                    student_id = EXCLUDED.student_id,
                    topic = EXCLUDED.topic,
                    rag_score = EXCLUDED.rag_score,
                    agent_score = EXCLUDED.agent_score,
                    outcome_score = EXCLUDED.outcome_score,
                    system_score = EXCLUDED.system_score,
                    full_report_json = EXCLUDED.full_report_json
                """,
                session_id,
                student_id,
                topic or "",
                rag_score,
                agent_score,
                outcome_score,
                system_score,
                _json_payload(full_report),
            )
    except Exception as exc:
        logger.warning("eval session report write failed: {}", exc)

    try:
        txt_report_path = save_metrics_txt_report(report_payload)
        logger.info("eval TXT report saved: {}", txt_report_path)
    except Exception as exc:
        logger.warning("eval TXT report write failed: {}", exc)


async def write_aggregated_report(
    period_type: str,
    period_start: datetime,
    period_end: datetime,
    avg_rag_score: float | None,
    avg_agent_score: float | None,
    avg_system_score: float | None,
    sessions_counted: int,
    students_counted: int,
    full_report: dict,
) -> None:
    """Write a weekly/monthly aggregate report."""
    try:
        async with get_conn() as conn:
            await conn.execute(
                """
                INSERT INTO eval_aggregated_reports
                    (period_type, period_start, period_end, avg_rag_score,
                     avg_agent_score, avg_system_score, sessions_counted,
                     students_counted, full_report_json)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                period_type,
                period_start,
                period_end,
                avg_rag_score,
                avg_agent_score,
                avg_system_score,
                sessions_counted,
                students_counted,
                _json_payload(full_report),
            )
    except Exception as exc:
        logger.warning("eval aggregated report write failed: {}", exc)
