from __future__ import annotations

from typing import Any

from loguru import logger

from config import settings

_scheduler: Any | None = None


async def _run_weekly_aggregation() -> None:
    """Read session reports from the last 7 days and write an aggregate report."""
    try:
        from evaluation.runner import build_aggregated_report

        await build_aggregated_report(period_type="weekly")
    except Exception as exc:
        logger.warning("Weekly eval aggregation failed: {}", exc)


async def _run_monthly_aggregation() -> None:
    """Read session reports from the last 30 days and write an aggregate report."""
    try:
        from evaluation.runner import build_aggregated_report

        await build_aggregated_report(period_type="monthly")
    except Exception as exc:
        logger.warning("Monthly eval aggregation failed: {}", exc)


def start_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        return

    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger
    except Exception as exc:
        logger.warning("Evaluation scheduler unavailable: {}", exc)
        return

    _scheduler = AsyncIOScheduler(timezone=settings.eval_schedule_timezone)

    if settings.eval_schedule_weekly:
        _scheduler.add_job(
            _run_weekly_aggregation,
            CronTrigger(day_of_week="sun", hour=2, minute=0),
            id="weekly_eval",
            replace_existing=True,
            misfire_grace_time=3600,
        )

    if settings.eval_schedule_monthly:
        _scheduler.add_job(
            _run_monthly_aggregation,
            CronTrigger(day=1, hour=3, minute=0),
            id="monthly_eval",
            replace_existing=True,
            misfire_grace_time=3600,
        )

    _scheduler.start()
    logger.info(
        "Evaluation scheduler started (weekly={}, monthly={})",
        settings.eval_schedule_weekly,
        settings.eval_schedule_monthly,
    )


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None

