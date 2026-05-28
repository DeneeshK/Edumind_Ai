from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from loguru import logger

from db.postgres import get_conn
from evaluation.collector import (
    record_metric,
    write_aggregated_report,
    write_session_report,
)
from evaluation.metrics.agent_metrics import (
    curriculum_coverage_score,
    curriculum_ordering_score,
    lesson_quality_score,
    question_quality_score,
    routing_accuracy_score,
    scoring_consistency_score,
)
from evaluation.metrics.outcome_metrics import (
    calibration_quality_score,
    mastery_progression_rate,
    session_efficiency_score,
)
from evaluation.metrics.rag_metrics import (
    chromadb_precision_at_k,
    hyde_quality_score,
    rag_faithfulness_score,
    reranker_gain_score,
    tavily_relevance_score,
)


def _avg(values: list[float]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


class EvaluationRunner:
    """
    Collects evaluation data during a session and runs metrics at hook points.
    Public methods are intended to be called via asyncio.create_task().
    """

    def __init__(self, session_id: str, student_id: str, topic: str, pace: str):
        self.session_id = session_id
        self.student_id = student_id
        self.topic = topic
        self.pace = pace

        self._rag_scores: list[float] = []
        self._agent_scores: list[float] = []
        self._outcome_scores: list[float] = []
        self._metric_results: dict[str, list[dict]] = {}
        self._reteach_count: int = 0
        self._modules_attempted: int = 0
        self._modules_mastered: int = 0
        self._total_modules: int = 0
        self._calibration_deltas: list[float] = []
        self._session_start: datetime = datetime.now(timezone.utc)

    def _remember_scores(self, bucket: list[float], results: list) -> None:
        for result in results:
            if isinstance(result, Exception):
                continue
            if isinstance(result, dict) and "score" in result:
                try:
                    bucket.append(float(result["score"]))
                except (TypeError, ValueError):
                    pass

    def _remember_metric_results(self, named_results: list[tuple[str, object]]) -> None:
        for metric_name, result in named_results:
            if isinstance(result, Exception):
                continue
            if isinstance(result, dict):
                self._metric_results.setdefault(metric_name, []).append(result)

    async def on_rag_retrieve(
        self,
        query: str,
        hypothetical: str,
        concept_card: str,
        chroma_chunks_before_rerank: list[str],
        chroma_chunks_after_rerank: list[str],
        tavily_results: list[dict],
    ) -> None:
        """Run HyDE, precision, Tavily, and reranker metrics for one retrieve() call."""
        try:
            results = await asyncio.gather(
                hyde_quality_score(
                    query,
                    hypothetical,
                    concept_card,
                    self.session_id,
                    self.student_id,
                ),
                chromadb_precision_at_k(
                    query,
                    chroma_chunks_after_rerank,
                    self.session_id,
                    self.student_id,
                ),
                tavily_relevance_score(
                    query,
                    tavily_results,
                    self.session_id,
                    self.student_id,
                ),
                reranker_gain_score(
                    query,
                    chroma_chunks_before_rerank,
                    chroma_chunks_after_rerank,
                    self.session_id,
                    self.student_id,
                ),
                return_exceptions=True,
            )
            self._remember_metric_results([
                ("hyde_quality", results[0]),
                ("chromadb_precision_at_k", results[1]),
                ("tavily_relevance", results[2]),
                ("reranker_gain", results[3]),
            ])
            self._remember_scores(self._rag_scores, results)
        except Exception as exc:
            logger.warning("EvaluationRunner.on_rag_retrieve failed: {}", exc)

    async def on_lesson_delivered(
        self,
        lesson_text: str,
        retrieved_chunks: list[str],
        module: dict,
    ) -> None:
        """Run lesson faithfulness and quality metrics."""
        try:
            results = await asyncio.gather(
                rag_faithfulness_score(
                    lesson_text,
                    retrieved_chunks,
                    self.session_id,
                    self.student_id,
                ),
                lesson_quality_score(
                    lesson_text,
                    module,
                    self.pace,
                    self.session_id,
                    self.student_id,
                ),
                return_exceptions=True,
            )
            if not self.topic:
                self.topic = module.get("concept", "")
            self._remember_metric_results([
                ("rag_faithfulness", results[0]),
                ("lesson_quality", results[1]),
            ])
            self._remember_scores(self._rag_scores, results[:1])
            self._remember_scores(self._agent_scores, results[1:])
        except Exception as exc:
            logger.warning("EvaluationRunner.on_lesson_delivered failed: {}", exc)

    async def on_curriculum_built(self, modules: list[dict]) -> None:
        """Run curriculum coverage and ordering metrics."""
        try:
            self._total_modules = max(self._total_modules, len(modules))
            results = await asyncio.gather(
                curriculum_coverage_score(
                    self.topic,
                    "",
                    modules,
                    self.session_id,
                    self.student_id,
                ),
                curriculum_ordering_score(
                    modules,
                    self.session_id,
                    self.student_id,
                ),
                return_exceptions=True,
            )
            self._remember_metric_results([
                ("curriculum_coverage", results[0]),
                ("curriculum_ordering", results[1]),
            ])
            self._remember_scores(self._agent_scores, results)
        except Exception as exc:
            logger.warning("EvaluationRunner.on_curriculum_built failed: {}", exc)

    async def on_evaluation_complete(
        self,
        concept: str,
        qa_log: list[dict],
        questions_asked: list[dict],
        lesson_text: str,
        mastery_score: float,
        advance_threshold: float,
        actual_action: str,
        misconception_type: str | None,
        reteach_count: int,
        calibration_delta: float,
        mastery_history: list[float],
        modules_attempted: int,
        modules_mastered: int,
    ) -> None:
        """Run evaluator and learning outcome metrics after one EvaluationReport."""
        try:
            self._modules_attempted = max(self._modules_attempted, int(modules_attempted))
            self._modules_mastered = max(self._modules_mastered, int(modules_mastered))
            self._reteach_count = max(self._reteach_count, int(reteach_count))
            self._calibration_deltas.append(float(calibration_delta))

            routing = routing_accuracy_score(
                mastery_score,
                advance_threshold,
                actual_action,
                misconception_type,
                reteach_count,
                self.session_id,
                self.student_id,
            )
            progression = mastery_progression_rate(
                mastery_history,
                concept,
                self.session_id,
                self.student_id,
            )

            async_results = await asyncio.gather(
                question_quality_score(
                    questions_asked,
                    lesson_text,
                    self.session_id,
                    self.student_id,
                ),
                scoring_consistency_score(
                    qa_log,
                    concept,
                    self.session_id,
                    self.student_id,
                ),
                return_exceptions=True,
            )

            self._remember_metric_results([
                ("routing_accuracy", routing),
                ("question_quality", async_results[0]),
                ("scoring_consistency", async_results[1]),
                ("mastery_progression_rate", progression),
            ])
            self._remember_scores(self._agent_scores, [routing] + async_results)
            self._remember_scores(self._outcome_scores, [progression])
            if (actual_action or "").upper() == "RETEACH":
                self._reteach_count += 1
        except Exception as exc:
            logger.warning("EvaluationRunner.on_evaluation_complete failed: {}", exc)

    async def on_session_end(
        self,
        modules_mastered: int,
        total_curriculum_modules: int,
        reteach_events: int,
        calibration_deltas: list[float],
    ) -> None:
        """Compute and write the final session report."""
        try:
            duration = (
                datetime.now(timezone.utc) - self._session_start
            ).total_seconds() / 60.0
            self._modules_mastered = max(self._modules_mastered, int(modules_mastered))
            self._total_modules = max(self._total_modules, int(total_curriculum_modules))
            self._reteach_count = max(self._reteach_count, int(reteach_events))
            if calibration_deltas:
                self._calibration_deltas = [float(v) for v in calibration_deltas]

            calibration = calibration_quality_score(
                self._calibration_deltas,
                self.session_id,
                self.student_id,
            )
            efficiency = session_efficiency_score(
                self._modules_attempted or len(self._calibration_deltas),
                self._modules_mastered,
                self._total_modules,
                self._reteach_count,
                duration,
                self.pace,
                self.session_id,
                self.student_id,
            )
            self._remember_scores(self._outcome_scores, [calibration, efficiency])
            self._remember_metric_results([
                ("calibration_quality", calibration),
                ("session_efficiency", efficiency),
            ])

            rag_score = _avg(self._rag_scores)
            agent_score = _avg(self._agent_scores)
            outcome_score = _avg(self._outcome_scores)
            system_score = self._compute_system_score(rag_score, agent_score, outcome_score)

            full_report = {
                "session_id": self.session_id,
                "student_id": self.student_id,
                "topic": self.topic,
                "pace": self.pace,
                "rag_scores": self._rag_scores,
                "agent_scores": self._agent_scores,
                "outcome_scores": self._outcome_scores,
                "calculated_metrics": self._metric_results,
                "duration_minutes": duration,
                "modules_attempted": self._modules_attempted,
                "modules_mastered": self._modules_mastered,
                "total_curriculum_modules": self._total_modules,
                "reteach_events": self._reteach_count,
                "calibration_deltas": self._calibration_deltas,
            }

            if system_score is not None:
                await record_metric(
                    "session_system_score",
                    "system",
                    system_score,
                    full_report,
                    self.session_id,
                    self.student_id,
                )

            await write_session_report(
                self.session_id,
                self.student_id,
                self.topic,
                rag_score,
                agent_score,
                outcome_score,
                system_score,
                full_report,
            )
        except Exception as exc:
            logger.warning("EvaluationRunner.on_session_end failed: {}", exc)

    def _compute_system_score(
        self,
        rag_score: float | None,
        agent_score: float | None,
        outcome_score: float | None,
    ) -> float | None:
        """
        Weighted combination:
        system_score = 0.35 * rag + 0.40 * agent + 0.25 * outcome.
        Missing components are re-normalized across available scores.
        """
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


async def build_aggregated_report(period_type: str = "weekly") -> dict:
    """Aggregate recent eval_session_reports and write eval_aggregated_reports."""
    now = datetime.now(timezone.utc)
    days = 30 if period_type == "monthly" else 7
    period_start = now - timedelta(days=days)

    try:
        async with get_conn() as conn:
            rows = await conn.fetch(
                """
                SELECT student_id, rag_score, agent_score, system_score
                FROM eval_session_reports
                WHERE created_at >= $1 AND created_at < $2
                """,
                period_start,
                now,
            )

        rag_values = [float(r["rag_score"]) for r in rows if r["rag_score"] is not None]
        agent_values = [float(r["agent_score"]) for r in rows if r["agent_score"] is not None]
        system_values = [float(r["system_score"]) for r in rows if r["system_score"] is not None]
        students = {r["student_id"] for r in rows if r["student_id"]}
        report = {
            "period_type": period_type,
            "period_start": period_start.isoformat(),
            "period_end": now.isoformat(),
            "sessions_counted": len(rows),
            "students_counted": len(students),
        }

        await write_aggregated_report(
            period_type,
            period_start,
            now,
            _avg(rag_values),
            _avg(agent_values),
            _avg(system_values),
            len(rows),
            len(students),
            report,
        )
        return report
    except Exception as exc:
        logger.warning("build_aggregated_report failed: {}", exc)
        return {"error": str(exc), "period_type": period_type}
