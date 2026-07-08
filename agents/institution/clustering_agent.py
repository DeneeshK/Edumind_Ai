"""
agents/institution/clustering_agent.py

Groups classroom students into meaningful, named learning clusters
("fast learners", "needs revision", "high practice low accuracy", ...) from
deterministic per-student feature vectors, and proposes a task per cluster.

The numbers come from classroom_analytics.student_table(); the LLM only
labels and explains groups. Student ids are validated against the roster so a
hallucinated id can never leak into a cluster.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from clients.groq_client import generate
from config import settings

_SYSTEM = """
You are EduMind's student-grouping specialist for teachers.
You receive per-student metrics (0-1 scales unless noted). Group ALL students
into 2-5 meaningful clusters. Each student appears in exactly one cluster.

Good cluster labels describe learning behaviour, e.g.:
"Fast learners", "Exam ready", "Needs revision", "Weak fundamentals",
"High engagement, low accuracy", "Low engagement — re-activate",
"Excellent conceptual understanding", "Needs mentoring".

Rules:
- Use ONLY the student_id values given. Never invent ids.
- rationale explains the shared pattern using the metrics.
- recommended_task is a concrete task the teacher can assign this cluster.

Return ONLY valid JSON:
{
  "clusters": [
    {
      "label": "cluster name",
      "student_ids": ["..."],
      "rationale": "1-2 sentences",
      "recommended_task": "one concrete task"
    }
  ]
}
"""


async def cluster_students(student_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Cluster students from analytics feature rows. Returns {"clusters": [...]}."""
    if len(student_rows) < 2:
        raise ValueError("Clustering needs at least 2 active students with data")

    features = [
        {
            "student_id": s["student_id"],
            "name": s["name"],
            "avg_progress": s["avg_progress"],
            "avg_mastery": s["avg_mastery"],
            "avg_test_score": s["avg_test_score"],
            "weak_concepts": s["weak_concepts"],
            "doubt_count": s["doubt_count"],
            "active_days_30": s["active_days_30"],
            "calibration_delta": s.get("avg_calibration_delta"),
            "risk": s["risk"],
        }
        for s in student_rows
    ]
    raw = await generate(
        messages=[{"role": "user", "content": "Students:\n" + json.dumps(features)}],
        model=settings.reasoning_model,
        system=_SYSTEM,
        json_mode=True,
        max_tokens=2500,
        _caller="clustering_agent",
    )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Clustering agent returned invalid JSON: {}", exc)
        raise ValueError("Clustering failed. Please try again.") from exc

    valid_ids = {s["student_id"] for s in student_rows}
    names = {s["student_id"]: s["name"] for s in student_rows}
    seen: set[str] = set()
    clusters = []
    for item in data.get("clusters") or []:
        if not isinstance(item, dict):
            continue
        ids = [
            sid for sid in (item.get("student_ids") or [])
            if sid in valid_ids and sid not in seen
        ]
        if not ids:
            continue
        seen.update(ids)
        clusters.append({
            "label": str(item.get("label") or "Group").strip()[:80],
            "student_ids": ids,
            "students": [{"student_id": sid, "name": names[sid]} for sid in ids],
            "rationale": str(item.get("rationale") or "").strip()[:400],
            "recommended_task": str(item.get("recommended_task") or "").strip()[:300],
        })

    leftover = valid_ids - seen
    if leftover:
        clusters.append({
            "label": "Unclustered",
            "student_ids": sorted(leftover),
            "students": [{"student_id": sid, "name": names[sid]} for sid in sorted(leftover)],
            "rationale": "Not enough signal yet to place these students in a group.",
            "recommended_task": "Check in individually to gather more learning data.",
        })
    if not clusters:
        raise ValueError("No usable clusters were generated. Please try again.")
    return {"clusters": clusters}
