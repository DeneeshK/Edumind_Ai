"""
agents/institution/recommendation_agent.py

Per-student personalized recommendations: next lesson, what to revise,
practice focus, and study materials — derived from the student's
classroom-scoped drilldown data. Uses the small task model: this runs per
student, so it must stay cheap.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from clients.groq_client import generate
from config import settings

_SYSTEM = """
You are EduMind's personal learning coach.
You receive one student's learning data (weak/strong concepts, test history,
course progress, misconceptions). Produce focused recommendations.

Rules:
- Be specific: name the concepts and lessons from the data, never generic
  advice like "study more".
- Address the student directly ("you").
- If a misconception is listed, target it in revision items.

Return ONLY valid JSON:
{
  "next_step": "one sentence — the single most valuable next action",
  "revise": [{"concept": "...", "reason": "..."}],
  "practice": ["specific practice suggestion"],
  "focus_tip": "one sentence habit/strategy tip",
  "encouragement": "one genuine sentence grounded in their real progress"
}
"""


async def recommend_for_student(drilldown: dict[str, Any]) -> dict[str, Any]:
    """Generate personalized recommendations from a student drilldown payload."""
    compact = {
        "name": drilldown.get("name"),
        "summary": drilldown.get("summary"),
        "weak_concepts": drilldown.get("weak_concepts"),
        "strong_concepts": drilldown.get("strong_concepts"),
        "test_history": drilldown.get("test_history"),
        "misconceptions": drilldown.get("misconceptions"),
        "assignments": [
            {"title": a.get("classroom_course_title"), "progress": a.get("progress"),
             "completed_modules": a.get("completed_modules"),
             "module_count": a.get("module_count")}
            for a in (drilldown.get("assignments") or [])
        ],
    }
    raw = await generate(
        messages=[{"role": "user", "content": "Student data:\n" + json.dumps(compact, default=str)}],
        model=settings.small_task_model,
        system=_SYSTEM,
        json_mode=True,
        max_tokens=1200,
        _caller="recommendation_agent",
    )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Recommendation agent returned invalid JSON: {}", exc)
        raise ValueError("Recommendation generation failed. Please try again.") from exc

    revise = []
    for item in data.get("revise") or []:
        if isinstance(item, dict) and str(item.get("concept") or "").strip():
            revise.append({
                "concept": str(item["concept"]).strip()[:120],
                "reason": str(item.get("reason") or "").strip()[:240],
            })
    return {
        "next_step": str(data.get("next_step") or "").strip()[:300],
        "revise": revise[:6],
        "practice": [str(p).strip()[:240] for p in (data.get("practice") or []) if str(p).strip()][:5],
        "focus_tip": str(data.get("focus_tip") or "").strip()[:240],
        "encouragement": str(data.get("encouragement") or "").strip()[:240],
    }
