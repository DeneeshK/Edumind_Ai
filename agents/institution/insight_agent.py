"""
agents/institution/insight_agent.py

Turns the deterministic classroom analytics snapshot into short, actionable
natural-language insights for teachers ("65% of students repeatedly fail
rotational dynamics — schedule one revision before the next chapter").
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from clients.groq_client import generate
from config import settings

_SYSTEM = """
You are EduMind's classroom insight analyst.
You receive a JSON analytics snapshot of one classroom. Produce 4-8 concrete,
evidence-backed insights a teacher can act on tomorrow.

Rules:
- Every insight cites the numbers that support it (percentages, counts, names
  of concepts). Never invent data that is not in the snapshot.
- Refer to students by name when the snapshot provides names.
- severity: "info" (positive/neutral), "warning" (needs attention),
  "critical" (act now).
- suggested_action is one imperative sentence.

Return ONLY valid JSON:
{
  "insights": [
    {
      "title": "short headline",
      "detail": "2-3 sentences with the supporting numbers",
      "severity": "info" | "warning" | "critical",
      "suggested_action": "one imperative sentence"
    }
  ],
  "summary": "2-3 sentence overall state of the class"
}
"""


async def generate_insights(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Generate insight cards from an analytics snapshot."""
    raw = await generate(
        messages=[{
            "role": "user",
            "content": "Classroom analytics snapshot:\n" + json.dumps(snapshot, default=str)[:12000],
        }],
        model=settings.reasoning_model,
        system=_SYSTEM,
        json_mode=True,
        max_tokens=2500,
        _caller="insight_agent",
    )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Insight agent returned invalid JSON: {}", exc)
        raise ValueError("Insight generation failed. Please try again.") from exc

    insights = []
    for item in data.get("insights") or []:
        if not isinstance(item, dict) or not str(item.get("title") or "").strip():
            continue
        severity = str(item.get("severity") or "info").lower()
        insights.append({
            "title": str(item["title"]).strip()[:140],
            "detail": str(item.get("detail") or "").strip()[:600],
            "severity": severity if severity in ("info", "warning", "critical") else "info",
            "suggested_action": str(item.get("suggested_action") or "").strip()[:240],
        })
    if not insights:
        raise ValueError("No usable insights were generated. Please try again.")
    return {"insights": insights, "summary": str(data.get("summary") or "").strip()[:600]}
