"""
agents/institution/revision_planner_agent.py

Builds a class-wide revision plan from the analytics snapshot: daily and
weekly revision blocks, priority concepts, practice suggestions, recommended
tests, and pacing advice. Teachers review the plan and can publish it to the
classroom stream.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from loguru import logger

from clients.groq_client import generate
from config import settings

_SYSTEM = """
You are EduMind's revision planning specialist for classroom teachers.
You receive a classroom analytics snapshot (weak concepts, student groups,
doubt hotspots, test performance). Design a practical revision plan for the
requested number of days.

Rules:
- Prioritise the weakest concepts and the most-asked doubt topics.
- Daily entries are short and realistic (30-60 min of revision per day).
- Include which student group each daily focus helps most when relevant.
- recommended_tests are topics the teacher should generate tests for.
- pace_advice comments on whether the class should slow down, keep pace, or
  accelerate, citing the snapshot numbers.

Return ONLY valid JSON:
{
  "title": "plan title",
  "priority_concepts": ["concept", "..."],
  "daily_plan": [
    {"day": 1, "focus": "concept/topic", "activities": ["..."], "target_group": "everyone|group name"}
  ],
  "weekly_goals": ["..."],
  "practice_suggestions": ["..."],
  "recommended_tests": ["topic", "..."],
  "pace_advice": "2-3 sentences"
}
"""


async def generate_revision_plan(
    snapshot: dict[str, Any],
    days: int = 7,
) -> dict[str, Any]:
    """Generate a class revision plan for the next `days` days."""
    days = max(3, min(int(days), 30))
    prompt = (
        f"Plan revision for the next {days} days starting {date.today().isoformat()}.\n"
        "Classroom analytics snapshot:\n" + json.dumps(snapshot, default=str)[:12000]
    )
    raw = await generate(
        messages=[{"role": "user", "content": prompt}],
        model=settings.reasoning_model,
        system=_SYSTEM,
        json_mode=True,
        max_tokens=3000,
        _caller="revision_planner_agent",
    )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Revision planner returned invalid JSON: {}", exc)
        raise ValueError("Revision plan generation failed. Please try again.") from exc

    daily = []
    for item in data.get("daily_plan") or []:
        if not isinstance(item, dict):
            continue
        activities = [str(a).strip() for a in (item.get("activities") or []) if str(a).strip()]
        daily.append({
            "day": int(item.get("day") or len(daily) + 1),
            "focus": str(item.get("focus") or "").strip()[:160],
            "activities": activities[:5],
            "target_group": str(item.get("target_group") or "everyone").strip()[:80],
        })
    if not daily:
        raise ValueError("No usable revision plan was generated. Please try again.")

    def _str_list(key: str, cap: int) -> list[str]:
        return [str(x).strip()[:200] for x in (data.get(key) or []) if str(x).strip()][:cap]

    return {
        "title": str(data.get("title") or "Class Revision Plan").strip()[:120],
        "days": days,
        "priority_concepts": _str_list("priority_concepts", 10),
        "daily_plan": daily[:days],
        "weekly_goals": _str_list("weekly_goals", 6),
        "practice_suggestions": _str_list("practice_suggestions", 8),
        "recommended_tests": _str_list("recommended_tests", 5),
        "pace_advice": str(data.get("pace_advice") or "").strip()[:600],
    }


def plan_to_markdown(plan: dict[str, Any]) -> str:
    """Render a revision plan as markdown for publishing to the class stream."""
    lines = [f"# {plan['title']}", ""]
    if plan.get("pace_advice"):
        lines += [f"> {plan['pace_advice']}", ""]
    if plan.get("priority_concepts"):
        lines += ["**Priority concepts:** " + ", ".join(plan["priority_concepts"]), ""]
    lines.append("## Daily plan")
    for day in plan.get("daily_plan") or []:
        lines.append(f"- **Day {day['day']} — {day['focus']}**"
                     + (f" _(for: {day['target_group']})_" if day.get("target_group") not in ("", "everyone") else ""))
        for activity in day.get("activities") or []:
            lines.append(f"  - {activity}")
    if plan.get("weekly_goals"):
        lines += ["", "## Weekly goals"] + [f"- {g}" for g in plan["weekly_goals"]]
    if plan.get("practice_suggestions"):
        lines += ["", "## Practice"] + [f"- {p}" for p in plan["practice_suggestions"]]
    if plan.get("recommended_tests"):
        lines += ["", "## Recommended tests"] + [f"- {t}" for t in plan["recommended_tests"]]
    return "\n".join(lines)
