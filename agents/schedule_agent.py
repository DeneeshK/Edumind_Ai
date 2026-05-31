"""
agents/schedule_agent.py

Generates a day-by-day, slot-aware learning timetable for a course.

Flow:
  1. Evenly distribute modules across available study days.
  2. Assign start_time / end_time to each module based on chosen study slots.
  3. Call the reasoning model to enrich each day with a theme, study tip,
     overall advice, and weekly milestones.
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import date, timedelta
from typing import Any

from loguru import logger

from clients.groq_client import generate
from config import settings


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_duration_to_days(value: int, unit: str) -> int:
    unit = (unit or "days").lower().strip()
    if unit in ("week", "weeks"):
        return value * 7
    if unit in ("month", "months"):
        return value * 30
    return value


def _minutes_to_hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _slot_start_minutes(slot: str) -> int:
    defaults = {"morning": 360, "afternoon": 780, "evening": 1080, "night": 1260}
    s = slot.strip().lower()
    if s in defaults:
        return defaults[s]
    if ":" in s:
        try:
            h, m = s.split(":")
            return int(h) * 60 + int(m)
        except ValueError:
            pass
    return 360  # fallback to morning


def _format_minutes(minutes: int) -> str:
    """Convert raw minutes to a readable string e.g. '2 hrs 30 min'."""
    if minutes <= 0:
        return "0 min"
    h, m = divmod(minutes, 60)
    if h and m:
        return f"{h} hr{'s' if h > 1 else ''} {m} min"
    if h:
        return f"{h} hr{'s' if h > 1 else ''}"
    return f"{m} min"


# ── Packer — even distribution ────────────────────────────────────────────────

def _pack_modules_into_days(
    modules: list[dict[str, Any]],
    total_days: int,
    hours_per_day: float,
) -> list[dict[str, Any]]:
    """
    Distribute modules evenly across days.

    Strategy:
    - Calculate how many study days we actually need based on total module
      minutes vs daily capacity.
    - Spread modules as evenly as possible across those days so no day is
      overstuffed and later days aren't empty rest days.
    - Cap at total_days — if there are more days than needed the extras become
      rest days naturally.
    """
    capacity_mins = int(hours_per_day * 60)

    # Normalise module minutes — cap each module at capacity so one module
    # never exceeds a full day
    items = []
    for mod in modules:
        mins = int(mod.get("estimated_minutes") or 30)
        mins = max(5, min(mins, capacity_mins))   # floor 5 min, cap at day capacity
        items.append({
            "module_id":         mod.get("id") or mod.get("module_id") or "",
            "module_title":      mod.get("title") or mod.get("module_title") or "Module",
            "concept":           mod.get("concept") or "",
            "estimated_minutes": mins,
            "difficulty":        mod.get("difficulty") or "standard",
            "completed":         False,
        })

    if not items:
        return []

    total_content_mins = sum(it["estimated_minutes"] for it in items)

    # How many days do we actually need to fit everything?
    days_needed = math.ceil(total_content_mins / capacity_mins)
    # Use the smaller of (days_needed, total_days) as actual study days
    study_days = min(days_needed, total_days)
    study_days = max(1, study_days)

    # Target minutes per day — spread evenly
    target_per_day = math.ceil(total_content_mins / study_days)
    # Don't exceed capacity
    target_per_day = min(target_per_day, capacity_mins)

    days: list[dict[str, Any]] = []
    current_items: list[dict] = []
    used = 0

    for item in items:
        m = item["estimated_minutes"]
        # Start a new day if adding this module exceeds target AND we still
        # have days left
        if used + m > target_per_day and current_items and len(days) < study_days - 1:
            days.append({"items": current_items, "used_minutes": used})
            current_items, used = [], 0
        current_items.append(item)
        used += m

    if current_items:
        days.append({"items": current_items, "used_minutes": used})

    return days


# ── Time-slot assigner ────────────────────────────────────────────────────────

def _assign_times(
    items: list[dict[str, Any]],
    slots: list[str],
    actual_minutes: int,
) -> list[dict[str, Any]]:
    """
    Assign start_time / end_time to each module across the given slots.

    actual_minutes is the real number of content minutes for this day
    (not the full daily capacity). Each slot gets an equal share.
    A 5-minute break is added between consecutive modules within a slot.
    """
    if not slots:
        slots = ["morning"]

    total_budget_mins = actual_minutes
    # Give each slot an equal share — no rounding loss
    slot_budgets = {}
    base = total_budget_mins // len(slots)
    remainder = total_budget_mins % len(slots)
    for i, slot in enumerate(slots):
        slot_budgets[slot] = base + (1 if i < remainder else 0)

    result: list[dict[str, Any]] = []
    queue = list(items)

    for slot in slots:
        cursor = _slot_start_minutes(slot)
        budget = slot_budgets[slot]
        slot_items_count = 0

        while queue:
            item = queue[0]
            m = item["estimated_minutes"]
            # Budget only counts module minutes — gaps are cosmetic and don't
            # eat into study time budget
            if m <= budget:
                queue.pop(0)
                gap = 5 if slot_items_count > 0 else 0
                cursor += gap   # advance cursor by gap (cosmetic spacing)
                result.append({
                    **item,
                    "slot":       slot,
                    "start_time": _minutes_to_hhmm(cursor),
                    "end_time":   _minutes_to_hhmm(cursor + m),
                })
                cursor += m
                budget -= m     # only deduct module time from budget
                slot_items_count += 1
            else:
                break  # move to next slot

    # Anything left over — append to the last slot continuing from where it ended
    if queue:
        last_slot = slots[-1]
        # Find where the last item in that slot ended
        last_items_in_slot = [r for r in result if r.get("slot") == last_slot]
        if last_items_in_slot:
            last_end = last_items_in_slot[-1]["end_time"]
            h, m_str = last_end.split(":")
            last_cursor = int(h) * 60 + int(m_str) + 5
        else:
            last_cursor = _slot_start_minutes(last_slot)
        for leftover in queue:
            m = leftover["estimated_minutes"]
            result.append({
                **leftover,
                "slot":       last_slot,
                "start_time": _minutes_to_hhmm(last_cursor),
                "end_time":   _minutes_to_hhmm(last_cursor + m),
            })
            last_cursor += m + 5

    return result


# ── LLM enrichment ────────────────────────────────────────────────────────────

_SYSTEM = """
You are EduMind's expert learning schedule planner.
Given a student's course topic, goal, pace, and a compact day-by-day module
summary, enrich the schedule with motivational guidance.

Return ONLY valid JSON — no markdown fences, no preamble.
Schema:
{
  "overall_advice": "3-5 sentence paragraph",
  "weekly_milestones": ["one milestone per 7-day block or whole schedule"],
  "days": [
    {
      "day_number": 1,
      "day_theme": "short evocative title e.g. Foundation Day",
      "study_tip": "one actionable sentence specific to today's modules"
    }
  ]
}
"""


async def _enrich_with_llm(
    days: list[dict[str, Any]],
    topic: str,
    goal: str,
    pace: str,
) -> dict[str, Any]:
    # Only send days that have actual content to the LLM
    summaries = [
        {
            "day_number": d["day_number"],
            "modules": [it["module_title"] for it in d.get("timetable_items", [])],
            "total_minutes": d.get("total_study_minutes", 0),
        }
        for d in days
        if d.get("timetable_items")
    ]

    prompt = (
        f"Topic: {topic}\nGoal: {goal}\nPace: {pace}\n\n"
        f"Day summaries:\n{json.dumps(summaries, indent=2)}\n\n"
        "Enrich this schedule."
    )

    try:
        raw = await generate(
            messages=[{"role": "user", "content": prompt}],
            model=settings.reasoning_model,
            system=_SYSTEM,
            json_mode=True,
            max_tokens=2000,
            _caller="schedule_agent",
        )
        return json.loads(raw)
    except Exception as exc:
        logger.warning("Schedule LLM enrichment failed ({}), using defaults.", exc)
        return {
            "overall_advice": (
                f"This schedule is designed to help you master {topic} at a steady pace. "
                "Follow the daily plan consistently and review completed modules regularly."
            ),
            "weekly_milestones": ["Complete all scheduled modules and review your progress."],
            "days": [
                {
                    "day_number": d["day_number"],
                    "day_theme":  f"Day {d['day_number']}",
                    "study_tip":  "Stay focused, take short breaks, and review after each module.",
                }
                for d in days
            ],
        }


# ── Main public function ──────────────────────────────────────────────────────

async def generate_learning_schedule(
    *,
    course: dict[str, Any],
    modules: list[dict[str, Any]],
    duration_value: int,
    duration_unit: str,        # "days" | "weeks" | "months"
    hours_per_day: float,      # hour basis only (e.g. 2.0)
    study_slots: list[str],    # ["morning","evening"] or ["06:00","18:00"]
    start_date: str | None = None,
) -> dict[str, Any]:
    """
    Generate a full learning schedule for a course.

    Returns a dict with schedule_id, course_id, total_days, hours_per_day,
    study_slots, start_date, end_date, overall_advice, weekly_milestones,
    total_study_minutes_formatted, and days[].
    """
    topic = course.get("topic") or "your subject"
    goal  = course.get("goal")  or "general learning"
    pace  = course.get("pace")  or "medium"

    total_days = max(1, _parse_duration_to_days(duration_value, duration_unit))

    try:
        start = date.fromisoformat(start_date) if start_date else date.today()
    except ValueError:
        start = date.today()
    end = start + timedelta(days=total_days - 1)

    # 1. Pack modules evenly across days
    raw_days = _pack_modules_into_days(modules, total_days, hours_per_day)

    # 2. Assign time slots
    enriched: list[dict[str, Any]] = []
    for idx, day in enumerate(raw_days):
        items_timed = _assign_times(day["items"], study_slots, day["used_minutes"])
        enriched.append({
            "day_number":              idx + 1,
            "date":                    (start + timedelta(days=idx)).isoformat(),
            "total_study_minutes":     day["used_minutes"],
            "total_study_formatted":   _format_minutes(day["used_minutes"]),
            "timetable_items":         items_timed,
        })

    # Pad remaining days as rest days
    for idx in range(len(enriched), total_days):
        enriched.append({
            "day_number":              idx + 1,
            "date":                    (start + timedelta(days=idx)).isoformat(),
            "total_study_minutes":     0,
            "total_study_formatted":   "Rest Day",
            "timetable_items":         [],
        })

    # 3. LLM enrichment
    llm = await _enrich_with_llm(enriched, topic, goal, pace)
    llm_days = {d["day_number"]: d for d in llm.get("days", [])}

    for day in enriched:
        ld = llm_days.get(day["day_number"], {})
        day["day_theme"] = ld.get("day_theme") or (
            "Rest Day" if not day["timetable_items"] else f"Day {day['day_number']}"
        )
        day["study_tip"] = ld.get("study_tip") or (
            "Take a break — rest is part of learning." if not day["timetable_items"]
            else "Stay focused and review after each module."
        )

    # Total content minutes across all days
    total_content_mins = sum(
        d["total_study_minutes"] for d in enriched if d["timetable_items"]
    )

    return {
        "schedule_id":                    str(uuid.uuid4()),
        "course_id":                      course.get("id"),
        "total_days":                     total_days,
        "hours_per_day":                  hours_per_day,
        "study_slots":                    study_slots,
        "start_date":                     start.isoformat(),
        "end_date":                       end.isoformat(),
        "total_content_minutes":          total_content_mins,
        "total_content_formatted":        _format_minutes(total_content_mins),
        "overall_advice":                 llm.get("overall_advice", ""),
        "weekly_milestones":              llm.get("weekly_milestones", []),
        "days":                           enriched,
    }