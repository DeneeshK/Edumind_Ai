"""
core/roadmap_service.py
Personalized roadmap/study-plan generation for saved courses.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from core.curriculum_quality import relevant_history_concepts


PACE_LABELS = {
    "fast": "Fast",
    "medium": "Balanced",
    "deep": "Deep",
}


def _clean_list(values: list[Any] | None) -> list[str]:
    """Normalize optional roadmap list fields into unique strings."""
    result: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        if isinstance(value, dict):
            text = str(value.get("concept") or value.get("title") or "").strip()
        else:
            text = str(value or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _title_case(value: str | None) -> str:
    """Capitalize a display title without changing the remaining text."""
    value = (value or "").strip()
    return value[:1].upper() + value[1:] if value else "Course"


def _target_label(profile: dict[str, Any], course: dict[str, Any]) -> str:
    """Return the target-context label used in roadmap headings."""
    target = profile.get("target_context") or profile.get("learning_goal") or course.get("goal")
    if not target:
        return "Personalized Learning"
    if str(target).lower() in {"jee", "neet", "gate"}:
        return str(target).upper()
    return _title_case(str(target))


def _roadmap_title(profile: dict[str, Any], course: dict[str, Any]) -> str:
    """Build the personalized roadmap title shown to the learner."""
    topic = profile.get("course_scope") or profile.get("exact_subject") or profile.get("topic") or course.get("topic") or "Course"
    target = _target_label(profile, course)
    pace = PACE_LABELS.get(profile.get("pace") or course.get("pace") or "medium", "Balanced")
    learner = str(profile.get("learner_level") or "").lower()
    learner_label = "Beginner" if "beginner" in learner or "fresh" in learner else ""
    depth = str(profile.get("depth_preference") or "").lower()
    depth_label = "Overview" if "overview" in depth or "high-level" in depth or "high level" in depth else ""
    labels = [pace, learner_label, depth_label]
    clean_labels: list[str] = []
    topic_lower = str(topic).lower()
    for label in labels:
        if label and label.lower() not in topic_lower and label not in clean_labels:
            clean_labels.append(label)
    suffix = " ".join(clean_labels + ["Roadmap"])
    base = _title_case(str(topic))
    target_lower = str(target).lower()
    if (
        not profile.get("course_scope")
        and target_lower not in {"personalized learning", "general learning"}
        and target_lower not in topic_lower
    ):
        base += " for " + str(target)
    return f"{base}: {suffix}"


def _module_minutes(module: dict[str, Any], profile: dict[str, Any]) -> int:
    """Choose a roadmap estimate that respects pace-specific minimums."""
    pace = profile.get("pace") or "medium"
    context = (profile.get("target_context") or profile.get("learning_goal") or "").lower()
    floor = {"fast": 25, "medium": 35, "deep": 55}.get(pace, 35)
    if pace == "fast" and any(k in context for k in ("jee", "neet", "gate", "exam")):
        floor = 45
    estimated = int(module.get("estimated_minutes") or 0)
    return max(estimated, floor)


def _difficulty(module: dict[str, Any], profile: dict[str, Any]) -> str:
    """Return the roadmap difficulty label for a module."""
    if module.get("difficulty"):
        return str(module["difficulty"])
    depth = module.get("depth_level") or profile.get("pace") or "standard"
    return {
        "surface": "focused",
        "standard": "moderate",
        "deep": "advanced",
        "fast": "focused",
        "medium": "moderate",
    }.get(str(depth), "moderate")


def _why_module_matters(module: dict[str, Any], profile: dict[str, Any]) -> str:
    """
    Use the module's own metadata to explain why it matters.
    No hardcoded domain checks — works for any subject.
    """
    concept = module.get("concept") or module.get("title") or "this concept"
    context = (profile.get("target_context") or profile.get("learning_goal") or "").lower()
    weak = _clean_list(profile.get("weak_concepts"))

    # If the module has its own explanation, use it — it was written by the LLM with full context
    why_it_matters = str(module.get("why_it_matters_for_goal") or module.get("goal_alignment") or "").strip()
    if why_it_matters and len(why_it_matters) > 20:
        return why_it_matters

    # Weak concept signal — works for any domain
    if any(str(concept).lower() in item.lower() or item.lower() in str(concept).lower() for item in weak):
        return f"This targets a weak area — mastering it now prevents gaps in later modules."

    # Exam context signal — works for any exam/subject
    if any(k in context for k in ("jee", "neet", "gate", "exam", "test", "competitive")):
        return f"{concept} appears in exam patterns and is needed for solving problems under time pressure."

    # Use the module's own why_now if available
    why_now = str(module.get("why_now") or "").strip()
    if why_now and len(why_now) > 20:
        return why_now

    # Fallback — generic but accurate
    return f"{concept} builds the foundation that the next modules depend on."


def _already_known(profile: dict[str, Any], history: dict[str, Any] | None) -> list[str]:
    """Summarize verified known concepts for the roadmap introduction."""
    known = _clean_list(profile.get("assumed_known_concepts") or profile.get("known_concepts"))
    if known:
        return known
    if profile.get("relevant_history"):
        return ["EduMind does not have verified mastery for this new course, so this roadmap starts with a short foundation check."]
    if history:
        relevant = relevant_history_concepts(profile, history)
        mastered = _clean_list(relevant.get("known"))
        if mastered:
            return mastered[:6]
    return ["EduMind does not have enough previous learning history yet, so this roadmap starts with a short foundation check."]


def _skipped_or_reduced(profile: dict[str, Any], already_known: list[str]) -> list[str]:
    """Describe topics that should be compressed because of pace or prior mastery."""
    pace = profile.get("pace") or "medium"
    context = (profile.get("target_context") or profile.get("learning_goal") or "").lower()
    items: list[str] = []
    known = [item for item in already_known if not item.startswith("EduMind does not")]
    if known:
        items.append("Detailed basics of " + ", ".join(known[:4]) + " will be reduced and refreshed only when needed.")
    if pace == "fast":
        if any(k in context for k in ("jee", "neet", "gate", "exam")):
            items.append("Long college-level derivations will be reduced in favor of formulas, traps, and problem patterns.")
        else:
            items.append("Slow theoretical detours will be reduced so the course stays focused on essentials.")
    if "skip detailed personalization" in _clean_list(profile.get("course_constraints")):
        items.append("Extra preference probing was skipped, so EduMind will use sensible defaults.")
    return items or ["Nothing major is skipped yet; early checks will decide what can be compressed."]


def _emphasis(profile: dict[str, Any]) -> list[str]:
    """
    Derive emphasis from what the student actually told us.
    No hardcoded domain keywords — works for any subject.
    """
    context = (profile.get("target_context") or profile.get("learning_goal") or "").lower()
    weak = _clean_list(profile.get("weak_concepts"))
    emphasis: list[str] = []

    # Exam context: works for any exam, any subject
    if any(k in context for k in ("jee", "neet", "gate", "exam", "test", "competitive", "board")):
        emphasis.extend([
            "High-yield concepts",
            "Formula and rule intuition",
            "Common exam traps and misconceptions",
            "Timed problem-solving practice",
        ])
    else:
        emphasis.extend(["Conceptual understanding", "Worked examples", "Practice and application"])

    if weak:
        emphasis.append("Targeted weak areas: " + ", ".join(weak[:4]))
    return emphasis


def _study_advice(profile: dict[str, Any]) -> list[str]:
    """Return study advice tailored to pace and target context."""
    advice = [
        "Take a 10-15 minute break after every 45-60 minutes.",
        "Try the check questions before moving to the next module.",
        "Take a short recap before starting a new day of study.",
    ]
    context = (profile.get("target_context") or profile.get("learning_goal") or "").lower()
    if any(k in context for k in ("jee", "neet", "gate", "exam")):
        advice.insert(1, "Revise formulas before practice modules, then solve without looking at notes.")
        advice.append("Do not rush problem-solving modules; speed comes after pattern recognition.")
    elif "machine learning" in context or "computer science" in context:
        advice.insert(1, "After each concept, connect it to one concrete application or implementation.")
    return advice


def _learner_is_beginner(profile: dict[str, Any]) -> bool:
    """Infer whether the profile describes a beginner learner."""
    text = " ".join(
        str(profile.get(key) or "").lower()
        for key in ("learner_level", "prior_knowledge_summary", "prior_knowledge")
    )
    return any(k in text for k in ("beginner", "fresh", "no prior", "complete beginner"))


def _daily_capacity(profile: dict[str, Any]) -> int:
    """Estimate daily roadmap study capacity in minutes."""
    pace = profile.get("pace") or "medium"
    beginner = _learner_is_beginner(profile)
    if pace == "deep":
        return 135 if beginner else 165
    if pace == "fast":
        return 95 if beginner else 140
    return 115 if beginner else 155


def _max_modules_per_day(profile: dict[str, Any], total_modules: int = 0) -> int:
    """
    Dynamic daily cap based on total curriculum size.
    A 25-module Python course at 2/day = 12.5 days — reasonable.
    A 10-module French Revolution at 2/day = 5 days — fine.
    Scale up for larger curricula to avoid the schedule looking padded.
    """
    pace = profile.get("pace") or "medium"
    beginner = _learner_is_beginner(profile)

    # Base cap by pace and experience
    if pace == "deep":
        base = 2 if beginner else 3
    elif pace == "fast":
        base = 3 if beginner else 4
    else:  # medium
        base = 2 if beginner else 3

    # Scale up for larger curricula — keeps schedule length reasonable
    if total_modules > 25:
        base = min(base + 1, 4)
    if total_modules > 35:
        base = min(base + 1, 5)

    return base


def _build_schedule(timeline: list[dict[str, Any]], total_minutes: int, profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Group roadmap modules into day-level schedule items."""
    capacity = _daily_capacity(profile)
    max_modules = _max_modules_per_day(profile, total_modules=len(timeline))
    days: list[dict[str, Any]] = []

    current = {
        "day": 1,
        "title": "",
        "items": [],
        "review_minutes": 0,
        "practice_minutes": 0,
        "break_minutes": 0,
        "total_minutes": 0,
    }
    current_minutes = 0
    current_modules = 0

    def finish_day() -> None:
        """Finalize the current study day and reset the day accumulator."""
        nonlocal current, current_minutes, current_modules
        if not current["items"]:
            return
        current["review_minutes"] = 10 if current_modules else 0
        current["practice_minutes"] = 10 if current_modules >= 2 else 5 if current_modules else 0
        current["break_minutes"] = 10 if current_minutes >= 75 else 0
        current["total_minutes"] = (
            current_minutes
            + current["review_minutes"]
            + current["practice_minutes"]
            + current["break_minutes"]
        )
        current["title"] = current["items"][0]["module_title"][:72]
        days.append(current)
        current = {
            "day": len(days) + 1,
            "title": "",
            "items": [],
            "review_minutes": 0,
            "practice_minutes": 0,
            "break_minutes": 0,
            "total_minutes": 0,
        }
        current_minutes = 0
        current_modules = 0

    for module in timeline:
        minutes = int(module["estimated_minutes"])
        projected = current_minutes + minutes + 20
        if current["items"] and (
            current_modules >= max_modules or projected > capacity
        ):
            finish_day()
        current["items"].append({
            "module_id": module["module_id"],
            "module_title": module["title"],
            "item_type": "module",
            "estimated_minutes": minutes,
            "break_after": minutes >= 45,
            "break_minutes": 10 if minutes >= 45 else 0,
        })
        current_minutes += minutes
        current_modules += 1

    finish_day()
    return days


class CourseRoadmapService:
    """Builds a deterministic, personalized roadmap from course modules."""

    def build(
        self,
        course: dict[str, Any],
        modules: list[dict[str, Any]],
        profile: dict[str, Any] | None = None,
        student_history: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the complete frontend roadmap payload for a saved course."""
        profile = dict(profile or {})
        profile.setdefault("topic", course.get("topic"))
        profile.setdefault("learning_goal", course.get("goal"))
        profile.setdefault("pace", course.get("pace") or "medium")
        scope_analysis = (
            profile.get("scope_analysis")
            or course.get("scope_analysis")
            or course.get("personalization_profile", {}).get("scope_analysis")
            or {}
        )
        concept_inventory = profile.get("concept_inventory") or {}
        prerequisite_graph = profile.get("prerequisite_graph") or {}
        learning_path = profile.get("learning_path") or []
        roadmap_steps = profile.get("roadmap_steps") or []
        architect_schedule = profile.get("schedule_plan") or []
        validation_result = (
            profile.get("validation_result")
            or course.get("validation_result")
            or course.get("personalization_profile", {}).get("validation_result")
            or {}
        )

        topic = profile.get("topic") or course.get("topic") or "Course"
        pace = profile.get("pace") or course.get("pace") or "medium"
        target = _target_label(profile, course)
        title = _roadmap_title(profile, course)
        already_known = _already_known(profile, student_history)

        timeline: list[dict[str, Any]] = []
        for idx, module in enumerate(modules):
            minutes = _module_minutes(module, profile)
            concepts_taught = _clean_list(module.get("concepts_taught") or module.get("must_teach") or [module.get("concept")])
            depends_on = _clean_list(module.get("depends_on_concepts") or module.get("prerequisites"))
            timeline.append({
                "module_id": module.get("id"),
                "module_number": idx + 1,
                "title": module.get("title") or f"Module {idx + 1}",
                "concept": module.get("concept") or "",
                "concepts_taught": concepts_taught,
                "depends_on_concepts": depends_on,
                "unlocks_concepts": _clean_list(module.get("unlocks_concepts")),
                "module_goal": module.get("module_goal") or module.get("purpose") or "",
                "why_now": module.get("why_now") or "",
                "what_this_module_will_not_cover": _clean_list(module.get("what_this_module_will_not_cover")),
                "question_scope": _clean_list(module.get("question_scope")),
                "why_this_module_matters": _why_module_matters(module, profile),
                "estimated_minutes": minutes,
                "difficulty": _difficulty(module, profile),
                "status": module.get("status") or "not_started",
                "recommended_next": bool(module.get("recommended")) or idx == 0,
                "prerequisites": depends_on,
            })

        estimated_total = sum(item["estimated_minutes"] for item in timeline)
        if timeline:
            estimated_total += 10 * max(0, len(timeline) - 1)
        schedule = _build_schedule(timeline, estimated_total, profile)
        if not roadmap_steps:
            roadmap_steps = [
                (
                    ("First, " if idx == 0 else "Then, " if idx < len(timeline) - 1 else "Finally, ")
                    + "we learn "
                    + (item.get("concept") or item.get("title") or "the next concept")
                    + "."
                )
                for idx, item in enumerate(timeline)
            ]
        strategy = profile.get("recommended_strategy") or ""

        intention = (
            "This course is designed to help you learn "
            + str(topic)
            + " for "
            + str(target)
            + ". It will follow a "
            + str(pace)
            + " path and prioritize "
            + (strategy[0].lower() + strategy[1:] if strategy else "the concepts and practice that matter most for your goal.")
        )

        return {
            "course_id": course.get("id"),
            "title": title,
            "course_intention": intention,
            "personalization_summary": {
                "topic": profile.get("topic"),
                "goal_context": profile.get("learning_goal") or profile.get("target_context"),
                "pace": pace,
                "depth_preference": profile.get("depth_preference"),
                "time_constraint": profile.get("time_constraint"),
                "known_concepts": _clean_list(profile.get("assumed_known_concepts") or profile.get("known_concepts")),
                "weak_concepts": _clean_list(profile.get("weak_concepts")),
                "preferred_strategy": strategy,
                "student_history_used": bool(profile.get("student_history_used")),
            },
            "scope_analysis": scope_analysis,
            "scope_analysis_summary": {
                "topic_breadth": scope_analysis.get("topic_breadth"),
                "topic_type": scope_analysis.get("topic_type"),
                "learner_goal_type": scope_analysis.get("learner_goal_type"),
                "recommended_module_count": scope_analysis.get("recommended_module_count"),
                "reason_for_module_count": scope_analysis.get("reason_for_module_count"),
                "coverage_strategy": scope_analysis.get("coverage_strategy"),
            },
            "validation_result": validation_result,
            "roadmap_steps": roadmap_steps,
            "what_we_will_learn_first": roadmap_steps[:2],
            "what_comes_next": roadmap_steps[2:5],
            "what_is_delayed_or_skipped": _clean_list(
                scope_analysis.get("what_to_delay_until_later")
                or []
            ) + _clean_list(scope_analysis.get("what_to_exclude") or []),
            "concept_inventory": concept_inventory,
            "prerequisite_graph": prerequisite_graph,
            "learning_path": learning_path,
            "architect_schedule_plan": architect_schedule,
            "already_known": already_known,
            "skipped_or_reduced": _skipped_or_reduced(profile, already_known),
            "emphasized_topics": _emphasis(profile),
            "estimated_total_time_minutes": estimated_total,
            "recommended_schedule": schedule,
            "study_advice": _study_advice(profile),
            "break_recommendations": [
                "Take a 10-15 minute break after 45-60 minutes.",
                "Use breaks before practice-heavy modules, not in the middle of one.",
            ],
            "module_timeline": timeline,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
