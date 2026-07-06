"""
core/course_service.py
Frontend-facing course/module orchestration.

This layer coordinates course planning, roadmap persistence, lazy lesson
generation, grounded question creation, module chat, and completion state for
the frontend API.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncGenerator
from typing import Any
from urllib.parse import parse_qs, urlparse

from loguru import logger
from core.metrics import metrics as _metrics

from agents.curriculum_architect import CurriculumArchitectAgent
from clients.groq_client import generate, stream, tool_call_loop
from clients import mcp_search_client
from clients.tavily_client import search as tavily_search  # V2: re-enable for LLM-triggered Tavily
from config import settings
from core.curriculum_quality import (
    concept_appears_in_text,
    filter_relevant_student_history,
    is_garbage_topic,
    is_related_to_profile,
    is_unreliable_generated_concept,
    parse_json_object,
    profile_has_no_prior_experience,
    relevant_history_concepts,
    validate_lesson_quality,
    validate_questions_grounded,
)
# from core.rag_pipeline import retrieve  # V2: re-enable when RAG is back
from core.roadmap_service import CourseRoadmapService
from core.student_model import StudentState
from db.postgres import (
    bulk_write_decisions,
    create_course_from_plan,
    ensure_student,
    get_course,
    get_course_module,
    get_module_questions,
    get_student_dashboard,
    get_student_doubts,
    get_student_skills,
    load_metacognition,
    latest_curriculum_for_student,
    list_course_modules,
    list_courses,
    list_module_chat_history,
    list_module_chat_history_for_student,
    record_doubt,
    record_module_chat_message,
    save_course_roadmap,
    save_master_roadmap,
    save_metacognition,
    save_module_content,
    save_module_questions,
    set_module_status,
    upsert_concept_mastery,
    upsert_student,
    upsert_student_skill,
    write_evaluation,
)


YOUTUBE_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([A-Za-z0-9_-]{6,})"
)

PLACEHOLDER_QUESTION_PATTERNS = (
    r"\baccording to the lesson\b",
    r"\bwhat is the key idea\b",
    r"\bwhat detail from the lesson explains\b",
    r"\bwhy does .+ matter\??$",
)


def infer_domain(topic: str, goal: str) -> str:
    """Infer a broad learning domain from the requested topic and goal."""
    text = f"{topic} {goal}".lower()
    domain_map = [
        ("machine learning", ("machine learning", "ml", "ai", "model", "neural")),
        ("software engineering", ("software", "web", "backend", "frontend", "app")),
        ("data science", ("data", "analytics", "statistics")),
        ("interview preparation", ("interview", "exam", "test")),
    ]
    for domain, keywords in domain_map:
        if any(k in text for k in keywords):
            return domain
    return goal or topic or "general learning"


def _clean_list(values: list[Any] | Any | None) -> list[str]:
    """Normalize scalar/list/dict values into a de-duplicated string list."""
    if values is None:
        values = []
    elif not isinstance(values, list):
        values = [values]
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, dict):
            text = str(value.get("concept") or value.get("topic") or value.get("title") or "").strip()
        else:
            text = str(value or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def module_required_concepts(module: dict[str, Any]) -> list[str]:
    """Return the concepts a module is expected to explicitly teach."""
    metadata = module.get("module_metadata") or {}
    concepts = _clean_list(module.get("concepts_taught") or metadata.get("concepts_taught"))
    if concepts:
        return concepts
    concepts = _clean_list(module.get("must_teach") or metadata.get("must_teach"))
    if concepts:
        return concepts
    concept = str(module.get("concept") or "").strip()
    return [concept] if concept else []


def _module_question_scope(module: dict[str, Any], fallback: list[str] | None = None) -> list[str]:
    """Return the concepts that generated questions are allowed to test."""
    metadata = module.get("module_metadata") or {}
    scope = _clean_list(module.get("question_scope") or metadata.get("question_scope"))
    return scope or list(fallback or [])


def _concepts_present_in_lesson(concepts: list[str], content: str) -> list[str]:
    """Filter planned concepts down to concepts that appear in lesson content."""
    return [concept for concept in concepts if concept_appears_in_text(content, concept)]


def _section_sentence_for_concept(
    section_sentences: list[tuple[str, str]],
    concept: str,
    used_quotes: set[str],
) -> tuple[str, str] | None:
    """Find a lesson sentence that can ground a question for one concept."""
    for section, sentence in section_sentences:
        if sentence in used_quotes:
            continue
        if concept_appears_in_text(sentence, concept) or concept_appears_in_text(section, concept):
            used_quotes.add(sentence)
            return section, sentence
    for section, sentence in section_sentences:
        if concept_appears_in_text(sentence, concept) or concept_appears_in_text(section, concept):
            return section, sentence
    return None


def _profile_value(course: dict[str, Any], *keys: str) -> str:
    """Read a profile, intent, or course-level value by priority order."""
    profile = course.get("personalization_profile") or {}
    intent = profile.get("current_intent") or {}
    for key in keys:
        value = intent.get(key) or profile.get(key) or course.get(key)
        if value:
            return str(value).strip()
    return ""


def _youtube_video_id(url: str) -> str:
    """Extract a YouTube video id from common watch, short, and embed URLs."""
    parsed = urlparse((url or "").strip())
    host = parsed.netloc.lower().replace("www.", "").replace("m.", "")
    path = parsed.path.strip("/")
    if host == "youtu.be" and path:
        return path.split("/")[0]
    if host.endswith("youtube.com"):
        if path == "watch":
            return (parse_qs(parsed.query).get("v") or [""])[0]
        if path.startswith("embed/") or path.startswith("shorts/"):
            return path.split("/")[1] if "/" in path else ""
    match = YOUTUBE_URL_RE.search(url or "")
    return match.group(1) if match else ""


def youtube_watch_url_to_embed_url(url: str) -> str:
    """Convert a YouTube watch/short URL into an embeddable player URL."""
    video_id = _youtube_video_id(url)
    if not video_id:
        return ""
    return f"https://www.youtube.com/embed/{video_id}"


def _extract_youtube_url(result: dict[str, Any]) -> str:
    """Extract a usable YouTube watch URL from a Tavily result."""
    url = str(result.get("url") or "").strip()
    if youtube_watch_url_to_embed_url(url):
        return url
    text = " ".join(
        str(result.get(key) or "")
        for key in ("title", "content", "raw_content", "snippet")
    )
    match = YOUTUBE_URL_RE.search(text)
    if not match:
        return ""
    return f"https://www.youtube.com/watch?v={match.group(1)}"


def _is_youtube_playlist_spam(url: str, result: dict[str, Any]) -> bool:
    """Detect playlist pages that are poor single-lesson video recommendations."""
    title = str(result.get("title") or "").lower()
    url_lower = (url or "").lower()
    if "youtube.com/playlist" in url_lower or "/playlist" in url_lower:
        return True
    return "playlist" in title and "watch?v=" not in url_lower


def _video_result_matches_level_and_pace(
    course: dict[str, Any],
    result: dict[str, Any],
) -> bool:
    """Reject video results that are clearly mismatched to learner level or pace."""
    text = " ".join(
        str(result.get(key) or "")
        for key in ("title", "content", "raw_content", "snippet")
    ).lower()
    level = _profile_value(course, "learner_level", "current_level").lower()
    pace = str(course.get("pace") or _profile_value(course, "pace") or "medium").lower()
    if "beginner" in level and re.search(r"\b(advanced|expert|intermediate)\b", text):
        return False
    if pace == "fast" and re.search(
        r"\b(full course|complete course|playlist|[2-9]\s*(?:hour|hr|hours|hrs))\b",
        text,
    ):
        return False
    return True


def _youtube_search_query(course: dict[str, Any], module: dict[str, Any]) -> str:
    """Build a YouTube-focused search query for optional lesson videos."""
    topic = str(course.get("topic") or "").strip()
    title = str(module.get("title") or "").strip()
    concept = str(module.get("concept") or "").strip()
    level = _profile_value(course, "learner_level", "current_level") or "beginner"
    pace = str(course.get("pace") or _profile_value(course, "pace") or "medium").strip()
    goal = str(course.get("goal") or _profile_value(course, "goal", "learning_goal") or "").strip()
    target_context = _profile_value(course, "target_context")
    descriptors = ["beginner explanation" if "beginner" in level.lower() else f"{level} explanation"]
    if pace == "fast":
        descriptors.append("short practical")
    elif pace == "deep":
        descriptors.append("detailed worked example")
    else:
        descriptors.append("worked example")
    parts = [
        "site:youtube.com/watch",
        topic,
        title,
        concept,
        target_context,
        " ".join(descriptors),
        goal,
    ]
    return " ".join(part for part in parts if part).strip()


def _lesson_videos_from_module(module: dict[str, Any]) -> list[dict[str, Any]]:
    """Read persisted lesson videos from module fields or metadata."""
    metadata = module.get("module_metadata") or {}
    videos = module.get("videos") or module.get("lesson_videos") or metadata.get("lesson_videos") or []
    return videos if isinstance(videos, list) else []


def lesson_videos_from_module(module: dict[str, Any]) -> list[dict[str, Any]]:
    """Public wrapper used by routers to expose persisted lesson videos."""
    return _lesson_videos_from_module(module)


def youtube_videos_from_tavily_results(
    course: dict[str, Any],
    module: dict[str, Any],
    results: list[dict[str, Any]] | None,
    max_results: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert Tavily search results into de-duplicated YouTube video cards."""
    videos: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    stats: dict[str, Any] = {
        "raw_result_count": len(results or []),
        "youtube_url_count": 0,
        "selected_video_count": 0,
        "rejection_counts": {},
    }

    def reject(reason: str) -> None:
        """Track why a candidate video was excluded."""
        counts = stats["rejection_counts"]
        counts[reason] = counts.get(reason, 0) + 1

    for result in results or []:
        if not isinstance(result, dict):
            reject("non_dict_result")
            continue
        url = _extract_youtube_url(result)
        video_id = _youtube_video_id(url)
        if not url or not video_id:
            reject("no_youtube_watch_url")
            continue
        stats["youtube_url_count"] += 1
        if video_id in seen_ids:
            reject("duplicate_video")
            continue
        if _is_youtube_playlist_spam(url, result):
            reject("playlist_or_playlist_spam")
            continue
        if not _video_result_matches_level_and_pace(course, result):
            reject("level_or_pace_mismatch")
            continue

        seen_ids.add(video_id)
        title = str(result.get("title") or "Recommended YouTube lesson").strip()
        canonical_url = f"https://www.youtube.com/watch?v={video_id}"
        item = {
            "title": title,
            "url": canonical_url,
            "embed_url": f"https://www.youtube.com/embed/{video_id}",
            "source": "youtube",
            "reason": "Relevant beginner explanation for this module.",
        }
        duration = result.get("duration") or result.get("video_duration")
        if duration:
            item["duration"] = str(duration)
        videos.append(item)
        if len(videos) >= max_results:
            break

    stats["selected_video_count"] = len(videos)
    if videos:
        stats["first_selected"] = {
            "title": videos[0].get("title"),
            "url": videos[0].get("url"),
            "embed_url": videos[0].get("embed_url"),
        }
    return videos, stats


async def search_youtube_videos_for_module(
    course: dict[str, Any],
    module: dict[str, Any],
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search for optional YouTube lesson videos and return selected cards."""
    query = _youtube_search_query(course, module)
    if not query:
        return []
    logger.info(
        "youtube_search_query module_id='{}' youtube_search_query='{}'",
        module.get("id"),
        query,
    )
    try:
        results = await asyncio.to_thread(tavily_search, query, 6)
    except Exception as exc:
        logger.warning("YouTube video search failed for module_id='{}': {}", module.get("id"), exc)
        return []

    videos, stats = youtube_videos_from_tavily_results(course, module, results, max_results)
    logger.info(
        "youtube_video_results module_id='{}' raw_result_count={} youtube_url_count={} selected_video_count={}",
        module.get("id"),
        stats.get("raw_result_count", 0),
        stats.get("youtube_url_count", 0),
        stats.get("selected_video_count", 0),
    )
    first_selected = stats.get("first_selected") or {}
    if first_selected:
        logger.info(
            "youtube_first_selected_video module_id='{}' title='{}' url='{}' embed_url='{}'",
            module.get("id"),
            first_selected.get("title"),
            first_selected.get("url"),
            first_selected.get("embed_url"),
        )
    elif stats.get("raw_result_count", 0) and not stats.get("youtube_url_count", 0):
        logger.info(
            "youtube_no_urls_extracted module_id='{}' rejection_counts={}",
            module.get("id"),
            stats.get("rejection_counts", {}),
        )
    elif stats.get("raw_result_count", 0):
        logger.info(
            "youtube_no_videos_selected module_id='{}' rejection_counts={}",
            module.get("id"),
            stats.get("rejection_counts", {}),
        )
    return videos


async def _optional_youtube_videos_for_module(
    course: dict[str, Any],
    module: dict[str, Any],
) -> list[dict[str, Any]]:
    """Best-effort video lookup that never blocks lesson generation."""
    try:
        return await search_youtube_videos_for_module(course, module)
    except Exception as exc:
        logger.warning("YouTube video search failed for module_id='{}': {}", module.get("id"), exc)
        return []


def _python_intent_guardrails(topic: str, goal: str, target_context: str) -> list[str]:
    """Return topics to avoid for beginner pure-Python course intents."""
    text = f"{topic} {goal} {target_context}".lower()
    if "python" not in text:
        return []
    if any(k in text for k in ("machine learning", " ml", "data science", "pandas", "sklearn", "numpy")):
        return []
    return [
        "thermodynamics",
        "linear algebra",
        "machine learning",
        "pandas",
        "sklearn",
        "scikit-learn",
        "advanced OOP",
        "decorators",
        "async",
        "deployment",
    ]


def _trusted_strategy(profile: dict[str, Any]) -> str:
    """Summarize the planning strategy implied by trusted profile fields."""
    intent = profile.get("current_intent") or {}
    topic = str(intent.get("exact_subject") or intent.get("topic") or profile.get("topic") or "the topic")
    target = str(intent.get("target_context") or profile.get("target_context") or "").lower()
    goal = str(intent.get("goal") or profile.get("learning_goal") or "")
    pace = str(intent.get("pace") or profile.get("pace") or "medium")
    depth = str(intent.get("depth_preference") or profile.get("depth_preference") or "")
    time_constraint = str(intent.get("time_constraint") or profile.get("time_constraint") or "")
    assumed = _clean_list(profile.get("assumed_known_concepts"))
    weak = _clean_list(profile.get("weak_concepts"))
    no_prior = profile_has_no_prior_experience(profile)
    python_guardrails = _python_intent_guardrails(topic, goal, target)

    if python_guardrails and no_prior:
        first = (
            "Start from zero. Teach pure Python fundamentals step by step. "
            "Do not specialize into ML, web, automation, or data science yet."
        )
    elif python_guardrails:
        first = "Teach general pure Python programming in prerequisite order before optional libraries or specializations."
    elif any(k in target or k in goal.lower() for k in ("jee", "neet", "gate", "exam")):
        first = "Prioritize the target exam's prerequisite flow, formulas, common traps, and problem patterns."
    elif any(k in target or k in goal.lower() for k in ("machine learning", "ml", "data science")):
        first = "Build the prerequisite toolchain before model APIs, using prior knowledge only when verified."
    else:
        first = "Build a practical prerequisite-aware path through " + topic + " without unrelated detours."

    parts = [first]
    if assumed:
        parts.append("Compress only verified known concepts: " + ", ".join(assumed[:3]) + ".")
    if weak:
        parts.append("Spend extra practice on " + ", ".join(weak[:3]) + ".")
    if pace:
        parts.append("Use a " + pace + " pace without overloading daily study.")
    if depth:
        parts.append("Match the depth preference: " + depth + ".")
    if time_constraint:
        parts.append("Respect the learner's available time: " + time_constraint + ".")
    return " ".join(parts)


def _current_intent(
    topic: str,
    goal: str,
    pace: str,
    prior_knowledge: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Create the normalized course intent block used by planning prompts."""
    exact_subject = str(data.get("exact_subject") or data.get("topic") or topic or "").strip()
    learning_goal = str(data.get("learning_goal") or data.get("goal") or goal or "").strip()
    goal_description = str(data.get("goal_description") or "").strip()
    target_context = str(
        data.get("target_context")
        or goal_description
        or infer_domain(exact_subject or topic, learning_goal)
    ).strip()
    topic_text = f"{exact_subject} {learning_goal} {target_context}".lower()
    if (
        "python" in topic_text
        and not any(k in topic_text for k in ("machine learning", " ml", "data science", "pandas", "sklearn", "numpy"))
        and any(k in topic_text for k in ("programming", "code", "coding", "pure", "fundamental", "beginner", "general"))
    ):
        target_context = "general pure Python"
    prior_summary = str(
        data.get("prior_knowledge_summary")
        or data.get("prior_knowledge")
        or prior_knowledge
        or ""
    ).strip()
    prior_experience = str(data.get("prior_experience") or "").strip()
    current_level = str(data.get("current_level") or "").strip()
    learner_level = str(data.get("learner_level") or "").strip()
    if not learner_level and current_level:
        learner_level = {
            "complete_beginner": "complete beginner",
            "basic": "some basic knowledge",
            "intermediate": "intermediate",
            "advanced": "advanced",
            "not_sure": "not sure",
        }.get(current_level, current_level.replace("_", " "))
    if not learner_level and re.search(r"\b(no prior|no background|complete beginner|fresh student|beginner)\b", prior_summary, re.I):
        learner_level = "complete beginner"

    # Build an enriched prior_knowledge_summary that combines every piece of
    # prior-knowledge signal into one structured paragraph.  This is what gets
    # injected verbatim into every LLM call — a richer paragraph means the
    # model personalises the curriculum far more accurately.
    known_concepts_raw = _clean_list(data.get("assumed_known_concepts") or data.get("known_concepts") or [])
    weak_concepts_raw  = _clean_list(data.get("weak_concepts") or [])
    pk_parts: list[str] = []
    if learner_level and learner_level not in ("not sure", "not_sure"):
        pk_parts.append(f"Learner level: {learner_level}.")
    if prior_summary:
        pk_parts.append(prior_summary)
    if prior_experience and prior_experience.lower() not in prior_summary.lower():
        pk_parts.append(f"Prior experience: {prior_experience}.")
    if known_concepts_raw:
        pk_parts.append(f"Already knows: {', '.join(known_concepts_raw[:10])}.")
    if weak_concepts_raw:
        pk_parts.append(f"Struggles with: {', '.join(weak_concepts_raw[:10])}.")
    enriched_prior_summary = " ".join(pk_parts).strip() or prior_summary

    return {
        "topic": str(data.get("topic") or topic or exact_subject).strip(),
        "exact_subject": exact_subject or str(topic or "").strip(),
        "goal": learning_goal,
        "goal_description": goal_description,
        "target_context": target_context,
        "current_level": current_level,
        "learner_level": learner_level,
        "pace": str(data.get("pace") or pace or "medium").strip(),
        "depth_preference": str(data.get("depth_preference") or "").strip(),
        "prior_knowledge_summary": enriched_prior_summary,
        "prior_experience": prior_experience,
        "time_constraint": str(data.get("time_constraint") or "").strip(),
        "time_commitment": data.get("time_commitment") or {},
        "deadline": str(data.get("deadline") or "").strip(),
    }


def _trusted_declared_known(data: dict[str, Any], intent: dict[str, Any]) -> list[str]:
    """Return declared known concepts only when they fit the current course intent."""
    profile_for_relevance = {
        "topic": intent.get("topic"),
        "exact_subject": intent.get("exact_subject"),
        "learning_goal": intent.get("goal"),
        "target_context": intent.get("target_context"),
    }
    no_prior = profile_has_no_prior_experience({"current_intent": intent, **profile_for_relevance})
    if no_prior:
        return []
    declared = _clean_list(data.get("assumed_known_concepts") or data.get("known_concepts"))
    return [
        concept for concept in declared
        if is_related_to_profile(concept, profile_for_relevance)
        and not is_unreliable_generated_concept(concept)
    ][:8]


def _trusted_declared_weak(data: dict[str, Any], intent: dict[str, Any]) -> list[str]:
    """Return declared weak concepts that are relevant to the current course intent."""
    profile_for_relevance = {
        "topic": intent.get("topic"),
        "exact_subject": intent.get("exact_subject"),
        "learning_goal": intent.get("goal"),
        "target_context": intent.get("target_context"),
    }
    return [
        concept for concept in _clean_list(data.get("weak_concepts"))
        if is_related_to_profile(concept, profile_for_relevance)
        and not is_unreliable_generated_concept(concept)
    ][:8]


def normalise_personalization_profile(
    topic: str,
    goal: str,
    pace: str,
    prior_knowledge: str = "",
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a trusted, current-intent-first personalization profile."""
    incoming = dict(profile or {})
    data = dict(incoming)
    intent = _current_intent(topic, goal, pace, prior_knowledge, data)
    data["current_intent"] = intent
    data["topic"] = intent["topic"]
    data["exact_subject"] = intent["exact_subject"]
    data["learning_goal"] = intent["goal"]
    data["goal_description"] = intent["goal_description"] or data.get("goal_description")
    data["target_context"] = intent["target_context"]
    data["current_level"] = intent["current_level"] or data.get("current_level")
    data["learner_level"] = intent["learner_level"] or data.get("learner_level")
    data["pace"] = intent["pace"] or "medium"
    data["depth_preference"] = data.get("depth_preference") or {
        "fast": "essentials and examples",
        "medium": "balanced understanding and practice",
        "deep": "deep conceptual understanding",
    }.get(data["pace"], "balanced understanding and practice")
    data["current_intent"]["depth_preference"] = data["depth_preference"]
    data["time_constraint"] = intent["time_constraint"] or data.get("time_constraint")
    data["time_commitment"] = intent["time_commitment"] or data.get("time_commitment") or {}
    data["deadline"] = intent["deadline"] or data.get("deadline")
    data["prior_experience"] = intent["prior_experience"] or data.get("prior_experience")
    # Use the enriched summary built in _current_intent (includes level + known/weak concepts).
    data["prior_knowledge_summary"] = intent["prior_knowledge_summary"]
    data["current_intent"]["prior_knowledge_summary"] = intent["prior_knowledge_summary"]
    declared_known = _trusted_declared_known(incoming, intent)
    data["assumed_known_concepts"] = declared_known
    data["known_concepts"] = list(declared_known)
    data["weak_concepts"] = _trusted_declared_weak(incoming, intent)
    data["must_include"] = _clean_list(data.get("must_include"))
    data["should_skip"] = _clean_list(data.get("should_skip"))
    data["preferred_teaching_style"] = data.get("preferred_teaching_style")
    data["assessment_preference"] = data.get("assessment_preference")
    data["expected_outcome"] = data.get("expected_outcome")
    data["course_constraints"] = _clean_list(data.get("course_constraints"))
    data["student_history_used"] = False
    data["student_history_relevant_concepts"] = []
    data["relevant_history"] = {
        "concepts": [],
        "courses": [],
        "reason": "History has not been filtered yet.",
    }
    data["skip_or_reduce"] = list(declared_known)
    data["do_not_include"] = _clean_list(data.get("do_not_include")) + _python_intent_guardrails(
        data["exact_subject"], data["learning_goal"], data["target_context"]
    )
    data["do_not_include"] = _clean_list(data["do_not_include"])
    data["recommended_strategy"] = _trusted_strategy(data)
    data["missing_fields"] = _clean_list(data.get("missing_fields"))
    data["confidence"] = float(data.get("confidence") or 0.7)
    data["ready_to_create"] = bool(data.get("ready_to_create", True))
    if is_garbage_topic(data["topic"]):
        raise ValueError("Invalid course request: topic is missing or looks like confirmation text")
    return data


async def get_student_history_snapshot(student_id: str) -> dict[str, Any]:
    """Best-effort learner history for course setup and roadmap planning."""
    try:
        dashboard = await get_student_dashboard(student_id)
    except Exception as exc:
        logger.warning("Student dashboard history unavailable for '{}': {}", student_id, exc)
        dashboard = {}

    try:
        skills = await get_student_skills(student_id)
    except Exception as exc:
        logger.warning("Student skill history unavailable for '{}': {}", student_id, exc)
        skills = {}

    try:
        doubts = await get_student_doubts(student_id)
    except Exception as exc:
        logger.warning("Student doubt history unavailable for '{}': {}", student_id, exc)
        doubts = []

    mastered = skills.get("strong_concepts") or dashboard.get("mastered_concepts") or []
    weak = skills.get("weak_concepts") or dashboard.get("weak_concepts") or []
    courses = dashboard.get("courses") or []
    return {
        "student_id": student_id,
        "previous_courses": courses,
        "completed_modules": [
            course for course in courses if int(course.get("completed_modules") or 0) > 0
        ],
        "mastered_concepts": mastered,
        "weak_concepts": weak,
        "doubt_history": doubts,
        "skill_graph": skills,
        "summary": dashboard.get("summary") or {},
    }


def apply_relevant_history_filter(
    profile: dict[str, Any],
    student_history: dict[str, Any] | None,
) -> dict[str, Any]:
    """Classify stored student history as relevant or irrelevant to the new course."""
    intent = profile.get("current_intent") or _current_intent(
        str(profile.get("topic") or ""),
        str(profile.get("learning_goal") or ""),
        str(profile.get("pace") or "medium"),
        str(profile.get("prior_knowledge_summary") or ""),
        profile,
    )
    filtered = filter_relevant_student_history(intent, student_history)
    declared_known = _clean_list(profile.get("assumed_known_concepts"))
    assumed = _clean_list(declared_known + filtered.get("assumed_known", []))
    weak = _clean_list(filtered.get("weak", []) + _clean_list(profile.get("weak_concepts")))
    possible = filtered.get("possibly_related_but_not_assumed_known") or []
    relevant_concepts = _clean_list(filtered.get("concepts") or [])

    profile["current_intent"] = intent
    profile["assumed_known_concepts"] = assumed
    profile["known_concepts"] = assumed
    profile["weak_concepts"] = weak
    profile["student_history_relevant_concepts"] = relevant_concepts
    profile["student_history_used"] = bool(relevant_concepts or possible)
    profile["relevant_history"] = {
        "concepts": relevant_concepts,
        "courses": _clean_list(filtered.get("courses") or []),
        "possibly_related_but_not_assumed_known": possible,
        "assumed_known": assumed,
        "reason": filtered.get("reasoning_summary") or "",
    }
    profile["student_history"] = {
        "available": bool(student_history),
        "raw_history_count": sum(
            len(student_history.get(key) or [])
            for key in ("previous_courses", "mastered_concepts", "weak_concepts", "doubt_history")
        ) if student_history else 0,
    }
    profile["skip_or_reduce"] = list(assumed)
    profile["recommended_strategy"] = _trusted_strategy(profile)
    return filtered


def filtered_history_snapshot(profile: dict[str, Any], history_filter: dict[str, Any]) -> dict[str, Any]:
    """Build the compact history payload passed into curriculum planning."""
    return {
        "filter": history_filter,
        "mastered_concepts": [{"concept": c, "source": "verified_history_filter"} for c in profile.get("assumed_known_concepts") or []],
        "weak_concepts": [{"concept": c, "source": "relevant_history_filter"} for c in profile.get("weak_concepts") or []],
        "previous_courses": [
            {"topic": c, "source": "possibly_related_history"}
            for c in (profile.get("relevant_history") or {}).get("courses", [])
        ],
    }


def sanitize_trusted_profile_for_course(profile: dict[str, Any]) -> dict[str, Any]:
    """Remove unrelated concepts from trusted profile fields before planning."""
    do_not_include = [str(x).lower().strip() for x in profile.get("do_not_include") or []]
    
    def _is_clean(concept: str) -> bool:
        """Return whether a profile concept survives the do-not-include filter."""
        c_lower = str(concept).lower().strip()
        return not any(dni in c_lower or c_lower in dni for dni in do_not_include)
    
    removed = []
    
    for key in ["assumed_known_concepts", "known_concepts", "weak_concepts", "student_history_relevant_concepts", "skip_or_reduce"]:
        original = profile.get(key) or []
        cleaned = [c for c in original if _is_clean(c)]
        profile[key] = cleaned
        for c in original:
            if c not in cleaned:
                removed.append(c)

    if profile.get("relevant_history"):
        for key in ["concepts", "assumed_known"]:
            original = profile["relevant_history"].get(key) or []
            profile["relevant_history"][key] = [c for c in original if _is_clean(c)]

    if removed:
        logger.info("Sanitizer removed contaminated concepts: {}", list(set(removed)))
        
    return profile


async def create_course(
    student_id: str,
    topic: str,
    goal: str,
    pace: str = "medium",
    prior_knowledge: str = "",
    name: str = "Student",
    personalization_profile: dict[str, Any] | None = None,
    web_search_enabled: bool = False,
) -> dict[str, Any]:
    """
    Create a persisted course, modules, master roadmap, and frontend roadmap.

    The function normalizes personalization data, builds a curriculum through
    the curriculum architect, persists the resulting course rows, and records
    planning metadata used later by lesson generation.
    """
    pace = pace if pace in ("fast", "medium", "deep") else "medium"
    profile = normalise_personalization_profile(
        topic=topic,
        goal=goal,
        pace=pace,
        prior_knowledge=prior_knowledge,
        profile=personalization_profile,
    )
    topic = str(profile.get("topic") or topic)
    goal = str(profile.get("learning_goal") or goal)
    pace = str(profile.get("pace") or pace)
    domain = str(profile.get("target_context") or infer_domain(topic, goal))
    history = await get_student_history_snapshot(student_id)
    history_filter = apply_relevant_history_filter(profile, history)
    planner_history = filtered_history_snapshot(profile, history_filter)
    logger.info(
        "Course creation intent: topic='{}', exact='{}', goal='{}', target='{}', level='{}', pace='{}', depth='{}'",
        profile.get("current_intent", {}).get("topic"),
        profile.get("current_intent", {}).get("exact_subject"),
        profile.get("current_intent", {}).get("goal"),
        profile.get("current_intent", {}).get("target_context"),
        profile.get("current_intent", {}).get("learner_level"),
        profile.get("current_intent", {}).get("pace"),
        profile.get("current_intent", {}).get("depth_preference"),
    )
    logger.info(
        "History filter: raw={}, relevant={}, assumed_known={}, rejected={}",
        profile.get("student_history", {}).get("raw_history_count", 0),
        len(profile.get("student_history_relevant_concepts") or []),
        len(profile.get("assumed_known_concepts") or []),
        len(history_filter.get("irrelevant") or []),
    )
    logger.info(
        "Planner input: topic='{}', goal='{}', assumed_known={}, do_not_include={}",
        topic,
        goal,
        profile.get("assumed_known_concepts") or [],
        profile.get("do_not_include") or [],
    )
    await upsert_student(student_id, name, domain, goal, pace)

    # Classify topic for metrics label
    _topic_lower = topic.lower()
    _topic_cat = (
        "programming" if any(k in _topic_lower for k in ("python","java","javascript","typescript","c++","rust","go","swift","kotlin","php","ruby")) else
        "science"     if any(k in _topic_lower for k in ("physics","chemistry","biology","chemistry","astronomy")) else
        "math"        if any(k in _topic_lower for k in ("math","calculus","algebra","geometry","statistics")) else
        "humanities"  if any(k in _topic_lower for k in ("history","philosophy","literature","economics","sociology")) else
        "other"
    )
    _metrics.active_course_creations.inc()

    try:
        profile = sanitize_trusted_profile_for_course(profile)
        state = await StudentState.load(student_id)
        state.domain = domain
        state.goal = goal
        state.pace = pace
        state.curriculum = None
        for concept in profile.get("assumed_known_concepts") or []:
            state.concept_mastery.setdefault(str(concept), 0.78)
            state.concept_depth.setdefault(str(concept), 0.55)
        for concept in profile.get("weak_concepts") or []:
            state.concept_mastery.setdefault(str(concept), 0.25)
            state.concept_depth.setdefault(str(concept), 0.2)
        architect = CurriculumArchitectAgent(state)
        setattr(architect, "personalization_profile", profile)
        setattr(architect, "student_history_snapshot", planner_history)
        plan = await architect.build_curriculum(topic)
        profile["scope_analysis"] = plan.scope_analysis
        profile["concept_inventory"] = plan.concept_inventory
        profile["prerequisite_graph"] = plan.prerequisite_graph
        profile["learning_path"] = plan.learning_path
        profile["roadmap_steps"] = plan.roadmap_steps
        profile["schedule_plan"] = plan.schedule_plan
        profile["validation_result"] = plan.validation_result
        # Read the curriculum_id that build_curriculum stored directly on the
        # architect — no extra DB query needed (avoids racing the ChromaDB
        # background task that starts loading BGE-M3 at this exact moment).
        curriculum_id = getattr(architect, "_curriculum_id", None)
        if curriculum_id is None:
            # Fallback: shouldn't normally happen, but guard gracefully.
            curriculum_row = await latest_curriculum_for_student(student_id)
            curriculum_id = int(curriculum_row["id"])
    except (asyncio.TimeoutError, TimeoutError) as exc:
        logger.warning("Course planner timed out for student='{}': {}", student_id, exc)
        await ensure_student(student_id, name, domain, goal, pace)
        _metrics.active_course_creations.dec()
        _metrics.course_creation_failures.labels(reason="llm_rate_limit").inc()
        raise ValueError("The AI model is temporarily unavailable. Please retry in a moment.") from exc
    except Exception as exc:
        logger.warning("Course planner failed for student='{}': {}", student_id, exc)
        await ensure_student(student_id, name, domain, goal, pace)
        details = str(exc)
        _metrics.active_course_creations.dec()
        if "Curriculum validation failed:" in details:
            _metrics.course_creation_failures.labels(reason="validation").inc()
            details = details.split("Curriculum validation failed:", 1)[1].strip()
            raise ValueError(
                "EduMind could not create a high-quality roadmap. Validation issues: "
                + details
            ) from exc
        if (
            "rate limit" in details.lower()
            or "timeout" in details.lower()
            or "timed out" in details.lower()
            or "availability" in details.lower()
            or type(exc).__name__ in ("GroqRateLimitError", "GroqTimeoutError")
        ):
            _metrics.course_creation_failures.labels(reason="llm_rate_limit").inc()
            raise ValueError("The AI model is temporarily rate-limited. Please retry in a moment.") from exc
        _metrics.course_creation_failures.labels(reason="other").inc()
        raise ValueError(
            "EduMind could not create a high-quality roadmap. Planner error: "
            + details
        ) from exc

    _metrics.active_course_creations.dec()
    _metrics.courses_created.labels(topic_category=_topic_cat).inc()

    course = await create_course_from_plan(
        student_id=student_id,
        curriculum_id=curriculum_id,
        plan=plan,
        pace=pace,
        prior_knowledge=prior_knowledge,
        personalization_profile=profile,
        web_search_enabled=web_search_enabled,
    )

    master_roadmap = getattr(architect, "_master_roadmap", None)
    if master_roadmap:
        try:
            roadmap_payload = (
                master_roadmap.model_dump()
                if hasattr(master_roadmap, "model_dump")
                else dict(master_roadmap)
            )
            await save_master_roadmap(course["id"], roadmap_payload)
        except Exception as exc:
            logger.warning("Master roadmap save failed for course '{}': {}", course["id"], exc)

    session_decisions = list(getattr(state, "session_decisions", []) or [])
    if session_decisions:
        course_id = course["id"]
        decision_records = [
            {
                "student_id": getattr(state, "student_id", student_id),
                "session_id": "course:" + course_id,
                "course_id": course_id,
                "agent": getattr(d, "agent", "curriculum_architect"),
                "action": getattr(d, "action", ""),
                "rationale": getattr(d, "reason", ""),
                "payload": d.model_dump() if hasattr(d, "model_dump") else dict(d),
            }
            for d in session_decisions
        ]
        try:
            await bulk_write_decisions(decision_records, course_id=course_id)
        except Exception as exc:
            logger.warning("Decision log flush failed for course '{}': {}", course_id, exc)

    course["modules"] = await list_course_modules(course["id"])
    roadmap = CourseRoadmapService().build(course, course["modules"], profile, planner_history)
    course["roadmap"] = await save_course_roadmap(course["id"], roadmap)
    course["roadmap_ready"] = True
    course["redirect_url"] = f"/courses/{course['id']}/roadmap"
    return course


async def create_course_events(
    student_id: str,
    topic: str,
    goal: str,
    pace: str = "medium",
    prior_knowledge: str = "",
    name: str = "Student",
    personalization_profile: dict[str, Any] | None = None,
    web_search_enabled: bool = False,
) -> AsyncGenerator[dict[str, Any], None]:
    """Stream course-creation progress events and finish with the saved course."""
    yield {"event": "connected", "data": {"message": "connected"}}
    yield {
        "event": "understanding_started",
        "data": {"message": "Understanding your goal", "topic": topic},
    }
    yield {
        "event": "history_checked",
        "data": {"message": "Checking your learning history"},
    }
    yield {
        "event": "research_started",
        "data": {"message": "Finding relevant sources", "topic": topic},
    }
    yield {
        "event": "source_found",
        "data": {
            "message": "Tavily and Chroma context will be used when available",
            "topic": topic,
        },
    }
    yield {
        "event": "curriculum_started",
        "data": {"message": "Designing your module path"},
    }
    try:
        course = await create_course(
            student_id=student_id,
            topic=topic,
            goal=goal,
            pace=pace,
            prior_knowledge=prior_knowledge,
            name=name,
            personalization_profile=personalization_profile,
            web_search_enabled=web_search_enabled,
        )
        for module in course.get("modules", []):
            yield {"event": "module_planned", "data": module}
        yield {
            "event": "roadmap_started",
            "data": {"message": "Creating your roadmap"},
        }
        yield {
            "event": "roadmap_ready",
            "data": {"message": "Roadmap ready", "roadmap": course.get("roadmap")},
        }
        yield {
            "event": "first_module_prepared",
            "data": {"message": "Preparing first module outline"},
        }
        yield {
            "event": "saved",
            "data": {
                "message": "Saving course",
                "course_id": course["id"],
                "redirect_url": course.get("redirect_url"),
            },
        }
        yield {
            "event": "course_ready",
            "data": {"message": "Course ready", "course_id": course["id"]},
        }
        yield {"event": "done", "data": course}
    except Exception as exc:
        logger.exception("Course creation stream failed")
        msg = str(exc)
        if "rate-limited" in msg or "rate limit" in msg.lower():
            yield {"event": "model_unavailable", "data": {"message": "The AI model is temporarily rate-limited. Please retry in a moment."}}
        else:
            yield {"event": "error", "data": {"message": msg}}


def _sentence_candidates(markdown: str) -> list[str]:
    """Split markdown into sentence candidates suitable for source quotes."""
    # Keep evidence text close to the final lesson so source_quote validation
    # can verify an actual substring instead of a markdown-stripped paraphrase.
    pieces = re.split(r"(?<=[.!?])\s+", markdown)
    return [
        re.sub(r"\s+", " ", p).strip()
        for p in pieces
        if len(p.split()) >= 7
    ]


def _section_sentence_candidates(markdown: str) -> list[tuple[str, str]]:
    """Return sentence candidates grouped by their nearest markdown heading."""
    current_section = "Lesson"
    result: list[tuple[str, str]] = []
    buffer: list[str] = []

    def flush(section: str, lines: list[str]) -> None:
        """Move buffered lesson lines into sentence candidates for one section."""
        text = "\n".join(lines).strip()
        if not text:
            return
        for sentence in _sentence_candidates(text):
            result.append((section, sentence))

    for line in (markdown or "").splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
        if match:
            flush(current_section, buffer)
            buffer = []
            current_section = re.sub(r"[*_`#]", "", match.group(1)).strip() or "Lesson"
        else:
            buffer.append(line)
    flush(current_section, buffer)
    if not result:
        result = [("Lesson", sentence) for sentence in _sentence_candidates(markdown)]
    return result


def _is_placeholder_question_text(text: str) -> bool:
    """Detect generic question phrasing that is not useful for grounded checks."""
    clean = re.sub(r"\s+", " ", text or "").strip().lower()
    return any(re.search(pattern, clean, flags=re.I) for pattern in PLACEHOLDER_QUESTION_PATTERNS)


def _question_auxiliary_for_concept(concept: str) -> str:
    """Choose a simple auxiliary verb for concept-specific question stems."""
    clean = str(concept or "").strip().lower()
    return "do" if clean.endswith("s") and not clean.endswith("ss") else "does"


def grounded_questions_from_content(
    content: str,
    module: dict[str, Any],
    pace: str,
) -> list[dict[str, Any]]:
    """Create deterministic lesson-grounded questions from saved markdown content."""
    target = {"fast": 2, "medium": 3, "deep": 5}.get(pace, 3)
    concepts_taught = module_required_concepts(module)
    question_scope = _module_question_scope(module, concepts_taught)
    lesson_concepts = _concepts_present_in_lesson(question_scope, content)
    if not lesson_concepts:
        lesson_concepts = _concepts_present_in_lesson(concepts_taught, content)
    concept = lesson_concepts[0] if lesson_concepts else (
        concepts_taught[0] if concepts_taught else module.get("concept", "this concept")
    )
    section_sentences = _section_sentence_candidates(content)
    if not section_sentences:
        section_sentences = [
            ("Lesson", f"{concept} means {module.get('description') or module.get('title')}."),
            ("Lesson", f"The module frames {concept} as {module.get('description') or concept}."),
        ]

    questions = []
    stems = [
        "What {auxiliary} {concept} let you do in this module?",
        "What does the lesson show about {concept}?",
        "Which beginner mistake should you avoid with {concept}?",
        "What would you change in the mini task to practice {concept}?",
        "Explain {concept} in your own words using the lesson example.",
    ]
    used_quotes: set[str] = set()
    for idx in range(target):
        tested = (
            lesson_concepts[idx % len(lesson_concepts)]
            if lesson_concepts else concept
        )
        grounded_sentence = _section_sentence_for_concept(section_sentences, tested, used_quotes)
        if grounded_sentence:
            source_section, quote = grounded_sentence
        else:
            source_section, quote = section_sentences[min(idx, len(section_sentences) - 1)]
        questions.append({
            "id": f"{module.get('id', 'module')}:q{idx + 1}",
            "question_text": stems[idx].format(
                concept=tested,
                auxiliary=_question_auxiliary_for_concept(tested),
            ),
            "expected_answer": quote,
            "source_quote": quote[:220],
            "concepts_tested": [tested],
            "source_section": source_section,
            "is_answerable_from_lesson": True,
            "difficulty": "simple" if idx < 2 else "applied",
        })
    return questions


def normalize_question_ids(
    course_id: str,
    module_id: str,
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign stable course/module-scoped ids to generated question objects."""
    normalized = []
    for idx, question in enumerate(questions, start=1):
        item = dict(question)
        item["id"] = f"{course_id}:{module_id}:q{idx}"
        normalized.append(item)
    return normalized


STRICT_QUESTION_RETRY_INSTRUCTION = (
    "Generate questions only from explicit facts/concepts present in the lesson content. "
    "Do not ask questions about anything not directly taught in the lesson. "
    "Do not use placeholder/meta-question phrasing such as 'According to the lesson', "
    "'What is the key idea', or 'What detail from the lesson explains'."
)


def question_generation_prompt(
    course: dict[str, Any],
    module: dict[str, Any],
    lesson_content: str,
    validation_issues: list[str],
) -> str:
    """Build the strict JSON prompt for retrying grounded question generation."""
    concepts_taught = module_required_concepts(module)
    question_scope = _module_question_scope(module, concepts_taught)
    lesson_concepts = _concepts_present_in_lesson(question_scope, lesson_content)
    if not lesson_concepts:
        lesson_concepts = _concepts_present_in_lesson(concepts_taught, lesson_content)
    target = {"fast": 2, "medium": 3, "deep": 5}.get(course.get("pace"), 3)
    return f"""Return STRICT JSON only. No markdown.

Create grounded check questions for this lesson.

{STRICT_QUESTION_RETRY_INSTRUCTION}

Course: {course.get('topic')}
Goal: {course.get('goal')}
Module title: {module.get('title')}
Module concept: {module.get('concept')}
Allowed concepts_tested: {json.dumps(lesson_concepts, default=str)}
Target question count: {target}

Previous validation issues:
{json.dumps(validation_issues, default=str)}

Lesson content:
\"\"\"
{lesson_content}
\"\"\"

Return this exact JSON shape:
{{
  "questions": [
    {{
      "question_text": "...",
      "expected_answer": "Use a short phrase or sentence copied exactly from the lesson.",
      "source_quote": "Copy one exact contiguous quote from the lesson that supports the answer.",
      "concepts_tested": ["one allowed concept"],
      "source_section": "Lesson",
      "is_answerable_from_lesson": true,
      "difficulty": "simple"
    }}
  ]
}}

Rules:
- source_quote must be copied verbatim from the lesson content.
- expected_answer must either be copied verbatim from the lesson or be fully supported by source_quote.
- concepts_tested must use only Allowed concepts_tested.
- Allowed concepts_tested has already been filtered to concepts explicitly present in the lesson.
- Do not mention or test a module concept that is absent from the lesson text.
- Do not use external knowledge, retrieved context, or future-course concepts.
- Ban placeholder/meta-question patterns:
  "According to the lesson...", "What is the key idea about...",
  "What detail from the lesson explains...", and generic "Why does X matter?"
- Questions should test understanding, application, prediction, common mistake recognition,
  or explanation in the learner's own words.
- For coding lessons, prefer concrete prompts such as output prediction, what a line does,
  what command to run, what small code change to make, or what beginner mistake causes failure.
- For math/science lessons, ask what a variable represents, which formula/idea applies,
  what changes the result, or which common mistake breaks the reasoning.
- For humanities lessons, ask about causes, consequences, actors, sequence, evidence,
  or what changed after the event/decision.
- If you cannot produce enough grounded questions, return fewer questions.
"""


def _coerce_question_list(raw_questions: Any, module: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize LLM question JSON into the backend question schema subset."""
    if not isinstance(raw_questions, list):
        return []
    default_concepts = _module_question_scope(module, module_required_concepts(module))
    default_concept = default_concepts[0] if default_concepts else str(module.get("concept") or "this concept")
    coerced: list[dict[str, Any]] = []
    for item in raw_questions:
        if not isinstance(item, dict):
            continue
        question_text = str(item.get("question_text") or item.get("question") or "").strip()
        expected = str(item.get("expected_answer") or item.get("answer") or "").strip()
        quote = str(item.get("source_quote") or item.get("evidence") or item.get("quote") or "").strip()
        concepts = item.get("concepts_tested") or item.get("concepts") or [default_concept]
        if not isinstance(concepts, list):
            concepts = [concepts]
        concepts = [str(concept).strip() for concept in concepts if str(concept).strip()]
        if not question_text or _is_placeholder_question_text(question_text):
            continue
        coerced.append({
            "question_text": question_text,
            "expected_answer": expected,
            "source_quote": quote,
            "concepts_tested": concepts or [default_concept],
            "source_section": str(item.get("source_section") or "Lesson").strip() or "Lesson",
            "is_answerable_from_lesson": bool(item.get("is_answerable_from_lesson", True)),
            "difficulty": str(item.get("difficulty") or "simple").strip() or "simple",
        })
    return coerced


async def _retry_grounded_questions_with_prompt(
    course: dict[str, Any],
    module: dict[str, Any],
    content: str,
    validation_issues: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retry question generation with explicit validation issues as feedback."""
    course_id = str(course.get("id") or "")
    module_id = str(module.get("id") or "")
    try:
        raw = await generate(
            messages=[{
                "role": "user",
                "content": question_generation_prompt(course, module, content, validation_issues),
            }],
            model=settings.generation_model,
            system="You are EduMind's grounded question writer. Return strict JSON only.",
        )
        data = parse_json_object(raw)
        questions = normalize_question_ids(
            course_id,
            module_id,
            _coerce_question_list(data.get("questions"), module),
        )
        validation = validate_questions_grounded(questions, content, module)
        return questions, validation
    except Exception as exc:
        logger.warning(
            "Question retry failed for course_id='{}' module_id='{}': {}",
            course_id,
            module_id,
            exc,
        )
        return [], {
            "passed": False,
            "issues": [str(exc)],
            "quality_score": 0.0,
            "regenerate_required": True,
            "regenerate": True,
        }


async def generate_validated_questions_for_lesson(
    course: dict[str, Any],
    module: dict[str, Any],
    content: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate grounded questions, retry once, and fall back to no questions."""
    course_id = str(course.get("id") or "")
    module_id = str(module.get("id") or "")
    questions = normalize_question_ids(
        course_id,
        module_id,
        grounded_questions_from_content(content, module, course.get("pace", "medium")),
    )
    validation = validate_questions_grounded(questions, content, module)
    if validation["passed"]:
        return questions, {"status": "passed", "issues": [], "retry_happened": False}

    first_issues = [str(issue) for issue in validation.get("issues") or []]
    logger.warning(
        "Question validation failed for course_id='{}' module_id='{}': issues={} retry_happened={} fallback_saved_without_questions={}",
        course_id,
        module_id,
        first_issues,
        False,
        False,
    )

    retry_questions, retry_validation = await _retry_grounded_questions_with_prompt(
        course,
        module,
        content,
        first_issues,
    )
    if retry_validation["passed"]:
        logger.info(
            "Question validation retry succeeded for course_id='{}' module_id='{}'",
            course_id,
            module_id,
        )
        return retry_questions, {"status": "passed_after_retry", "issues": [], "retry_happened": True}

    retry_issues = [str(issue) for issue in retry_validation.get("issues") or []]
    logger.warning(
        "Question validation retry failed for course_id='{}' module_id='{}': issues={} retry_happened={} fallback_saved_without_questions={}",
        course_id,
        module_id,
        retry_issues,
        True,
        True,
    )
    return [], {
        "status": "failed",
        "issues": first_issues + retry_issues,
        "retry_happened": True,
    }


async def adaptation_context_for_module(
    student_id: str,
    course_id: str,
    module: dict[str, Any],
) -> dict[str, Any]:
    """Collect student skill, doubt, and adaptation signals for lesson generation."""
    from db.postgres import get_adaptation_summary, get_compact_doubt_summary
    course = await get_course(course_id)
    profile = (course or {}).get("personalization_profile") or {}
    skills = await get_student_skills(student_id)
    weak = [
        n.get("concept")
        for n in skills.get("weak_concepts", [])
        if n.get("concept") and (
            not profile or n.get("concept") in relevant_history_concepts(profile, {"weak_concepts": [n]})["weak"]
        )
    ][:5]
    history = await list_module_chat_history_for_student(course_id, module["id"], student_id)
    recent_doubts = [m for m in history if m.get("role") == "user"][-5:]

    # Pull in compact adaptation summary from previous evaluations
    adaptation_summary = await get_adaptation_summary(student_id, course_id) or {}
    doubt_summary = await get_compact_doubt_summary(course_id, module["id"], student_id)

    # Build a rich doubt signal for the lesson generator.
    # Doubts logged in chat during lesson consumption represent genuine confusion.
    # We inject them so the lesson can pre-empt similar confusion.
    doubt_messages = [m.get("content", "") for m in recent_doubts if m.get("content")]
    doubt_concepts: list[str] = []
    doubt_types: list[str] = []
    if doubt_summary and isinstance(doubt_summary, dict):
        doubt_concepts = list(doubt_summary.get("concepts", []))[:6]
        doubt_types = list(doubt_summary.get("types", []))[:4]

    # Build specific, actionable teaching adjustments from all signals
    adjustments: list[str] = []
    if weak:
        adjustments.append(
            f"Student has weak mastery of: {', '.join(weak[:5])}. "
            "Add a short prerequisite bridge before introducing concepts that depend on these."
        )
    if doubt_concepts:
        adjustments.append(
            f"Student asked questions about: {', '.join(doubt_concepts)}. "
            "Pre-emptively address these areas with clearer explanations and extra examples."
        )
    if doubt_types and "misconception" in doubt_types:
        adjustments.append(
            "Student has shown misconception-type doubts. "
            "Add a 'Common mistakes' or 'Watch out for' section to this module."
        )
    if adaptation_summary.get("example_preference") == "more":
        adjustments.append("Student prefers more worked examples — add at least 2 additional examples.")
    if adaptation_summary.get("pace_adjustment") == "slower":
        adjustments.append("Student finds the pace fast — break each concept into smaller steps.")
    prev_weak = list(adaptation_summary.get("weak_concepts", []))[:5]
    if prev_weak:
        adjustments.append(
            f"Previous evaluations flagged weakness in: {', '.join(prev_weak)}. "
            "If this module builds on those concepts, add a 1-paragraph recap before proceeding."
        )

    return {
        "current_module": module.get("title"),
        "recent_doubt_messages": doubt_messages,
        "doubt_concepts": doubt_concepts,
        "doubt_summary": doubt_summary,
        "weak_concepts": weak,
        "student_mastery": {
            n.get("concept"): n.get("mastery_score")
            for n in skills.get("nodes", [])
            if n.get("concept")
        },
        "adaptation_summary": adaptation_summary,
        "recommended_teaching_adjustments": adjustments,
    }


def lesson_prompt(
    course: dict[str, Any],
    module: dict[str, Any],
    context_chunks: list[str],
    adaptation_context: dict[str, Any],
    previous_modules: list[dict[str, Any]] | None = None,
) -> str:
    """Build the lesson-generation prompt from course, module, and adaptation data."""
    pace = course.get("pace", "medium")
    concept = module.get("concept", "")
    profile = course.get("personalization_profile") or {}
    scope = profile.get("scope_analysis") or {}
    roadmap_steps = profile.get("roadmap_steps") or []
    metadata = module.get("module_metadata") or {}
    concepts_taught = module_required_concepts(module) or [concept]
    depends_on = module.get("depends_on_concepts") or metadata.get("depends_on_concepts") or module.get("prerequisites") or []
    question_scope = _module_question_scope(module, concepts_taught)
    lesson_requirements = module.get("lesson_requirements") or metadata.get("lesson_requirements") or []
    practice_requirements = module.get("practice_requirements") or metadata.get("practice_requirements") or []
    focus_concepts = _clean_list(
        module.get("focus_concepts")
        or metadata.get("focus_concepts")
        or module.get("focus_concept")
        or metadata.get("focus_concept")
    )
    if len(concepts_taught) > 1 and not focus_concepts:
        concept_coverage_requirements = (
            "This module has multiple required concepts. Teach every required concept by name: "
            + ", ".join(concepts_taught)
            + ". For each one, include what it is, when to use it, a simple example or demonstration, "
            "and a common beginner mistake. Then compare when to choose each concept before the practice task."
        )
    elif focus_concepts:
        concept_coverage_requirements = (
            "This module metadata narrows the lesson focus to: "
            + ", ".join(focus_concepts)
            + ". Keep other planned concepts only as brief context unless the metadata also requires teaching them."
        )
    else:
        concept_coverage_requirements = (
            "Teach the required concept by name and make the worked example actually perform it: "
            + ", ".join(concepts_taught)
            + "."
        )
    not_cover = (
        module.get("what_this_module_will_not_cover")
        or metadata.get("what_this_module_will_not_cover")
        or scope.get("what_to_exclude")
        or []
    )
    previous_summary = [
        {
            "title": item.get("title"),
            "concepts_taught": item.get("concepts_taught") or item.get("must_teach") or [item.get("concept")],
        }
        for item in previous_modules or []
    ]
    if pace == "fast":
        pace_requirements = """PACE: FAST — Student is time-constrained and needs capsule learning.

Your job: deliver the essential mental model and one strong worked example. Nothing more.

Content behavior:
- Open with a one-sentence "why this matters" hook
- State the core idea in 2-3 bullet points — no prose paragraphs
- One concrete worked example that demonstrates the concept directly
- One "watch out" — the single most common mistake
- One mini practice task (2-3 sentences max)
- Short 3-bullet recap

Do NOT write flowing paragraphs. Do NOT add background, history, or theory.
Every sentence must earn its place. If it can be cut, cut it.
The student should finish in under 5 minutes and walk away with the key idea locked in."""

    elif pace == "deep":
        pace_requirements = """PACE: DEEP — Student is in researcher mode. They want mastery, not familiarity.

Your job: treat this concept the way a university professor or subject expert would
in a dedicated lecture. The student has time and genuine curiosity. Reward it.

Content behavior:
- Open with context: where this concept sits in the broader subject, why it matters
- Explain the core idea fully, then go deeper — cover the "why behind the why"
- Break the concept into its sub-components and explain each one individually
- Include the historical or scientific origin of the idea where relevant
- Cover at least 3 worked examples at increasing complexity
- Address competing interpretations, edge cases, or exceptions
- Connect explicitly to adjacent concepts the student will encounter later
- Include what experts find interesting, counterintuitive, or still debated
- Surface common misconceptions at a deeper level than "beginners confuse X with Y"
- Practice task should require genuine reasoning, not just recall

Do NOT summarize. Do NOT give a surface overview and call it done.
If a sub-topic deserves its own section, give it one.
The student expects the depth of a textbook chapter combined with a mentor's clarity.
Length is a natural byproduct of real depth — write until the concept is truly covered."""

    else:
        pace_requirements = """PACE: MEDIUM — Standard academic treatment. Clear, complete, supported.

Your job: teach this concept the way a good school or university course would —
enough to fully understand and apply it, without overwhelming detail.

Content behavior:
- Clear explanation of what the concept is and why it matters
- Build intuition before introducing formal definitions or formulas
- Two worked examples: one simple, one slightly more applied
- Address one common misconception
- A guided practice task with an expected answer
- Connect briefly to what comes next in the course

Write in flowing prose with clear structure. Not too brief, not exhaustive.
The student should finish feeling they genuinely understand the concept
and could explain it to someone else."""

    return f"""Write a polished markdown lesson for an AI learning platform.

The lesson should feel like a human mentor teaching a focused course page:
clear, practical, warm, and specific to this learner. Use the planning metadata
below to decide what to teach, but do not expose raw metadata labels as
student-facing headings.

Course topic: {course.get('topic')}
Student goal: {course.get('goal')}
Pace: {pace}
Module title: {module.get('title')}
Concept: {concept}

Internal planning metadata. Use this as guidance only; do not turn these keys
into lesson sections:
{json.dumps({
    "course_scope_analysis": scope,
    "roadmap_steps": roadmap_steps,
    "previous_modules_already_taught": previous_summary,
    "concepts_taught_in_this_module_only": concepts_taught,
    "depends_on_concepts": depends_on,
    "question_scope_for_later_checks": question_scope,
    "module_description": module.get("description"),
    "purpose": module.get("purpose") or module.get("module_metadata", {}).get("purpose"),
    "why_it_matters_for_goal": module.get("why_it_matters_for_goal") or module.get("module_metadata", {}).get("why_it_matters_for_goal"),
    "must_teach": module.get("must_teach") or module.get("module_metadata", {}).get("must_teach") or [],
    "examples_to_include": module.get("examples_to_include") or module.get("module_metadata", {}).get("examples_to_include") or [],
    "practice_type": module.get("practice_type") or module.get("module_metadata", {}).get("practice_type"),
    "prerequisites": module.get("prerequisites"),
    "why_now": module.get("why_now") or module.get("module_metadata", {}).get("why_now"),
    "this_module_will_not_cover": not_cover,
    "lesson_requirements": lesson_requirements,
    "practice_requirements": practice_requirements,
}, default=str)}

Adaptation context:
{json.dumps(adaptation_context, default=str)}

ACTION REQUIRED — apply ALL teaching adjustments listed in recommended_teaching_adjustments.
{chr(10).join(adaptation_context.get("recommended_teaching_adjustments") or []) or "No specific adjustments."}
If adaptation_summary contains weak_concepts, add a brief recap of those before the main explanation.
If adaptation_summary contains example_preference=more, include an extra worked example.
If adaptation_summary contains pace_adjustment=slower, use smaller steps and more line-by-line explanation.
If doubt_concepts is non-empty, pre-emptively address each of those concepts with extra clarity.
If recent_doubt_messages is non-empty, those are questions the student actually asked — answer them inline within the relevant section of this lesson.

Retrieved context:
{chr(10).join(context_chunks[:4])}

Required teaching flow:
1. Mentor-style opening / hook that gives the learner one clear reason to care.
2. What you will be able to do by the end.
3. Mental model: explain the core idea in plain language before details.
4. Step-by-step explanation in prerequisite order.
5. Worked example / demonstration that actually performs the concept.
6. Line-by-line explanation if code, math, formulas, or structured evidence appears.
7. Common beginner mistake and how to avoid it.
8. Mini practice task.
9. Expected output, expected answer, or solution sketch for that task.
10. Short recap.

Required concept coverage:
{concept_coverage_requirements}

Rules:
- Teach only Concepts taught in this module, explicitly listed dependencies, and tiny recaps of previous modules.
- Teach the concrete content listed in the internal must_teach and lesson_requirements metadata.
- Do not count a shared umbrella word as coverage. For example, teaching only "for loops" does not cover "while loops".
- Do not introduce concepts outside question_scope_for_later_checks except as a clearly labeled one-sentence preview.
- Never use excluded/delayed topics from "This module will not cover" as examples, exercises, or questions.
- Include examples appropriate to the course topic and target context.
- For programming or coding courses, include concrete runnable code blocks,
  expected output, a line-by-line explanation, an output prediction moment,
  and one small modification task. A programming lesson without code is incomplete.
- For math, physics, chemistry, or science courses, include intuition, formula
  meaning, concrete quantities, a worked example, a common mistake, and a practice problem.
- For history, humanities, or social science courses, include context, cause-effect flow,
  timeline or actors when relevant, evidence/examples, and a misconception to avoid.
- Treat Retrieved context as optional evidence. Ignore any retrieved chunk that conflicts with this module boundary.
- Do not include "Any doubts?" or interruptive chat prompts.
- Any in-lesson check questions must be answerable from this lesson alone.
- The backend generates the saved check-question objects separately after this lesson exists;
  do not emit JSON or metadata for those questions inside the lesson.
- Avoid out-of-syllabus terms unless you define them in the lesson first.
- Make the lesson feel like a real course page, not a tiny note.
- Do not write a generic template where the example says only "identify,
  apply, interpret"; the worked example must actually perform the concept.
- Do not use student-facing headings named "Must Teach", "Lesson Requirements",
  "Concepts Taught in this Module", "Practice Requirements", "Question Scope",
  "Module Goal", or "Why It Matters for Goal".
- Use markdown headings, but choose natural learner-facing headings.

Pace-specific requirements:
{pace_requirements}
"""


async def generate_module_lesson(
    course_id: str,
    module_id: str,
    student_id: str | None = None,
) -> dict[str, Any]:
    """
    Generate and persist lesson markdown for one module.

    Existing lesson content is returned without regeneration. New content is
    validated, saved with optional videos, and returned with an empty question
    list because questions are generated only after lesson content exists.
    """
    course = await get_course(course_id, student_id)
    module = await get_course_module(course_id, module_id)
    if not course or not module:
        raise ValueError("Course or module not found")
    if module.get("content_markdown"):
        return {**module, "questions": [], "videos": _lesson_videos_from_module(module)}

    await set_module_status(course_id, module_id, "in_progress")
    try:
        all_modules = await list_course_modules(course_id)
    except Exception as exc:
        logger.warning("Previous-module context unavailable for '{}': {}", module_id, exc)
        all_modules = []
    current_index = int(module.get("module_index") or 0)
    previous_modules = [
        item for item in all_modules
        if int(item.get("module_index") or 0) < current_index
    ]
    adaptation_context = await adaptation_context_for_module(
        course["student_id"], course_id, module
    )
    context_chunks: list[str] = []  # V1: RAG disabled
    # try:                                                     # V2
    #     context_chunks = await retrieve(                     # V2
    #         query=f"{module['concept']} {course['topic']} {course['goal']}",  # V2
    #         domain=infer_domain(course["topic"], course["goal"]),  # V2
    #         top_k=4,                                         # V2
    #         course_id=course_id,                             # V2
    #         student_id=course["student_id"],                 # V2
    #         topic=course.get("topic"),                       # V2
    #         module_id=module_id,                             # V2
    #     )                                                    # V2
    # except Exception as exc:                                 # V2
    #     logger.warning("Lesson RAG failed for '{}': {}", module["concept"], exc)  # V2
    #     context_chunks = []                                  # V2

    try:
        content = await generate(
            messages=[{
                "role": "user",
                "content": lesson_prompt(course, module, context_chunks, adaptation_context, previous_modules),
            }],
            model=settings.generation_model,
            system="You are EduMind's expert course writer. Return markdown only.",
        )
    except Exception as exc:
        logger.warning("Lesson generation failed for '{}': {}", module["concept"], exc)
        return {
            "error": "rate_limited",
            "message": "The AI model is temporarily unavailable. Please retry.",
            "lesson": None
        }

    lesson_validation = validate_lesson_quality(content, course, module, context_chunks)
    if not lesson_validation["passed"]:
        try:
            content = await generate(
                messages=[{
	                    "role": "user",
	                    "content": (
	                        lesson_prompt(course, module, context_chunks, adaptation_context, previous_modules)
	                        + "\n\nPrevious lesson failed validation:\n"
	                        + "\n".join("- " + issue for issue in lesson_validation["issues"])
	                        + "\nRegenerate a corrected lesson."
	                    ),
                }],
                model=settings.generation_model,
                system="You are EduMind's expert course writer. Return corrected markdown only.",
            )
            lesson_validation = validate_lesson_quality(content, course, module, context_chunks)
        except Exception as exc:
            logger.warning("Lesson validation retry failed for '{}': {}", module["concept"], exc)
    if not lesson_validation["passed"]:
        # Never block delivery — student always gets content.
        # Validation issues are signals for improvement on retry, not reasons to crash.
        logger.warning(
            "Lesson for \'{}\' has quality issues (delivering best attempt): {}",
            module.get("concept"), "; ".join(lesson_validation["issues"])
        )

    videos = await _optional_youtube_videos_for_module(course, module)
    await save_module_content(course_id, module_id, content, [], videos=videos)
    logger.info(
        "youtube_saved_videos course_id='{}' module_id='{}' saved_video_count={}",
        course_id,
        module_id,
        len(videos or []),
    )
    updated = await get_course_module(course_id, module_id)
    response_videos = videos or _lesson_videos_from_module(updated or {})
    logger.info(
        "module_lesson_response_videos course_id='{}' module_id='{}' response_video_count={}",
        course_id,
        module_id,
        len(response_videos or []),
    )
    return {**updated, "questions": [], "videos": response_videos}


async def generate_module_lesson_events(
    course_id: str,
    module_id: str,
    student_id: str | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Stream lesson generation, persist the final lesson, and emit saved state."""
    yield {"event": "connected", "data": {"message": "connected"}}
    course = await get_course(course_id, student_id)
    module = await get_course_module(course_id, module_id)
    if not course or not module:
        yield {"event": "error", "data": {"message": "Course or module not found"}}
        return
    if module.get("content_markdown"):
        yield {"event": "done", "data": {**module, "questions": [], "videos": _lesson_videos_from_module(module)}}
        return

    await set_module_status(course_id, module_id, "in_progress")
    yield {"event": "lesson_started", "data": {"module_id": module_id}}
    try:
        all_modules = await list_course_modules(course_id)
    except Exception:
        all_modules = []
    current_index = int(module.get("module_index") or 0)
    previous_modules = [
        item for item in all_modules
        if int(item.get("module_index") or 0) < current_index
    ]
    adaptation_context = await adaptation_context_for_module(
        course["student_id"], course_id, module
    )
    context_chunks: list[str] = []  # V1: RAG disabled
    # try:                                                     # V2
    #     context_chunks = await retrieve(                     # V2
    #         query=f"{module['concept']} {course['topic']} {course['goal']}",  # V2
    #         domain=infer_domain(course["topic"], course["goal"]),  # V2
    #         top_k=4,                                         # V2
    #         course_id=course_id,                             # V2
    #         student_id=course["student_id"],                 # V2
    #         topic=course.get("topic"),                       # V2
    #         module_id=module_id,                             # V2
    #     )                                                    # V2
    # except Exception:                                        # V2
    #     context_chunks = []                                  # V2

    chunks: list[str] = []
    try:
        async for chunk in stream(
	            messages=[{
	                "role": "user",
	                "content": lesson_prompt(course, module, context_chunks, adaptation_context, previous_modules),
	            }],
            model=settings.generation_model,
            system="You are EduMind's expert course writer. Return markdown only.",
        ):
            chunks.append(chunk)
        content = "".join(chunks).strip()
        if not content:
            raise ValueError("Empty streamed lesson")
    except Exception as exc:
        logger.warning("Streamed lesson generation failed for '{}': {}", module["concept"], exc)
        yield {
            "event": "error",
            "data": {
                "message": "The AI model could not finish this lesson. Please retry in a moment."
            },
        }
        return

    lesson_validation = validate_lesson_quality(content, course, module, context_chunks)
    if not lesson_validation["passed"]:
        try:
            content = await generate(
                messages=[{
                    "role": "user",
	                    "content": (
	                        lesson_prompt(course, module, context_chunks, adaptation_context, previous_modules)
	                        + "\n\nPrevious lesson failed validation:\n"
                        + "\n".join("- " + issue for issue in lesson_validation["issues"])
                        + "\nRegenerate a corrected lesson."
                    ),
                }],
                model=settings.generation_model,
                system="You are EduMind's expert course writer. Return corrected markdown only.",
            )
            lesson_validation = validate_lesson_quality(content, course, module, context_chunks)
        except Exception as exc:
            logger.warning("Stream lesson validation retry failed for '{}': {}", module["concept"], exc)
    if not lesson_validation["passed"]:
        # Never block delivery — log the issues and continue with best available content.
        logger.warning(
            "Streamed lesson for \'{}\' has quality issues (delivering best attempt): {}",
            module.get("concept"), "; ".join(lesson_validation["issues"])
        )
    yield {"event": "chunk", "data": content}

    videos = await _optional_youtube_videos_for_module(course, module)
    await save_module_content(course_id, module_id, content, [], videos=videos)
    logger.info(
        "youtube_saved_videos course_id='{}' module_id='{}' saved_video_count={}",
        course_id,
        module_id,
        len(videos or []),
    )
    updated = await get_course_module(course_id, module_id)
    response_videos = videos or _lesson_videos_from_module(updated or {})
    logger.info(
        "module_lesson_response_videos course_id='{}' module_id='{}' response_video_count={}",
        course_id,
        module_id,
        len(response_videos or []),
    )
    yield {"event": "saved", "data": {"module_id": module_id}}
    yield {"event": "done", "data": {**updated, "questions": [], "videos": response_videos}}


async def get_or_create_module_questions(
    course: dict[str, Any],
    module: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return saved module questions or generate grounded questions from content."""
    questions = await get_module_questions(course["id"], module["id"])
    if questions:
        return questions
    content = module.get("content_markdown", "")
    if not content:
        raise ValueError("Cannot generate grounded questions before lesson content exists.")
    generated, question_status = await generate_validated_questions_for_lesson(
        course,
        module,
        content,
    )
    if question_status.get("status") == "failed":
        logger.warning(
            "Returning module without questions for course_id='{}' module_id='{}': issues={} retry_happened={} fallback_saved_without_questions={}",
            course.get("id"),
            module.get("id"),
            question_status.get("issues") or [],
            question_status.get("retry_happened", False),
            True,
        )
        return []
    await save_module_questions(course["id"], module["id"], generated)
    return await get_module_questions(course["id"], module["id"])


def classify_doubt(message: str, module: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    """Classify a module chat message into a doubt type and related concepts."""
    text = message.lower()
    prereqs = [str(p) for p in module.get("prerequisites") or []]
    related = [module.get("concept", "")]
    missing: list[str] = []
    if any(pr.lower() in text for pr in prereqs):
        dtype = "prerequisite gap"
        missing = [pr for pr in prereqs if pr.lower() in text]
    elif any(word in text for word in ("why", "how", "confused", "understand")):
        dtype = "conceptual"
    elif any(word in text for word in ("example", "worked", "step")):
        dtype = "example confusion"
    elif re.search(r"\b(apply|application|use|using|real world|practice)\b", text):
        dtype = "application confusion"
    elif any(word in text for word in ("term", "word", "mean", "definition", "symbol")):
        dtype = "terminology confusion"
    else:
        dtype = "conceptual"
    if dtype == "prerequisite gap" and not missing:
        missing = prereqs[:2]
    related.extend(missing)
    return dtype, [r for r in related if r], missing


async def update_metacognition_from_doubt(
    student_id: str,
    concept: str,
    doubt_type: str,
) -> None:
    """Update long-term doubt counters in the student's metacognition profile."""
    profile = await load_metacognition(student_id) or {}
    doubt_profile = profile.setdefault("doubt_profile", {})
    by_type = doubt_profile.setdefault("by_type", {})
    by_concept = doubt_profile.setdefault("by_concept", {})
    by_type[doubt_type] = int(by_type.get(doubt_type, 0)) + 1
    by_concept[concept] = int(by_concept.get(concept, 0)) + 1
    weak_types = profile.setdefault("weak_concept_types", [])
    if doubt_type not in weak_types:
        weak_types.append(doubt_type)
    await save_metacognition(student_id, profile)


async def _answer_doubt_with_web_search(
    course: dict[str, Any],
    module: dict[str, Any],
    message: str,
    dtype: str,
    context: str,
    history: list[dict[str, Any]],
) -> str | None:
    """
    Answer a doubt with an LLM-driven web-search tool loop.

    The model is grounded in the module content and may call the MCP web-search
    tools, but only when it decides the question needs external knowledge. Chunks
    are scoped to this course's namespace. Returns the reply, or None to signal
    the caller to fall back to the default grounded path (e.g. MCP unreachable or
    the model produced no answer).
    """
    course_id = course.get("id") or ""
    ctx = (
        f"Course: {course.get('topic')} | Module concept: {module.get('concept')} "
        f"| Pace: {course.get('pace', 'medium')}"
    )
    system = (
        "You are EduMind's module chat assistant. Answer the student's doubt clearly "
        "and stay grounded in the current module content below.\n\n"
        "You have web-search tools. Use them ONLY when the student's question involves "
        "a concept you do not recognize, or needs current/external detail the module "
        "does not cover. In that case call smoke_search first to orient, then "
        "research_web to fetch grounded sources, then answer. If the module already "
        "answers the question, do NOT search — just answer.\n"
        "Always finish by calling the `answer` tool with your final reply. Label any "
        'content beyond the module as "extra context". Do not invent facts.\n\n'
        f'MODULE CONTENT:\n"""\n{context[:6000]}\n"""\n\n'
        f"RECENT CHAT:\n{json.dumps(history[-6:], default=str)}"
    )
    tools = mcp_search_client.groq_tools() + [
        {
            "type": "function",
            "function": {
                "name": "answer",
                "description": "Provide the final answer to the student's doubt.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reply": {"type": "string", "description": "The final answer to show the student."}
                    },
                    "required": ["reply"],
                },
            },
        }
    ]
    executor = mcp_search_client.make_tool_executor(namespace=course_id, context=ctx)
    try:
        result = await tool_call_loop(
            system=system,
            user_message=f"Student doubt: {message}",
            tools=tools,
            terminal_tool_name="answer",
            model=settings.generation_model,
            tool_executor=executor,
        )
        reply = (result or {}).get("reply")
        return reply.strip() if isinstance(reply, str) and reply.strip() else None
    except Exception as exc:
        logger.warning("Web-search doubt loop failed: {} — falling back to grounded answer.", exc)
        return None


async def answer_module_chat(
    course_id: str,
    module_id: str,
    student_id: str,
    message: str,
) -> dict[str, Any]:
    """
    Answer a student's module chat question and persist doubt evidence.

    The response is grounded in the saved module lesson and records both the
    user message and assistant reply for future adaptation context.
    """
    course = await get_course(course_id, student_id)
    module = await get_course_module(course_id, module_id)
    if not course or not module:
        raise ValueError("Course or module not found")

    dtype, related, missing = classify_doubt(message, module)
    history = await list_module_chat_history_for_student(course_id, module_id, student_id)
    context = module.get("content_markdown") or ""
    if not context:
        raise ValueError("Generate the lesson before opening module chat.")
    # Web-search RAG is opt-in per course. When ON, the LLM itself decides
    # whether it needs to search the web (via the MCP tools) — it only does so
    # for concepts it doesn't recognize or that need current/external detail.
    reply: str | None = None
    if bool(course.get("web_search_enabled")) and mcp_search_client.is_enabled():
        reply = await _answer_doubt_with_web_search(
            course=course, module=module, message=message,
            dtype=dtype, context=context, history=history,
        )

    if reply is None:
        # Default grounded path (unchanged): answer from the saved module only.
        prompt = f"""A student asked a doubt in the side chat.

Course: {course.get('topic')}
Module: {module.get('title')} / {module.get('concept')}
Doubt type: {dtype}
Student message: {message}

Current module content is the primary source:
\"\"\"
{context[:6000]}
\"\"\"

Recent chat:
{json.dumps(history[-6:], default=str)}

Answer simply and clearly. Stay grounded in the current module. If you add
anything beyond the module, label it as "extra context". Do not invent facts.
"""
        try:
            reply = await generate(
                messages=[{"role": "user", "content": prompt}],
                model=settings.generation_model,
                system="You are EduMind's module chat assistant.",
            )
        except Exception as exc:
            logger.warning("Module chat generation failed: {}", exc)
            reply = "I could not generate a reliable answer right now. Please retry in a moment."

    await record_module_chat_message(
        student_id, course_id, module_id, "user", message,
        dtype, related, missing,
    )
    await record_module_chat_message(
        student_id, course_id, module_id, "assistant", reply,
        dtype, related, missing,
    )
    await record_doubt(
        student_id=student_id,
        course_id=course_id,
        concept=module["concept"],
        doubt_text=message,
        doubt_type=dtype,
    )
    await upsert_student_skill(
        student_id=student_id,
        concept=module["concept"],
        mastery_score=0.35,
        depth_score=0.25,
        source="doubt",
        status="weak" if dtype == "prerequisite gap" else "learning",
        evidence={
            "course_id": course_id,
            "module_id": module_id,
            "doubt_type": dtype,
            "message": message,
        },
    )
    await update_metacognition_from_doubt(student_id, module["concept"], dtype)

    return {
        "reply": reply,
        "doubt_type": dtype,
        "related_concepts": related,
        "possible_missing_prerequisites": missing,
        "saved": True,
    }


async def evaluate_module_answer(
    course_id: str,
    module_id: str,
    student_id: str,
    question_id: str,
    answer: str,
    confidence: int = 3,
) -> dict[str, Any]:
    """Evaluate one saved check-question answer and persist mastery signals."""
    course = await get_course(course_id, student_id)
    module = await get_course_module(course_id, module_id)
    if not course or not module:
        raise ValueError("Course or module not found")
    questions = await get_or_create_module_questions(course, module)
    question = next((q for q in questions if q["id"] == question_id), None)
    if not question:
        raise ValueError("Question not found")

    correctness = 0.0
    depth = 0.0
    mastery = round(0.6 * correctness + 0.4 * depth, 3)
    threshold = {"fast": 0.60, "medium": 0.72, "deep": 0.85}.get(
        course.get("pace"), 0.72
    )
    action = "MOVE_FORWARD" if mastery >= threshold else "RETEACH"
    if mastery >= threshold - 0.1 and action != "MOVE_FORWARD":
        action = "MOVE_FORWARD_WITH_FLAG"

    await write_evaluation({
        "student_id": student_id,
        "session_id": "course:" + course_id,
        "concept": module["concept"],
        "correctness_score": correctness,
        "depth_score": depth,
        "mastery_score": mastery,
        "misconception_type": None if mastery >= threshold else "conceptual",
        "misconception_detail": "" if mastery >= threshold else "Answer missed some grounded lesson details.",
        "confidence_stated": max(1, min(5, int(confidence or 3))),
        "calibration_delta": round(max(1, min(5, int(confidence or 3))) / 5 - mastery, 4),
        "questions_asked": 1,
        "recommended_action": action,
    })
    await upsert_concept_mastery(student_id, module["concept"], correctness, depth)
    await upsert_student_skill(
        student_id=student_id,
        concept=module["concept"],
        mastery_score=mastery,
        depth_score=depth,
        source="evaluation",
        status="mastered" if mastery >= threshold else "learning",
        evidence={
            "course_id": course_id,
            "module_id": module_id,
            "question_id": question_id,
            "answer": answer,
        },
    )
    if mastery >= threshold:
        await set_module_status(course_id, module_id, "completed")

    return {
        "correctness_score": correctness,
        "depth_score": depth,
        "mastery_score": mastery,
        "feedback": (
            "Good work. Your answer matches the lesson's key idea."
            if mastery >= threshold
            else "You are close, but review the source quote and connect your answer more directly to the lesson."
        ),
        "misconceptions": [] if mastery >= threshold else ["conceptual"],
        "recommended_action": action,
        "next_step": "Continue to next module" if mastery >= threshold else "Review this module and try another question",
    }


async def complete_module(course_id: str, module_id: str, student_id: str | None = None) -> dict[str, Any]:
    """
    Mark module completed. Returns:
      - updated module
      - next_module / prev_module for navigation
      - evaluation_prompt: whether to show the evaluation popup
      - course_complete: True if this was the last module
    """
    from db.postgres import get_next_module, get_prev_module
    course = await get_course(course_id, student_id)
    if not course:
        raise ValueError("Course not found")
    module = await get_course_module(course_id, module_id)
    if not module:
        raise ValueError("Module not found")

    await set_module_status(course_id, module_id, "completed")

    # Optimistic skill save at completion (will be refined by evaluation)
    real_student_id = student_id or course.get("student_id", "")
    await upsert_student_skill(
        student_id=real_student_id,
        concept=module["concept"],
        mastery_score=0.65,
        depth_score=0.55,
        source="completion",
        status="learning",
        evidence={"course_id": course_id, "module_id": module_id},
    )

    updated_module = await get_course_module(course_id, module_id)
    next_mod = await get_next_module(course_id, module_id)
    prev_mod = await get_prev_module(course_id, module_id)

    # Check if course is now fully complete
    updated_course = await get_course(course_id, real_student_id)
    course_complete = (updated_course or {}).get("status") == "completed"

    return {
        "module": updated_module,
        "next_module": next_mod,
        "prev_module": prev_mod,
        "course_complete": course_complete,
        "evaluation_prompt": {
            "show": True,
            "message": "Would you like to take a quick evaluation to test your understanding?",
            "optional": True,
        },
    }


__all__ = [
    "answer_module_chat",
    "complete_module",
    "create_course",
    "create_course_events",
    "evaluate_module_answer",
    "generate_module_lesson",
    "generate_module_lesson_events",
    "get_or_create_module_questions",
    "get_student_dashboard",
    "get_student_doubts",
    "get_student_skills",
    "grounded_questions_from_content",
    "lesson_videos_from_module",
    "list_courses",
    "module_required_concepts",
    "search_youtube_videos_for_module",
    "youtube_watch_url_to_embed_url",
    "youtube_videos_from_tavily_results",
]
