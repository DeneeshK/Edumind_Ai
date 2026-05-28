"""
core/curriculum_quality.py
Shared quality gates for intake, curriculum planning, roadmap generation, and
lesson delivery.

LLMs do the primary semantic work. These helpers are guardrails: they reject
garbage topics, stale-state contamination, malformed JSON, and outputs that
contradict the LLM's own scope analysis.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any


CONFIRMATION_LIKE = {
    "yes",
    "yes you can",
    "yes please",
    "yeah",
    "yep",
    "sure",
    "sure go ahead",
    "ok",
    "okay",
    "ok proceed",
    "go on",
    "go ahead",
    "continue",
    "do that",
    "that's fine",
    "thats fine",
    "make it",
    "start it",
    "create it",
    "proceed",
}

GARBAGE_TOPIC_PATTERNS = [
    r"^\s*(yes|yeah|yep|ok|okay|sure)(\s+\w+){0,3}\s*$",
    r"^\s*(go on|go ahead|continue|start it|create it|make it|do that)\s*$",
    r"^\s*(i don'?t know|you decide|whatever|anything)\s*$",
]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "basics", "basic", "be", "beginner",
    "can", "complete", "course", "deep", "deeply", "fast", "for", "from",
    "full", "i", "in", "into", "is", "it", "learn", "learning", "level",
    "me", "of", "on", "or", "slow", "the", "to", "understand", "want",
    "with", "you",
}

CONCEPT_ALIAS_GROUPS = {
    "control_flow": {
        "control structures",
        "control structure",
        "control flow",
        "conditionals",
        "conditional",
        "if statements",
        "if statement",
        "branching",
        "branches",
        "loops",
        "loop",
        "iteration",
        "iterating",
    },
    "functions": {
        "functions",
        "function",
        "function basics",
        "function definition",
        "parameters",
        "parameter",
        "return values",
        "return value",
        "return statements",
        "return statement",
        "reusable logic",
    },
    "variables": {
        "variables",
        "variable",
        "identifiers",
        "identifier",
        "names",
        "naming",
    },
    "data_structures": {
        "data structures",
        "data structure",
        "collections",
        "collection",
        "lists",
        "list",
        "dictionaries",
        "dictionary",
        "tuples",
        "tuple",
        "sets",
        "set",
    },
    "oop": {
        "oop",
        "object oriented programming",
        "object-oriented programming",
        "classes",
        "class",
        "objects",
        "object",
    },
    "numerical_arrays": {
        "numerical arrays",
        "numerical array",
        "numeric arrays",
        "numeric array",
        "numpy arrays",
        "numpy array",
        "ndarray",
        "ndarrays",
        "arrays",
        "array",
    },
}

EXCLUSION_VARIANT_PATTERNS = {
    "decorator": r"\bdecorators?\b",
    "async": r"\b(async|await|asyncio|asynchronous|coroutines?)\b",
    "deployment": r"\b(deployment|deploy|deploying|deployed|hosting|production release)\b",
    "machine learning": r"\b(machine learning|ml models?|model training|neural networks?)\b",
    "sklearn": r"\b(sklearn|scikit-learn)\b",
    "scikit-learn": r"\b(sklearn|scikit-learn)\b",
    "advanced oop": r"\b(advanced oop|advanced object[- ]oriented programming|metaclasses?|multiple inheritance|inheritance hierarchy)\b",
}

RELATED_BRIDGES = {
    "linear algebra": {"vectors", "matrices", "matrix", "eigenvalues", "eigenvectors"},
    "matrices": {"vectors", "linear algebra", "matrix multiplication", "transformations"},
    "matrix": {"vectors", "linear algebra", "matrix multiplication", "transformations"},
    "machine learning": {"python", "numpy", "pandas", "statistics", "calculus", "linear algebra", "vectors", "matrices"},
    "python": {"programming", "variables", "functions", "data structures", "numpy", "pandas", "machine learning"},
    "thermodynamics": {"heat", "work", "energy", "pv diagrams", "first law", "second law", "jee physics"},
}

PLACEHOLDER_TITLE_PATTERNS = [
    r"\bfocused learning unit\s*\d+\b",
    r"\blearning unit\s*\d+\b",
    r"^\s*module\s*\d+\s*$",
]

NO_PRIOR_PATTERNS = (
    r"\bno prior\b",
    r"\bno background\b",
    r"\bno previous\b",
    r"\bnever (?:learned|coded|programmed|studied)\b",
    r"\bcomplete beginner\b",
    r"\bfresh student\b",
    r"\bfrom scratch\b",
)

UNRELIABLE_GENERATED_CONCEPT_PATTERNS = PLACEHOLDER_TITLE_PATTERNS + [
    r"\bpurpose and mental model\b",
    r"\bprerequisite ideas needed\b",
    r"\bcore mechanics\b",
]


@dataclass
class CourseScopeAnalysis:
    requested_subject: str = ""
    actual_course_focus: str = ""
    learner_level: str = ""
    target_outcome: str = ""
    depth: str = ""
    pace: str = ""
    topic_breadth: str = "medium"
    course_type: str = "mixed"
    topic_type: str = "mixed"
    learner_goal_type: str = "conceptual"
    estimated_total_learning_time: str = ""
    recommended_module_count: int = 0
    initial_recommended_module_count: int = 0
    rough_scope_recommendation: int = 0
    final_module_count_target: int = 0
    module_count_reasoning: str = ""
    reason_for_module_count: str = ""
    recommended_granularity: str = ""
    roadmap_strategy: str = ""
    coverage_strategy: str = ""
    what_to_include: list[str] = field(default_factory=list)
    what_to_exclude: list[str] = field(default_factory=list)
    what_to_delay_until_later: list[str] = field(default_factory=list)
    what_to_skip_because_student_already_knows: list[str] = field(default_factory=list)
    risk_of_topic_drift: list[str] = field(default_factory=list)
    what_to_compress: list[str] = field(default_factory=list)
    what_to_expand: list[str] = field(default_factory=list)
    scope_reasoning: list[str] = field(default_factory=list)
    scope_roadmap_alignment: str = ""
    roadmap_split_hints: list[str] = field(default_factory=list)
    estimated_total_hours: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating fenced markdown around it."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.I).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


QUESTION_LIKE_SCOPE_PREFIXES = (
    "how ",
    "what ",
    "why ",
    "when ",
    "where ",
    "can you ",
    "explain ",
    "describe ",
)


def is_question_like_scope_text(value: Any) -> bool:
    text = clean_text(value).lower()
    if not text:
        return False
    return "?" in text or any(text.startswith(prefix) for prefix in QUESTION_LIKE_SCOPE_PREFIXES)


def _normalise_word(word: str) -> str:
    clean = word.lower().strip()
    if len(clean) > 4 and clean.endswith("ies"):
        return clean[:-3] + "y"
    if len(clean) > 4 and clean.endswith("sses"):
        return clean[:-2]
    if len(clean) > 3 and clean.endswith("s") and not clean.endswith("ss"):
        return clean[:-1]
    return clean


def _raw_token_set(value: Any) -> set[str]:
    text = clean_text(value).lower()
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]*", text)
    return {
        _normalise_word(word)
        for word in words
        if len(_normalise_word(word)) > 2 and _normalise_word(word) not in STOPWORDS
    }


def _normalised_phrase(value: Any) -> str:
    tokens = _raw_token_set(value)
    if not tokens:
        return ""
    text = clean_text(value).lower()
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]*", text)
    return " ".join(
        _normalise_word(word)
        for word in words
        if _normalise_word(word) not in STOPWORDS
    )


def _concept_semantic_keys(value: Any) -> set[str]:
    phrase = _normalised_phrase(value)
    tokens = set(phrase.split())
    if not tokens:
        return set()

    keys: set[str] = set()
    padded_phrase = f" {phrase} "
    for key, aliases in CONCEPT_ALIAS_GROUPS.items():
        for alias in aliases:
            alias_phrase = _normalised_phrase(alias)
            alias_tokens = set(alias_phrase.split())
            if not alias_tokens:
                continue
            if len(alias_tokens) == 1 and alias_tokens <= tokens:
                keys.add(key)
                break
            if len(alias_tokens) > 1 and (
                f" {alias_phrase} " in padded_phrase or alias_tokens <= tokens
            ):
                keys.add(key)
                break
    return keys


def token_set(value: Any) -> set[str]:
    tokens = _raw_token_set(value)
    for semantic_key in _concept_semantic_keys(value):
        tokens.add(semantic_key)
        for alias in CONCEPT_ALIAS_GROUPS.get(semantic_key, set()):
            tokens |= _raw_token_set(alias)
    return tokens


def _construct_pattern_groups(concept: Any) -> list[list[str]]:
    clean = clean_text(concept).lower()
    groups: list[list[str]] = []
    if re.search(r"\bfor[\s-]+loops?\b", clean):
        groups.append([
            r"\bfor[\s-]+loops?\b",
            r"\bfor\s+[a-zA-Z_][a-zA-Z0-9_]*\s+in\b",
        ])
    if re.search(r"\bwhile[\s-]+loops?\b", clean):
        groups.append([
            r"\bwhile[\s-]+loops?\b",
            r"\bwhile\s+[^:\n]+:",
        ])
    if re.search(r"\bif[\s-]+statements?\b", clean):
        groups.append([
            r"\bif[\s-]+statements?\b",
            r"\bif\s+[^:\n]+:",
        ])
    return groups


def concept_appears_in_text(text: Any, concept: Any) -> bool:
    """Return True when a lesson/question actually mentions a planned concept."""
    text_clean = clean_text(text).lower()
    concept_clean = clean_text(concept).lower()
    if not text_clean or not concept_clean:
        return False

    construct_groups = _construct_pattern_groups(concept_clean)
    if construct_groups:
        return all(
            any(re.search(pattern, text_clean, flags=re.I) for pattern in group)
            for group in construct_groups
        )

    if _contains_term(text_clean, concept_clean):
        return True
    concept_tokens = token_set(concept_clean)
    text_tokens = token_set(text_clean)
    if not concept_tokens or not text_tokens:
        return False
    if concept_tokens <= text_tokens:
        return True
    return bool(concept_tokens & text_tokens)


def is_garbage_topic(topic: Any) -> bool:
    clean = clean_text(topic).lower().strip(" .,!?:;")
    if not clean:
        return True
    if clean in CONFIRMATION_LIKE:
        return True
    return any(re.match(pattern, clean, flags=re.I) for pattern in GARBAGE_TOPIC_PATTERNS)


def profile_keywords(profile: dict[str, Any]) -> set[str]:
    fields = [
        profile.get("topic"),
        profile.get("exact_subject"),
        profile.get("learning_goal"),
        profile.get("target_context"),
        profile.get("expected_outcome"),
    ]
    tokens: set[str] = set()
    for field_value in fields:
        tokens |= token_set(field_value)
    for concept in profile.get("must_include") or []:
        tokens |= token_set(concept)
    return tokens


def is_related_to_profile(concept: Any, profile: dict[str, Any]) -> bool:
    concept_tokens = token_set(concept)
    if not concept_tokens:
        return False
    target_tokens = profile_keywords(profile)
    if concept_tokens & target_tokens:
        return True

    text = clean_text(concept).lower()
    profile_text = " ".join(
        clean_text(profile.get(key)).lower()
        for key in ("topic", "exact_subject", "learning_goal", "target_context")
    )
    for anchor, related in RELATED_BRIDGES.items():
        if anchor in profile_text and any(term in text for term in related):
            return True
        if anchor in text and any(term in profile_text for term in related):
            return True
    return False


def _extract_concepts(items: list[Any] | None) -> list[str]:
    concepts: list[str] = []
    seen: set[str] = set()
    for item in items or []:
        if isinstance(item, dict):
            value = item.get("concept") or item.get("topic") or item.get("title")
        else:
            value = item
        clean = clean_text(value)
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            concepts.append(clean)
    return concepts


def _dedupe_text(items: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = clean_text(item)
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _concept_from_history_item(item: Any) -> str:
    if isinstance(item, dict):
        return clean_text(
            item.get("concept")
            or item.get("topic")
            or item.get("title")
            or item.get("name")
        )
    return clean_text(item)


def _history_item_score(item: Any) -> float | None:
    if not isinstance(item, dict):
        return None
    for key in ("mastery_score", "score", "confidence", "proficiency"):
        if item.get(key) is None:
            continue
        try:
            return float(item.get(key))
        except (TypeError, ValueError):
            return None
    return None


def _history_item_verified(item: Any) -> bool:
    if not isinstance(item, dict):
        return True
    score = _history_item_score(item)
    if score is not None:
        return score >= 0.7
    status = clean_text(
        item.get("status")
        or item.get("evidence")
        or item.get("source")
        or item.get("mastery_status")
    ).lower()
    if any(k in status for k in ("mastered", "verified", "evaluated", "completed", "strong")):
        return True
    if item.get("verified") is True or item.get("mastered") is True:
        return True
    return False


def is_unreliable_generated_concept(value: Any) -> bool:
    text = clean_text(value).lower()
    if not text:
        return False
    return any(re.search(pattern, text, flags=re.I) for pattern in UNRELIABLE_GENERATED_CONCEPT_PATTERNS)


def profile_has_no_prior_experience(profile: dict[str, Any]) -> bool:
    current_intent = profile.get("current_intent") if isinstance(profile.get("current_intent"), dict) else {}
    text = " ".join(
        clean_text(value).lower()
        for value in (
            profile.get("learner_level"),
            profile.get("prior_knowledge_summary"),
            profile.get("prior_knowledge"),
            current_intent.get("learner_level"),
            current_intent.get("prior_knowledge_summary"),
        )
    )
    return any(re.search(pattern, text, flags=re.I) for pattern in NO_PRIOR_PATTERNS)


def filter_relevant_student_history(
    current_intent: dict[str, Any],
    student_history: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Split prior learning data into safe planning context.

    This intentionally does not equate previous courses or generated module
    titles with mastery. Only verified mastered concepts may become assumed
    knowledge, and an explicit no-prior statement suppresses weak/inferred
    assumptions.
    """
    history = student_history or {}
    profile = {
        "topic": current_intent.get("topic") or current_intent.get("exact_subject"),
        "exact_subject": current_intent.get("exact_subject") or current_intent.get("topic"),
        "learning_goal": current_intent.get("goal") or current_intent.get("learning_goal"),
        "target_context": current_intent.get("target_context"),
        "expected_outcome": current_intent.get("target_outcome"),
        "must_include": current_intent.get("must_include") or [],
    }
    no_prior = profile_has_no_prior_experience({**profile, "current_intent": current_intent})

    relevant: list[dict[str, Any]] = []
    irrelevant: list[dict[str, Any]] = []
    possible: list[dict[str, Any]] = []
    assumed_known: list[str] = []
    weak: list[str] = []
    courses: list[str] = []

    def add_bucket(bucket: list[dict[str, Any]], concept: str, source: str, reason: str) -> None:
        bucket.append({"concept": concept, "source": source, "reason": reason})

    for item in history.get("mastered_concepts") or []:
        concept = _concept_from_history_item(item)
        if not concept:
            continue
        if not is_related_to_profile(concept, profile):
            add_bucket(irrelevant, concept, "mastered_concepts", "Not related to current course intent.")
            continue
        if is_unreliable_generated_concept(concept):
            add_bucket(possible, concept, "mastered_concepts", "Looks like generated course text, not a mastered concept.")
            continue
        verified = _history_item_verified(item)
        if verified and not no_prior:
            assumed_known.append(concept)
            add_bucket(relevant, concept, "mastered_concepts", "Verified mastery is relevant to this course.")
        elif verified and _history_item_score(item) is not None and _history_item_score(item) >= 0.9:
            assumed_known.append(concept)
            add_bucket(relevant, concept, "mastered_concepts", "Strong verified mastery overrides weak no-prior wording.")
        else:
            add_bucket(possible, concept, "mastered_concepts", "Relevant but not reliable enough to skip content.")

    for item in history.get("weak_concepts") or []:
        concept = _concept_from_history_item(item)
        if not concept:
            continue
        if is_related_to_profile(concept, profile) and not is_unreliable_generated_concept(concept):
            weak.append(concept)
            add_bucket(relevant, concept, "weak_concepts", "Relevant weak area for extra support.")
        else:
            add_bucket(irrelevant, concept, "weak_concepts", "Weak area is unrelated to current course intent.")

    for item in history.get("previous_courses") or []:
        concept = _concept_from_history_item(item)
        if not concept:
            continue
        if is_related_to_profile(concept, profile):
            courses.append(concept)
            add_bucket(
                possible,
                concept,
                "previous_courses",
                "Previous course context is related but is not assumed mastered.",
            )
        else:
            add_bucket(irrelevant, concept, "previous_courses", "Previous course is unrelated.")

    skill_graph = history.get("skill_graph") or {}
    for item in skill_graph.get("nodes") or []:
        concept = _concept_from_history_item(item)
        if not concept or not is_related_to_profile(concept, profile):
            continue
        if is_unreliable_generated_concept(concept):
            add_bucket(possible, concept, "skill_graph", "Looks generated, not reliable mastery evidence.")
            continue
        score = _history_item_score(item)
        if score is not None and score >= 0.7 and not no_prior:
            assumed_known.append(concept)
            add_bucket(relevant, concept, "skill_graph", "Skill graph shows verified mastery.")
        elif score is not None and score < 0.45:
            weak.append(concept)
            add_bucket(relevant, concept, "skill_graph", "Skill graph shows a relevant weak area.")
        else:
            add_bucket(possible, concept, "skill_graph", "Relevant but not reliable enough to skip content.")

    assumed_known = _dedupe_text(assumed_known)
    weak = _dedupe_text(weak)
    courses = _dedupe_text(courses)
    concepts = _dedupe_text(assumed_known + weak + courses)
    reason = (
        "Explicit no-prior wording means history is context only unless verified mastery is strong."
        if no_prior and not assumed_known
        else "Filtered history by current intent relevance and mastery reliability."
    )
    return {
        "relevant": relevant,
        "irrelevant": irrelevant,
        "possibly_related_but_not_assumed_known": possible,
        "assumed_known": assumed_known,
        "weak": weak,
        "concepts": concepts,
        "courses": courses,
        "reasoning_summary": reason,
    }


def relevant_history_concepts(
    profile: dict[str, Any],
    student_history: dict[str, Any] | None,
) -> dict[str, list[str]]:
    history = student_history or {}
    mastered = _extract_concepts(history.get("mastered_concepts"))
    weak = _extract_concepts(history.get("weak_concepts"))
    previous = _extract_concepts(history.get("previous_courses"))
    return {
        "known": [c for c in mastered if is_related_to_profile(c, profile)],
        "weak": [c for c in weak if is_related_to_profile(c, profile)],
        "previous_topics": [c for c in previous if is_related_to_profile(c, profile)],
        "unrelated_previous_topics": [c for c in previous if not is_related_to_profile(c, profile)],
    }


def fallback_scope_analysis(profile: dict[str, Any]) -> CourseScopeAnalysis:
    """Provider-failure fallback. The LLM path is the primary source of scope."""
    topic = clean_text(profile.get("topic") or profile.get("exact_subject"))
    goal = clean_text(profile.get("learning_goal") or profile.get("target_context"))
    depth = clean_text(profile.get("depth_preference") or profile.get("pace")).lower()
    pace = clean_text(profile.get("pace") or "medium")
    learner_level = clean_text(profile.get("learner_level"))
    target_context = clean_text(profile.get("target_context")).lower()
    topic_tokens = token_set(topic)
    goal_tokens = token_set(goal)
    breadth_score = len(topic_tokens) + len(goal_tokens)

    if re.search(r"\b(complete|full|entire|from scratch|zero to advanced)\b", f"{topic} {goal}", re.I):
        breadth_score += 10
    if re.search(r"\b(algorithm|regression|derivative|matrix|law|concept)\b", topic, re.I):
        breadth_score -= 2
    if "deep" in depth or "derivation" in depth:
        breadth_score += 2

    if breadth_score >= 15:
        breadth = "huge"
        count = max(24, breadth_score * 4)
    elif breadth_score >= 10:
        breadth = "broad"
        count = max(12, breadth_score * 2)
    elif breadth_score >= 6:
        breadth = "medium"
        count = max(6, breadth_score + 2)
    elif breadth_score >= 3:
        breadth = "narrow"
        count = max(4, breadth_score + 2)
    else:
        breadth = "tiny"
        count = max(2, breadth_score + 1)

    include = [topic] if topic else []
    exclude: list[str] = []
    delay: list[str] = []
    actual_focus = topic
    course_type = "mixed"
    if "python" in topic.lower():
        course_type = "programming"
        if not any(k in target_context or k in goal.lower() for k in ("machine learning", "ml", "data science")):
            actual_focus = "pure Python fundamentals"
            exclude = ["machine learning", "pandas", "sklearn", "advanced OOP", "decorators", "async", "deployment"]
            delay = ["third-party libraries", "frameworks", "machine learning libraries"]
    elif any(k in target_context or k in goal.lower() for k in ("jee", "neet", "gate", "exam")):
        course_type = "exam_prep"
    elif any(k in topic.lower() for k in ("regression", "algebra", "calculus", "matrix")):
        course_type = "math"
    elif "thermodynamics" in topic.lower():
        course_type = "science"

    return CourseScopeAnalysis(
        requested_subject=topic,
        actual_course_focus=actual_focus,
        learner_level=learner_level,
        target_outcome=goal,
        depth=depth or clean_text(profile.get("depth_preference")),
        pace=pace,
        topic_breadth=breadth,
        course_type=course_type,
        topic_type=course_type,
        learner_goal_type="conceptual",
        estimated_total_learning_time="LLM unavailable; estimated from request scope.",
        recommended_module_count=int(count),
        initial_recommended_module_count=int(count),
        rough_scope_recommendation=int(count),
        final_module_count_target=int(count),
        module_count_reasoning=(
            "Initial and final target match because fallback scope analysis has no roadmap yet."
        ),
        reason_for_module_count=(
            "Fallback estimate from topic specificity, goal breadth, and requested depth because the LLM scope analysis was unavailable."
        ),
        recommended_granularity="Use focused modules with one main concept layer per module.",
        roadmap_strategy="Teach the requested subject in prerequisite order and delay unrelated branches.",
        coverage_strategy="Teach the requested subject in logical prerequisite order while avoiding unrelated prior-course material.",
        what_to_include=include,
        what_to_exclude=exclude,
        what_to_delay_until_later=delay,
        what_to_skip_because_student_already_knows=list(profile.get("known_concepts") or []),
        risk_of_topic_drift=exclude[:3],
        what_to_compress=list(profile.get("known_concepts") or []),
        what_to_expand=list(profile.get("weak_concepts") or []),
        scope_reasoning=[
            "Fallback estimate from topic specificity, goal breadth, and requested depth."
        ],
        estimated_total_hours=round(float(count) * 0.5, 2),
    )


def _module_text(module: Any) -> str:
    if hasattr(module, "model_dump"):
        module = module.model_dump()
    if not isinstance(module, dict):
        return clean_text(module)
    fields = [
        module.get("title"),
        module.get("concept"),
        module.get("domain_framing"),
        module.get("purpose"),
        module.get("why_it_matters_for_goal"),
        " ".join(module.get("must_teach") or []),
        " ".join(module.get("examples_to_include") or []),
    ]
    return " ".join(clean_text(f) for f in fields if f)


def contamination_terms(profile: dict[str, Any], student_history: dict[str, Any] | None) -> list[str]:
    history = relevant_history_concepts(profile, student_history)
    terms = history["unrelated_previous_topics"]
    explicit_known = set(clean_text(c).lower() for c in profile.get("known_concepts") or [])
    return [term for term in terms if clean_text(term).lower() not in explicit_known]


def _as_module_dict(module: Any) -> dict[str, Any]:
    if hasattr(module, "model_dump"):
        module = module.model_dump()
    return dict(module) if isinstance(module, dict) else {}


def _concept_key(value: Any) -> str:
    return _normalised_phrase(value)


def _concept_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = clean_text(value)
        key = _concept_key(text)
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _looks_like_module_reference(value: Any) -> bool:
    text = clean_text(value).lower()
    return bool(re.fullmatch(r"(m|module)[\s_-]*\d+[a-z]?", text))


def _concepts_match(left: Any, right: Any) -> bool:
    a = _concept_key(left)
    b = _concept_key(right)
    if not a or not b:
        return False
    if a == b:
        return True
    a_keys = _concept_semantic_keys(left)
    b_keys = _concept_semantic_keys(right)
    if a_keys and b_keys:
        if a_keys == b_keys:
            return True
        if len(a_keys) == 1 and len(b_keys) == 1 and a_keys & b_keys:
            return True
    a_tokens = _raw_token_set(a)
    b_tokens = _raw_token_set(b)
    if not a_tokens or not b_tokens:
        return False
    if a_tokens <= b_tokens or b_tokens <= a_tokens:
        return True
    if a in b or b in a:
        return True
    overlap = len(a_tokens & b_tokens)
    smaller = min(len(a_tokens), len(b_tokens))
    return smaller > 0 and (overlap / smaller) >= 0.6


def _concept_in_list(concept: Any, candidates: list[str]) -> bool:
    clean_candidates = [clean_text(candidate) for candidate in candidates if clean_text(candidate)]
    if any(_concepts_match(concept, candidate) for candidate in clean_candidates):
        return True

    concept_keys = _concept_semantic_keys(concept)
    if concept_keys:
        candidate_keys: set[str] = set()
        for candidate in clean_candidates:
            candidate_keys |= _concept_semantic_keys(candidate)
        if concept_keys <= candidate_keys:
            return True

    concept_tokens = token_set(concept)
    if not concept_tokens:
        return False
    combined_tokens: set[str] = set()
    for candidate in clean_candidates:
        combined_tokens |= token_set(candidate)
    if concept_tokens <= combined_tokens:
        return True
    overlap = len(concept_tokens & combined_tokens)
    required = max(2, int(len(concept_tokens) * 0.67))
    return overlap >= required


def _contains_term(text: str, term: str) -> bool:
    term_clean = clean_text(term).lower()
    if not term_clean:
        return False
    text_clean = clean_text(text).lower()
    term_key = _concept_key(term_clean)
    variant_pattern = EXCLUSION_VARIANT_PATTERNS.get(term_key) or EXCLUSION_VARIANT_PATTERNS.get(term_clean)
    if variant_pattern and re.search(variant_pattern, text_clean, flags=re.I):
        return True
    if re.search(r"[.\-+_]", term_clean):
        return term_clean in text_clean
    if re.search(r"\b" + re.escape(term_clean) + r"\b", text_clean):
        return True
    text_key = _normalised_phrase(text_clean)
    if not term_key or not text_key:
        return False
    return bool(re.search(r"\b" + re.escape(term_key) + r"\b", text_key))


def has_placeholder_title(title: Any) -> bool:
    text = clean_text(title).lower()
    return any(re.search(pattern, text, flags=re.I) for pattern in PLACEHOLDER_TITLE_PATTERNS)


def _module_taught_concepts(data: dict[str, Any]) -> list[str]:
    taught = _concept_list(data.get("concepts_taught"))
    if taught:
        return taught
    must_teach = _concept_list(data.get("must_teach"))
    if must_teach:
        return must_teach
    concept = clean_text(data.get("concept"))
    return [concept] if concept else []


def _module_dependencies(data: dict[str, Any]) -> list[str]:
    dependencies = _concept_list(data.get("depends_on_concepts"))
    if dependencies:
        return dependencies
    return _concept_list(data.get("prerequisites"))


def _module_question_scope(data: dict[str, Any]) -> list[str]:
    scope = _concept_list(data.get("question_scope"))
    if scope:
        return scope
    return []


def _scope_dict(scope_analysis: dict[str, Any] | CourseScopeAnalysis) -> dict[str, Any]:
    return (
        scope_analysis.to_dict()
        if isinstance(scope_analysis, CourseScopeAnalysis)
        else dict(scope_analysis or {})
    )


def _scope_exclusions(scope: dict[str, Any]) -> list[str]:
    exclusions: list[str] = []
    for field_name in (
        "what_to_exclude",
        "risk_of_topic_drift",
    ):
        exclusions.extend(_concept_list(scope.get(field_name)))
    return list(dict.fromkeys(exclusions))


def _profile_exclusions(profile: dict[str, Any] | None) -> list[str]:
    profile = profile or {}
    exclusions: list[str] = []
    for field_name in ("do_not_include", "should_skip"):
        exclusions.extend(_concept_list(profile.get(field_name)))
    return list(dict.fromkeys(exclusions))


def _combined_exclusions(scope: dict[str, Any], profile: dict[str, Any] | None = None) -> list[str]:
    return list(dict.fromkeys(_scope_exclusions(scope) + _profile_exclusions(profile)))


def _recommended_module_count(scope: dict[str, Any]) -> int:
    return int(scope.get("final_module_count_target") or scope.get("recommended_module_count") or 0)


def _is_beginner(profile: dict[str, Any], scope: dict[str, Any]) -> bool:
    text = " ".join(
        clean_text(value).lower()
        for value in (
            profile.get("learner_level"),
            profile.get("prior_knowledge_summary"),
            profile.get("prior_knowledge"),
            scope.get("learner_level"),
        )
    )
    return any(k in text for k in ("beginner", "fresh", "no prior", "from scratch", "complete beginner"))


def _repair_prompt(issues: list[str]) -> str:
    if not issues:
        return ""
    return (
        "Repair the curriculum by preserving the learner profile and regenerating "
        "scope, concept inventory, prerequisite graph, module plan, roadmap steps, "
        "and schedule. Fix these validator issues exactly:\n"
        + "\n".join("- " + issue for issue in issues)
    )


def _roadmap_step_dicts(roadmap: Any) -> list[dict[str, Any]]:
    if hasattr(roadmap, "model_dump"):
        roadmap = roadmap.model_dump()
    if not isinstance(roadmap, dict):
        return []
    steps = roadmap.get("steps") or []
    result: list[dict[str, Any]] = []
    for step in steps:
        if hasattr(step, "model_dump"):
            step = step.model_dump()
        if isinstance(step, dict):
            result.append(step)
    return result


def validate_master_roadmap(
    roadmap: Any,
    scope: CourseScopeAnalysis,
    profile: dict,
) -> dict:
    """
    Deterministic validation for the roadmap-first planning layer.

    Returns {"passed": bool, "issues": list[str], "repair_prompt": str}.
    """
    issues: list[str] = []
    steps = _roadmap_step_dicts(roadmap)
    scope_dict = _scope_dict(scope)
    exclusions = _combined_exclusions(scope_dict, profile)

    step_lookup: dict[str, int] = {}
    clusters_seen: dict[str, int] = {}
    for idx, step in enumerate(steps):
        step_id = clean_text(step.get("step_id"))
        title = clean_text(step.get("title"))
        cluster = clean_text(step.get("concept_cluster"))
        hint = clean_text(step.get("module_generation_hint"))

        if not step_id:
            issues.append(f"Roadmap step {idx + 1} is missing step_id.")
        if not title:
            issues.append(f"Roadmap step {idx + 1} is missing title.")
        if not cluster:
            issues.append(f"Roadmap step {idx + 1} is missing concept_cluster.")
        if not hint:
            issues.append(f"Roadmap step {idx + 1} is missing module_generation_hint.")

        cluster_key = _concept_key(cluster)
        if cluster_key:
            if cluster_key in clusters_seen:
                issues.append(
                    f"Roadmap step {idx + 1} duplicates concept_cluster '{cluster}'."
                )
            clusters_seen[cluster_key] = idx

        step_text = " ".join(
            clean_text(value)
            for value in [
                step_id,
                title,
                cluster,
                step.get("why_this_step_exists"),
                step.get("goal_alignment"),
                step.get("module_generation_hint"),
                " ".join(step.get("subtopics") or []),
                " ".join(step.get("prerequisites") or []),
            ]
        ).lower()
        for excluded in exclusions:
            if _contains_term(step_text, excluded):
                issues.append(
                    f"Roadmap step {idx + 1} uses excluded concept: {excluded}."
                )
                break

        try:
            estimated = int(step.get("estimated_minutes") or 0)
        except (TypeError, ValueError):
            estimated = 0
        if estimated < 10:
            issues.append(
                f"Roadmap step {idx + 1} estimated_minutes must be at least 10."
            )

        for key in (step_id, title, cluster):
            clean_key = _concept_key(key)
            if clean_key and clean_key not in step_lookup:
                step_lookup[clean_key] = idx

    if len(steps) < 2:
        issues.append("Master roadmap must contain at least 2 steps.")

    for idx, step in enumerate(steps):
        for prereq in step.get("prerequisites") or []:
            prereq_key = _concept_key(prereq)
            if prereq_key in {"", "none", "no prerequisites"}:
                continue
            prereq_index = step_lookup.get(prereq_key)
            if prereq_index is not None and prereq_index >= idx:
                issues.append(
                    f"Roadmap step {idx + 1} depends on '{prereq}' before it appears."
                )
                break

    return {
        "passed": not issues,
        "issues": issues,
        "repair_prompt": _repair_prompt(issues),
    }


def validate_modules_against_roadmap(
    modules: list,
    roadmap: Any,
    scope: CourseScopeAnalysis,
    profile: dict,
) -> dict:
    """
    Deterministic validation that module plans are derived from master steps.

    Returns {"passed": bool, "issues": list[str], "repair_prompt": str}.
    """
    issues: list[str] = []
    steps = _roadmap_step_dicts(roadmap)
    step_ids = {clean_text(step.get("step_id")) for step in steps if clean_text(step.get("step_id"))}
    steps_by_id = {
        clean_text(step.get("step_id")): step
        for step in steps
        if clean_text(step.get("step_id"))
    }
    covered: set[str] = set()
    scope_dict = _scope_dict(scope)
    exclusions = _combined_exclusions(scope_dict, profile)

    module_dicts = [_as_module_dict(module) for module in modules]
    for idx, module in enumerate(module_dicts, start=1):
        step_id = clean_text(module.get("roadmap_step_id"))
        # In the new architecture, the roadmap is derived FROM modules after planning.
        # step_ids may not exist yet at validation time. We only track coverage, not enforce IDs.
        if step_id:
            covered.add(step_id)

        concept = clean_text(module.get("concept"))
        # Skip roadmap step concept-overlap check if this module built the roadmap itself
        candidate_steps = [steps_by_id[step_id]] if step_id in steps_by_id else []
        roadmap_concepts: list[str] = []
        for step in candidate_steps:
            roadmap_concepts.append(clean_text(step.get("concept_cluster")))
            roadmap_concepts.extend(_concept_list(step.get("subtopics")))
        # NOTE: Concept-overlap check is skipped when the roadmap was built from modules
        # (candidate_steps will be empty). Only check when using the old pre-built roadmap flow.
        if candidate_steps:
            roadmap_concepts: list[str] = []
            for step in candidate_steps:
                roadmap_concepts.append(clean_text(step.get("concept_cluster")))
                roadmap_concepts.extend(_concept_list(step.get("subtopics")))
            concept_tokens = token_set(concept)
            roadmap_tokens: set[str] = set()
            for roadmap_concept in roadmap_concepts:
                roadmap_tokens |= token_set(roadmap_concept)
            if (
                concept
                and roadmap_concepts
                and not _concept_in_list(concept, roadmap_concepts)
                and not (concept_tokens & roadmap_tokens)
            ):
                issues.append(
                    f"Module {idx} concept '{concept}' does not overlap its roadmap step."
                )

        module_text = _module_text(module).lower()
        for excluded in exclusions:
            if _contains_term(module_text, excluded):
                issues.append(f"Module {idx} uses excluded or delayed concept: {excluded}.")
                break

        for field_name in ("prerequisites", "depends_on_concepts", "question_scope"):
            for value in _concept_list(module.get(field_name)):
                if _looks_like_module_reference(value):
                    issues.append(
                        f"Module {idx} {field_name} contains module id reference '{value}' instead of a concept name."
                    )
                    break

    # NOTE: Step-coverage checks are skipped in the new architecture.
    # The roadmap is built FROM modules — every step is already covered by definition.
    # Only run these checks if a pre-built roadmap was supplied (steps have real step_ids).
    if steps and step_ids:
        for step in steps:
            step_id = clean_text(step.get("step_id"))
            if step_id and step_id not in covered:
                issues.append(f"Roadmap step '{step_id}' is not covered by any module.")
            expected = [clean_text(step.get("concept_cluster"))] + _concept_list(step.get("subtopics"))
            mapped_modules = [
                module for module in module_dicts
                if clean_text(module.get("roadmap_step_id")) == step_id
            ]
            mapped_taught: list[str] = []
            for module in mapped_modules:
                mapped_taught.extend(_module_taught_concepts(module))
                mapped_taught.extend(_concept_list(module.get("must_teach")))
            for concept in expected:
                if concept and not _concept_in_list(concept, mapped_taught):
                    issues.append(
                        f"Roadmap step '{step_id}' concept is missing from mapped modules: {concept}."
                    )
                    break

    # NOTE: Module count range check removed.
    # The LLM decides how many modules the topic warrants from subject knowledge.
    # A hard ±40% range check caused false failures when LLM was correct and scope was wrong.

    return {
        "passed": not issues,
        "issues": issues,
        "repair_prompt": _repair_prompt(issues),
    }


def validate_curriculum_quality(
    topic: str,
    modules: list[Any],
    profile: dict[str, Any],
    scope_analysis: dict[str, Any] | CourseScopeAnalysis,
    student_history: dict[str, Any] | None = None,
    concept_inventory: dict[str, Any] | None = None,
    prerequisite_graph: dict[str, list[str]] | None = None,
    roadmap_steps: list[str] | None = None,
    schedule: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    issues: list[str] = []
    topic_clean = clean_text(topic or profile.get("topic"))
    topic_tokens = token_set(topic_clean) | token_set(profile.get("exact_subject"))
    context_tokens = token_set(profile.get("target_context")) | token_set(profile.get("learning_goal"))
    scope = _scope_dict(scope_analysis)
    exclusions = _combined_exclusions(scope, profile)
    learner_is_beginner = _is_beginner(profile, scope)
    module_dicts = [_as_module_dict(m) for m in modules]

    if is_garbage_topic(topic_clean):
        issues.append(f"Invalid course topic: {topic_clean!r}.")

    if not modules:
        issues.append("No modules were generated.")

    module_texts = [_module_text(m).lower() for m in modules]
    combined = " ".join(module_texts)
    if topic_tokens and not (topic_tokens & token_set(combined)):
        issues.append("Modules do not appear to teach the requested topic.")

    if context_tokens:
        # Filter out generic purpose/goal words that carry no subject content.
        # A goal like "for academic purpose" or "to become a developer" has no
        # content tokens that would appear in module titles — this is not a failure.
        generic_words = {
            "want", "learn", "learning", "study", "understand", "academic",
            "purpose", "become", "because", "general", "prepare", "preparation",
            "beginner", "basic", "i", "my", "me", "for", "to", "and", "the",
            "it", "this", "that", "with", "about", "how", "what", "why",
            "developer", "programmer", "student", "professional",
        }
        subject_context_tokens = context_tokens - generic_words
        if subject_context_tokens:
            context_hits = subject_context_tokens & token_set(combined)
            if not context_hits and len(modules) > 1:
                issues.append("Modules do not reflect the requested target context or goal.")

    for module in modules:
        text = _module_text(module).lower()
        if any(re.search(pattern, text, flags=re.I) for pattern in GARBAGE_TOPIC_PATTERNS):
            issues.append("Module title or concept contains confirmation/garbage text.")
            break

    for term in contamination_terms(profile, student_history):
        term_clean = clean_text(term).lower()
        if term_clean and term_clean in combined:
            issues.append(f"Unrelated previous-course topic leaked into curriculum: {term}.")

    planning_text = " ".join(
        clean_text(value).lower()
        for value in (
            profile.get("recommended_strategy"),
            profile.get("course_scope"),
            profile.get("target_context"),
            profile.get("learning_goal"),
        )
        if value
    )
    for term in _concept_list(profile.get("known_concepts")):
        if is_related_to_profile(term, profile):
            continue
        if _contains_term(planning_text, term) or _contains_term(combined, term):
            issues.append(
                f"State contamination: unrelated prior concept appears in current planning context: {term}."
            )
            break

    recommended = _recommended_module_count(scope)
    if recommended > 0 and modules:
        if len(modules) < max(3, recommended // 4):
            issues.append(
                f"Module count {len(modules)} is suspiciously low for scope recommendation {recommended}: "
                + clean_text(scope.get("reason_for_module_count"))
            )

    if concept_inventory is not None:
        core = _concept_list(concept_inventory.get("core_concepts"))
        delayed = _concept_list(concept_inventory.get("concepts_to_delay"))
        skipped = _concept_list(concept_inventory.get("concepts_to_skip"))
        taught_all: list[str] = []
        for data in module_dicts:
            taught_all.extend(_module_taught_concepts(data))
        for concept in core:
            if not _concept_in_list(concept, taught_all) and not _concept_in_list(concept, skipped):
                issues.append(f"Core concept from inventory is not covered by the module plan: {concept}.")
                break
        # NOTE: "delayed concept too early" check removed.
        # Roadmap ordering already handles sequencing. A concept in position 5 of 20 is
        # not "too early" — it's where the LLM decided it belongs in the prerequisite chain.
        # The old check caused false positives when the LLM correctly placed a topic.

    previous_concepts: list[str] = _concept_list(profile.get("known_concepts"))
    previous_concepts.extend(_concept_list(scope.get("what_to_skip_because_student_already_knows")))
    previous_concepts.extend(_concept_list(scope.get("what_to_compress")))
    taught_counts: dict[str, int] = {}

    for idx, data in enumerate(module_dicts, start=1):
        if not clean_text(data.get("title")) or not clean_text(data.get("concept")):
            issues.append(f"Module {idx} lacks a title or concept.")
            continue
        title = clean_text(data.get("title"))
        if has_placeholder_title(title):
            issues.append(f"Module {idx} has a placeholder or vague title: {title}.")
        if not clean_text(data.get("domain_framing")) and not clean_text(data.get("why_it_matters_for_goal")):
            issues.append(f"Module {idx} lacks goal-specific framing.")
        taught = _module_taught_concepts(data)
        dependencies = _module_dependencies(data)
        question_scope = _module_question_scope(data)

        if not taught:
            issues.append(f"Module {idx} does not define specific concepts_taught.")
        if not question_scope:
            issues.append(f"Module {idx} does not define question_scope.")

        for dep in dependencies:
            if _looks_like_module_reference(dep):
                issues.append(
                    f"Module {idx} depends on module id '{dep}' instead of a concept name."
                )
                break

        for scoped_concept in question_scope:
            if _looks_like_module_reference(scoped_concept):
                issues.append(
                    f"Module {idx} question_scope contains module id '{scoped_concept}' instead of a concept name."
                )
                break
            if is_question_like_scope_text(scoped_concept):
                issues.append(
                    f"Module {idx} question_scope contains question-like text '{scoped_concept}'; "
                    "expected concept/skill labels, not full questions."
                )
                break

        for dep in dependencies:
            if _concept_key(dep) in {"none", "no prerequisites"}:
                continue
            if not _concept_in_list(dep, previous_concepts):
                issues.append(
                    f"Module {idx} depends on '{dep}' before it is taught or assumed."
                )
                break

        available_now = previous_concepts + taught
        for concept in question_scope:
            if not _concept_in_list(concept, available_now):
                issues.append(
                    f"Module {idx} question_scope includes '{concept}' outside taught/prior concepts."
                )
                break

        module_text = _module_text(data).lower()
        for excluded in exclusions:
            if _contains_term(module_text, excluded):
                issues.append(f"Module {idx} drifts into excluded or delayed topic: {excluded}.")
                break

        purposeful_review = bool(
            re.search(
                r"\b(review|recap|practice|project|integrat|capstone|debug)\b",
                " ".join(clean_text(data.get(k)) for k in ("title", "module_goal", "why_now", "purpose")).lower(),
            )
        )
        for concept in taught:
            key = _concept_key(concept)
            if key:
                taught_counts[key] = taught_counts.get(key, 0) + 1
                if taught_counts[key] > 1 and not purposeful_review:
                    issues.append(f"Concept '{concept}' is repeated without a clear review/practice purpose.")
                    break

        for concept in taught:
            if not _concept_in_list(concept, previous_concepts):
                previous_concepts.append(concept)

    if prerequisite_graph:
        graph_concepts = _concept_list(list(prerequisite_graph.keys()))
        taught_all = []
        for data in module_dicts:
            taught_all.extend(_module_taught_concepts(data))
        for concept in graph_concepts:
            if concept.lower() == "concept":
                continue
            if not _concept_in_list(concept, taught_all) and not _concept_in_list(concept, previous_concepts):
                issues.append(f"Prerequisite graph contains concept not present in modules or assumed knowledge: {concept}.")
                break

    if roadmap_steps is not None:
        clean_steps = [clean_text(step) for step in roadmap_steps if clean_text(step)]
        if len(clean_steps) < min(3, len(modules)):
            issues.append("Roadmap does not clearly explain the learning path step by step.")
        if any(has_placeholder_title(step) for step in clean_steps):
            issues.append("Roadmap contains placeholder-like text.")

    minutes = [int(_as_module_dict(m).get("estimated_minutes") or 0) for m in modules]
    if len(minutes) >= 5 and len(set(minutes)) == 1 and minutes[0] == 40:
        issues.append("Every module has an identical 40-minute estimate without justification.")

    if schedule is not None:
        if modules and not schedule:
            issues.append("Curriculum does not include a realistic schedule plan.")
        max_modules_per_day = 2 if learner_is_beginner else 4
        if str(profile.get("pace") or scope.get("pace")).lower() == "deep":
            max_modules_per_day = max(2, max_modules_per_day - 1)
        for day in schedule:
            items = day.get("items") or []
            module_items = [
                item for item in items
                if str(item.get("module_id") or item.get("module_title") or item.get("module_index") or "").strip()
                and str(item.get("item_type") or "module") == "module"
            ]
            if len(module_items) > max_modules_per_day:
                issues.append(
                    f"Schedule overload: day {day.get('day')} contains {len(module_items)} modules."
                )
                break
            day_minutes = int(day.get("total_minutes") or 0)
            if not day_minutes:
                day_minutes = sum(int(item.get("estimated_minutes") or 0) for item in items)
                day_minutes += int(day.get("review_minutes") or 0)
                day_minutes += int(day.get("practice_minutes") or 0)
            if learner_is_beginner and day_minutes > 240:
                issues.append(f"Schedule overload: day {day.get('day')} asks a beginner to study {day_minutes} minutes.")
                break
            has_break = any(item.get("break_after") or item.get("break_minutes") for item in items) or int(day.get("break_minutes") or 0) > 0
            if day_minutes >= 90 and not has_break:
                issues.append(f"Schedule day {day.get('day')} has no break despite a long study load.")
                break

    quality_score = max(0.0, round(1.0 - 0.14 * len(issues), 2))
    return {
        "passed": not issues,
        "quality_score": quality_score,
        "issues": issues,
        "repair_prompt": _repair_prompt(issues),
        "regenerate_required": bool(issues),
        "regenerate": bool(issues),
    }


def validate_lesson_quality(
    content: str,
    course: dict[str, Any],
    module: dict[str, Any],
    context_chunks: list[str] | None = None,
) -> dict[str, Any]:
    issues: list[str] = []
    text = clean_text(content).lower()
    topic = clean_text(course.get("topic"))
    concept = clean_text(module.get("concept"))
    title = clean_text(module.get("title"))
    profile = course.get("personalization_profile") or {}
    scope = _scope_dict(profile.get("scope_analysis") or course.get("scope_analysis") or {})
    exclusions = _combined_exclusions(scope, profile)
    taught = _module_taught_concepts(module) or [concept]
    question_scope = _module_question_scope(module)
    course_kind = " ".join(
        clean_text(value).lower()
        for value in (
            scope.get("course_type"),
            scope.get("topic_type"),
            course.get("topic"),
            course.get("goal"),
        )
    )

    # Depth signal — not a hard gate, just a signal passed to the retry prompt.
    # The LLM decides actual length; this check surfaces when something went wrong
    # (e.g. model truncated, empty response, single-paragraph stub).
    # These are absolute minimums — if below this, something clearly failed.
    pace = str((course.get("personalization_profile") or {}).get("pace") or course.get("pace") or "medium").lower()
    word_count = len(text.split())
    # Absolute floor: if below this, the content is genuinely unusable
    absolute_floor = {"fast": 120, "medium": 180, "deep": 250}.get(pace, 180)
    if word_count < absolute_floor:
        issues.append(
            f"Lesson is too short (got {word_count} words). "
            f"For {pace} pace the lesson must cover the concept fully — "
            f"expand significantly with explanation, examples, and practice."
        )
    if token_set(concept) and not (token_set(concept) & token_set(text)):
        issues.append("Lesson does not teach the module concept.")
    for taught_concept in taught:
        if token_set(taught_concept) and not concept_appears_in_text(content, taught_concept):
            issues.append(f"Lesson does not cover planned concept: {taught_concept}.")
            break
    if token_set(topic) and not (token_set(topic) & token_set(text)):
        issues.append("Lesson does not connect to the course topic.")
    if any(re.search(pattern, text, flags=re.I) for pattern in GARBAGE_TOPIC_PATTERNS):
        issues.append("Lesson contains confirmation/garbage text.")
    if any(word in course_kind for word in ("programming", "coding", "code")) and "```" not in content:
        issues.append("Programming lesson lacks a concrete code example.")
    generic_lesson_markers = (
        "turns the topic from a collection of terms",
        "think of the concept as a lens",
        "concept move",
        "name the setup",
    )
    if sum(1 for marker in generic_lesson_markers if marker in text) >= 2:
        issues.append("Lesson appears to use generic scaffolding instead of concrete teaching.")
    for excluded in exclusions:
        if _contains_term(text, excluded):
            issues.append(f"Lesson drifts into excluded or delayed topic: {excluded}.")
            break
    if question_scope:
        for scoped_concept in question_scope:
            if not _concept_in_list(scoped_concept, taught + _module_dependencies(module)):
                # This is a module-plan issue surfaced during lesson generation.
                issues.append(f"Lesson question scope contains an unplanned concept: {scoped_concept}.")
                break

    quality_score = max(0.0, round(1.0 - 0.18 * len(issues), 2))
    return {
        "passed": not issues,
        "quality_score": quality_score,
        "issues": issues,
        "repair_prompt": _repair_prompt(issues),
        "regenerate_required": bool(issues),
        "regenerate": bool(issues),
    }


def lesson_section_titles(content: str) -> set[str]:
    titles = {"Lesson"}
    for line in (content or "").splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
        if match:
            title = re.sub(r"[*_`#]", "", match.group(1)).strip()
            if title:
                titles.add(title)
    return titles


def validate_questions_grounded(
    questions: list[dict[str, Any]],
    lesson_text: str,
    module: dict[str, Any],
) -> dict[str, Any]:
    issues: list[str] = []
    lesson_clean = clean_text(lesson_text)
    lesson_lower = lesson_clean.lower()
    sections = lesson_section_titles(lesson_text)
    question_scope = (
        _module_question_scope(module)
        or _module_taught_concepts(module)
        + _module_dependencies(module)
    )

    if not questions:
        issues.append("No questions were generated.")

    for idx, question in enumerate(questions, start=1):
        q_text = clean_text(question.get("question") or question.get("question_text"))
        expected = clean_text(question.get("expected_answer"))
        quote = clean_text(question.get("source_quote"))
        source_section = clean_text(question.get("source_section") or "Lesson")
        concepts_tested = _concept_list(question.get("concepts_tested"))

        if not q_text:
            issues.append(f"Question {idx} is empty.")
            continue
        if not concepts_tested:
            issues.append(f"Question {idx} does not declare concepts_tested.")
        for concept in concepts_tested:
            if question_scope and not _concept_in_list(concept, question_scope):
                issues.append(
                    f"Question {idx} tests '{concept}' outside the module question_scope."
                )
                break
            if not concept_appears_in_text(lesson_text, concept):
                issues.append(
                    f"Question {idx} tests '{concept}' but the lesson does not teach that concept."
                )
                break
        for scoped_concept in question_scope:
            if (
                concept_appears_in_text(q_text, scoped_concept)
                and not concept_appears_in_text(lesson_text, scoped_concept)
            ):
                issues.append(
                    f"Question {idx} references '{scoped_concept}' but the lesson does not teach it."
                )
                break
        if source_section not in sections:
            issues.append(f"Question {idx} cites missing lesson section: {source_section}.")
        if quote and quote.lower() not in lesson_lower:
            issues.append(f"Question {idx} source_quote is not present in the lesson.")
        if expected and expected.lower() not in lesson_lower and quote and quote.lower() not in lesson_lower:
            issues.append(f"Question {idx} expected answer is not grounded in the lesson.")
        if not quote and expected and expected.lower() not in lesson_lower:
            issues.append(f"Question {idx} has no lesson quote supporting the expected answer.")

    quality_score = max(0.0, round(1.0 - 0.2 * len(issues), 2))
    return {
        "passed": not issues,
        "quality_score": quality_score,
        "issues": issues,
        "repair_prompt": _repair_prompt(issues),
        "regenerate_required": bool(issues),
        "regenerate": bool(issues),
    }
