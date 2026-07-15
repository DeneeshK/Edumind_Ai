"""
core/llm_schemas.py
Pydantic schemas for the LLM JSON the LIVE flow consumes.

Before this module, model output was parsed with ``parse_json_object`` into a
plain dict and read with ``.get()`` defaults, so a malformed or adversarial
diagnosis silently degraded to ``mastery_signal="uncertain"`` with no signal that
validation had failed. These schemas make that failure OBSERVABLE: ``parse_llm_json``
validates the parsed object, and on failure it logs a WARNING, increments the
``edumind_llm_schema_failures_total`` counter, and returns ``None`` so the call site
falls back to its existing safe default — now a visible fallback instead of a silent one.

Curriculum shapes are deliberately NOT modelled here: core/curriculum_quality.py
already validates roadmap / coverage / sequencing output. This module covers only
the evaluation and question-generation shapes those validators do not touch.
"""

from __future__ import annotations

from typing import Any, Literal

from loguru import logger
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from core.metrics import metrics as _metrics


# ── Reusable coercers ─────────────────────────────────────────────────────────
def _clamp_unit(value: Any) -> float:
    """Coerce to float and clamp into [0.0, 1.0]; raises for non-numeric input."""
    number = float(value)  # non-numeric -> ValueError -> ValidationError
    return max(0.0, min(1.0, number))


def _as_str_list(value: Any) -> list[str]:
    """Normalize None / scalar / list into a list of non-empty trimmed strings."""
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [s for s in (str(v).strip() for v in value) if s]


def _as_text(value: Any) -> str:
    """Normalize None / list / scalar into a single string (lists are joined)."""
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(v).strip() for v in value if str(v).strip())
    return str(value)


# ── Evaluation: answer diagnosis ──────────────────────────────────────────────
class AnswerDiagnosis(BaseModel):
    """Validated diagnosis of one student answer (evaluation_agent.diagnose)."""

    model_config = ConfigDict(extra="ignore")

    # Required and constrained: a missing / out-of-vocab signal must fail
    # validation so the caller falls back observably instead of trusting garbage.
    mastery_signal: Literal["clear", "uncertain", "weak"]
    correct_concepts: list[str] = Field(default_factory=list)
    weak_concepts: list[str] = Field(default_factory=list)
    missing_reasoning: str = ""
    vague_parts: str = ""
    suspicious_parts: str = ""
    evidence_from_answer: str = ""
    confidence_score: float = 0.0
    probe_worthy: bool = False

    _v_lists = field_validator(
        "correct_concepts", "weak_concepts", mode="before"
    )(_as_str_list)
    _v_text = field_validator(
        "missing_reasoning", "vague_parts", "suspicious_parts", "evidence_from_answer",
        mode="before",
    )(_as_text)
    _v_score = field_validator("confidence_score", mode="before")(_clamp_unit)


# ── Evaluation: targeted probe question ───────────────────────────────────────
class ProbeQuestion(BaseModel):
    """Validated targeted follow-up question (evaluation_agent._generate_probe)."""

    model_config = ConfigDict(extra="ignore")

    question_text: str
    id: str = ""
    type: str = "misconception_probe"
    concepts_tested: list[str] = Field(default_factory=list)
    difficulty: str = "applied"
    probing_for: str = ""

    _v_concepts = field_validator("concepts_tested", mode="before")(_as_str_list)

    @field_validator("question_text", mode="before")
    @classmethod
    def _require_question_text(cls, value: Any) -> str:
        """A probe with no question text is useless — reject so the caller skips it."""
        text = "" if value is None else str(value).strip()
        if not text:
            raise ValueError("question_text must be non-empty")
        return text


# ── Evaluation: final report ──────────────────────────────────────────────────
class FinalReport(BaseModel):
    """Validated final evaluation report (evaluation_agent._finalize).

    Fields are lenient (all defaulted) so a well-formed report validates
    unchanged; ``decision`` is left a free string because the call site already
    re-checks it against DECISION_ENUM and the nested blocks vary in shape.
    """

    model_config = ConfigDict(extra="ignore")

    strengths: list[str] = Field(default_factory=list)
    weak_concepts: list[str] = Field(default_factory=list)
    misconceptions: list[str] = Field(default_factory=list)
    mastery_score: float = 0.0
    confidence_trend: str = ""
    decision: str = "ADVANCE"
    decision_reasoning: str = ""
    motivational_feedback: str = ""
    transition_feedback: str = ""
    reteach_data: dict[str, Any] = Field(default_factory=dict)
    adaptation_summary: dict[str, Any] = Field(default_factory=dict)

    _v_lists = field_validator(
        "strengths", "weak_concepts", "misconceptions", mode="before"
    )(_as_str_list)
    _v_score = field_validator("mastery_score", mode="before")(_clamp_unit)


# ── Course service: generated check questions ─────────────────────────────────
class GeneratedQuestion(BaseModel):
    """One grounded check question from the question-generation retry prompt."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    question_text: str = Field(
        default="", validation_alias=AliasChoices("question_text", "question")
    )
    expected_answer: str = Field(
        default="", validation_alias=AliasChoices("expected_answer", "answer")
    )
    source_quote: str = Field(
        default="", validation_alias=AliasChoices("source_quote", "evidence", "quote")
    )
    concepts_tested: list[str] = Field(
        default_factory=list, validation_alias=AliasChoices("concepts_tested", "concepts")
    )
    source_section: str = "Lesson"
    is_answerable_from_lesson: bool = True
    difficulty: str = "simple"

    _v_concepts = field_validator("concepts_tested", mode="before")(_as_str_list)


class GeneratedQuestionList(BaseModel):
    """Wrapper for the ``{"questions": [...]}`` shape the retry prompt returns."""

    model_config = ConfigDict(extra="ignore")

    questions: list[GeneratedQuestion] = Field(default_factory=list)


# ── Parse helper ──────────────────────────────────────────────────────────────
def parse_llm_json(
    raw: str,
    model_cls: type[BaseModel],
    *,
    caller: str,
) -> tuple[BaseModel | None, str | None]:
    """Parse + validate LLM JSON into ``model_cls``; observable on failure.

    Returns ``(instance, None)`` on success, or ``(None, error)`` on failure —
    after logging a WARNING and incrementing
    ``edumind_llm_schema_failures_total{caller, schema}``. The call site is
    expected to fall back to its existing safe default when the instance is None.
    """
    # Import here to avoid a core/ import cycle (curriculum_quality imports light).
    from core.curriculum_quality import parse_json_object

    schema = model_cls.__name__
    parsed = parse_json_object(raw)
    if not parsed:
        return _fail(caller, schema, "no JSON object could be parsed from model output")
    try:
        return model_cls.model_validate(parsed), None
    except ValidationError as exc:
        return _fail(caller, schema, _summarize_validation_error(exc))


def _fail(caller: str, schema: str, error: str) -> tuple[None, str]:
    """Record an observable schema failure and return the (None, error) tuple."""
    logger.warning(
        "LLM schema validation failed caller='{}' schema='{}': {}", caller, schema, error
    )
    _metrics.llm_schema_failures.labels(caller=caller, schema=schema).inc()
    return None, error


def _summarize_validation_error(exc: ValidationError) -> str:
    """Compact one-line summary of a pydantic ValidationError for logs."""
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")
    return "; ".join(parts[:5])
