"""
agents/evaluation_agent.py
Agentic evaluation after lesson completion.

Triggered: ONLY when student clicks Next/Complete after reading the lesson.
Never during course creation, lesson generation, or module opening.

Flow:
  1. start_session()   → generate 2-3 base questions from lesson content
  2. submit_answer()   → diagnose; if anomaly detected, generate targeted probe;
                         else move to next base question; finalize when done
  3. finalize()        → final report, decision, feedback, save to DB

Pace rules:
  fast  → 2 base questions; 1 probe only if clear anomaly
  medium → 3 base questions; up to 1 probe per weak answer
  deep  → 4 base questions; probes on any weak/uncertain answer
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from loguru import logger
from core.metrics import metrics as _metrics

from clients.groq_client import generate
from config import settings
from core.curriculum_quality import parse_json_object
from prompts import get_prompt
from db.postgres import (
    get_adaptation_summary,
    get_compact_doubt_summary,
    get_course,
    get_course_module,
    save_adaptation_summary,
    save_evaluation_session,
    upsert_concept_mastery,
    upsert_student_skill,
    write_evaluation,
)

DECISION_ENUM = {
    "ADVANCE",
    "ADVANCE_WITH_LIGHT_REVIEW",
    "RETEACH_WEAK_CONCEPTS",
    "REPEAT_MODULE",
    "ADJUST_FUTURE_LESSON_DIFFICULTY",
}

# Registry-backed (prompts/evaluation.py). Rendered once at import; the string is
# identical to the former inline literal (snapshot-proven).
_EVAL_SYSTEM_PROMPT = get_prompt("evaluation_system")
_EVAL_SYSTEM = _EVAL_SYSTEM_PROMPT.render()

# Per-pace config
# fast:   2 base questions + up to 1 chained probe = max 3 total
#         probe only if a clear anomaly is detected in one of the base answers
# medium: 3 base questions + up to 2 chained probes = max 5 total (can reach 6 at hard limit)
#         probe whenever a weakness or vagueness is detected
# deep:   4 base questions + up to 3 chained probes = max 7 total
#         probe on any answer that isn't fully clear
_PACE_CONFIG = {
    "fast":   {"base_q": 2, "max_probes": 1,  "probe_threshold": "weak",      "max_total": 3},
    "medium": {"base_q": 3, "max_probes": 2,  "probe_threshold": "weak",      "max_total": 6},
    "deep":   {"base_q": 4, "max_probes": 3,  "probe_threshold": "uncertain", "max_total": 7},
}

MAX_QUESTIONS = 7  # hard ceiling — never exceeded regardless of pace


def _lesson_excerpt(content: str, max_chars: int = 1500) -> str:
    """Trim lesson markdown to the portion sent into evaluation prompts."""
    return content[:max_chars].strip() if content else ""


def _module_context(module: dict[str, Any]) -> dict[str, Any]:
    """Normalize module metadata into the scope used by evaluation prompts."""
    metadata = module.get("module_metadata") or {}
    return {
        "title": module.get("title", ""),
        "concept": module.get("concept", ""),
        "concepts_taught": (
            module.get("concepts_taught")
            or metadata.get("concepts_taught")
            or module.get("must_teach")
            or metadata.get("must_teach")
            or [module.get("concept", "")]
        ),
        "question_scope": (
            module.get("question_scope")
            or metadata.get("question_scope")
            or []
        ),
        "must_teach": (
            module.get("must_teach")
            or metadata.get("must_teach")
            or []
        ),
        "depth_level": module.get("depth_level", "standard"),
    }


async def _generate_probe_question(
    mod_ctx: dict[str, Any],
    lesson_content: str,
    trigger_answer: dict[str, Any],
    probe_number: int,
) -> dict[str, Any] | None:
    """
    Generate one targeted follow-up question based on a weak/uncertain answer.
    Like an interviewer who notices a gap and probes exactly that spot.
    """
    diagnosis = trigger_answer.get("diagnosis") or {}
    weak_concepts = diagnosis.get("weak_concepts") or []
    missing_reasoning = diagnosis.get("missing_reasoning") or ""
    vague_parts = diagnosis.get("vague_parts") or ""
    suspicious_parts = diagnosis.get("suspicious_parts") or ""

    if not (weak_concepts or missing_reasoning or vague_parts):
        return None

    prompt = json.dumps({
        "task": "generate_targeted_probe_question",
        "module": mod_ctx,
        "lesson_excerpt": _lesson_excerpt(lesson_content),
        "trigger_question": trigger_answer.get("question_text", ""),
        "trigger_answer": trigger_answer.get("answer_text", "")[:400],
        "detected_weakness": {
            "weak_concepts": weak_concepts,
            "missing_reasoning": missing_reasoning,
            "vague_parts": vague_parts,
            "suspicious_parts": suspicious_parts,
        },
        "probe_number": probe_number,
        "instructions": get_prompt("evaluation_probe_instructions").render(),
        "return_schema": {
            "id": f"probe_{probe_number}",
            "question_text": "...",
            "type": "misconception_probe",
            "concepts_tested": ["..."],
            "difficulty": "applied",
            "probing_for": "what gap this targets",
        },
    }, default=str)

    try:
        raw = await generate(
            messages=[{"role": "user", "content": prompt}],
            model=getattr(settings, "adaptation_model", settings.reasoning_model),
            system=_EVAL_SYSTEM,
            json_mode=True,
            _caller="evaluation",
            _prompt_name="evaluation_probe_instructions",
            _prompt_version=get_prompt("evaluation_probe_instructions").version,
        )
        data = parse_json_object(raw)
        if data.get("question_text"):
            return data
    except Exception as exc:
        logger.warning("Probe question generation failed: {}", exc)
    return None


async def diagnose_student_answer(
    mod_ctx: dict[str, Any],
    lesson_content: str,
    question: dict[str, Any],
    answer_text: str,
    confidence: int = 3,
    previous_answers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Diagnose one student answer against the lesson and question.

    Thin seam extracted from ``_submit_answer_impl`` so the same live diagnosis
    code path can be exercised without a DB-backed evaluation session (used by
    the golden-eval runner). Behaviour is identical to the inline block it
    replaced. Returns the parsed diagnosis dict, or a safe fallback on failure.
    """
    previous_answers = previous_answers or []
    diagnosis_prompt = json.dumps({
        "task": "diagnose_student_answer",
        "module": mod_ctx,
        "lesson_excerpt": _lesson_excerpt(lesson_content),
        "question": question,
        "student_answer": answer_text,
        "confidence_stated": confidence,
        "previous_answers": previous_answers[-3:],
        "instructions": get_prompt("evaluation_diagnose_instructions").render(),
        "return_schema": {
            "correct_concepts": ["..."],
            "weak_concepts": ["..."],
            "missing_reasoning": "one sentence describing what reasoning was absent",
            "vague_parts": "what was said vaguely",
            "suspicious_parts": "what might indicate a misconception",
            "confidence_score": 0.0,
            "mastery_signal": "clear | uncertain | weak",
            "evidence_from_answer": "a brief quote or paraphrase from student answer",
            "probe_worthy": True,
        },
    }, default=str)

    try:
        raw = await generate(
            messages=[{"role": "user", "content": diagnosis_prompt}],
            model=getattr(settings, "adaptation_model", settings.reasoning_model),
            system=_EVAL_SYSTEM,
            json_mode=True,
            _caller="evaluation",
            _prompt_name="evaluation_diagnose_instructions",
            _prompt_version=get_prompt("evaluation_diagnose_instructions").version,
        )
        return parse_json_object(raw)
    except Exception as exc:
        logger.warning("Answer diagnosis failed: {}", exc)
        return {
            "correct_concepts": [],
            "weak_concepts": mod_ctx.get("concepts_taught", []),
            "missing_reasoning": "Could not analyze answer.",
            "mastery_signal": "uncertain",
            "confidence_score": 0.5,
            "probe_worthy": False,
        }


async def start_session(
    course_id: str,
    module_id: str,
    student_id: str,
) -> dict[str, Any]:
    """Workflow-span wrapper: groups the whole evaluation session start."""
    from core.tracing import get_tracer

    with get_tracer().start_as_current_span("workflow.eval.start_session") as span:
        if span.is_recording():
            span.set_attribute("edumind.course_id", str(course_id))
            span.set_attribute("edumind.module_id", str(module_id))
            span.set_attribute("edumind.student_id", str(student_id))
        result = await _start_session_impl(course_id, module_id, student_id)
        if span.is_recording() and isinstance(result, dict) and result.get("session_id"):
            span.set_attribute("edumind.session_id", str(result["session_id"]))
        return result


async def _start_session_impl(
    course_id: str,
    module_id: str,
    student_id: str,
) -> dict[str, Any]:
    """
    Start an evaluation session after the student clicks Next/Complete.
    Returns session_id and the first batch of base questions.
    """
    course = await get_course(course_id, student_id)
    module = await get_course_module(course_id, module_id)
    if not course or not module:
        raise ValueError("Course or module not found")

    lesson_content = module.get("content_markdown", "")
    if not lesson_content:
        raise ValueError(
            "Cannot start evaluation before lesson content exists. "
            "Open the module first to generate the lesson."
        )

    session_id = str(uuid.uuid4())
    pace = course.get("pace", "medium")
    pace_cfg = _PACE_CONFIG.get(pace, _PACE_CONFIG["medium"])
    mod_ctx = _module_context(module)
    doubt_summary = await get_compact_doubt_summary(course_id, module_id, student_id)
    adaptation_summary = await get_adaptation_summary(student_id, course_id) or {}
    target_q = pace_cfg["base_q"]

    prompt = json.dumps({
        "task": "generate_initial_evaluation_questions",
        "course_topic": course.get("topic", ""),
        "student_goal": course.get("goal", ""),
        "pace": pace,
        "module": mod_ctx,
        "lesson_excerpt": _lesson_excerpt(lesson_content),
        "doubt_summary": doubt_summary,
        "adaptation_notes": adaptation_summary.get("notes", ""),
        "target_question_count": target_q,
        "instructions": (
            f"Generate exactly {target_q} evaluation questions for the module '{mod_ctx['title']}'. "
            "━━━ SCOPE RULE (CRITICAL) ━━━ "
            "Every base question MUST test ONLY concepts listed in concepts_taught. "
            "DO NOT ask about anything not covered in this module's lesson content. "
            "If you are unsure whether a concept was taught — DO NOT ask about it. "
            "━━━ QUESTION MIX ━━━ "
            "  Q1: RECALL — ask the student to explain or define one specific thing from the lesson. "
            "       Example: 'What does X do?' or 'How does Y work?' "
            "  Q2: APPLICATION — ask the student to apply or trace through what was taught. "
            "       For code modules: give a short 3-5 line code snippet and ask what it does or outputs. "
            "       For math/science: give numbers and ask them to apply the formula or method. "
            "       For theory: give a scenario and ask them to apply the concept. "
            "  Q3+ (if target >= 3): MISCONCEPTION CHECK — state a common wrong belief about this "
            "       specific concept and ask if it is correct and why. "
            "       The misconception must be relevant to THIS module's concept only. "
            "━━━ QUALITY RULES ━━━ "
            "  - Questions must be specific, not vague. Bad: 'Tell me about X'. Good: 'What happens when...' "
            "  - For coding: always include a short runnable code example in Q2. "
            "  - Calibrate difficulty to the student's level in the student_context. "
            "━━━ BONUS QUESTION ━━━ "
            "Add 1 bonus question that requires the student to USE the concept in a new scenario. "
            "This tests transfer. Mark with 'is_bonus': true and 'is_base_question': false. "
            "The bonus does NOT affect pass/fail. "
            "Return JSON only."
        ),
        "pace_context": (
            f"Pace is '{pace}'. "
            "fast pace → questions should be crisp, direct, testing the essential core only. "
            "medium pace → questions should match textbook depth — test understanding, not memorization. "
            "deep pace → questions should probe nuance, edge cases, and deeper reasoning."
        ),
        "return_schema": {
            "questions": [
                {
                    "id": "q1",
                    "question_text": "...",
                    "type": "recall | conceptual | application | misconception_probe",
                    "concepts_tested": ["..."],
                    "difficulty": "simple | applied",
                    "is_base_question": True,
                }
            ]
        },
    }, default=str)

    try:
        raw = await generate(
            messages=[{"role": "user", "content": prompt}],
            model=getattr(settings, "adaptation_model", settings.reasoning_model),
            system=_EVAL_SYSTEM,
            json_mode=True,
            _caller="evaluation",
            _prompt_name=_EVAL_SYSTEM_PROMPT.name,
            _prompt_version=_EVAL_SYSTEM_PROMPT.version,
        )
        data = parse_json_object(raw)
        questions = data.get("questions") or []
    except Exception as exc:
        logger.warning("Evaluation question generation failed: {}", exc)
        questions = []

    # Fallback base questions
    if not questions:
        concept = mod_ctx["concept"] or "this concept"
        questions = [
            {
                "id": "q1",
                "question_text": f"In your own words, what is {concept} and how does it work?",
                "type": "recall",
                "concepts_tested": [concept],
                "difficulty": "simple",
                "is_base_question": True,
            },
            {
                "id": "q2",
                "question_text": f"What is one common mistake students make when learning {concept}?",
                "type": "misconception_probe",
                "concepts_tested": [concept],
                "difficulty": "applied",
                "is_base_question": True,
            },
        ]

    # Mark all as base questions, cap at target
    for i, q in enumerate(questions):
        q["id"] = f"q{i+1}"
        q["is_base_question"] = True
    questions = questions[:target_q]

    session = {
        "session_id": session_id,
        "course_id": course_id,
        "module_id": module_id,
        "student_id": student_id,
        "status": "active",
        "pace": pace,
        "questions_asked": 0,
        "probes_used": 0,
        "questions": questions,
        "answers": [],
        "final_report": {},
        "decision": "",
        "motivational_feedback": "",
        "transition_feedback": "",
        "reteach_data": {},
    }
    await save_evaluation_session(session)
    _metrics.eval_sessions_started.labels(pace=pace).inc()
    for _q in questions:
        _qtype = "bonus" if _q.get("is_bonus") else _q.get("type", "recall")
        _metrics.eval_questions_asked.labels(pace=pace, question_type=_qtype).inc()

    return {
        "session_id": session_id,
        "questions": questions,
        "total_questions": len(questions),
        "module_title": module.get("title", ""),
        "message": (
            f"Great — you've completed this module! "
            f"I have {len(questions)} quick question(s) to check your understanding."
        ),
    }


async def submit_answer(
    session_id: str,
    question_id: str,
    answer_text: str,
    confidence: int = 3,
) -> dict[str, Any]:
    """Workflow-span wrapper around one answer submission.

    Records the answer LENGTH only — never the answer text — per the learner
    privacy rule for span attributes.
    """
    from core.tracing import get_tracer

    with get_tracer().start_as_current_span("workflow.eval.submit_answer") as span:
        if span.is_recording():
            span.set_attribute("edumind.session_id", str(session_id))
            span.set_attribute("edumind.question_id", str(question_id))
            span.set_attribute("edumind.answer_len", len(answer_text or ""))
            span.set_attribute("edumind.confidence", int(confidence))
        return await _submit_answer_impl(session_id, question_id, answer_text, confidence)


async def _submit_answer_impl(
    session_id: str,
    question_id: str,
    answer_text: str,
    confidence: int = 3,
) -> dict[str, Any]:
    """
    Submit one answer.

    After diagnosing the answer:
    - If a weakness/anomaly is detected AND probe budget allows → generate targeted probe question
    - Else → advance to next base question
    - When all base questions answered (+ any probes) → finalize

    Returns diagnosis plus either: next_question (probe or base) OR final report.
    """
    from db.postgres import get_evaluation_session
    session = await get_evaluation_session(session_id)
    if not session:
        raise ValueError("Evaluation session not found")
    if session.get("status") != "active":
        raise ValueError("Evaluation session is already completed")

    questions = session.get("questions_json") or []
    answers = session.get("answers_json") or []
    course_id = session["course_id"]
    module_id = session["module_id"]
    student_id = session["student_id"]
    pace = session.get("pace", "medium")
    pace_cfg = _PACE_CONFIG.get(pace, _PACE_CONFIG["medium"])
    probes_used = int(session.get("probes_used") or 0)

    question = next((q for q in questions if q.get("id") == question_id), None)
    if not question:
        raise ValueError(f"Question '{question_id}' not found in session")

    course = await get_course(course_id)
    module = await get_course_module(course_id, module_id)
    lesson_content = (module or {}).get("content_markdown", "")
    mod_ctx = _module_context(module or {})

    # --- Diagnose this answer ---
    diagnosis = await diagnose_student_answer(
        mod_ctx=mod_ctx,
        lesson_content=lesson_content,
        question=question,
        answer_text=answer_text,
        confidence=confidence,
        previous_answers=answers,
    )

    answers.append({
        "question_id": question_id,
        "question_text": question.get("question_text", ""),
        "answer_text": answer_text,
        "confidence": confidence,
        "diagnosis": diagnosis,
        "is_probe": not question.get("is_base_question", True),
    })

    questions_asked = len(answers)
    mastery_signal = diagnosis.get("mastery_signal", "uncertain")
    has_weakness = bool(
        diagnosis.get("weak_concepts")
        or diagnosis.get("missing_reasoning")
        or diagnosis.get("vague_parts")
        or diagnosis.get("suspicious_parts")
    )
    probe_worthy = bool(diagnosis.get("probe_worthy", has_weakness))

    # How many base questions remain unanswered?
    answered_ids = {a["question_id"] for a in answers}
    base_questions = [q for q in questions if q.get("is_base_question", True)]
    remaining_base = [q for q in base_questions if q["id"] not in answered_ids]

    # Probe budget: can we insert a follow-up?
    probe_signal_strong_enough = mastery_signal in ("weak", "uncertain") if pace == "fast" else mastery_signal in ("weak", "uncertain")
    if pace == "deep":
        probe_signal_strong_enough = mastery_signal != "clear"

    can_probe = (
        probe_worthy
        and probe_signal_strong_enough
        and probes_used < pace_cfg["max_probes"]
        and questions_asked < MAX_QUESTIONS
    )

    # Use pace-specific total cap (not global MAX_QUESTIONS)
    pace_max_total = pace_cfg.get("max_total", MAX_QUESTIONS)

    # Check if this was the last question (base or probe)
    all_base_done = len(remaining_base) == 0
    at_hard_limit = questions_asked >= pace_max_total
    should_finalize = at_hard_limit or (all_base_done and not can_probe)

    # --- Generate probe question if warranted ---
    # Probes chain from the triggering answer: if base Q1 shows weakness,
    # the probe targets exactly that weakness — like an interviewer digging in.
    # Probes are never generated off other probes (no infinite chains).
    probe_question = None
    if not should_finalize and can_probe and questions_asked < pace_max_total:
        # Only probe off base questions (not off probes themselves)
        if question.get("is_base_question", True):
            probe_number = probes_used + 1
            probe_question = await _generate_probe_question(
                mod_ctx, lesson_content, answers[-1], probe_number
            )
            if probe_question:
                probe_question["is_base_question"] = False
                questions.append(probe_question)
                probes_used += 1
                _metrics.eval_probe_triggered.labels(pace=pace).inc()
                _metrics.eval_questions_asked.labels(pace=pace, question_type="probe").inc()

    # Re-evaluate finalize after potential probe
    answered_ids = {a["question_id"] for a in answers}
    remaining_questions = [q for q in questions if q["id"] not in answered_ids]
    should_finalize = at_hard_limit or len(remaining_questions) == 0

    # Save progress
    updated_session = {
        **{k: session[k] for k in session},
        "session_id": session_id,
        "questions_asked": questions_asked,
        "probes_used": probes_used,
        "questions": questions,
        "answers": answers,
        "status": "completed" if should_finalize else "active",
    }

    if should_finalize:
        final = await _finalize(
            session_id=session_id,
            course=course or {},
            module=module or {},
            mod_ctx=mod_ctx,
            lesson_content=lesson_content,
            answers=answers,
            student_id=student_id,
            course_id=course_id,
            module_id=module_id,
        )
        updated_session.update(final)
        await save_evaluation_session(updated_session)
        return {
            "session_id": session_id,
            "question_id": question_id,
            "diagnosis": diagnosis,
            "session_complete": True,
            "questions_asked": questions_asked,
            "final_report": final.get("final_report", {}),
            "decision": final.get("decision", "ADVANCE"),
            "motivational_feedback": final.get("motivational_feedback", ""),
            "transition_feedback": final.get("transition_feedback", ""),
            "reteach_data": final.get("reteach_data", {}),
        }

    # Next question: probe if generated, otherwise next base
    next_q = probe_question if probe_question else remaining_questions[0] if remaining_questions else None

    await save_evaluation_session(updated_session)
    return {
        "session_id": session_id,
        "question_id": question_id,
        "diagnosis": diagnosis,
        "session_complete": False,
        "questions_asked": questions_asked,
        "is_probe": probe_question is not None,
        "probe_reason": (
            f"You mentioned {diagnosis.get('vague_parts') or diagnosis.get('missing_reasoning') or 'something unclear'} — let me check that."
            if probe_question else None
        ),
        "next_question": next_q,
    }


async def _finalize(
    *,
    session_id: str,
    course: dict[str, Any],
    module: dict[str, Any],
    mod_ctx: dict[str, Any],
    lesson_content: str,
    answers: list[dict[str, Any]],
    student_id: str,
    course_id: str,
    module_id: str,
) -> dict[str, Any]:
    """Workflow-span wrapper around session finalization (report + decision)."""
    from core.tracing import get_tracer

    with get_tracer().start_as_current_span("workflow.eval.finalize") as span:
        if span.is_recording():
            span.set_attribute("edumind.session_id", str(session_id))
            span.set_attribute("edumind.course_id", str(course_id))
            span.set_attribute("edumind.module_id", str(module_id))
            span.set_attribute("edumind.student_id", str(student_id))
            span.set_attribute("edumind.answer_count", len(answers or []))
        return await _finalize_impl(
            session_id=session_id,
            course=course,
            module=module,
            mod_ctx=mod_ctx,
            lesson_content=lesson_content,
            answers=answers,
            student_id=student_id,
            course_id=course_id,
            module_id=module_id,
        )


async def _finalize_impl(
    *,
    session_id: str,
    course: dict[str, Any],
    module: dict[str, Any],
    mod_ctx: dict[str, Any],
    lesson_content: str,
    answers: list[dict[str, Any]],
    student_id: str,
    course_id: str,
    module_id: str,
) -> dict[str, Any]:
    """Generate final report, decision, feedback, and save all progress data."""
    pace = course.get("pace", "medium")
    threshold = {"fast": 0.60, "medium": 0.72, "deep": 0.85}.get(pace, 0.72)

    all_correct: list[str] = []
    all_weak: list[str] = []
    mastery_signals = []
    for ans in answers:
        d = ans.get("diagnosis") or {}
        all_correct.extend(d.get("correct_concepts") or [])
        all_weak.extend(d.get("weak_concepts") or [])
        mastery_signals.append(d.get("mastery_signal", "uncertain"))

    clear_count = mastery_signals.count("clear")
    weak_count = mastery_signals.count("weak")
    total = len(mastery_signals) or 1
    mastery_score = round(
        (clear_count * 1.0 + (total - clear_count - weak_count) * 0.6 + weak_count * 0.2) / total, 3
    )

    # Separate base vs probe answers for richer reporting
    base_answers = [a for a in answers if not a.get("is_probe")]
    probe_answers = [a for a in answers if a.get("is_probe")]

    report_prompt = json.dumps({
        "task": "generate_final_evaluation_report",
        "module": mod_ctx,
        "course_topic": course.get("topic", ""),
        "student_goal": course.get("goal", ""),
        "pace": pace,
        "mastery_threshold": threshold,
        "computed_mastery_score": mastery_score,
        "correct_concepts": list(dict.fromkeys(all_correct)),
        "weak_concepts": list(dict.fromkeys(all_weak)),
        "mastery_signals": mastery_signals,
        "base_answers_count": len(base_answers),
        "probe_answers_count": len(probe_answers),
        "answers_summary": [
            {
                "q": a["question_text"],
                "a": a["answer_text"][:300],
                "signal": (a.get("diagnosis") or {}).get("mastery_signal", "uncertain"),
                "is_probe": a.get("is_probe", False),
            }
            for a in answers
        ],
        "decision_options": list(DECISION_ENUM),
        "instructions": get_prompt("evaluation_finalize_instructions").render(),
        "return_schema": {
            "strengths": ["..."],
            "weak_concepts": ["..."],
            "misconceptions": ["..."],
            "mastery_score": 0.0,
            "confidence_trend": "...",
            "decision": "ADVANCE",
            "decision_reasoning": "...",
            "motivational_feedback": "...",
            "transition_feedback": "...",
            "reteach_data": {
                "message": "...",
                "weak_concepts": ["..."],
                "recommended_action": "reteach",
                "reteach_focus": ["..."],
                "next_lesson_adjustment": "...",
            },
            "adaptation_summary": {
                "notes": "...",
                "preferred_style": "...",
                "weak_concepts": ["..."],
                "depth_adjustment": "none | increase | decrease",
                "example_preference": "more | same | fewer",
                "pace_adjustment": "none | slower | faster",
            },
        },
    }, default=str)

    try:
        raw = await generate(
            messages=[{"role": "user", "content": report_prompt}],
            model=getattr(settings, "adaptation_model", settings.reasoning_model),
            system=_EVAL_SYSTEM,
            json_mode=True,
            _caller="evaluation",
            _prompt_name="evaluation_finalize_instructions",
            _prompt_version=get_prompt("evaluation_finalize_instructions").version,
        )
        report_data = parse_json_object(raw)
    except Exception as exc:
        logger.warning("Final evaluation report generation failed: {}", exc)
        report_data = {
            "strengths": list(dict.fromkeys(all_correct)),
            "weak_concepts": list(dict.fromkeys(all_weak)),
            "misconceptions": [],
            "mastery_score": mastery_score,
            "confidence_trend": "unknown",
            "decision": "ADVANCE" if mastery_score >= threshold else "RETEACH_WEAK_CONCEPTS",
            "motivational_feedback": "You've completed the evaluation. Keep going!",
            "transition_feedback": "Moving to the next module.",
            "adaptation_summary": {"notes": "", "weak_concepts": list(dict.fromkeys(all_weak))},
        }

    decision = report_data.get("decision", "ADVANCE")
    if decision not in DECISION_ENUM:
        decision = "ADVANCE" if mastery_score >= threshold else "RETEACH_WEAK_CONCEPTS"

    final_mastery = float(report_data.get("mastery_score") or mastery_score)
    misconception = report_data.get("misconceptions") or []
    misconception_type = misconception[0] if misconception else None

    concept = module.get("concept", mod_ctx.get("concept", ""))
    if concept:
        correctness = min(1.0, final_mastery + 0.1)
        depth = max(0.0, final_mastery - 0.1)
        await upsert_concept_mastery(student_id, concept, correctness, depth)
        await upsert_student_skill(
            student_id=student_id,
            concept=concept,
            mastery_score=final_mastery,
            depth_score=depth,
            source="evaluation",
            status="mastered" if final_mastery >= threshold else "learning",
            evidence={
                "course_id": course_id,
                "module_id": module_id,
                "session_id": session_id,
                "decision": decision,
            },
        )

    await write_evaluation({
        "student_id": student_id,
        "session_id": f"eval:{session_id}",
        "concept": concept,
        "correctness_score": min(1.0, final_mastery + 0.1),
        "depth_score": max(0.0, final_mastery - 0.1),
        "mastery_score": final_mastery,
        "misconception_type": misconception_type,
        "misconception_detail": "; ".join(str(m) for m in misconception[:3]),
        "confidence_stated": 3,
        "calibration_delta": 0.0,
        "questions_asked": len(answers),
        "recommended_action": decision,
    })

    adaptation_summary = report_data.get("adaptation_summary") or {}
    adaptation_summary["weak_concepts"] = list(dict.fromkeys(
        (adaptation_summary.get("weak_concepts") or []) + list(dict.fromkeys(all_weak))
    ))
    adaptation_summary["last_decision"] = decision
    adaptation_summary["last_module_id"] = module_id
    await save_adaptation_summary(student_id, course_id, module_id, adaptation_summary)

    return {
        "final_report": report_data,
        "decision": decision,
        "motivational_feedback": report_data.get("motivational_feedback", ""),
        "transition_feedback": report_data.get("transition_feedback", ""),
        "reteach_data": report_data.get("reteach_data") or {},
        "status": "completed",
    }


async def get_session_report(session_id: str) -> dict[str, Any]:
    """Retrieve a completed evaluation session report."""
    from db.postgres import get_evaluation_session
    session = await get_evaluation_session(session_id)
    if not session:
        raise ValueError("Evaluation session not found")
    return {
        "session_id": session_id,
        "status": session.get("status"),
        "questions_asked": session.get("questions_asked", 0),
        "decision": session.get("decision", ""),
        "final_report": session.get("final_report_json") or {},
        "motivational_feedback": session.get("motivational_feedback", ""),
        "transition_feedback": session.get("transition_feedback", ""),
        "reteach_data": session.get("reteach_data_json") or {},
    }
