"""
agents/course_report_agent.py

Generates the final course performance report when all modules are completed.

Triggered: when the last module is completed (recalculate_course_progress sets
           course status to 'completed'), OR on explicit GET request.

The report is:
  - A mentor-style performance summary: what you are good at, what needs work
  - Skill categorisation: mastered / learning / weak — saved to student_skills
  - Honest, specific feedback. Not generic praise.
  - Reasoning for each skill verdict
"""
from __future__ import annotations

import json
from typing import Any

from loguru import logger

from clients.groq_client import generate
from config import settings
from core.curriculum_quality import parse_json_object
from db.postgres import (
    get_course,
    get_course_completion_report,
    get_student_skills,
    list_course_modules,
    save_course_completion_report,
    upsert_student_skill,
)

_REPORT_SYSTEM = (
    "You are EduMind's course mentor. Return STRICT JSON only. No markdown. "
    "Generate honest, specific, encouraging course completion feedback. "
    "Base all claims on the data provided. Do not invent skills or weaknesses."
)


async def generate_course_report(
    course_id: str,
    student_id: str,
) -> dict[str, Any]:
    """
    Generate (or return cached) final course report.
    Called when all modules are done and student views the completion screen.
    """
    # Return cached report if it already exists
    cached = await get_course_completion_report(course_id, student_id)
    if cached and cached.get("report"):
        return cached["report"]

    course = await get_course(course_id, student_id)
    if not course:
        raise ValueError("Course not found")

    modules = await list_course_modules(course_id)
    skills = await get_student_skills(student_id)

    # Pull all eval session data for this course from student_skills
    # (already written per-module by evaluation_agent._finalize)
    course_skills = [
        n for n in (skills.get("nodes") or [])
        if (n.get("evidence_json") or {}).get("course_id") == course_id
    ]

    mastered = [n for n in course_skills if n.get("status") == "mastered"]
    learning = [n for n in course_skills if n.get("status") == "learning"]
    weak = [n for n in course_skills if n.get("status") == "weak"]

    # Build module completion summary
    module_summary = []
    for m in modules:
        module_summary.append({
            "title": m.get("title", ""),
            "concept": m.get("concept", ""),
            "status": m.get("status", ""),
            "module_index": m.get("module_index", 0),
        })

    prompt = json.dumps({
        "task": "generate_course_completion_report",
        "course_topic": course.get("topic", ""),
        "student_goal": course.get("goal", ""),
        "pace": course.get("pace", "medium"),
        "total_modules": len(modules),
        "completed_modules": sum(1 for m in modules if m.get("status") == "completed"),
        "module_summary": module_summary,
        "mastered_skills": [
            {"concept": n["concept"], "mastery_score": n.get("mastery_score", 0)}
            for n in mastered
        ],
        "learning_skills": [
            {"concept": n["concept"], "mastery_score": n.get("mastery_score", 0)}
            for n in learning
        ],
        "weak_skills": [
            {"concept": n["concept"], "mastery_score": n.get("mastery_score", 0)}
            for n in weak
        ],
        "instructions": (
            "Generate a final course performance report. "
            "overall_summary: 3-4 sentences. Like a mentor talking to a student. "
            "Mention their actual strengths (mastered skills) and actual gaps (weak skills) by name. "
            "Do NOT say 'great job' generically. Be honest and specific. "
            "strengths_narrative: 1-2 sentences about what they genuinely learned well. "
            "growth_areas_narrative: 1-2 sentences about what needs more practice, with actionable advice. "
            "skill_verdict: for EACH skill (mastered/learning/weak), provide a 1-sentence reason. "
            "personality_insight: 1 sentence about their learning style based on the data "
            "(e.g. fast but sometimes surface-level, thorough but slow, etc.). "
            "next_steps: 2-3 concrete recommendations for what to do next. "
            "Return JSON only."
        ),
        "return_schema": {
            "overall_summary": "...",
            "strengths_narrative": "...",
            "growth_areas_narrative": "...",
            "personality_insight": "...",
            "skill_verdicts": [
                {
                    "concept": "...",
                    "status": "mastered | learning | weak",
                    "mastery_score": 0.0,
                    "reason": "one sentence why this verdict",
                }
            ],
            "mastered_skills": ["..."],
            "learning_skills": ["..."],
            "weak_skills": ["..."],
            "next_steps": ["..."],
            "completion_badge": "Completed | Completed with distinction | Needs review",
        },
    }, default=str)

    try:
        raw = await generate(
            messages=[{"role": "user", "content": prompt}],
            model=getattr(settings, "adaptation_model", settings.reasoning_model),
            system=_REPORT_SYSTEM,
        )
        report = parse_json_object(raw)
    except Exception as exc:
        logger.warning("Course report generation failed: {}", exc)
        report = {
            "overall_summary": (
                f"You have completed {course.get('topic', 'the course')}. "
                f"You mastered {len(mastered)} concept(s), are still building on {len(learning)}, "
                f"and {len(weak)} concept(s) need more work."
            ),
            "strengths_narrative": (
                f"Strong areas: {', '.join(n['concept'] for n in mastered[:3])}."
                if mastered else "Keep practicing — all skills are still developing."
            ),
            "growth_areas_narrative": (
                f"Focus on: {', '.join(n['concept'] for n in weak[:3])}."
                if weak else "No major weak areas detected."
            ),
            "personality_insight": "You worked steadily through the course.",
            "skill_verdicts": [
                {"concept": n["concept"], "status": n["status"],
                 "mastery_score": n.get("mastery_score", 0), "reason": "Based on evaluation results."}
                for n in course_skills
            ],
            "mastered_skills": [n["concept"] for n in mastered],
            "learning_skills": [n["concept"] for n in learning],
            "weak_skills": [n["concept"] for n in weak],
            "next_steps": [
                f"Review {n['concept']} with additional practice problems."
                for n in weak[:3]
            ] or ["Explore an advanced course on this topic."],
            "completion_badge": "Completed",
        }

    # Ensure skill verdicts are also saved/updated in student_skills with course tag
    for verdict in (report.get("skill_verdicts") or []):
        concept = verdict.get("concept")
        status = verdict.get("status", "learning")
        score = float(verdict.get("mastery_score") or 0)
        if concept:
            try:
                await upsert_student_skill(
                    student_id=student_id,
                    concept=concept,
                    mastery_score=score,
                    depth_score=max(0.0, score - 0.1),
                    source="course_report",
                    status=status,
                    evidence={"course_id": course_id, "report": True},
                )
            except Exception as exc:
                logger.warning("Skill upsert in course report failed for {}: {}", concept, exc)

    # Cache it
    await save_course_completion_report(course_id, student_id, report)
    return report
