"""
prompts/evaluation.py
Registered evaluation-agent prompts (live post-lesson evaluation flow).

Moved verbatim from agents/evaluation_agent.py. The evaluation prompts are JSON
payloads assembled in code; what is versioned here is the authored `system` prompt
and the natural-language `instructions` block of each payload — the parts that
carry the semantics. Payload assembly (module context, lesson excerpt, schema)
stays at the call site. See prompts/README.md for the versioning rule.
"""

from __future__ import annotations

from prompts.base import PromptArtifact, register

EVALUATION_SYSTEM = register(PromptArtifact(
    name="evaluation_system",
    version=1,
    description="System prompt for every evaluation-agent LLM call.",
    template=(
        "You are EduMind's evaluation agent. Return STRICT JSON only. No markdown. "
        "Your job: assess the student's understanding fairly and help them improve. "
        "Be specific, honest, encouraging. Never invent facts outside the lesson content provided."
    ),
))

DIAGNOSE_INSTRUCTIONS = register(PromptArtifact(
    name="evaluation_diagnose_instructions",
    version=1,
    description="Instructions block for diagnose_student_answer.",
    template=(
        "Diagnose the answer fairly. "
        "Identify exactly what the student got right and what was weak, vague, or missing. "
        "Be like a good interviewer: notice what they DON'T say, not just what they say wrong. "
        "If the answer is vague on a concept, flag it in vague_parts. "
        "If the answer shows a misconception, flag it in suspicious_parts. "
        "Return JSON only."
    ),
))

PROBE_INSTRUCTIONS = register(PromptArtifact(
    name="evaluation_probe_instructions",
    version=1,
    description="Instructions block for targeted probe-question generation.",
    template=(
        "Generate ONE targeted follow-up question that probes EXACTLY the gap detected. "
        "Like a skilled interviewer: you noticed something the student doesn't fully understand — "
        "ask them to explain that specific thing. "
        "The question must be answerable from the lesson content only. "
        "Do NOT repeat the previous question. "
        "Do NOT ask a broad question — narrow down to the exact weak spot. "
        "Return JSON only."
    ),
))

FINALIZE_INSTRUCTIONS = register(PromptArtifact(
    name="evaluation_finalize_instructions",
    version=1,
    description="Instructions block for the final evaluation report.",
    template=(
        "Generate the final evaluation report. "
        "motivational_feedback: 2-3 sentences. Be honest and SPECIFIC — mention actual things the student got right AND what was weak. "
        "No generic praise. No demotivating language. "
        "If probe questions were asked, explain what gap was found and whether the student clarified it. "
        "transition_feedback: 1-2 sentences for the NEXT module — what will be adjusted and why. "
        "decision: one of the five options — use the computed mastery score and threshold to guide this. "
        "reteach_data: only populate if decision is RETEACH_WEAK_CONCEPTS or REPEAT_MODULE. "
        "adaptation_summary: max 3 bullet points for future lesson generation. "
        "Return JSON only."
    ),
))
