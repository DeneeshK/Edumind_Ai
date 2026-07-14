"""
agents/institution/teacher_assistant_agent.py

Conversational assistant for teachers, grounded in live classroom analytics.
Uses the existing groq_client.tool_call_loop: the model decides which
read-only analytics tools to call, then answers through the terminal tool.

The assistant only ever reads data and drafts content (homework, lesson
plans). It never publishes, assigns, or modifies anything — the teacher does
that explicitly in the UI.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from clients.groq_client import tool_call_loop
from config import settings
from core import classroom_analytics as analytics


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    """Groq tool spec builder (same shape as BaseAgent.build_tool)."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOLS = [
    _tool(
        "get_class_overview",
        "Headline KPIs: average score, progress, mastery, active students, weekly score trend.",
        {}, [],
    ),
    _tool(
        "get_students_table",
        "Per-student metrics: progress, mastery, test average, doubts, activity, risk flag, rank.",
        {}, [],
    ),
    _tool(
        "get_concept_heatmap",
        "Class-average mastery per concept — identifies weakest and strongest concepts.",
        {}, [],
    ),
    _tool(
        "get_doubts",
        "Most frequently asked doubt concepts and recent doubt text samples.",
        {}, [],
    ),
    _tool(
        "get_student_detail",
        "Deep dive on one student: weak/strong concepts, test history, misconceptions, course progress.",
        {"student_id": {"type": "string", "description": "The student's id from get_students_table"}},
        ["student_id"],
    ),
    _tool(
        "final_answer",
        "Deliver the final reply to the teacher. Call this exactly once when ready.",
        {
            "answer_markdown": {
                "type": "string",
                "description": "The complete answer in markdown. Cite real numbers/names from tools.",
            },
            "suggested_actions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "0-3 short follow-up actions the teacher could take next.",
            },
        },
        ["answer_markdown"],
    ),
]

_SYSTEM_TEMPLATE = """
You are EduMind's teaching assistant for the classroom "{classroom_name}"
(subject: {subject}). You help the teacher understand their class and prepare
teaching material.

CRITICAL TOOL CALLING RULES:
- You MUST call tools using the structured JSON tool-call format ONLY.
- NEVER use XML tags like <function=tool_name> or plain-text function calls.
- Use ONLY the provided tool names.

How to work:
1. For questions about the class (who needs help, weakest chapter, who hasn't
   finished, rankings, engagement), ALWAYS call the relevant analytics tools
   first — never guess.
2. For drafting requests (homework, lesson plans, practice questions,
   assignments), first check the analytics that make the draft relevant
   (e.g. weakest concepts), then write the full draft in your final answer.
3. Ground every claim in tool data. Cite names and numbers.
4. Keep answers concise and skimmable: short paragraphs, bullet lists,
   bold key names/numbers.
5. Finish by calling final_answer exactly once with the complete reply.
"""


async def ask_teacher_assistant(
    *,
    classroom_id: str,
    classroom_name: str,
    subject: str,
    question: str,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Answer one teacher question with analytics-grounded tools.
    Returns {"answer_markdown", "suggested_actions", "tool_trace"}.
    """
    trace: list[dict[str, Any]] = []

    async def _execute(tool_name: str, args: dict) -> str:
        """Run one analytics tool and return its JSON result for the model."""
        try:
            if tool_name == "get_class_overview":
                result: Any = await analytics.overview(classroom_id)
            elif tool_name == "get_students_table":
                result = await analytics.student_table(classroom_id)
            elif tool_name == "get_concept_heatmap":
                heatmap = await analytics.concept_heatmap(classroom_id, max_concepts=15)
                result = {
                    "concepts": [
                        {"concept": heatmap["concepts"][i], "class_avg": heatmap["class_avg"][i]}
                        for i in range(len(heatmap.get("concepts") or []))
                    ]
                }
            elif tool_name == "get_doubts":
                result = await analytics.doubt_analytics(classroom_id)
            elif tool_name == "get_student_detail":
                result = await analytics.student_drilldown(
                    classroom_id, str(args.get("student_id") or "")
                )
                if not result:
                    result = {"error": "student not found in this classroom"}
            else:
                return f"Unknown tool '{tool_name}'."
        except Exception as exc:
            logger.error("Teacher assistant tool {} failed: {}", tool_name, exc)
            return json.dumps({"error": str(exc)})

        trace.append({"tool": tool_name, "args": args})
        return json.dumps(result, default=str)[:9000]

    context = ""
    if history:
        turns = [
            f"{'Teacher' if m['role'] == 'user' else 'Assistant'}: {m['message'][:500]}"
            for m in history[-6:]
        ]
        context = "Recent conversation:\n" + "\n".join(turns)

    result = await tool_call_loop(
        system=_SYSTEM_TEMPLATE.format(
            classroom_name=classroom_name, subject=subject or "general"
        ),
        user_message=question,
        tools=TOOLS,
        context=context,
        terminal_tool_name="final_answer",
        model=settings.reasoning_model,
        tool_executor=_execute,
        _caller="teacher_assistant",
    )

    answer = str(result.get("answer_markdown") or "").strip()
    if not answer:
        answer = (
            "I couldn't put together an answer this time — please rephrase the "
            "question or try again."
        )
    actions = [
        str(a).strip()[:160]
        for a in (result.get("suggested_actions") or [])
        if str(a).strip()
    ][:3]
    return {"answer_markdown": answer, "suggested_actions": actions, "tool_trace": trace}
