"""
agents/base_agent.py
BaseAgent — inherited by all 5 EduMind agents.

Provides:
  build_tool()      — type-safe Groq tool spec builder
  run()             — calls tool_call_loop(), returns terminal args dict
  _execute_tool()   — override in subclass to handle non-terminal tools

Every subclass must define:
  NAME              — str, agent identifier for logs and decision_log
  TOOLS             — list[dict], built with build_tool()
  TERMINAL_TOOL     — str, name of the tool that exits the loop
  _execute_tool()   — handles all non-terminal tool calls
  run()             — builds system prompt + user message, calls super().run()
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from clients.groq_client import tool_call_loop
from core.student_model import StudentState
from config import settings


class BaseAgent:
    NAME: str = "base_agent"
    TOOLS: list[dict] = []
    TERMINAL_TOOL: str = ""

    def __init__(self, state: StudentState):
        self.state = state

    # ── Tool spec builder ─────────────────────────────────────────────────────

    @staticmethod
    def build_tool(
        name: str,
        description: str,
        properties: dict[str, dict],
        required: list[str],
    ) -> dict:
        """
        Build a Groq-compatible tool spec dict.

        Args:
            name:        tool function name (snake_case)
            description: what the LLM reads to decide when to call this tool
            properties:  {param_name: {"type": ..., "description": ..., "enum": [...]}}
            required:    list of required param names

        Example:
            build_tool(
                name="submit_evaluation",
                description="Submit final evaluation scores.",
                properties={
                    "correctness_score": {"type": "number", "description": "0.0-1.0"},
                    "depth_score":       {"type": "number", "description": "0.0-1.0"},
                },
                required=["correctness_score", "depth_score"],
            )
        """
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    # ── Tool executor (override in subclass) ──────────────────────────────────

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        """
        Execute a non-terminal tool call and return a result string.
        Subclasses override this to implement tool logic.

        Args:
            tool_name: name of the tool called by the LLM
            args:      parsed dict of tool arguments

        Returns:
            str result fed back to the LLM as the tool response
        """
        logger.warning("{}: unhandled tool '{}' — returning empty result", self.NAME, tool_name)
        return f"Tool '{tool_name}' not implemented in {self.NAME}."

    def _tool_executor_wrapper(self, tool_name: str, args_str: str) -> str:
        """Internal wrapper — parses args_str JSON before passing to _execute_tool."""
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {"raw": args_str}
        return self._execute_tool(tool_name, args)

    # ── Core run loop ─────────────────────────────────────────────────────────

    def run(
        self,
        system: str,
        user_message: str,
        context: str = "",
        model: str | None = None,
    ) -> dict:
        """
        Run the agentic loop for this agent.

        Args:
            system:       system prompt (agent rules + student context)
            user_message: trigger message for this agent's task
            context:      optional extra context appended to user_message
            model:        override model (defaults to reasoning_model)

        Returns:
            dict of terminal tool arguments
        """
        logger.info("▶ {} starting (student={})", self.NAME, self.state.student_id)

        result = tool_call_loop(
            system=system,
            user_message=user_message,
            tools=self.TOOLS,
            context=context,
            terminal_tool_name=self.TERMINAL_TOOL,
            model=model or settings.reasoning_model,
            tool_executor=self._tool_executor_wrapper,
        )

        logger.info("◀ {} finished → keys={}", self.NAME, list(result.keys()))
        return result

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _student_context(self) -> str:
        """Return student context string for injection into system prompts."""
        return self.state.as_prompt_context()

    def _log_decision(self, action: str, reason: str, payload: dict | None = None) -> None:
        """Record a decision to state.session_decisions (flushed to DB at session end)."""
        from core.student_model import AdaptationDecision
        decision = AdaptationDecision(
            action=action,
            reason=reason,
            metacognition_updates=payload or {},
        )
        self.state.add_decision(decision)
        logger.debug("{}: decision logged — action={}", self.NAME, action)

    def _current_module(self):
        """Return the current curriculum module or None."""
        if self.state.curriculum is None:
            return None
        idx = self.state.curriculum.current_index
        modules = self.state.curriculum.modules
        if idx >= len(modules):
            return None
        return modules[idx]
