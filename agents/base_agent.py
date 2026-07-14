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
import time
import functools
from typing import Any

from loguru import logger

from clients.groq_client import tool_call_loop
from core.student_model import StudentState
from config import settings


# ── Observability helpers ─────────────────────────────────────────────────────
# Lightweight built-in tracing — no external dependency required.
# Emits structured JSON trace records via loguru so they can be shipped to
# any log aggregator (ELK, Datadog, Loki). Each agent run gets a trace_id
# that groups all tool calls, token counts, and latencies in one record.

def _trace_agent_run(run_fn):
    """
    Decorator for BaseAgent.run().
    Records: agent name, student_id, wall-clock latency, tool call count,
    approximate token usage (from response usage field when available).
    Emits as a structured JSON log line at INFO level.
    """
    @functools.wraps(run_fn)
    async def wrapper(self, system, user_message, context="", model=None):
        """Run one traced agent call and emit its final trace record."""
        import uuid
        trace_id = str(uuid.uuid4())[:8]
        t0 = time.perf_counter()
        self._trace_id = trace_id
        self._tool_calls_this_run: list[dict] = []

        try:
            result = await run_fn(self, system, user_message, context, model)
        except Exception as exc:
            elapsed = round(time.perf_counter() - t0, 3)
            logger.opt(lazy=True).info(
                "TRACE agent={agent} trace={trace} student={student} "
                "status=error error={error} latency_s={lat}",
                agent=lambda: self.NAME,
                trace=lambda: trace_id,
                student=lambda: self.state.student_id,
                error=lambda: str(exc),
                lat=lambda: elapsed,
            )
            raise

        elapsed = round(time.perf_counter() - t0, 3)
        logger.info(
            "TRACE | agent={} trace={} student={} status=ok "
            "tool_calls={} latency_s={} tools={}",
            self.NAME,
            trace_id,
            self.state.student_id,
            len(self._tool_calls_this_run),
            elapsed,
            [t["name"] for t in self._tool_calls_this_run],
        )
        return result
    return wrapper


class BaseAgent:
    """Shared tool-loop, tracing, and tool-schema helpers for EduMind agents."""

    NAME: str = "base_agent"
    TOOLS: list[dict] = []
    TERMINAL_TOOL: str = ""

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Enforce that every concrete subclass declares a non-empty TERMINAL_TOOL.
        Catches the mistake at class-definition time, not at runtime inside a loop."""
        super().__init_subclass__(**kwargs)
        # Skip the check for abstract intermediate classes that don't set NAME
        if cls.NAME != "base_agent" and not cls.TERMINAL_TOOL:
            raise TypeError(
                f"{cls.__name__} must define a non-empty TERMINAL_TOOL class attribute. "
                f"Without it, tool_call_loop() will never exit."
            )

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

    async def _execute_tool(self, tool_name: str, args: dict) -> str:
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

    async def _tool_executor_wrapper(self, tool_name: str, args_input: dict | str | bytes | None) -> str:
        """Internal wrapper that normalises tool args before _execute_tool().

        groq_client.tool_call_loop() now parses JSON arguments before calling
        the executor. Older/direct callers may still pass a raw JSON string, so
        keep this tolerant instead of assuming exactly one representation.
        """
        if isinstance(args_input, dict):
            args = args_input
        elif args_input is None:
            args = {}
        else:
            raw = args_input.decode() if isinstance(args_input, bytes) else str(args_input)
            try:
                parsed = json.loads(raw)
                args = parsed if isinstance(parsed, dict) else {"raw": parsed}
            except (json.JSONDecodeError, TypeError):
                args = {"raw": raw}

        t0 = time.perf_counter()
        result = await self._execute_tool(tool_name, args)
        elapsed = round(time.perf_counter() - t0, 3)

        # Record into trace
        if hasattr(self, "_tool_calls_this_run"):
            self._tool_calls_this_run.append({
                "name": tool_name,
                "latency_s": elapsed,
                "result_len": len(result) if result else 0,
            })
        logger.debug(
            "TOOL | agent={} tool={} latency_s={} result_len={}",
            self.NAME, tool_name, elapsed, len(result) if result else 0,
        )
        return result

    # ── Core run loop ─────────────────────────────────────────────────────────

    @_trace_agent_run
    async def run(
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

        # Prepend tool-format instruction to EVERY agent system prompt.
        # This prevents models from using XML <function=...> format which
        # Groq rejects with a 400 tool_use_failed error.
        tool_format_instruction = (
            "CRITICAL TOOL CALLING RULES:\n"
            "- You MUST call tools using the structured JSON tool-call format ONLY.\n"
            "- NEVER use XML tags like <function=tool_name> or <function_calls>.\n"
            "- NEVER write function calls as plain text or code blocks.\n"
            "- Use ONLY the tool definitions provided. Do not invent tool names.\n"
            "- Always provide ALL required fields when calling a tool.\n\n"
        )
        full_system = tool_format_instruction + system

        result = await tool_call_loop(
            system=full_system,
            user_message=user_message,
            tools=self.TOOLS,
            context=context,
            terminal_tool_name=self.TERMINAL_TOOL,
            model=model or settings.reasoning_model,
            tool_executor=self._tool_executor_wrapper,
            _caller=self.NAME,
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
            agent=self.NAME,
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
