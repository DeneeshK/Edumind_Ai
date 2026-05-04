"""
clients/groq_client.py
Three methods:
  generate()        — single completion, returns str
  tool_call_loop()  — agentic loop, LLM decides tools, returns terminal tool args
  stream()          — streaming generator, yields str chunks

Retry: 3x exponential backoff on 429 only
Timeout: 30s, raises GroqTimeoutError
"""

from __future__ import annotations

import json
import time
from collections.abc import Generator
from typing import Any, Callable

from groq import Groq
from groq import RateLimitError, APITimeoutError
from loguru import logger

from config import settings


# ── Custom exceptions ─────────────────────────────────────────────────────────

class GroqTimeoutError(Exception):
    pass

class GroqRateLimitError(Exception):
    pass


# ── Client singleton ──────────────────────────────────────────────────────────

_client: Groq | None = None

def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.groq_api_key)
    return _client


# ── Retry helper ──────────────────────────────────────────────────────────────

def _with_retry(fn: Callable, *args, **kwargs) -> Any:
    """
    Call fn(*args, **kwargs) with up to 3 retries on 429.
    Raises GroqTimeoutError on timeout, GroqRateLimitError if retries exhausted.
    """
    delays = [1, 2, 4]  # exponential backoff seconds
    for attempt, delay in enumerate(delays, 1):
        try:
            return fn(*args, **kwargs)
        except APITimeoutError as e:
            raise GroqTimeoutError(f"Groq timed out after {settings.groq_timeout_seconds}s") from e
        except RateLimitError as e:
            if attempt == len(delays):
                raise GroqRateLimitError("Groq rate limit exceeded after 3 retries") from e
            logger.warning("Rate limited by Groq (attempt {}). Retrying in {}s…", attempt, delay)
            time.sleep(delay)
        except Exception as e:
            raise


# ── generate() ───────────────────────────────────────────────────────────────

def generate(
    messages: list[dict],
    model: str | None = None,
    system: str | None = None,
) -> str:
    """
    Single completion. Returns the assistant's text response.
    
    Args:
        messages: list of {role, content} dicts (user/assistant turns)
        model: override model (defaults to generation_model)
        system: optional system prompt prepended to messages
    """
    client = get_client()
    model = model or settings.generation_model

    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    def _call():
        return client.chat.completions.create(
            model=model,
            messages=full_messages,
            timeout=settings.groq_timeout_seconds,
        )

    response = _with_retry(_call)
    return response.choices[0].message.content


# ── tool_call_loop() ──────────────────────────────────────────────────────────

def tool_call_loop(
    system: str,
    user_message: str,
    tools: list[dict],
    context: str = "",
    terminal_tool_name: str = "",
    model: str | None = None,
    tool_executor: Callable[[str, str], Any] | None = None,
) -> dict:
    """
    The main agentic loop. Python orchestrates, LLM decides.

    - LLM sees tools and decides which to call and with what args
    - Python executes the chosen tool and feeds result back
    - Loop continues until LLM calls terminal_tool_name or gives a text response
    - Returns the terminal tool's arguments as a dict

    Args:
        system:             system prompt (contains student context + rules)
        user_message:       the trigger message for this agent
        tools:              list of Groq tool dicts (built by base_agent)
        context:            extra context appended to user_message
        terminal_tool_name: when LLM calls this tool, loop exits
        model:              override model (defaults to reasoning_model)
        tool_executor:      fn(tool_name, args_json_str) -> result_str
                            if None, tools are recorded but not executed
    """
    client = get_client()
    model = model or settings.reasoning_model

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message + ("\n\n" + context if context else "")},
    ]

    terminal_result: dict = {}
    max_iterations = 20  # safety cap — prevent infinite loops

    for iteration in range(max_iterations):
        logger.debug("tool_call_loop iteration {}", iteration + 1)

        def _call():
            return client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                timeout=settings.groq_timeout_seconds,
            )

        response = _with_retry(_call)
        message = response.choices[0].message

        # ── LLM chose to call tool(s) ─────────────────────────────────────────
        if message.tool_calls:
            # Append assistant message with tool_calls
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            })

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args_str = tool_call.function.arguments
                logger.info("LLM called tool: {} args: {}", name, args_str[:120])

                # Check for terminal tool BEFORE executing
                if name == terminal_tool_name:
                    try:
                        terminal_result = json.loads(args_str)
                    except json.JSONDecodeError:
                        terminal_result = {"raw": args_str}
                    logger.info("Terminal tool '{}' called — exiting loop.", name)
                    # Append a tool result so conversation stays valid
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "done",
                    })
                    return terminal_result

                # Execute non-terminal tool
                if tool_executor:
                    try:
                        result = tool_executor(name, args_str)
                        result_str = str(result)
                    except Exception as e:
                        result_str = f"ERROR: {e}"
                        logger.warning("Tool '{}' raised: {}", name, e)
                else:
                    result_str = f"Tool '{name}' executed (no executor provided)."

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                })

        # ── LLM chose to respond with text (no tool call) ────────────────────
        else:
            logger.info("LLM gave text response — loop ends.")
            return {"text_response": message.content}

    logger.warning("tool_call_loop hit max_iterations ({}) — returning empty.", max_iterations)
    return {}


# ── stream() ──────────────────────────────────────────────────────────────────

def stream(
    messages: list[dict],
    model: str | None = None,
    system: str | None = None,
) -> Generator[str, None, None]:
    """
    Streaming completion. Yields text chunks as they arrive.
    Used by the Tutor agent to stream lesson content token-by-token.

    Args:
        messages: list of {role, content} dicts
        model:    override model (defaults to generation_model)
        system:   optional system prompt
    """
    client = get_client()
    model = model or settings.generation_model

    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=full_messages,
            stream=True,
            timeout=settings.groq_timeout_seconds,
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    except APITimeoutError as e:
        raise GroqTimeoutError("Groq stream timed out") from e
    except RateLimitError:
        raise GroqRateLimitError("Groq rate limited during stream")
