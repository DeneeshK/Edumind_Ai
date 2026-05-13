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
import asyncio
import inspect
from collections.abc import Generator
from typing import Any, Callable

from groq import Groq
from groq import RateLimitError, APITimeoutError, BadRequestError
from loguru import logger

from config import settings


# ── Custom exceptions ───────────────────────────────────────────────────

class GroqTimeoutError(Exception):
    pass


class GroqRateLimitError(Exception):
    pass


class GroqBadRequestError(Exception):
    pass


# ── Client singleton ────────────────────────────────────────────────────

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.groq_api_key)
    return _client


# ── Retry helper ────────────────────────────────────────────────────────

async def _with_retry(fn: Callable, *args, **kwargs) -> Any:
    """
    Call fn(*args, **kwargs) with up to 3 retries on 429 and 400 (malformed tool calls).
    Uses asyncio.sleep so the event loop is never blocked.
    Raises GroqTimeoutError on timeout, GroqRateLimitError/GroqBadRequestError if retries exhausted.
    """
    delays = [2, 5, 10, 20]  # exponential backoff seconds
    for attempt, delay in enumerate(delays, 1):
        try:
            return fn(*args, **kwargs)
        except APITimeoutError as e:
            raise GroqTimeoutError(
                f"Groq timed out after {settings.groq_timeout_seconds}s") from e
        except RateLimitError as e:
            if attempt == len(delays):
                raise GroqRateLimitError(
                    "Groq rate limit exceeded after 3 retries") from e
            logger.warning(
                "Rate limited by Groq (attempt {}). Retrying in {}s…",
                attempt,
                delay)
            await asyncio.sleep(delay)   # non-blocking — event loop stays free
        except BadRequestError as e:
            if attempt == len(delays):
                raise GroqBadRequestError(
                    f"Groq bad request (e.g. malformed JSON) after 3 retries: {e}") from e
            logger.warning(
                "Bad request from Groq (attempt {}). Likely a malformed tool call. Retrying in {}s…",
                attempt,
                delay)
            await asyncio.sleep(delay)
        except Exception:
            raise


# ── generate() ───────────────────────────────────────────────────────────────

async def generate(
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

    response = await _with_retry(_call)
    return response.choices[0].message.content


# ── tool_call_loop() ────────────────────────────────────────────────────

async def tool_call_loop(
    system: str,
    user_message: str,
    tools: list[dict],
    context: str = "",
    terminal_tool_name: str = "",
    model: str | None = None,
    tool_executor: Callable[[str, dict], Any] | None = None,
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
        tool_executor:      fn(tool_name, args_dict) -> result_str
                            if None, tools are recorded but not executed
    """
    client = get_client()
    model = model or settings.reasoning_model

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message +
            ("\n\n" + context if context else "")},
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

        response = await _with_retry(_call)
        message = response.choices[0].message

        # ── LLM chose to call tool(s) ────────────────────────────────────────
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
                logger.info("LLM called tool: {} args: {}",
                            name, args_str[:120])

                # Parse args_str to dict — with safe fallback for malformed JSON.
                # Strategy: fix known Python-literal issues first (True/False/None),
                # then parse. Never use regex to strip structure — that
                # corrupts valid args.
                args_dict: dict = {}
                try:
                    import re as _re
                    cleaned = _re.sub(r'\bTrue\b', 'true', args_str)
                    cleaned = _re.sub(r'\bFalse\b', 'false', cleaned)
                    cleaned = _re.sub(r'\bNone\b', 'null', cleaned)
                    # If the model wrapped the JSON in an outer array, unwrap
                    # it
                    cleaned = cleaned.strip()
                    if cleaned.startswith('[') and cleaned.endswith(']'):
                        inner = cleaned[1:-1].strip()
                        if inner.startswith('{'):
                            cleaned = inner
                    args_dict = json.loads(cleaned)
                    if not isinstance(args_dict, dict):
                        logger.warning(
                            "Tool '{}' args parsed to non-dict ({}), using {{}}",
                            name, type(args_dict).__name__
                        )
                        args_dict = {}
                except json.JSONDecodeError as je:
                    logger.warning(
                        "Tool '{}' args JSON parse failed: {} — raw: {}",
                        name, je, args_str[:120]
                    )
                    args_dict = {}

                # Check for terminal tool BEFORE executing
                if name == terminal_tool_name:
                    # Strip to only fields defined in the terminal tool schema
                    terminal_tool_schema = next(
                        (t for t in tools if t["function"]
                         ["name"] == name), None
                    )
                    if terminal_tool_schema:
                        allowed = set(
                            terminal_tool_schema["function"]["parameters"]
                            .get("properties", {}).keys()
                        )
                        terminal_result = {
                            k: v for k, v in args_dict.items() if k in allowed}
                    else:
                        terminal_result = args_dict
                    logger.info(
                        "Terminal tool '{}' called — exiting loop.", name)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "done",
                    })
                    return terminal_result

                # Execute non-terminal tool — pass parsed dict, not raw string
                if tool_executor:
                    try:
                        result = tool_executor(name, args_dict)
                        if inspect.isawaitable(result):
                            result = await result
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

    logger.warning(
        "tool_call_loop hit max_iterations ({}) — returning empty.",
        max_iterations)
    return {}


# ── stream() ────────────────────────────────────────────────────────────

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
