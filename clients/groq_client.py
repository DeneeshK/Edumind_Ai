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
import ast
import re
import time
from collections.abc import AsyncGenerator, Generator
from typing import Any, Callable

from groq import Groq
from groq import RateLimitError, APITimeoutError
from loguru import logger

from config import settings, compute_llm_cost_usd


# ── Custom exceptions ─────────────────────────────────────────────────────────

class GroqTimeoutError(Exception):
    """Raised when a Groq completion or stream exceeds the configured timeout."""

    pass

class GroqRateLimitError(Exception):
    """Raised after Groq rate-limit retries are exhausted."""

    pass


def _parse_tool_args(raw: str) -> dict | list:
    """Parse tool args from model output, tolerating common non-JSON slips."""
    cleaned = raw.strip()
    if cleaned.startswith("="):
        cleaned = cleaned[1:].strip()
    if cleaned.endswith(">"):
        cleaned = cleaned[:-1].strip()

    cleaned = re.sub(r"\bTrue\b", "true", cleaned)
    cleaned = re.sub(r"\bFalse\b", "false", cleaned)
    cleaned = re.sub(r"\bNone\b", "null", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(raw.strip())
            return parsed if isinstance(parsed, (dict, list)) else {}
        except (ValueError, SyntaxError):
            logger.warning("Tool args parse failed ({} chars).", len(raw))
            return {}


def _extract_failed_generation(exc: Exception) -> str:
    """Groq 400 tool_use_failed responses include the rejected generation."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        failed = err.get("failed_generation")
        if failed:
            return str(failed)

    text = str(exc)
    start = text.find("{'error':")
    if start != -1:
        try:
            payload = ast.literal_eval(text[start:])
            failed = payload.get("error", {}).get("failed_generation")
            if failed:
                return str(failed)
        except (ValueError, SyntaxError):
            pass

    return ""


def _recover_tool_call_from_text(text: str) -> tuple[str, dict | list] | None:
    """Recover XML/plain-text tool calls emitted by a model."""
    xml_match = re.search(
        r"<function[=\s]+(?P<name>\w+)\s*(?:=|>)?\s*(?P<args>.*?)(?:</function>|$)",
        text,
        re.DOTALL,
    )
    if not xml_match:
        xml_match = re.search(
            r"function[=\s]+(?P<name>\w+)[^{\[]*(?P<args>[\[{].*)",
            text,
            re.DOTALL,
        )
    if not xml_match:
        return None

    name = xml_match.group("name")
    args = _parse_tool_args(xml_match.group("args"))
    return name, args


# ── Token + cost accounting ───────────────────────────────────────────────────

def _record_usage(
    model: str,
    caller: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    estimated: bool = False,
) -> float | None:
    """Record token counts (and cost when priced) to Prometheus.

    Increments edumind_llm_tokens_total for both directions and, when the model
    has a non-zero price in config.GROQ_MODEL_PRICES, edumind_llm_cost_usd_total.
    Returns the estimated USD cost (or None if the model is unpriced) so the
    caller can also attach it as a span attribute. Never raises — metrics must
    not break an LLM call.
    """
    from core.metrics import metrics as _m

    try:
        if prompt_tokens:
            _m.llm_tokens.labels(
                model=model, caller=caller, direction="prompt"
            ).inc(prompt_tokens)
        if completion_tokens:
            _m.llm_tokens.labels(
                model=model, caller=caller, direction="completion"
            ).inc(completion_tokens)

        cost = compute_llm_cost_usd(model, prompt_tokens, completion_tokens)
        if cost is not None:
            _m.llm_cost_usd.labels(model=model, caller=caller).inc(cost)
        logger.debug(
            "USAGE | model={} caller={} prompt={} completion={}{} cost_usd={}",
            model, caller, prompt_tokens, completion_tokens,
            " (est)" if estimated else "", cost,
        )
        return cost
    except Exception as exc:  # pragma: no cover - defensive; metrics never fatal
        logger.debug("Token/cost recording failed for {}: {}", model, exc)
        return None


def _extract_usage(response: Any) -> tuple[int, int]:
    """Pull (prompt_tokens, completion_tokens) from a Groq response.usage.

    Groq's CompletionUsage carries prompt_tokens/completion_tokens. Returns
    (0, 0) when usage is missing so callers can no-op cleanly.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    return prompt, completion


# ── Client singleton ──────────────────────────────────────────────────────────

_client: Groq | None = None

def get_client() -> Groq:
    """Return the process-wide Groq SDK client."""
    global _client
    if _client is None:
        _client = Groq(api_key=settings.groq_api_key)
    return _client


# ── Retry helper ──────────────────────────────────────────────────────────────

async def _with_retry(fn: Callable, *args, **kwargs) -> Any:
    """
    Call fn(*args, **kwargs) with configurable retries on 429.
    Uses asyncio.to_thread so the blocking Groq HTTP call never
    freezes the event loop. Uses asyncio.sleep for backoff.
    """
    max_retries = max(0, int(getattr(settings, "groq_max_retries", 3)))
    total_attempts = max_retries + 1

    for attempt in range(1, total_attempts + 1):
        try:
            # to_thread runs the blocking SDK call in a thread pool
            # so the asyncio event loop stays free for other sessions
            return await asyncio.to_thread(fn, *args, **kwargs)
        except APITimeoutError as e:
            raise GroqTimeoutError(
                f"Groq timed out after {settings.groq_timeout_seconds}s"
            ) from e
        except RateLimitError as e:
            if attempt == total_attempts:
                raise GroqRateLimitError(
                    f"Groq rate limit exceeded after {max_retries} retries"
                ) from e
            delay = min(2 ** (attempt - 1), 30)
            logger.warning(
                "Rate limited by Groq (retry {}/{}). Retrying in {}s…",
                attempt, max_retries, delay
            )
            from core.metrics import metrics as _m
            _m.llm_retries.labels(model="groq").inc()
            await asyncio.sleep(delay)
        except Exception:
            raise


# ── generate() ───────────────────────────────────────────────────────────────

async def generate(
    messages: list[dict],
    model: str | None = None,
    system: str | None = None,
    json_mode: bool = False,
    max_tokens: int | None = None,
    _caller: str = "unknown",
) -> str:
    """
    Single completion. Returns the assistant's text response.

    Args:
        messages:   list of {role, content} dicts (user/assistant turns)
        model:      override model (defaults to generation_model)
        system:     optional system prompt prepended to messages
        json_mode:  if True, sets response_format={"type":"json_object"}
        max_tokens: optional output token cap
        _caller:    label for metrics (e.g. "curriculum_architect", "tutor", "evaluation")
    """
    import time as _time
    from core.metrics import metrics as _m

    client = get_client()
    model = model or settings.generation_model

    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    def _call():
        """Perform one Groq chat completion request for plain generation."""
        kwargs: dict = {
            "model": model,
            "messages": full_messages,
            "timeout": settings.groq_timeout_seconds,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return client.chat.completions.create(**kwargs)

    _m.llm_requests.labels(model=model, caller=_caller).inc()
    _start = _time.perf_counter()
    try:
        response = await _with_retry(_call)
        _m.llm_latency.labels(model=model).observe(_time.perf_counter() - _start)
        prompt_tokens, completion_tokens = _extract_usage(response)
        _record_usage(model, _caller, prompt_tokens, completion_tokens)
        return response.choices[0].message.content
    except GroqTimeoutError:
        _m.llm_errors.labels(model=model, error_type="timeout").inc()
        raise
    except GroqRateLimitError:
        _m.llm_errors.labels(model=model, error_type="rate_limit").inc()
        raise
    except Exception:
        _m.llm_errors.labels(model=model, error_type="other").inc()
        raise


# ── tool_call_loop() ──────────────────────────────────────────────────────────

async def tool_call_loop(
    system: str,
    user_message: str,
    tools: list[dict],
    context: str = "",
    terminal_tool_name: str = "",
    model: str | None = None,
    tool_executor: Callable[[str, dict], Any] | None = None,
    _caller: str = "unknown",
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
        _caller:            label for token/cost metrics (usually the agent NAME)
    """
    client = get_client()
    model = model or settings.reasoning_model

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message + ("\n\n" + context if context else "")},
    ]

    terminal_result: dict = {}
    max_iterations = 20  # safety cap — prevent infinite loops

    def _filter_terminal_args(tool_name: str, args_obj: dict | list) -> dict:
        """Keep only arguments declared by the terminal tool schema."""
        args_dict = args_obj if isinstance(args_obj, dict) else {}
        terminal_tool_schema = next(
            (t for t in tools if t["function"]["name"] == tool_name), None
        )
        if terminal_tool_schema:
            allowed = set(
                terminal_tool_schema["function"]["parameters"]
                .get("properties", {}).keys()
            )
            return {k: v for k, v in args_dict.items() if k in allowed}
        return args_dict

    async def _execute_non_terminal(tool_name: str, args_obj: dict | list) -> str:
        """Execute one or more non-terminal tool calls and stringify their results."""
        if not tool_executor:
            return f"Tool '{tool_name}' executed (no executor provided)."

        arg_items = args_obj if isinstance(args_obj, list) else [args_obj]
        results = []
        for idx, item in enumerate(arg_items, 1):
            if not isinstance(item, dict):
                results.append(f"Skipped non-object args at index {idx}.")
                continue
            try:
                result = tool_executor(tool_name, item)
                if inspect.isawaitable(result):
                    result = await result
                results.append(str(result))
            except Exception as e:
                result_str = f"ERROR: {e}"
                logger.warning("Tool '{}' raised: {}", tool_name, e)
                results.append(result_str)
        return "\n".join(results)

    for iteration in range(max_iterations):
        logger.debug("tool_call_loop iteration {}", iteration + 1)

        def _call():
            """Perform one Groq chat completion request for the tool loop."""
            return client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="required",   # force tool call — prevents free-text XML responses
                timeout=settings.groq_timeout_seconds,
            )

        try:
            response = await _with_retry(_call)
        except Exception as exc:
            failed_generation = _extract_failed_generation(exc)
            recovered = (
                _recover_tool_call_from_text(failed_generation)
                if failed_generation else None
            )
            if not recovered:
                raise

            recovered_name, recovered_args = recovered
            logger.warning(
                "Recovered malformed Groq tool generation: tool='{}' args_type='{}'",
                recovered_name, type(recovered_args).__name__,
            )

            if recovered_name == terminal_tool_name:
                logger.info(
                    "Recovered terminal tool '{}' from failed_generation.",
                    recovered_name,
                )
                return _filter_terminal_args(recovered_name, recovered_args)

            result_str = await _execute_non_terminal(recovered_name, recovered_args)
            messages.append({
                "role": "user",
                "content": (
                    "Recovered and executed malformed tool call "
                    f"'{recovered_name}'. Tool result:\n{result_str}\n\n"
                    "Continue. Use the structured JSON tool-call format only."
                ),
            })
            continue

        prompt_tokens, completion_tokens = _extract_usage(response)
        _record_usage(model, _caller, prompt_tokens, completion_tokens)

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

                # Parse args_str to dict — with safe fallback for malformed JSON.
                # Strategy: fix known Python-literal issues first (True/False/None),
                # then parse. Never use regex to strip structure — that corrupts valid args.
                args_obj = _parse_tool_args(args_str)
                arg_keys = list(args_obj.keys()) if isinstance(args_obj, dict) else []
                logger.info(
                    "LLM called tool: {} args_type={} arg_keys={}",
                    name,
                    type(args_obj).__name__,
                    arg_keys,
                )

                # Check for terminal tool BEFORE executing
                if name == terminal_tool_name:
                    # Strip to only fields defined in the terminal tool schema
                    terminal_result = _filter_terminal_args(name, args_obj)
                    logger.info("Terminal tool '{}' called — exiting loop.", name)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "done",
                    })
                    return terminal_result

                # Execute non-terminal tool — pass parsed dict, not raw string
                result_str = await _execute_non_terminal(name, args_obj)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                })

        # ── LLM chose to respond with text (no tool call) ────────────────────
        else:
            text_content = message.content or ""

            recovered = _recover_tool_call_from_text(text_content)
            if recovered:
                recovered_name, recovered_args = recovered
                logger.warning(
                    "LLM used XML tool format instead of JSON tool calls. "
                    "Recovering: tool='{}' args_type='{}'",
                    recovered_name, type(recovered_args).__name__,
                )
                if recovered_name == terminal_tool_name:
                    logger.info("Recovered terminal tool '{}' from XML format.", recovered_name)
                    return _filter_terminal_args(recovered_name, recovered_args)
                result_str = await _execute_non_terminal(recovered_name, recovered_args)
                messages.append({"role": "assistant", "content": ""})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Tool result for recovered '{recovered_name}':\n"
                        f"{result_str}\n\nContinue using structured JSON tool calls."
                    ),
                })
                continue  # re-enter loop

            logger.info("LLM gave text response — loop ends.")
            return {"text_response": text_content}

    logger.warning("tool_call_loop hit max_iterations ({}) — returning empty.", max_iterations)
    return {}


# ── stream() ──────────────────────────────────────────────────────────────────

async def stream(
    messages: list[dict],
    model: str | None = None,
    system: str | None = None,
    _caller: str = "unknown",
) -> AsyncGenerator[str, None]:
    """
    Async streaming completion. Yields text chunks as they arrive.

    The blocking SDK call AND the entire chunk iteration run inside
    asyncio.to_thread via a queue bridge — the event loop is never
    blocked at any point, not even during per-chunk iteration.

    Args:
        messages: list of {role, content} dicts
        model:    override model (defaults to generation_model)
        system:   optional system prompt
        _caller:  label for token/cost metrics (e.g. "lesson", "worked_example")

    Note on usage: groq==0.9.0's completions.create() does not accept
    stream_options={"include_usage": True}, so the server never returns an exact
    token count for streamed calls. Completion tokens are therefore ESTIMATED as
    len(streamed_text)//4 and recorded with that caveat (see _record_usage's
    estimated flag / the gen_ai.usage.is_estimate span attribute). Prompt tokens
    are not counted for streams.
    """
    client = get_client()
    model = model or settings.generation_model

    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    # Bridge: thread puts chunks here; async generator reads them
    chunk_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=256)
    loop = asyncio.get_event_loop()

    def _blocking_stream_to_queue():
        """Run entirely in a thread — no event loop access from here."""
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
                    # thread-safe put into the asyncio queue
                    asyncio.run_coroutine_threadsafe(
                        chunk_queue.put(delta.content), loop
                    ).result()
        except APITimeoutError as e:
            asyncio.run_coroutine_threadsafe(
                chunk_queue.put(None), loop
            ).result()
            raise GroqTimeoutError("Groq stream timed out") from e
        except RateLimitError as e:
            asyncio.run_coroutine_threadsafe(
                chunk_queue.put(None), loop
            ).result()
            raise GroqRateLimitError("Groq rate limited during stream") from e
        finally:
            # Always signal completion
            asyncio.run_coroutine_threadsafe(
                chunk_queue.put(None), loop
            ).result()

    # Start streaming in a thread; don't await — let it run alongside us
    thread_future = asyncio.to_thread(_blocking_stream_to_queue)
    task = asyncio.create_task(thread_future)

    streamed_chars = 0
    try:
        while True:
            chunk = await asyncio.wait_for(chunk_queue.get(), timeout=settings.groq_timeout_seconds + 5)
            if chunk is None:
                break
            streamed_chars += len(chunk)
            yield chunk
    except asyncio.TimeoutError:
        logger.warning("stream(): chunk_queue timed out — cancelling stream task")
        task.cancel()
        raise GroqTimeoutError("Groq stream timed out waiting for chunk")
    finally:
        # Estimate completion tokens from streamed length — the SDK gives us no
        # exact count for streams (see the docstring). Prompt tokens unknown → 0.
        if streamed_chars:
            _record_usage(model, _caller, 0, streamed_chars // 4, estimated=True)
        # Ensure the thread task is awaited to avoid ResourceWarning
        try:
            await task
        except (GroqTimeoutError, GroqRateLimitError, asyncio.CancelledError):
            pass
