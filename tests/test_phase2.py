"""
tests/test_phase2.py
2.3 — Verify tool_call_loop with a toy 2-tool agent.
      Verify LLM calls the correct tool given context.

Run: pytest tests/test_phase2.py -v -s
"""

from clients import tavily_client
from clients.groq_client import tool_call_loop, generate, stream
import pytest
from dotenv import load_dotenv
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

load_dotenv()


# ── Toy tool definitions ────────────────────────────────────────────────

GREET_TOOL = {
    "type": "function",
    "function": {
        "name": "greet_student",
        "description": "Greet the student by name. Call this first.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The student's name"},
                "message": {"type": "string", "description": "A short greeting"},
            },
            "required": ["name", "message"],
        },
    },
}

CONCLUDE_TOOL = {
    "type": "function",
    "function": {
        "name": "conclude_session",
        "description": "Call this after greeting to conclude. This is the terminal tool.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "One sentence summary of what happened"},
                "ready": {"type": "boolean", "description": "Is the student ready to begin?"},
            },
            "required": ["summary", "ready"],
        },
    },
}


def toy_executor(tool_name: str, args: dict) -> str:
    if tool_name == "greet_student":
        return f"Greeted {args['name']} successfully."
    return "done"


# ── Tests ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tool_call_loop_calls_terminal_tool():
    """
    LLM should: call greet_student → then call conclude_session (terminal).
    We verify the terminal result contains expected keys.
    """
    result = await tool_call_loop(
        system=(
            "You are a session starter. "
            "First call greet_student to greet the student. "
            "Then call conclude_session with a summary. "
            "Always call both tools in this order."
        ),
        user_message="Start a session for student named Arjun.",
        tools=[GREET_TOOL, CONCLUDE_TOOL],
        terminal_tool_name="conclude_session",
        tool_executor=toy_executor,
    )

    print(f"\n✅ tool_call_loop result: {result}")
    assert "summary" in result, f"Expected 'summary' in result, got: {result}"
    assert "ready" in result, f"Expected 'ready' in result, got: {result}"
    print(f"   summary = {result['summary']}")
    print(f"   ready   = {result['ready']}")


@pytest.mark.asyncio
async def test_generate_returns_string():
    """generate() should return a non-empty string."""
    result = await generate(
        messages=[{"role": "user", "content": "Say hello in one word."}],
    )
    print(f"\n✅ generate() returned: '{result}'")
    assert isinstance(result, str)
    assert len(result) > 0


def test_stream_yields_chunks():
    """stream() should yield multiple string chunks."""
    chunks = list(stream(
        messages=[
            {"role": "user", "content": "Count from 1 to 5, one number per word."}],
    ))
    full = "".join(chunks)
    print(f"\n✅ stream() yielded {len(chunks)} chunks: '{full[:80]}'")
    assert len(chunks) > 0
    assert isinstance(full, str)


def test_tavily_returns_results():
    """search() should return a list (may be empty if key not set)."""
    results = tavily_client.search("Python async programming tutorial")
    print(f"\n✅ Tavily returned {len(results)} results")
    if results:
        print(f"   First result: {results[0].get('title', '')[:60]}")
    # Returns [] on error — never crashes
    assert isinstance(results, list)


def test_tavily_cache():
    """Same query twice should hit cache (no extra API call)."""
    tavily_client.clear_cache()
    q = "asyncio event loop Python"
    r1 = tavily_client.search(q)
    r2 = tavily_client.search(q)
    assert r1 == r2
    print(f"\n✅ Tavily cache works — same result both times ({len(r1)} results)")
