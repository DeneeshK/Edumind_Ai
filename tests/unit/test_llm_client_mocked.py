from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from clients import groq_client


pytestmark = pytest.mark.unit


def _text_response(text: str):
    message = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _tool_response(tool_name: str, arguments: dict, call_id: str = "call-1"):
    tool_call = SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=tool_name, arguments=json.dumps(arguments)),
    )
    message = SimpleNamespace(content="", tool_calls=[tool_call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.mark.asyncio
async def test_generate_returns_mocked_content_without_real_groq(monkeypatch):
    monkeypatch.setattr(groq_client, "get_client", lambda: object())

    async def fake_with_retry(fn, *args, **kwargs):
        return _text_response("mocked lesson")

    monkeypatch.setattr(groq_client, "_with_retry", fake_with_retry)

    result = await groq_client.generate(
        messages=[{"role": "user", "content": "Say hello"}],
        model="test-model",
    )

    assert result == "mocked lesson"


@pytest.mark.asyncio
async def test_generate_surfaces_timeout_from_mocked_client(monkeypatch):
    monkeypatch.setattr(groq_client, "get_client", lambda: object())

    async def fake_with_retry(fn, *args, **kwargs):
        raise groq_client.GroqTimeoutError("timeout")

    monkeypatch.setattr(groq_client, "_with_retry", fake_with_retry)

    with pytest.raises(groq_client.GroqTimeoutError):
        await groq_client.generate(messages=[{"role": "user", "content": "slow"}])


@pytest.mark.asyncio
async def test_tool_call_loop_returns_terminal_tool_args(monkeypatch):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Finish the flow.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "ready": {"type": "boolean"},
                    },
                    "required": ["summary", "ready"],
                },
            },
        }
    ]
    monkeypatch.setattr(groq_client, "get_client", lambda: object())

    async def fake_with_retry(fn, *args, **kwargs):
        return _tool_response(
            "finish",
            {"summary": "Done", "ready": True, "extra": "discarded"},
        )

    monkeypatch.setattr(groq_client, "_with_retry", fake_with_retry)

    result = await groq_client.tool_call_loop(
        system="Use tools.",
        user_message="Finish now.",
        tools=tools,
        terminal_tool_name="finish",
    )

    assert result == {"summary": "Done", "ready": True}


def test_tool_arg_parser_tolerates_python_literals():
    parsed = groq_client._parse_tool_args("{'ready': True, 'value': None}")

    assert parsed == {"ready": True, "value": None}
