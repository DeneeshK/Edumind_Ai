from __future__ import annotations

import pytest

from clients import mcp_search_client


pytestmark = pytest.mark.unit


def test_is_enabled_respects_kill_switch(monkeypatch):
    monkeypatch.setattr(mcp_search_client, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_search_client.settings, "mcp_search_enabled", True)
    assert mcp_search_client.is_enabled() is True

    monkeypatch.setattr(mcp_search_client.settings, "mcp_search_enabled", False)
    assert mcp_search_client.is_enabled() is False


def test_is_enabled_false_when_sdk_missing(monkeypatch):
    monkeypatch.setattr(mcp_search_client, "_MCP_AVAILABLE", False)
    monkeypatch.setattr(mcp_search_client.settings, "mcp_search_enabled", True)
    assert mcp_search_client.is_enabled() is False


@pytest.mark.asyncio
async def test_call_returns_empty_dict_when_disabled(monkeypatch):
    """The toggle being off must short-circuit before any network attempt."""
    monkeypatch.setattr(mcp_search_client, "is_enabled", lambda: False)

    async def fail_if_called(*a, **kw):
        raise AssertionError("sse_client should never be invoked when disabled")

    monkeypatch.setattr(mcp_search_client, "sse_client", fail_if_called)
    result = await mcp_search_client._call("health", {})
    assert result == {}


@pytest.mark.asyncio
async def test_call_degrades_to_empty_dict_on_transport_failure(monkeypatch):
    """A web-search hiccup must never raise into the caller (a lesson/doubt flow)."""
    monkeypatch.setattr(mcp_search_client, "is_enabled", lambda: True)

    def raise_sse(*a, **kw):
        raise ConnectionError("server unreachable")

    monkeypatch.setattr(mcp_search_client, "sse_client", raise_sse)
    result = await mcp_search_client._call("smoke_search", {"concept": "x"})
    assert result == {}


@pytest.mark.asyncio
async def test_make_tool_executor_collects_sources_from_research_web(monkeypatch):
    """
    The final doubt reply cites sources collected during the tool loop.
    Verifies the executor appends {title, url} for every chunk with a URL.
    """
    async def fake_research_web(concept, namespace, context=""):
        return {"chunks_stored": 2}

    async def fake_retrieve_full(query, namespace, context="", top_k=None):
        return [
            {"content": "chunk one text", "source_url": "https://a.com", "source_title": "A"},
            {"content": "chunk two text", "source_url": "https://b.com", "source_title": "B"},
            {"content": "chunk with no url", "source_url": "", "source_title": ""},
        ]

    monkeypatch.setattr(mcp_search_client, "research_web", fake_research_web)
    monkeypatch.setattr(mcp_search_client, "retrieve_full", fake_retrieve_full)

    collected: list[dict] = []
    executor = mcp_search_client.make_tool_executor(namespace="course-1", sources=collected)
    output = await executor("research_web", {"concept": "graphs", "query": "how do graphs work?"})

    assert "chunk one text" in output
    assert "chunk two text" in output
    assert collected == [
        {"title": "A", "url": "https://a.com"},
        {"title": "B", "url": "https://b.com"},
    ]


@pytest.mark.asyncio
async def test_make_tool_executor_smoke_search_reports_not_found(monkeypatch):
    async def fake_smoke_search(concept):
        return {"found": False, "summary": ""}

    monkeypatch.setattr(mcp_search_client, "smoke_search", fake_smoke_search)
    executor = mcp_search_client.make_tool_executor(namespace="course-1")
    output = await executor("smoke_search", {"concept": "an obscure topic"})
    assert "No useful web results" in output


def test_groq_tools_shape():
    tools = mcp_search_client.groq_tools()
    names = {t["function"]["name"] for t in tools}
    assert names == {"smoke_search", "research_web"}
    for t in tools:
        assert t["type"] == "function"
        assert "description" in t["function"]
        assert "parameters" in t["function"]
