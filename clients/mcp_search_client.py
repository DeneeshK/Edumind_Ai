"""
clients/mcp_search_client.py
Thin MCP client that lets EduMind agents call the standalone web-search RAG
server (edumind_mcp_search) over HTTP/SSE.

Two responsibilities:
  1. High-level async helpers (smoke_search, research_web, retrieve, health)
     that open a short-lived MCP session per call and degrade to empty results
     on any failure — a web-search hiccup must never break a lesson or a doubt.
  2. Groq tool specs + an executor, so the doubt/tutor tool-loop can let the LLM
     DECIDE when to search the web (the toggle only controls whether these tools
     are offered at all).

Nothing here loads embedding models or touches pgvector; that all lives in the
MCP server process, keeping this API's memory footprint unchanged.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from config import settings

# The `mcp` client SDK is only needed when web search is used. Import lazily so a
# deployment that never enables the toggle need not install it.
try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    _MCP_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    _MCP_AVAILABLE = False


def is_enabled() -> bool:
    """Whether the MCP client is importable and not globally killed."""
    return _MCP_AVAILABLE and settings.mcp_search_enabled


def _unwrap(result: Any) -> dict:
    """Extract a dict payload from an MCP CallToolResult (structured or text)."""
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except Exception:
                return {"text": text}
    return {}


async def _call(tool: str, args: dict) -> dict:
    """
    Open a short-lived MCP session, call one tool, and return its dict result.

    Returns an empty dict on any transport/tool error so callers can degrade to
    "no web context" rather than surfacing an error to the student.
    """
    if not is_enabled():
        return {}
    try:
        async with sse_client(settings.mcp_search_server_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool, args)
                return _unwrap(result)
    except Exception as exc:
        logger.warning("MCP search tool '{}' failed: {} — degrading to empty.", tool, exc)
        return {}


# ── High-level helpers ────────────────────────────────────────────────────────

async def health() -> bool:
    """Return True if the MCP server is reachable and ready."""
    result = await _call("health", {})
    return result.get("status") == "ok"


async def smoke_search(concept: str) -> dict:
    """Cheap orientation pass: {concept, summary, sources, found}."""
    return await _call("smoke_search", {"concept": concept})


async def research_web(concept: str, namespace: str, context: str = "") -> dict:
    """Full ingest (smoke → deep → chunk → embed → store). Returns ingest stats."""
    return await _call(
        "research_concept",
        {"concept": concept, "namespace": namespace, "context": context},
    )


async def retrieve(query: str, namespace: str, context: str = "", top_k: int | None = None) -> list[str]:
    """
    Retrieve grounding chunks for a query. Returns plain text chunks (possibly
    empty), matching the shape legacy callers expected from the old RAG path.
    """
    result = await _call(
        "retrieve",
        {
            "query": query,
            "namespace": namespace,
            "context": context,
            "top_k": top_k or settings.rag_top_k,
        },
    )
    chunks = result.get("chunks") or []
    return [c.get("content", "") for c in chunks if c.get("content")]


# ── Groq tool-loop integration ────────────────────────────────────────────────

def groq_tools() -> list[dict]:
    """
    Groq tool specs offered to the LLM when web search is ON for a course.

    The LLM decides whether to call these — that decision IS the "only search
    when I don't know this" gate the product wants.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "smoke_search",
                "description": (
                    "Do a quick web lookup to orient on a concept you are unsure about. "
                    "Returns a short summary. Call this first when you don't recognize a "
                    "concept, then decide whether a full search is worth it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "concept": {"type": "string", "description": "The concept you are unsure about."}
                    },
                    "required": ["concept"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "research_web",
                "description": (
                    "Do a full web search for a concept and retrieve grounded source "
                    "chunks to answer with. Use only after smoke_search shows the concept "
                    "is genuinely unfamiliar or needs current, external detail."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "concept": {"type": "string", "description": "The concept to research."},
                        "query": {"type": "string", "description": "The specific question to retrieve chunks for."},
                    },
                    "required": ["concept", "query"],
                },
            },
        },
    ]


def make_tool_executor(namespace: str, context: str = ""):
    """
    Build an executor closure that routes the web tools to the MCP server.

    Bound to a namespace (course id) so retrieved content is scoped per course.
    Returns compact strings the LLM can read, never raw dumps.
    """
    async def _executor(tool_name: str, args: dict) -> str:
        if tool_name == "smoke_search":
            res = await smoke_search(args.get("concept", ""))
            if not res.get("found"):
                return "No useful web results found for that concept."
            return f"Web summary: {res.get('summary', '')[:800]}"

        if tool_name == "research_web":
            concept = args.get("concept", "")
            query = args.get("query", concept)
            await research_web(concept, namespace=namespace, context=context)
            chunks = await retrieve(query, namespace=namespace, context=context)
            if not chunks:
                return "No relevant web content was found for that query."
            joined = "\n\n---\n\n".join(chunks[: settings.rag_top_k])
            return f"Retrieved web context (use only what fits):\n{joined[:4000]}"

        return f"Unknown tool: {tool_name}"

    return _executor
