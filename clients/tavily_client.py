"""
clients/tavily_client.py
search()          — single query, session-cached, returns list[dict]
search_multiple() — multiple queries deduplicated, returns list[dict]

Session cache: {hash(query): result} — resets when module is reloaded
10s timeout, returns [] on any failure (never crashes the caller)
"""

from __future__ import annotations

import hashlib
from typing import Any

from tavily import TavilyClient
from loguru import logger

from config import settings


# ── Session-scoped cache ──────────────────────────────────────────────────────
# Lives for the duration of the Python process (one student session)
_cache: dict[str, list[dict]] = {}

# ── Client singleton ──────────────────────────────────────────────────────────
_client: TavilyClient | None = None

def get_client() -> TavilyClient:
    global _client
    if _client is None:
        _client = TavilyClient(api_key=settings.tavily_api_key)
    return _client


def _cache_key(query: str) -> str:
    return hashlib.md5(query.strip().lower().encode()).hexdigest()


# ── search() ──────────────────────────────────────────────────────────────────

def search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search Tavily for query. Returns list of result dicts:
      [{"title": ..., "url": ..., "content": ...}, ...]

    - Session-cached: same query in same session costs 0 API calls
    - Returns [] on any error (caller degrades gracefully)
    - 10s timeout enforced by Tavily client
    """
    key = _cache_key(query)
    if key in _cache:
        logger.debug("Tavily cache hit: '{}'", query[:60])
        return _cache[key]

    try:
        client = get_client()
        logger.info("Tavily search: '{}'", query[:80])
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
        )
        results = response.get("results", [])
        _cache[key] = results
        logger.info("Tavily returned {} results for '{}'", len(results), query[:60])
        return results

    except Exception as e:
        logger.warning("Tavily search failed for '{}': {} — returning []", query[:60], e)
        return []


# ── search_multiple() ─────────────────────────────────────────────────────────

def search_multiple(queries: list[str], max_results_each: int = 3) -> list[dict]:
    """
    Run multiple search queries and return deduplicated results.
    Deduplication is by URL.

    Used by Curriculum Architect which fires 2-3 queries for better coverage.
    """
    seen_urls: set[str] = set()
    combined: list[dict] = []

    for query in queries:
        results = search(query, max_results=max_results_each)
        for r in results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                combined.append(r)

    logger.info(
        "search_multiple: {} queries → {} unique results",
        len(queries), len(combined)
    )
    return combined


# ── clear_cache() ─────────────────────────────────────────────────────────────

def clear_cache() -> None:
    """Call at session end to free memory."""
    global _cache
    _cache = {}
    logger.debug("Tavily session cache cleared.")
