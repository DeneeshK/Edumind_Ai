"""
clients/tavily_client.py
search()          — single query, TTL-cached (24h), returns list[dict]
search_multiple() — multiple queries deduplicated, returns list[dict]

TTL cache: results persist across sessions for 24 hours.
Same concept queried in a new session costs 0 Tavily API calls.
10s timeout, returns [] on any failure (never crashes the caller).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from tavily import TavilyClient
from loguru import logger

from config import settings


# ── TTL cache config ──────────────────────────────────────────────────────────
# 24-hour TTL: concept content changes rarely enough that one day is safe.
_TTL_SECONDS = 86_400  # 24 hours

# Persist cache to disk so it survives process restarts.
# Stored alongside ChromaDB so both are cleared together if needed.
_CACHE_PATH = Path(settings.chromadb_path) / "tavily_cache.json"

# In-memory layer: {key: {"results": [...], "expires_at": float}}
_cache: dict[str, dict] = {}
_cache_loaded = False


def _load_cache() -> None:
    """Load disk cache into memory once at first use."""
    global _cache, _cache_loaded
    if _cache_loaded:
        return
    _cache_loaded = True
    if _CACHE_PATH.exists():
        try:
            raw = json.loads(_CACHE_PATH.read_text())
            now = time.time()
            # Drop expired entries on load
            _cache = {k: v for k, v in raw.items() if v.get("expires_at", 0) > now}
            logger.info("Tavily cache loaded: {} valid entries", len(_cache))
        except Exception as e:
            logger.warning("Tavily cache load failed: {} — starting fresh", e)
            _cache = {}


def _save_cache() -> None:
    """Persist current cache to disk. Silent on failure."""
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(_cache))
    except Exception as e:
        logger.warning("Tavily cache save failed: {}", e)


# ── Client singleton ──────────────────────────────────────────────────────────
_client: TavilyClient | None = None

def _get_client() -> TavilyClient:
    """Return the process-wide Tavily SDK client."""
    global _client
    if _client is None:
        _client = TavilyClient(api_key=settings.tavily_api_key)
    return _client


def _cache_key(query: str) -> str:
    """Build a stable cache key for a normalized search query."""
    return hashlib.md5(query.strip().lower().encode()).hexdigest()


# ── search() ──────────────────────────────────────────────────────────────────

def search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search Tavily for query. Returns list of result dicts:
      [{"title": ..., "url": ..., "content": ...}, ...]

    - TTL-cached for 24 hours across sessions
    - Returns [] on any error (caller degrades gracefully)
    - 10s timeout enforced by Tavily client
    """
    _load_cache()
    key = _cache_key(query)
    now = time.time()

    entry = _cache.get(key)
    if entry and entry.get("expires_at", 0) > now:
        logger.debug("Tavily TTL cache hit: key={} query_chars={}", key[:12], len(query))
        return entry["results"]

    try:
        client = _get_client()
        logger.info("Tavily API call: key={} query_chars={}", key[:12], len(query))
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",  # Use advanced depth for richer content
        )
        results = response.get("results", [])
        _cache[key] = {
            "results": results,
            "expires_at": now + _TTL_SECONDS,
            "query": query[:120],  # stored for debug inspection
        }
        _save_cache()
        logger.info("Tavily: {} results cached for key={}", len(results), key[:12])
        return results

    except Exception as e:
        logger.warning("Tavily search failed for key={}: {} — returning []", key[:12], e)
        return []


# ── search_multiple() ─────────────────────────────────────────────────────────

def search_multiple(queries: list[str], max_results_each: int = 3) -> list[dict]:
    """
    Run multiple search queries and return deduplicated results.
    Deduplication is by URL.
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
        "search_multiple: {} queries -> {} unique results",
        len(queries), len(combined)
    )
    return combined


# ── clear_cache() ─────────────────────────────────────────────────────────────

def clear_cache(expired_only: bool = True) -> None:
    """
    Clear the in-memory cache.

    Args:
        expired_only: if True (default), only remove expired entries.
                      if False, wipe everything (use for testing only).
    """
    global _cache
    if expired_only:
        now = time.time()
        before = len(_cache)
        _cache = {k: v for k, v in _cache.items() if v.get("expires_at", 0) > now}
        removed = before - len(_cache)
        if removed:
            logger.debug("Tavily cache: removed {} expired entries", removed)
    else:
        _cache = {}
        _save_cache()
        logger.debug("Tavily cache fully cleared.")
