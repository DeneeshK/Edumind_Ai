"""
core/rag_pipeline.py
retrieve() — full RAG pipeline:
  HyDE → embed → ChromaDB → Tavily → combine → rerank → top 5 plain text chunks

hyde()     — generate a hypothetical answer to improve retrieval
"""

from __future__ import annotations

import asyncio

from loguru import logger

from clients.groq_client import generate
from clients.tavily_client import search as tavily_search
from core.curriculum_quality import token_set
from db.chromadb_client import search as chroma_search, rerank
from config import settings

_eval_runner_ref = None


def set_eval_runner(runner) -> None:
    global _eval_runner_ref
    _eval_runner_ref = runner


def clear_eval_runner() -> None:
    global _eval_runner_ref
    _eval_runner_ref = None


# ── HyDE — Hypothetical Document Embedding ────────────────────────────────────

async def hyde(query: str) -> str:
    """
    Generate a short hypothetical answer to the query.
    Embedding this hypothetical answer instead of the raw query
    pulls more relevant chunks from the vector store.
    """
    prompt = (
        f"Write a clear, concise 3-sentence explanation that directly answers: '{query}'. "
        f"Write as if you are a textbook. Do not include preamble."
    )
    hypothetical = await generate(
        messages=[{"role": "user", "content": prompt}],
        model=settings.generation_model,
    )
    logger.debug("HyDE generated ({} chars) for query: '{}'", len(hypothetical), query[:60])
    return hypothetical


# ── retrieve() ────────────────────────────────────────────────────────────────

def _is_relevant_chunk(chunk: str, query: str, topic: str | None = None) -> bool:
    chunk_tokens = token_set(chunk)
    query_tokens = token_set(query) | token_set(topic)
    if not query_tokens:
        return True
    return bool(chunk_tokens & query_tokens)


async def retrieve(
    query: str,
    domain: str,
    top_k: int = 5,
    course_id: str | None = None,
    student_id: str | None = None,
    topic: str | None = None,
    module_id: str | None = None,
) -> list[str]:
    """
    Full RAG pipeline. Returns top_k plain text chunks, reranked.

    Steps:
      1. HyDE: generate hypothetical answer to improve query embedding
      2. ChromaDB: vector search using the hypothetical answer
      3. Tavily: web search for live examples and current content
      4. Combine: merge ChromaDB + Tavily results
      5. Rerank: BGE Reranker selects final top_k

    Args:
        query:  the concept or question to retrieve content for
        domain: ChromaDB collection name (e.g. "machine_learning")
        top_k:  number of final chunks to return (default 5)

    Returns:
        list of plain text strings, best-first
    """
    chunks: list[str] = []
    hypothetical = ""
    tavily_results: list[dict] = []

    # ── Step 1 + 2: HyDE → ChromaDB ──────────────────────────────────────────
    try:
        hypothetical = await hyde(query)
        if course_id:
            where = {"course_id": course_id}
        elif student_id:
            where = {"student_id": student_id}
        else:
            where = None
        chroma_results = await chroma_search(
            query=hypothetical,
            domain=domain,
            top_k=top_k * 2,   # fetch more, reranker will trim
            where=where,
        )
        relevant_chroma = [
            chunk for chunk in chroma_results
            if _is_relevant_chunk(chunk, query, topic)
        ]
        rejected = len(chroma_results) - len(relevant_chroma)
        chunks.extend(relevant_chroma)
        logger.info(
            "ChromaDB contributed {} chunks for course_id={} module_id={} (rejected {})",
            len(relevant_chroma), course_id, module_id, rejected,
        )
    except Exception as e:
        logger.warning("ChromaDB retrieval failed: {} — continuing with Tavily only", e)

    # ── Step 3: Tavily web search ─────────────────────────────────────────────
    try:
        tavily_results = tavily_search(
            query=f"{query} {domain}",
            max_results=5,
        )
        rejected_web = 0
        for r in tavily_results:
            content = r.get("content", "").strip()
            if content and len(content) > 50 and _is_relevant_chunk(content, query, topic):
                chunks.append(content)
            elif content:
                rejected_web += 1
        logger.info("Tavily contributed {} raw chunks (rejected {})", len(tavily_results), rejected_web)
    except Exception as e:
        logger.warning("Tavily retrieval failed: {} — continuing with ChromaDB only", e)

    if not chunks:
        logger.warning("No chunks retrieved for query: '{}'", query)
        return []

    # ── Step 4 + 5: Combine → Rerank ─────────────────────────────────────────
    # Deduplicate by first 100 chars
    seen: set[str] = set()
    unique_chunks: list[str] = []
    for c in chunks:
        key = c[:100].strip()
        if key not in seen:
            seen.add(key)
            unique_chunks.append(c)

    final = await rerank(query=query, chunks=unique_chunks, top_k=top_k)
    logger.info("RAG pipeline complete: {} final chunks for '{}'", len(final), query[:60])

    if _eval_runner_ref is not None:
        asyncio.create_task(
            _eval_runner_ref.on_rag_retrieve(
                query=query,
                hypothetical=hypothetical,
                concept_card="",
                chroma_chunks_before_rerank=unique_chunks,
                chroma_chunks_after_rerank=final,
                tavily_results=tavily_results,
            )
        )

    return final
