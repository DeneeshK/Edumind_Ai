"""
core/rag_pipeline.py
retrieve() — full RAG pipeline:
  HyDE → embed → ChromaDB → Tavily → combine → rerank → top 5 plain text chunks

hyde()     — generate a hypothetical answer to improve retrieval
"""

from __future__ import annotations

from loguru import logger

from clients.groq_client import generate
from clients.tavily_client import search as tavily_search
from db.chromadb_client import search as chroma_search, rerank
from config import settings


# ── HyDE — Hypothetical Document Embedding ──────────────────────────────

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
    logger.debug("HyDE generated ({} chars) for query: '{}'",
                 len(hypothetical), query[:60])
    return hypothetical


# ── retrieve() ──────────────────────────────────────────────────────────

async def retrieve(query: str, domain: str, top_k: int = 5) -> list[str]:
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

    # ── Step 1 + 2: HyDE → ChromaDB ──────────────────────────────────────────
    try:
        hypothetical = await hyde(query)
        chroma_results = chroma_search(
            query=hypothetical,
            domain=domain,
            top_k=top_k * 2,   # fetch more, reranker will trim
        )
        chunks.extend(chroma_results)
        logger.info("ChromaDB contributed {} chunks", len(chroma_results))
    except Exception as e:
        logger.warning(
            "ChromaDB retrieval failed: {} — continuing with Tavily only", e)

    # ── Step 3: Tavily web search ───────────────────────────────────────────
    try:
        tavily_results = tavily_search(
            query=f"{query} {domain}",
            max_results=5,
        )
        for r in tavily_results:
            content = r.get("content", "").strip()
            if content and len(content) > 50:
                chunks.append(content)
        logger.info("Tavily contributed {} chunks", len(tavily_results))
    except Exception as e:
        logger.warning(
            "Tavily retrieval failed: {} — continuing with ChromaDB only", e)

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

    final = rerank(query=query, chunks=unique_chunks, top_k=top_k)
    logger.info("RAG pipeline complete: {} final chunks for '{}'",
                len(final), query[:60])
    return final
