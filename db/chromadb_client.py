"""
db/chromadb_client.py
embed()   — CPU BGE-M3 via sentence-transformers, returns list[float]
insert()  — store chunk in ChromaDB
search()  — vector search, returns list[str]
rerank()  — BGE Reranker Large cross-encoder, returns reordered list[str]

Both models run on CPU only — not in real-time path.
Used only during curriculum generation and RAG retrieval.

ROLLBACK: if BGE-M3 causes conflicts, revert _get_embedder() to
  SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
and _get_reranker() to
  CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
"""

from __future__ import annotations

import asyncio
import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from config import settings

_embedder = None
_reranker = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading BGE-M3 embedder (CPU)…")
        _embedder = SentenceTransformer("BAAI/bge-m3", device="cpu")
        logger.info("BGE-M3 loaded.")
    return _embedder

def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        logger.info("Loading BGE Reranker Large (CPU)…")
        _reranker = CrossEncoder("BAAI/bge-reranker-large", device="cpu")
        logger.info("BGE Reranker Large loaded.")
    return _reranker

_chroma_client = None
_collections: dict = {}

def _get_chroma():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=settings.chromadb_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info("ChromaDB client initialised at '{}'", settings.chromadb_path)
    return _chroma_client

def _get_collection(domain: str):
    # Sanitize: keep only alphanumerics/underscores/hyphens, collapse spaces.
    # Truncate to 40 chars before adding the "_bge_m3" suffix (7 chars)
    # so the final name stays within ChromaDB's 63-char limit.
    import re as _re
    safe_name = _re.sub(r"[^a-z0-9_-]+", "_", domain.lower().strip())
    safe_name = _re.sub(r"_+", "_", safe_name).strip("_")[:40]
    if not safe_name or not safe_name[0].isalnum():
        safe_name = "collection_" + safe_name.lstrip("_")[:30]
    # BGE-M3 produces 1024-dim vectors — suffix collection name so old
    # collections with different vector dimensions are never reused.
    collection_name = f"{safe_name}_bge_m3"
    if collection_name not in _collections:
        _collections[collection_name] = _get_chroma().get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    return _collections[collection_name]

async def embed(text: str) -> list[float]:
    """Async wrapper — model loading + encoding both run in a thread pool.
    Calling _get_embedder() on the event loop directly blocks for ~8s on first
    load (BGE-M3 is large); this keeps that work off the event loop entirely.
    """
    def _load_and_encode(t: str) -> list:
        return _get_embedder().encode(t, normalize_embeddings=True).tolist()

    return await asyncio.to_thread(_load_and_encode, text)

async def insert(
    chunk_id: str,
    domain: str,
    text: str,
    metadata: dict | None = None,
) -> None:
    vector = await embed(text)
    collection = _get_collection(domain)
    item_metadata = {"domain": domain, "chunk_id": chunk_id, **(metadata or {})}
    await asyncio.to_thread(
        lambda: collection.upsert(
            ids=[chunk_id],
            embeddings=[vector],
            documents=[text],
            metadatas=[item_metadata],
        )
    )
    logger.debug("Inserted chunk '{}' into domain '{}'", chunk_id, domain)

async def search(
    query: str,
    domain: str,
    top_k: int = 10,
    where: dict | None = None,
) -> list[str]:
    collection = _get_collection(domain)
    count = await asyncio.to_thread(collection.count)
    if count == 0:
        logger.warning("ChromaDB collection '{}' is empty.", domain)
        return []
    actual_k = min(top_k, count)
    query_vec = await embed(query)
    query_kwargs = {
        "query_embeddings": [query_vec],
        "n_results": actual_k,
        "include": ["documents"],
    }
    if where:
        query_kwargs["where"] = where
    results = await asyncio.to_thread(lambda: collection.query(**query_kwargs))
    docs = results.get("documents", [[]])[0]
    logger.info("ChromaDB search: {} results for '{}' in '{}'", len(docs), query[:60], domain)
    return docs

async def rerank(query: str, chunks: list[str], top_k: int = 5) -> list[str]:
    """Async wrapper — model loading + cross-encoder inference run in a thread pool."""
    if not chunks:
        return []
    if len(chunks) <= top_k:
        return chunks

    def _load_and_rerank(q: str, cs: list[str]) -> list[str]:
        reranker = _get_reranker()
        pairs = [[q, chunk] for chunk in cs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, cs), key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in ranked[:top_k]]

    top = await asyncio.to_thread(_load_and_rerank, query, chunks)
    logger.info("Reranked {} chunks → top {}", len(chunks), top_k)
    return top
