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
# import chromadb                                    # V2: re-enable with ChromaDB
# from chromadb.config import Settings as ChromaSettings  # V2: re-enable with ChromaDB
from loguru import logger

from config import settings

_embedder = None
_reranker = None

def _get_embedder():
    # V1: disabled — BGE-M3 uses ~2.2 GB RAM on CPU, causes EC2 crashes.
    # V2: re-enable when moving to a memory-optimised instance.
    raise RuntimeError("BGE-M3 embedder is disabled in V1. See db/chromadb_client.py.")
    # global _embedder                                          # V2
    # if _embedder is None:                                     # V2
    #     from sentence_transformers import SentenceTransformer # V2
    #     logger.info("Loading BGE-M3 embedder (CPU)…")        # V2
    #     _embedder = SentenceTransformer("BAAI/bge-m3", device="cpu")  # V2
    #     logger.info("BGE-M3 loaded.")                        # V2
    # return _embedder                                          # V2

def _get_reranker():
    # V1: disabled — BGE Reranker Large uses ~1.2 GB RAM on CPU.
    # V2: re-enable alongside _get_embedder().
    raise RuntimeError("BGE Reranker is disabled in V1. See db/chromadb_client.py.")
    # global _reranker                                          # V2
    # if _reranker is None:                                     # V2
    #     from sentence_transformers import CrossEncoder        # V2
    #     logger.info("Loading BGE Reranker Large (CPU)…")     # V2
    #     _reranker = CrossEncoder("BAAI/bge-reranker-large", device="cpu")  # V2
    #     logger.info("BGE Reranker Large loaded.")             # V2
    # return _reranker                                          # V2

_chroma_client = None
_collections: dict = {}

def _get_chroma():
    # V1: disabled — ChromaDB HNSW index uses ~300-500 MB RAM.
    # V2: re-enable with chromadb import above.
    raise RuntimeError("ChromaDB client is disabled in V1. See db/chromadb_client.py.")
    # global _chroma_client                                     # V2
    # if _chroma_client is None:                                # V2
    #     _chroma_client = chromadb.PersistentClient(           # V2
    #         path=settings.chromadb_path,                      # V2
    #         settings=ChromaSettings(anonymized_telemetry=False), # V2
    #     )                                                      # V2
    #     logger.info("ChromaDB client initialised at '{}'", settings.chromadb_path)  # V2
    # return _chroma_client                                     # V2

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
    """V1: disabled. V2: re-enable _get_embedder() to restore."""
    logger.debug("embed() called but embedder is disabled (V1).")
    return []
    # def _load_and_encode(t: str) -> list:                    # V2
    #     return _get_embedder().encode(t, normalize_embeddings=True).tolist()  # V2
    # return await asyncio.to_thread(_load_and_encode, text)   # V2

async def insert(
    chunk_id: str,
    domain: str,
    text: str,
    metadata: dict | None = None,
) -> None:
    """V1: disabled — silently skipped. V2: re-enable body below."""
    logger.debug("insert() called but ChromaDB is disabled (V1). chunk_id='{}'", chunk_id)
    return
    # vector = await embed(text)                               # V2
    # collection = _get_collection(domain)                     # V2
    # item_metadata = {"domain": domain, "chunk_id": chunk_id, **(metadata or {})}  # V2
    # await asyncio.to_thread(                                 # V2
    #     lambda: collection.upsert(                           # V2
    #         ids=[chunk_id],                                  # V2
    #         embeddings=[vector],                             # V2
    #         documents=[text],                                # V2
    #         metadatas=[item_metadata],                       # V2
    #     )                                                    # V2
    # )                                                        # V2
    # logger.debug("Inserted chunk '{}' into domain '{}'", chunk_id, domain)  # V2

async def search(
    query: str,
    domain: str,
    top_k: int = 10,
    where: dict | None = None,
) -> list[str]:
    """V1: disabled — returns []. V2: re-enable body below."""
    logger.debug("search() called but ChromaDB is disabled (V1). domain='{}'", domain)
    return []
    # collection = _get_collection(domain)                     # V2
    # count = await asyncio.to_thread(collection.count)        # V2
    # if count == 0:                                           # V2
    #     logger.warning("ChromaDB collection '{}' is empty.", domain)  # V2
    #     return []                                            # V2
    # actual_k = min(top_k, count)                             # V2
    # query_vec = await embed(query)                           # V2
    # query_kwargs = {                                         # V2
    #     "query_embeddings": [query_vec],                     # V2
    #     "n_results": actual_k,                               # V2
    #     "include": ["documents"],                            # V2
    # }                                                        # V2
    # if where:                                                # V2
    #     query_kwargs["where"] = where                        # V2
    # results = await asyncio.to_thread(lambda: collection.query(**query_kwargs))  # V2
    # docs = results.get("documents", [[]])[0]                 # V2
    # logger.info("ChromaDB search: {} results for '{}' in '{}'", len(docs), query[:60], domain)  # V2
    # return docs                                              # V2

async def rerank(query: str, chunks: list[str], top_k: int = 5) -> list[str]:
    """V1: reranker disabled — returns input unchanged up to top_k. V2: re-enable body below."""
    logger.debug("rerank() called but reranker is disabled (V1). Returning first {} chunks.", top_k)
    return chunks[:top_k]
    # if not chunks:                                           # V2
    #     return []                                            # V2
    # if len(chunks) <= top_k:                                 # V2
    #     return chunks                                        # V2
    # def _load_and_rerank(q: str, cs: list[str]) -> list[str]:  # V2
    #     reranker = _get_reranker()                           # V2
    #     pairs = [[q, chunk] for chunk in cs]                 # V2
    #     scores = reranker.predict(pairs)                     # V2
    #     ranked = sorted(zip(scores, cs), key=lambda x: x[0], reverse=True)  # V2
    #     return [chunk for _, chunk in ranked[:top_k]]        # V2
    # top = await asyncio.to_thread(_load_and_rerank, query, chunks)  # V2
    # logger.info("Reranked {} chunks → top {}", len(chunks), top_k)  # V2
    # return top                                               # V2