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
        logger.info("Loading all-mpnet-base-v2 embedder (CPU)…")
        _embedder = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2", device="cpu")
        logger.info("all-mpnet-base-v2 loaded.")
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
        logger.info(
            "ChromaDB client initialised at '{}'",
            settings.chromadb_path)
    return _chroma_client


def _get_collection(domain: str):
    safe_name = domain.replace(" ", "_").replace("/", "_").lower()[:60]
    # BGE-M3 produces 1024-dim vectors — suffix collection name so old
    # 384-dim MiniLM collections are never accidentally reused
    collection_name = f"{safe_name}_mpnet"
    if collection_name not in _collections:
        _collections[collection_name] = _get_chroma().get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    return _collections[collection_name]


def embed(text: str) -> list[float]:
    embedder = _get_embedder()
    vec = embedder.encode(text, normalize_embeddings=True)
    return vec.tolist()


def insert(chunk_id: str, domain: str, text: str) -> None:
    vector = embed(text)
    collection = _get_collection(domain)
    collection.upsert(
        ids=[chunk_id],
        embeddings=[vector],
        documents=[text],
        metadatas=[{"domain": domain, "chunk_id": chunk_id}],
    )
    logger.debug("Inserted chunk '{}' into domain '{}'", chunk_id, domain)


def search(query: str, domain: str, top_k: int = 10) -> list[str]:
    collection = _get_collection(domain)
    count = collection.count()
    if count == 0:
        logger.warning("ChromaDB collection '{}' is empty.", domain)
        return []
    actual_k = min(top_k, count)
    query_vec = embed(query)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=actual_k,
        include=["documents"],
    )
    docs = results.get("documents", [[]])[0]
    logger.info("ChromaDB search: {} results for '{}' in '{}'",
                len(docs), query[:60], domain)
    return docs


def rerank(query: str, chunks: list[str], top_k: int = 5) -> list[str]:
    if not chunks:
        return []
    if len(chunks) <= top_k:
        return chunks
    reranker = _get_reranker()
    pairs = [[query, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    top = [chunk for _, chunk in ranked[:top_k]]
    logger.info("Reranked {} chunks → top {}", len(chunks), top_k)
    return top
