"""
tests/test_phase3.py
3.3 — Insert 5 chunks → retrieve with a query → verify relevance

Run: pytest tests/test_phase3.py -v -s
NOTE: BGE-M3 downloads ~2GB on first run. This is a one-time download.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

import pytest
from db.chromadb_client import insert, search, rerank, embed
from core.rag_pipeline import hyde, retrieve

TEST_DOMAIN = "test_ml_domain"

# 5 chunks — 3 relevant to "attention mechanism", 2 irrelevant
CHUNKS = [
    ("chunk_001", "The attention mechanism allows a model to focus on relevant parts "
                  "of the input sequence when producing each output token. It computes "
                  "a weighted sum of values based on query-key similarity scores."),
    ("chunk_002", "Self-attention, also called intra-attention, relates different positions "
                  "of a single sequence to compute a representation of that sequence. "
                  "Transformers rely entirely on self-attention instead of recurrence."),
    ("chunk_003", "The softmax function converts raw attention scores into a probability "
                  "distribution that sums to 1, determining how much each token attends "
                  "to every other token in the sequence."),
    ("chunk_004", "Photosynthesis is the process by which green plants use sunlight to "
                  "synthesise nutrients from carbon dioxide and water. This is completely "
                  "unrelated to neural networks."),
    ("chunk_005", "The French Revolution began in 1789 and led to the rise of Napoleon. "
                  "This is also completely unrelated to machine learning attention."),
]


@pytest.mark.asyncio
async def test_embed_returns_vector():
    vec = await embed("test sentence")
    assert isinstance(vec, list)
    assert len(vec) > 100   # BGE-M3 produces 1024-dim vectors
    assert isinstance(vec[0], float)
    print(f"\n✅ embed() returned {len(vec)}-dim vector")


@pytest.mark.asyncio
async def test_insert_and_search():
    """Insert 5 chunks, search for attention, verify relevant chunks returned."""
    # Insert all 5
    for chunk_id, text in CHUNKS:
        await insert(chunk_id, TEST_DOMAIN, text)
    print(f"\n✅ Inserted {len(CHUNKS)} chunks into ChromaDB")

    # Search
    results = await search("how does attention mechanism work", TEST_DOMAIN, top_k=3)
    print(f"✅ Retrieved {len(results)} chunks")
    for i, r in enumerate(results):
        print(f"   [{i+1}] {r[:80]}...")

    assert len(results) > 0
    # At least one result should mention attention
    combined = " ".join(results).lower()
    assert "attention" in combined, "Expected attention-related chunks in top results"
    print("✅ Relevant chunks contain 'attention'")


@pytest.mark.asyncio
async def test_rerank_orders_by_relevance():
    """Reranker should put attention chunks above unrelated ones."""
    all_texts = [text for _, text in CHUNKS]
    query = "explain self-attention in transformers"
    reranked = await rerank(query, all_texts, top_k=3)

    print(f"\n✅ Reranked top 3:")
    for i, r in enumerate(reranked):
        print(f"   [{i+1}] {r[:80]}...")

    combined = " ".join(reranked).lower()
    assert "attention" in combined
    # Irrelevant chunks should not dominate top 3
    assert "photosynthesis" not in reranked[0].lower()


@pytest.mark.asyncio
async def test_hyde_generates_hypothetical():
    """HyDE should return a non-empty string."""
    hypo = await hyde("what is the query-key-value mechanism in transformers")
    print(f"\n✅ HyDE output ({len(hypo)} chars): '{hypo[:100]}...'")
    assert isinstance(hypo, str)
    assert len(hypo) > 50


@pytest.mark.asyncio
async def test_full_rag_pipeline():
    """Full retrieve() pipeline should return up to 5 relevant chunks."""
    results = await retrieve(
        query="how does self-attention work in transformers",
        domain=TEST_DOMAIN,
        top_k=5,
    )
    print(f"\n✅ RAG pipeline returned {len(results)} chunks")
    for i, r in enumerate(results):
        print(f"   [{i+1}] {r[:80]}...")

    assert len(results) > 0
    combined = " ".join(results).lower()
    assert "attention" in combined
    print("✅ Final chunks are relevant to attention mechanism")
