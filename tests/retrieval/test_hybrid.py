"""Tests for HybridRetriever (combining BM25, Dense, and RRF)."""

import pytest

from veritas.retrieval.bm25 import BM25Index
from veritas.retrieval.corpus import PassageStore
from veritas.retrieval.dense import DenseEncoder, DenseIndex, MockEmbedder
from veritas.retrieval.hybrid import HybridRetriever
from veritas.retrieval.schema import Passage


@pytest.fixture
def sample_hybrid_retriever():
    p1 = Passage(
        passage_id="p1",
        text="The World Health Organization confirmed that vaccination reduces severe COVID-19 hospitalizations.",
        url="https://who.int/report",
        domain="who.int",
    )
    p2 = Passage(
        passage_id="p2",
        text="The Bureau of Labor Statistics reported nonfarm payroll employment rose by 275000 in February.",
        url="https://bls.gov/news",
        domain="bls.gov",
    )
    p3 = Passage(
        passage_id="p3",
        text="NASA's James Webb Space Telescope detected atmospheric methane on a habitable-zone exoplanet.",
        url="https://nasa.gov/jwst",
        domain="nasa.gov",
    )
    store = PassageStore([p1, p2, p3])

    bm25_idx = BM25Index.build(store)
    encoder = DenseEncoder(model_name_or_path="mock-embedder", device="cpu", embedder=MockEmbedder(dimension=16))
    dense_idx = DenseIndex.build(store, encoder=encoder)

    return HybridRetriever(
        bm25_index=bm25_idx,
        dense_index=dense_idx,
        mode="hybrid",
        bm25_top_k=10,
        dense_top_k=10,
        rrf_k=60,
        top_k=10,
    )


def test_hybrid_mode_fuses_bm25_and_dense(sample_hybrid_retriever):
    retriever = sample_hybrid_retriever
    query = "vaccination reduces severe hospitalizations"
    results = retriever.retrieve(query)

    assert len(results) > 0
    # First result should match p1
    assert results[0].passage_id == "p1"
    assert results[0].rank == 1
    assert results[0].score > 0
    # Sources should record the retrieval methods that found it
    assert "bm25" in results[0].sources
    assert results[0].passage is not None
    assert results[0].passage.domain == "who.int"


def test_mode_override_bm25_only(sample_hybrid_retriever):
    retriever = sample_hybrid_retriever
    query = "nonfarm payroll employment rose in February"
    results = retriever.retrieve(query, mode="bm25")

    assert len(results) > 0
    assert results[0].passage_id == "p2"
    assert results[0].sources == ["bm25"]
    assert "bm25" in results[0].source_ranks


def test_mode_override_dense_only(sample_hybrid_retriever):
    retriever = sample_hybrid_retriever
    query = "James Webb Space Telescope methane"
    results = retriever.retrieve(query, mode="dense")

    assert len(results) > 0
    assert results[0].sources == ["dense"]
    assert "dense" in results[0].source_ranks


def test_multiple_queries_per_claim_fused_and_deduplicated(sample_hybrid_retriever):
    retriever = sample_hybrid_retriever
    # Claim with two query formulations
    queries = [
        "vaccination hospitalizations",
        "World Health Organization vaccines",
    ]
    results = retriever.retrieve(queries, mode="hybrid")

    assert len(results) > 0
    # Ensure all retrieved passage IDs are strictly unique (deduplicated)
    retrieved_ids = [r.passage_id for r in results]
    assert len(retrieved_ids) == len(set(retrieved_ids))
    assert results[0].passage_id == "p1"
    assert results[0].rank == 1


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        HybridRetriever(mode="invalid_mode")


def test_empty_query_returns_empty(sample_hybrid_retriever):
    retriever = sample_hybrid_retriever
    assert retriever.retrieve([]) == []
    assert retriever.retrieve("") == []
    assert retriever.retrieve(["", "   "]) == []
