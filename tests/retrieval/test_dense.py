"""Tests for VERITAS Dense Retrieval (SentenceTransformers + FAISS).

All tests run locally using deterministic fixtures without downloading large
model weights or accessing external networks.
"""

import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from veritas.retrieval.corpus import PassageStore, create_synthetic_corpus
from veritas.retrieval.dense import (
    DenseEncoder,
    DenseIndex,
    DenseIndexMetadata,
    MockEmbedder,
    compute_dense_config_fingerprint,
    resolve_model_prefixes,
    run_build_dense_index,
)
from veritas.retrieval.schema import Passage


def test_mock_embedder_dimension_and_normalization():
    embedder = MockEmbedder(dimension=32)
    texts = ["First test sentence.", "Second completely different passage."]
    vecs = embedder.encode(texts, normalize_embeddings=True)

    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (2, 32)
    assert vecs.dtype == np.float32

    # L2 norm must be approximately 1.0
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)


def test_mock_embedder_repeatability():
    embedder = MockEmbedder(dimension=16)
    vec1 = embedder.encode(["Repeatable claim statement."])
    vec2 = embedder.encode(["Repeatable claim statement."])
    np.testing.assert_array_equal(vec1, vec2)


def test_prefix_resolution():
    # BGE model prefix convention
    q_bge, d_bge = resolve_model_prefixes("BAAI/bge-small-en-v1.5")
    assert q_bge == "Represent this sentence for searching relevant passages: "
    assert d_bge == ""

    # E5 model prefix convention
    q_e5, d_e5 = resolve_model_prefixes("intfloat/e5-small-v2")
    assert q_e5 == "query: "
    assert d_e5 == "passage: "

    # MiniLM / General prefix convention
    q_mini, d_mini = resolve_model_prefixes("sentence-transformers/all-MiniLM-L6-v2")
    assert q_mini == ""
    assert d_mini == ""

    # Explicit override takes precedence
    q_exp, d_exp = resolve_model_prefixes("BAAI/bge-small-en-v1.5", query_prefix="custom: ", document_prefix="doc: ")
    assert q_exp == "custom: "
    assert d_exp == "doc: "


def test_dense_encoder_format_matching():
    embedder = MockEmbedder(dimension=16)
    encoder = DenseEncoder(
        model_name_or_path="test-model",
        device="cpu",
        normalize_embeddings=True,
        query_prefix="query: ",
        document_prefix="doc: ",
        embedder=embedder,
    )

    q_vecs = encoder.encode_queries(["Search query text"])
    d_vecs = encoder.encode_documents(["Document passage text"])

    assert q_vecs.shape == (1, 16)
    assert d_vecs.shape == (1, 16)
    assert q_vecs.dtype == np.float32
    assert d_vecs.dtype == np.float32

    # Verify both are normalized
    assert abs(np.linalg.norm(q_vecs[0]) - 1.0) < 1e-5
    assert abs(np.linalg.norm(d_vecs[0]) - 1.0) < 1e-5


def test_dense_index_embedding_dimension_and_mapping():
    p1 = Passage(passage_id="p1", text="First evidence passage.")
    p2 = Passage(passage_id="p2", text="Second evidence passage.")
    p3 = Passage(passage_id="p3", text="Third evidence passage.")
    store = PassageStore([p1, p2, p3])

    encoder = DenseEncoder(
        model_name_or_path="test-model",
        device="cpu",
        normalize_embeddings=True,
        embedder=MockEmbedder(dimension=24),
    )

    dense_index = DenseIndex.build(store, encoder=encoder, index_name="test_index")

    assert dense_index.index.d == 24
    assert dense_index.index.ntotal == 3
    assert dense_index.passage_ids == ["p1", "p2", "p3"]
    assert dense_index.metadata.dimension == 24
    assert dense_index.metadata.total_passages == 3
    assert dense_index.metadata.index_type == "IndexFlatIP"


def test_known_passage_retrieval():
    """Verify that a query matching a known passage retrieves that passage at rank 1."""
    p1 = Passage(
        passage_id="psg:cdc_flu",
        text="Annual influenza vaccination prevents severe respiratory disease and hospitalization.",
        url="https://cdc.gov/flu",
        date="2024-01-15",
        domain="cdc.gov",
    )
    p2 = Passage(
        passage_id="psg:bls_jobs",
        text="Total nonfarm payroll employment increased by 275000 in February across health care and government.",
        url="https://bls.gov/news",
        date="2024-03-08",
        domain="bls.gov",
    )
    p3 = Passage(
        passage_id="psg:nasa_jwst",
        text="The James Webb Space Telescope detected atmospheric methane on a habitable-zone exoplanet.",
        url="https://nasa.gov/jwst",
        date="2023-09-11",
        domain="nasa.gov",
    )

    store = PassageStore([p1, p2, p3])
    encoder = DenseEncoder(
        model_name_or_path="test-embedder",
        device="cpu",
        normalize_embeddings=True,
        embedder=MockEmbedder(dimension=32),
    )

    dense_index = DenseIndex.build(store, encoder=encoder)

    # Exact passage text query must produce identical vector and highest cosine similarity (~1.0)
    results = dense_index.search(
        "Total nonfarm payroll employment increased by 275000 in February across health care and government.",
        top_k=3,
    )

    assert len(results) > 0
    assert results[0].passage_id == "psg:bls_jobs"
    assert results[0].rank == 1
    # Normalized inner product with identical vector is 1.0
    assert results[0].score > 0.99
    assert results[0].passage.domain == "bls.gov"
    assert results[0].passage.date == "2024-03-08"


def test_dense_index_save_and_load_preserves_separate_passage_ids(tmp_path: Path):
    out_dir = tmp_path / "indexes" / "dense_save_load"
    p1 = Passage(passage_id="doc_a", text="Alpha text snippet", url="https://a.com", date="2024-01-01", domain="a.com")
    p2 = Passage(passage_id="doc_b", text="Beta text snippet", url="https://b.com", date="2024-02-01", domain="b.com")
    store = PassageStore([p1, p2])

    encoder = DenseEncoder(
        model_name_or_path="test-embedder",
        device="cpu",
        normalize_embeddings=True,
        embedder=MockEmbedder(dimension=16),
    )

    index1 = DenseIndex.build(store, encoder=encoder, index_name="dense_save_load")
    index1.save(out_dir)

    # Check that FAISS index, separate passage_ids.json, passages.jsonl, and metadata exist
    assert (out_dir / "dense_index.faiss").is_file()
    assert (out_dir / "passage_ids.json").is_file()
    assert (out_dir / "passages.jsonl").is_file()
    assert (out_dir / "index_metadata.json").is_file()

    # Verify separate passage_ids.json format
    with (out_dir / "passage_ids.json").open("r", encoding="utf-8") as f:
        ids_data = json.load(f)
    assert ids_data == ["doc_a", "doc_b"]

    # Verify metadata record
    with (out_dir / "index_metadata.json").open("r", encoding="utf-8") as f:
        meta_data = json.load(f)
    assert meta_data["dimension"] == 16
    assert meta_data["normalize_embeddings"] is True
    assert meta_data["index_type"] == "IndexFlatIP"
    assert meta_data["total_passages"] == 2

    # Load back
    index2 = DenseIndex.load(out_dir, encoder=encoder)
    assert index2.index.ntotal == 2
    assert index2.passage_ids == ["doc_a", "doc_b"]

    res = index2.search("Alpha text snippet", top_k=1)
    assert res[0].passage_id == "doc_a"
    assert res[0].passage.url == "https://a.com"


def test_build_or_load_reuses_cached_dense_index(tmp_path: Path, monkeypatch):
    out_dir = tmp_path / "indexes" / "dense_cache"
    store = PassageStore([Passage(passage_id="doc_1", text="Cached text")])

    encoder = DenseEncoder(
        model_name_or_path="mock-embedder",
        device="cpu",
        normalize_embeddings=True,
        embedder=MockEmbedder(dimension=16),
    )

    # 1. First build
    idx1 = DenseIndex.build_or_load(store, out_dir, encoder=encoder)
    assert (out_dir / "index_metadata.json").is_file()

    # 2. Track calls to DenseIndex.build
    build_called = False
    original_build = DenseIndex.build

    def tracked_build(*args, **kwargs):
        nonlocal build_called
        build_called = True
        return original_build(*args, **kwargs)

    monkeypatch.setattr(DenseIndex, "build", tracked_build)

    # 3. Second call with unchanged corpus and config -> should reuse cache without building
    idx2 = DenseIndex.build_or_load(store, out_dir, encoder=encoder, force_rebuild=False)
    assert not build_called
    assert idx2.index.ntotal == 1


def test_empty_corpus_search_returns_empty():
    store = PassageStore([])
    encoder = DenseEncoder(
        model_name_or_path="mock-embedder",
        device="cpu",
        embedder=MockEmbedder(dimension=16),
    )
    dense_index = DenseIndex.build(store, encoder=encoder)
    assert dense_index.search("query statement", top_k=5) == []


def test_run_build_dense_index_smoke_test(tmp_path: Path):
    out_dir = tmp_path / "indexes" / "smoke_dense"
    summary = run_build_dense_index(
        index_dir=out_dir,
        smoke_test=True,
        test_query="vaccination hospitalizations",
    )

    assert summary["status"] == "success"
    assert summary["total_passages"] == 3
    assert (out_dir / "dense_index.faiss").is_file()
    assert (out_dir / "passage_ids.json").is_file()
    assert (out_dir / "passages.jsonl").is_file()
    assert (out_dir / "index_metadata.json").is_file()

    results = summary.get("test_query_results", [])
    assert len(results) > 0
    assert "score" in results[0]
    assert "passage_id" in results[0]
