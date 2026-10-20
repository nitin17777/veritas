"""Tests for VERITAS BM25 Retrieval Index."""

import json
from pathlib import Path

import pytest

from veritas.retrieval.bm25 import BM25Index, compute_config_fingerprint, tokenize_bm25
from veritas.retrieval.corpus import PassageStore, create_synthetic_corpus
from veritas.retrieval.schema import Passage


def test_tokenize_bm25_consistency_and_case_insensitivity():
    text1 = "COVID-19 vaccination reduced hospitalizations by 90% in 2024!"
    text2 = "covid-19 VACCINATION reduced HOSPITALIZATIONS by 90% in 2024!"

    tokens1 = tokenize_bm25(text1)
    tokens2 = tokenize_bm25(text2)

    assert tokens1 == tokens2
    assert "covid" in tokens1
    assert "vaccination" in tokens1
    assert "90" in tokens1
    assert "2024" in tokens1
    # Verify all tokens are strictly lowercase
    assert all(t.islower() or t.isdigit() for t in tokens1)


def test_tokenize_bm25_empty_and_punctuation():
    assert tokenize_bm25("") == []
    assert tokenize_bm25("   ... ,,, !!! ???   ") == []
    assert tokenize_bm25(None) == []


def test_exact_sentence_retrieves_passage_at_rank_1():
    """CRITICAL TEST: A query containing a known passage's exact sentence retrieves that passage at rank 1."""
    p1 = Passage(
        passage_id="psg:cdc_flu",
        text="Annual influenza vaccination is recommended for everyone 6 months and older to prevent respiratory complications.",
        url="https://cdc.gov/flu",
        date="2024-01-15",
        domain="cdc.gov",
    )
    p2 = Passage(
        passage_id="psg:bls_jobs",
        text="Total nonfarm payroll employment increased by 275000 in February as the unemployment rate changed little.",
        url="https://bls.gov/news",
        date="2024-03-08",
        domain="bls.gov",
    )
    p3 = Passage(
        passage_id="psg:nasa_jwst",
        text="The James Webb Space Telescope observed atmospheric water vapor and carbon dioxide on a distant gas giant exoplanet.",
        url="https://nasa.gov/jwst",
        date="2023-10-12",
        domain="nasa.gov",
    )
    p4 = Passage(
        passage_id="psg:noaa_ocean",
        text="Global sea surface temperatures reached record highs throughout the boreal summer driven by marine heatwaves.",
        url="https://noaa.gov/oceans",
        date="2023-09-01",
        domain="noaa.gov",
    )

    store = PassageStore([p1, p2, p3, p4])
    index = BM25Index.build(store)

    # 1. Exact sentence query from p1
    query_p1 = "Annual influenza vaccination is recommended for everyone 6 months and older"
    results_p1 = index.search(query_p1, top_k=3)
    assert len(results_p1) > 0
    assert results_p1[0].passage_id == "psg:cdc_flu"
    assert results_p1[0].rank == 1
    assert results_p1[0].score > 0.0
    assert results_p1[0].passage.domain == "cdc.gov"

    # 2. Exact sentence query from p3 with mixed casing
    query_p3 = "THE JAMES WEBB SPACE TELESCOPE OBSERVED ATMOSPHERIC WATER VAPOR"
    results_p3 = index.search(query_p3, top_k=3)
    assert len(results_p3) > 0
    assert results_p3[0].passage_id == "psg:nasa_jwst"
    assert results_p3[0].rank == 1
    assert results_p3[0].score > 0.0
    assert results_p3[0].passage.url == "https://nasa.gov/jwst"

    # 3. Exact sentence query from p2
    query_p2 = "Total nonfarm payroll employment increased by 275000 in February"
    results_p2 = index.search(query_p2, top_k=3)
    assert len(results_p2) > 0
    assert results_p2[0].passage_id == "psg:bls_jobs"
    assert results_p2[0].rank == 1
    assert results_p2[0].score > 0.0


def test_passage_id_ordering_preserved_and_mapped_correctly():
    """Ensure search results map to the exact underlying Passage records, preserving ordering."""
    passages = [
        Passage(passage_id=f"doc_{i}", text=f"Distinct unique keyword_{i} and general context topic")
        for i in range(10)
    ]
    store = PassageStore(passages)
    index = BM25Index.build(store)

    for i in range(10):
        res = index.search(f"keyword_{i}", top_k=1)
        assert len(res) == 1
        assert res[0].passage_id == f"doc_{i}"
        assert res[0].passage.text == passages[i].text
        assert res[0].rank == 1


def test_empty_corpus_search_returns_empty():
    store = PassageStore([])
    index = BM25Index.build(store)
    results = index.search("any query", top_k=5)
    assert results == []


def test_empty_query_returns_empty():
    store = create_synthetic_corpus([Passage(passage_id="p1", text="Sample text")])
    index = BM25Index.build(store)
    assert index.search("", top_k=5) == []
    assert index.search("   ", top_k=5) == []
    assert index.search("!@#$%", top_k=5) == []


def test_index_serialization_and_metadata_saved_under_indexes(tmp_path: Path):
    index_dir = tmp_path / "indexes" / "test_bm25"
    p1 = Passage(
        passage_id="p1",
        text="Renewable energy generation exceeded fossil fuels in multiple European nations.",
        url="https://energy.eu/stat",
        date="2024-02-15",
        domain="energy.eu",
    )
    p2 = Passage(
        passage_id="p2",
        text="Federal Reserve kept target interest rates unchanged between 5.25 and 5.50 percent.",
        url="https://federalreserve.gov/monetary",
        date="2024-01-31",
        domain="federalreserve.gov",
    )
    store = PassageStore([p1, p2])

    index = BM25Index.build(
        store,
        k1=1.6,
        b=0.8,
        index_name="test_bm25",
        source_datasets=["custom_energy"],
    )
    saved_path = index.save(index_dir)

    assert saved_path == index_dir
    assert (index_dir / "passages.jsonl").is_file()
    assert (index_dir / "bm25_index.joblib").is_file()
    assert (index_dir / "index_metadata.json").is_file()

    # Verify metadata contents
    with (index_dir / "index_metadata.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["index_name"] == "test_bm25"
    assert meta["backend"] == "BM25Okapi"
    assert meta["total_passages"] == 2
    assert meta["k1"] == 1.6
    assert meta["b"] == 0.8
    assert "corpus_fingerprint" in meta
    assert meta["source_datasets"] == ["custom_energy"]

    # Test loading from disk
    loaded_index = BM25Index.load(index_dir)
    assert len(loaded_index.passage_store) == 2
    assert loaded_index.k1 == 1.6
    assert loaded_index.b == 0.8

    # Query loaded index
    results = loaded_index.search("Federal Reserve interest rates", top_k=2)
    assert len(results) > 0
    assert results[0].passage_id == "p2"
    assert results[0].passage.domain == "federalreserve.gov"


def test_build_or_load_reuses_cache_when_unchanged(tmp_path: Path, monkeypatch):
    index_dir = tmp_path / "indexes" / "cache_test"
    store = PassageStore(
        [
            Passage(passage_id="p1", text="Artificial intelligence models demonstrate reasoning."),
            Passage(passage_id="p2", text="Quantum computing explores error correction codes."),
        ]
    )

    # First build
    idx1 = BM25Index.build_or_load(store, index_dir, k1=1.5, b=0.75)
    assert (index_dir / "index_metadata.json").is_file()

    # Track calls to BM25Index.build to verify it is NOT called during reuse
    build_called = False
    original_build = BM25Index.build

    def tracked_build(*args, **kwargs):
        nonlocal build_called
        build_called = True
        return original_build(*args, **kwargs)

    monkeypatch.setattr(BM25Index, "build", tracked_build)

    # Second call with identical store and config -> should reuse without rebuilding!
    idx2 = BM25Index.build_or_load(store, index_dir, k1=1.5, b=0.75, force_rebuild=False)
    assert not build_called
    assert len(idx2.passage_store) == 2


def test_build_or_load_rebuilds_when_corpus_changes(tmp_path: Path):
    index_dir = tmp_path / "indexes" / "rebuild_corpus_test"
    store1 = PassageStore([Passage(passage_id="p1", text="Original document text")])
    BM25Index.build_or_load(store1, index_dir, k1=1.5, b=0.75)

    # Change corpus
    store2 = PassageStore(
        [
            Passage(passage_id="p1", text="Original document text"),
            Passage(passage_id="p2", text="Brand new second document"),
        ]
    )
    idx2 = BM25Index.build_or_load(store2, index_dir, k1=1.5, b=0.75, force_rebuild=False)
    assert len(idx2.passage_store) == 2
    assert idx2.metadata.total_passages == 2


def test_build_or_load_rebuilds_when_config_changes(tmp_path: Path):
    index_dir = tmp_path / "indexes" / "rebuild_config_test"
    store = PassageStore([Passage(passage_id="p1", text="Consistent passage")])
    BM25Index.build_or_load(store, index_dir, k1=1.5, b=0.75)

    # Change k1 parameter
    idx2 = BM25Index.build_or_load(store, index_dir, k1=1.8, b=0.75, force_rebuild=False)
    assert idx2.k1 == 1.8
    assert idx2.metadata.k1 == 1.8


def test_build_or_load_force_rebuild(tmp_path: Path, monkeypatch):
    index_dir = tmp_path / "indexes" / "force_test"
    store = PassageStore([Passage(passage_id="p1", text="Passage text")])
    BM25Index.build_or_load(store, index_dir, k1=1.5, b=0.75)

    build_called = False
    original_build = BM25Index.build

    def tracked_build(*args, **kwargs):
        nonlocal build_called
        build_called = True
        return original_build(*args, **kwargs)

    monkeypatch.setattr(BM25Index, "build", tracked_build)

    BM25Index.build_or_load(store, index_dir, k1=1.5, b=0.75, force_rebuild=True)
    assert build_called
