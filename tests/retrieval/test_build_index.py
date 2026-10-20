"""Tests for VERITAS BM25 index build command and CLI execution."""

import json
from pathlib import Path

import pytest

from veritas.retrieval.build_index import run_build_index


def test_run_build_index_smoke_test(tmp_path: Path):
    out_dir = tmp_path / "indexes" / "smoke_index"
    summary = run_build_index(
        index_dir=out_dir,
        smoke_test=True,
        test_query="vaccination hospitalizations",
    )

    assert summary["status"] == "success"
    assert summary["total_passages"] == 3
    assert (out_dir / "passages.jsonl").is_file()
    assert (out_dir / "bm25_index.joblib").is_file()
    assert (out_dir / "index_metadata.json").is_file()

    # Verify test query retrieved matching result
    query_results = summary.get("test_query_results", [])
    assert len(query_results) > 0
    assert query_results[0]["rank"] == 1
    assert query_results[0]["passage_id"] == "claim_test_1#e0"
    assert "WHO" in query_results[0]["passage"]["text"] or "vaccination" in query_results[0]["passage"]["text"]


def test_run_build_index_with_liar_only_creates_zero_passages(tmp_path: Path):
    """VERITAS constraint test: LIAR alone yields 0 passages, never fabricating evidence."""
    processed_dir = tmp_path / "data" / "processed"
    liar_dir = processed_dir / "liar"
    liar_dir.mkdir(parents=True)

    # Mock LIAR claim record with empty evidence
    rec = {
        "id": "liar:1001.json",
        "claim": "The tax cuts generated record federal revenue in two years.",
        "label": "False",
        "claim_date": None,
        "topic": "taxes",
        "split": "train",
        "evidence": [],
    }
    with (liar_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    out_dir = tmp_path / "indexes" / "liar_index"
    summary = run_build_index(
        processed_dir=processed_dir,
        datasets=["liar"],
        index_dir=out_dir,
        force_rebuild=True,
    )

    assert summary["total_passages"] == 0
    assert (out_dir / "passages.jsonl").is_file()
    assert (out_dir / "index_metadata.json").is_file()


def test_run_build_index_rebuild_and_reuse_cache(tmp_path: Path):
    out_dir = tmp_path / "indexes" / "cache_index"

    # 1. First run in smoke mode
    res1 = run_build_index(index_dir=out_dir, smoke_test=True)
    assert res1["status"] == "success"

    # 2. Second run without force_rebuild
    res2 = run_build_index(index_dir=out_dir, smoke_test=True, force_rebuild=False)
    assert res2["status"] == "success"
    assert res2["total_passages"] == res1["total_passages"]


def test_run_build_index_from_config(tmp_path: Path):
    config_file = tmp_path / "test_retrieval.yaml"
    out_dir = tmp_path / "indexes" / "cfg_index"

    yaml_content = f"""
project:
  name: VERITAS
  seed: 42

retrieval:
  backend: bm25
  index_dir: "{out_dir.as_posix()}"
  k1: 1.6
  b: 0.8
  chunk_size: 140
  chunk_overlap: 25
"""
    with config_file.open("w", encoding="utf-8") as f:
        f.write(yaml_content)

    summary = run_build_index(
        config_path=config_file,
        smoke_test=True,
    )

    assert summary["k1"] == 1.6
    assert summary["b"] == 0.8
    assert summary["chunk_size"] == 140
    assert summary["chunk_overlap"] == 25
