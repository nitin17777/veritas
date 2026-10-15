"""Tests for the unified data preparation pipeline command."""

import json
from pathlib import Path

import pytest

from veritas.data_prep.prepare import run_data_prep


def test_prepare_skips_missing_datasets_gracefully(tmp_path: Path):
    """Missing datasets must be skipped gracefully without downloading or crashing."""
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    raw_root.mkdir()

    summary = run_data_prep(
        datasets=["fever", "rawfc"],
        raw_root=raw_root,
        processed_root=processed_root,
        strict=False,
    )

    assert "fever" not in summary["results"]
    assert len(summary["skipped"]) == 2
    assert summary["skipped"][0]["dataset"] == "fever"
    assert summary["skipped"][1]["dataset"] == "rawfc"

    # Summary file must be saved
    summary_file = processed_root / "ingestion_summary.json"
    assert summary_file.is_file()


def test_prepare_processes_existing_dataset(tmp_path: Path):
    """When a dataset exists, prepare runs ingestion and outputs summary metrics."""
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    fever_raw = raw_root / "fever"
    fever_raw.mkdir(parents=True)

    # Write a tiny synthetic FEVER fixture
    sample_fever = [
        {"id": 1, "claim": "Sample claim one.", "label": "SUPPORTS"},
        {"id": 2, "claim": "Sample claim two.", "label": "REFUTES"},
    ]
    with (fever_raw / "train.jsonl").open("w", encoding="utf-8") as f:
        for item in sample_fever:
            f.write(json.dumps(item) + "\n")

    summary = run_data_prep(
        datasets=["fever"],
        raw_root=raw_root,
        processed_root=processed_root,
    )

    assert "fever" in summary["results"]
    assert summary["results"]["fever"]["status"] == "success"
    assert summary["results"]["fever"]["counts"] == {"train": 2}
    assert (processed_root / "fever" / "train.jsonl").is_file()
