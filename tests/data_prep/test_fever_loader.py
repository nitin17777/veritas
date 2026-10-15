"""Tests for FEVER dataset loader using synthetic fixtures."""

import json
from pathlib import Path

import pytest

from veritas.data_prep.fever_loader import (
    load_fever,
    load_fever_split,
    parse_fever_item,
)
from veritas.data_prep.schema import validate_claim_record


@pytest.fixture
def synthetic_fever_items():
    return [
        {
            "id": 101,
            "verifiable": "VERIFIABLE",
            "label": "SUPPORTS",
            "claim": "The Pacific Ocean is the largest ocean on Earth.",
            "evidence": [
                [[1, 1, "Pacific_Ocean", 0]]
            ],
        },
        {
            "id": 102,
            "verifiable": "VERIFIABLE",
            "label": "REFUTES",
            "claim": "The Moon is larger than Jupiter.",
            "evidence": [
                [[2, 2, "Moon", 3]]
            ],
        },
        {
            "id": 103,
            "verifiable": "NOT VERIFIABLE",
            "label": "NOT ENOUGH INFO",
            "claim": "A secret meeting occurred on Mars in 1940.",
            "evidence": [],
        },
    ]


def test_parse_fever_item_supports(synthetic_fever_items):
    """FEVER SUPPORTS maps to True, date is None, pointers in metadata."""
    item = synthetic_fever_items[0]
    record = parse_fever_item(item, split="train")

    assert record["id"] == "fever:101"
    assert record["label"] == "True"
    assert record["claim"] == "The Pacific Ocean is the largest ocean on Earth."
    assert record["claim_date"] is None
    assert record["split"] == "train"
    assert record["evidence"] == []
    assert record["metadata"]["source_id"] == "101"
    assert record["metadata"]["verifiable"] == "VERIFIABLE"
    assert record["metadata"]["raw_evidence"] == [[[1, 1, "Pacific_Ocean", 0]]]
    validate_claim_record(record)


def test_parse_fever_item_refutes(synthetic_fever_items):
    """FEVER REFUTES maps to False."""
    item = synthetic_fever_items[1]
    record = parse_fever_item(item, split="validation")

    assert record["id"] == "fever:102"
    assert record["label"] == "False"
    assert record["split"] == "validation"
    validate_claim_record(record)


def test_parse_fever_item_not_enough_info(synthetic_fever_items):
    """FEVER NOT ENOUGH INFO maps to Unverified."""
    item = synthetic_fever_items[2]
    record = parse_fever_item(item, split="test")

    assert record["id"] == "fever:103"
    assert record["label"] == "Unverified"
    assert record["split"] == "test"
    validate_claim_record(record)


def test_parse_fever_item_missing_required_fields():
    """Missing id, claim, or label must raise ValueError."""
    with pytest.raises(ValueError, match="Missing required FEVER field 'id'"):
        parse_fever_item({"claim": "test", "label": "SUPPORTS"}, split="train")

    with pytest.raises(ValueError, match="Missing required FEVER field 'claim'"):
        parse_fever_item({"id": 1, "label": "SUPPORTS"}, split="train")

    with pytest.raises(ValueError, match="Missing required FEVER field 'label'"):
        parse_fever_item({"id": 1, "claim": "test"}, split="train")


def test_parse_fever_item_empty_fields():
    """Empty claim or ID strings must raise ValueError."""
    with pytest.raises(ValueError, match="FEVER claim ID cannot be empty"):
        parse_fever_item({"id": "   ", "claim": "test", "label": "SUPPORTS"}, split="train")

    with pytest.raises(ValueError, match="FEVER claim statement cannot be empty"):
        parse_fever_item({"id": 1, "claim": "   ", "label": "SUPPORTS"}, split="train")


def test_load_fever_split_with_malformed_lines(tmp_path: Path):
    """Malformed JSON and invalid rows must be recorded in summary."""
    jsonl_file = tmp_path / "paper_dev.jsonl"
    lines = [
        json.dumps({"id": 1, "claim": "Valid claim", "label": "SUPPORTS"}),
        "INVALID_JSON_LINE",
        json.dumps({"id": 2, "claim": "Bad label", "label": "UNKNOWN_VERACITY"}),
        json.dumps({"id": 3, "claim": "Another valid", "label": "REFUTES"}),
    ]
    jsonl_file.write_text("\n".join(lines), encoding="utf-8")

    records, summary = load_fever_split(jsonl_file, split="validation", strict=False)

    assert len(records) == 2
    assert summary.valid_records == 2
    assert summary.malformed_rows == 2
    assert summary.records_processed == 4
    assert summary.label_distribution == {"True": 1, "False": 1}
    assert any("Unknown label" in u for u in summary.unknown_labels)


def test_load_fever_end_to_end_synthetic(tmp_path: Path, synthetic_fever_items):
    """End-to-end load_fever with synthetic directory fixture."""
    raw_dir = tmp_path / "raw_fever"
    out_dir = tmp_path / "processed_fever"
    raw_dir.mkdir()

    # Create train and dev split files
    train_file = raw_dir / "train.jsonl"
    dev_file = raw_dir / "paper_dev.jsonl"

    with train_file.open("w", encoding="utf-8") as f:
        for item in synthetic_fever_items:
            f.write(json.dumps(item) + "\n")

    with dev_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps(synthetic_fever_items[0]) + "\n")

    counts = load_fever(raw_dir=raw_dir, output_dir=out_dir)

    assert counts["train"] == 3
    assert counts["validation"] == 1
    assert "test" not in counts  # test split not present in raw fixture, skipped gracefully

    # Verify output JSONL files
    out_train = out_dir / "train.jsonl"
    assert out_train.is_file()
    with out_train.open("r", encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    assert len(recs) == 3
    for r in recs:
        validate_claim_record(r)
