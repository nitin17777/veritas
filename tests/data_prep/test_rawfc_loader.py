"""Tests for RAWFC dataset loader using synthetic fixtures."""

import json
from pathlib import Path

import pytest

from veritas.data_prep.label_map import PROPOSED_RAWFC_LABEL_MAP
from veritas.data_prep.rawfc_loader import (
    load_rawfc,
    load_rawfc_split,
    parse_rawfc_item,
)
from veritas.data_prep.schema import validate_claim_record


@pytest.fixture
def synthetic_rawfc_items():
    return [
        {
            "id": "snopes_1001",
            "claim": "A massive sea monster washed ashore on a Florida beach.",
            "label": "false",
            "date": "2019-06-12",
            "topic": "critters",
            "reports": [
                {
                    "text": "Marine biologists confirmed the object was a decayed basking shark.",
                    "url": "https://example.com/shark",
                    "date": "2019-06-13",
                    "domain": "example.com",
                }
            ],
        },
        {
            "id": "snopes_1002",
            "claim": "The state legislature passed a partial gas tax holiday.",
            "label": "mixture",
            "date": "2020-03-01",
            "topic": "politics",
            "reports": [],
        },
    ]


def test_rawfc_unactivated_label_map_raises_error(synthetic_rawfc_items):
    """Without an approved label_map, parsing must refuse and prompt for human review."""
    item = synthetic_rawfc_items[0]
    with pytest.raises(ValueError, match="RAWFC label mapping has not been activated"):
        parse_rawfc_item(item, split="train", label_map=None)


def test_rawfc_parsing_with_proposed_mapping(synthetic_rawfc_items):
    """With proposed mapping passed, RAWFC items are normalized to ClaimRecord."""
    item = synthetic_rawfc_items[0]
    record = parse_rawfc_item(item, split="train", label_map=PROPOSED_RAWFC_LABEL_MAP)

    assert record["id"] == "rawfc:snopes_1001"
    assert record["label"] == "False"
    assert record["claim"] == "A massive sea monster washed ashore on a Florida beach."
    assert record["claim_date"] == "2019-06-12"
    assert record["topic"] == "critters"
    assert record["split"] == "train"
    assert len(record["evidence"]) == 1
    assert "decayed basking shark" in record["evidence"][0]["text"]
    assert record["evidence"][0]["date"] == "2019-06-13"
    assert record["evidence"][0]["domain"] == "example.com"
    validate_claim_record(record)


def test_rawfc_mixture_label_mapping(synthetic_rawfc_items):
    """Snopes 'mixture' maps to 'Partially True' under the proposed mapping."""
    item = synthetic_rawfc_items[1]
    record = parse_rawfc_item(item, split="validation", label_map=PROPOSED_RAWFC_LABEL_MAP)

    assert record["id"] == "rawfc:snopes_1002"
    assert record["label"] == "Partially True"
    assert record["evidence"] == []
    validate_claim_record(record)


def test_rawfc_missing_fields_raises():
    """Missing required fields in RAWFC item must raise ValueError."""
    with pytest.raises(ValueError, match="Missing required RAWFC field 'id'"):
        parse_rawfc_item({"claim": "test", "label": "false"}, split="train", label_map=PROPOSED_RAWFC_LABEL_MAP)

    with pytest.raises(ValueError, match="Missing required RAWFC field 'claim'"):
        parse_rawfc_item({"id": "1", "label": "false"}, split="train", label_map=PROPOSED_RAWFC_LABEL_MAP)


def test_load_rawfc_split_end_to_end(tmp_path: Path, synthetic_rawfc_items):
    """Test loading a RAWFC split file from JSON format."""
    split_file = tmp_path / "train.json"
    with split_file.open("w", encoding="utf-8") as f:
        json.dump(synthetic_rawfc_items, f)

    records, summary = load_rawfc_split(
        split_file,
        split="train",
        label_map=PROPOSED_RAWFC_LABEL_MAP,
        strict=True,
    )

    assert len(records) == 2
    assert summary.valid_records == 2
    assert summary.malformed_rows == 0
    assert summary.label_distribution == {"False": 1, "Partially True": 1}
    assert summary.original_label_distribution == {"false": 1, "mixture": 1}
