"""Tests for AVeriTeC dataset loader using synthetic fixtures."""

import json
from pathlib import Path

import pytest

from veritas.data_prep.averitec_loader import (
    load_averitec,
    load_averitec_split,
    parse_averitec_item,
)
from veritas.data_prep.label_map import PROPOSED_AVERITEC_LABEL_MAP
from veritas.data_prep.schema import validate_claim_record


@pytest.fixture
def synthetic_averitec_items():
    return [
        {
            "claim_id": 501,
            "claim": "Renewable energy supplied 20 percent of electricity generation in the United States in 2020.",
            "label": "Supported",
            "claim_date": "2021-04-10",
            "speaker": "Energy Information Administration",
            "questions": [
                {
                    "question": "What percentage of US electricity was renewable in 2020?",
                    "answers": [
                        {
                            "answer": "Renewables accounted for approximately 20% of US electricity generation in 2020.",
                            "source_url": "https://www.eia.gov/todayinenergy/detail.php?id=48896",
                            "date": "2021-04-11",
                        }
                    ],
                }
            ],
        },
        {
            "claim_id": 502,
            "claim": "Coffee consumption reduces mortality risk by 90 percent.",
            "label": "Conflicting Evidence/Cherry-picking",
            "claim_date": "2018-07-02",
            "speaker": "Social Media Post",
            "questions": [
                {
                    "question": "Does coffee reduce mortality by 90%?",
                    "answers": [
                        {
                            "answer": "Studies show a modest 10-15% association with lower mortality, not 90%.",
                            "source_url": "https://jamanetwork.com/journals/jamainternalmedicine",
                            "date": "2018-07-03",
                        }
                    ],
                }
            ],
        },
    ]


def test_averitec_unactivated_label_map_raises_error(synthetic_averitec_items):
    """Without an approved label_map, parsing must refuse and prompt for human review."""
    item = synthetic_averitec_items[0]
    with pytest.raises(ValueError, match="AVeriTeC label mapping has not been activated"):
        parse_averitec_item(item, split="train", label_map=None)


def test_averitec_parsing_with_proposed_mapping(synthetic_averitec_items):
    """With proposed mapping passed, AVeriTeC items are normalized to ClaimRecord."""
    item = synthetic_averitec_items[0]
    record = parse_averitec_item(item, split="train", label_map=PROPOSED_AVERITEC_LABEL_MAP)

    assert record["id"] == "averitec:501"
    assert record["label"] == "True"
    assert "Renewable energy supplied" in record["claim"]
    assert record["claim_date"] == "2021-04-10"
    assert record["split"] == "train"
    assert len(record["evidence"]) == 1
    assert "Renewables accounted for approximately 20%" in record["evidence"][0]["text"]
    assert record["evidence"][0]["url"] == "https://www.eia.gov/todayinenergy/detail.php?id=48896"
    assert record["evidence"][0]["domain"] == "www.eia.gov"
    assert record["evidence"][0]["date"] == "2021-04-11"
    validate_claim_record(record)


def test_averitec_cherry_picking_maps_to_misleading(synthetic_averitec_items):
    """Cherry-picking rating maps to Misleading under proposed mapping."""
    item = synthetic_averitec_items[1]
    record = parse_averitec_item(item, split="validation", label_map=PROPOSED_AVERITEC_LABEL_MAP)

    assert record["id"] == "averitec:502"
    assert record["label"] == "Misleading"
    assert record["evidence"][0]["domain"] == "jamanetwork.com"
    validate_claim_record(record)


def test_averitec_missing_fields_raises():
    """Missing required fields in AVeriTeC item must raise ValueError."""
    with pytest.raises(ValueError, match="Missing required AVeriTeC field"):
        parse_averitec_item({"claim": "test", "label": "Supported"}, split="train", label_map=PROPOSED_AVERITEC_LABEL_MAP)


def test_load_averitec_split_end_to_end(tmp_path: Path, synthetic_averitec_items):
    """Test loading an AVeriTeC split file from JSON format."""
    split_file = tmp_path / "dev.json"
    with split_file.open("w", encoding="utf-8") as f:
        json.dump(synthetic_averitec_items, f)

    records, summary = load_averitec_split(
        split_file,
        split="validation",
        label_map=PROPOSED_AVERITEC_LABEL_MAP,
        strict=True,
    )

    assert len(records) == 2
    assert summary.valid_records == 2
    assert summary.malformed_rows == 0
    assert summary.label_distribution == {"True": 1, "Misleading": 1}
