"""Tests for VERITAS shared data contracts and ClaimRecord/EvidenceRecord schema."""

import json
from pathlib import Path

import pytest

from veritas.data_prep.schema import (
    REQUIRED_CLAIM_FIELDS,
    REQUIRED_EVIDENCE_FIELDS,
    VALID_LABELS,
    VALID_SPLITS,
    is_valid_iso_date,
    validate_claim_record,
    validate_evidence_record,
)


def _valid_minimal_record():
    return {
        "id": "claim:001",
        "claim": "The Earth orbits the Sun.",
        "label": "True",
        "claim_date": None,
        "topic": None,
        "split": "train",
        "evidence": [],
    }


def _valid_full_record():
    return {
        "id": "claim:002",
        "claim": "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        "label": "True",
        "claim_date": "2024-03-15",
        "topic": "science,physics",
        "split": "validation",
        "evidence": [
            {
                "text": "Water boils at 100 °C (212 °F) at 1 atm.",
                "url": "https://en.wikipedia.org/wiki/Water",
                "date": "2024-01-10",
                "domain": "en.wikipedia.org",
            },
            {
                "text": "Thermodynamic properties of water.",
                "url": None,
                "date": None,
                "domain": None,
            },
        ],
        "metadata": {
            "source_dataset": "synthetic_benchmark",
            "annotator_id": "ann_42",
        },
    }


# ==============================================================================
# 1. Valid Record Tests
# ==============================================================================


def test_valid_minimal_record():
    """Minimal claim record without optional fields must validate."""
    record = _valid_minimal_record()
    validate_claim_record(record)


def test_valid_full_record():
    """Full claim record with dates, evidence items, and metadata must validate."""
    record = _valid_full_record()
    validate_claim_record(record)


@pytest.mark.parametrize("label", sorted(VALID_LABELS))
def test_all_five_valid_labels_pass(label):
    """All 5 defined labels must pass validation."""
    record = _valid_minimal_record()
    record["label"] = label
    validate_claim_record(record)


@pytest.mark.parametrize("split", sorted(VALID_SPLITS))
def test_all_four_valid_splits_pass(split):
    """All valid split values must pass validation."""
    record = _valid_minimal_record()
    record["split"] = split
    validate_claim_record(record)


# ==============================================================================
# 2. Missing Required Fields Tests
# ==============================================================================


@pytest.mark.parametrize("field_name", sorted(REQUIRED_CLAIM_FIELDS))
def test_missing_required_claim_field_raises(field_name):
    """Omitting any required core field must raise ValueError."""
    record = _valid_minimal_record()
    del record[field_name]
    with pytest.raises(ValueError, match="Missing required fields"):
        validate_claim_record(record)


def test_record_must_be_dict():
    """Non-dict inputs must raise ValueError."""
    with pytest.raises(ValueError, match="Claim record must be a dictionary"):
        validate_claim_record(["not", "a", "dict"])  # type: ignore


# ==============================================================================
# 3. Metadata Preservation & Cannot Replace Required Fields
# ==============================================================================


def test_metadata_cannot_replace_claim_field():
    """Having claim inside metadata does not satisfy top-level required claim."""
    record = _valid_minimal_record()
    del record["claim"]
    record["metadata"] = {"claim": "I am in metadata, not top level"}
    with pytest.raises(ValueError, match="Missing required fields:.*'claim'"):
        validate_claim_record(record)


def test_metadata_cannot_replace_label_field():
    """Having label inside metadata does not satisfy top-level required label."""
    record = _valid_minimal_record()
    del record["label"]
    record["metadata"] = {"label": "True"}
    with pytest.raises(ValueError, match="Missing required fields:.*'label'"):
        validate_claim_record(record)


def test_metadata_must_be_dict():
    """If metadata is present, it must be a dictionary."""
    record = _valid_minimal_record()
    record["metadata"] = "not_a_dictionary"
    with pytest.raises(ValueError, match="metadata must be a dictionary"):
        validate_claim_record(record)


# ==============================================================================
# 4. Field Type & Non-Empty Validation Tests
# ==============================================================================


@pytest.mark.parametrize("bad_id", [None, 123, "", "   ", ["id"]])
def test_invalid_claim_id_raises(bad_id):
    """Claim ID must be a non-empty string."""
    record = _valid_minimal_record()
    record["id"] = bad_id
    with pytest.raises(ValueError, match="Claim id must be a non-empty string"):
        validate_claim_record(record)


@pytest.mark.parametrize("bad_claim", [None, 123, "", "   ", {"text": "claim"}])
def test_invalid_claim_text_raises(bad_claim):
    """Claim must be a non-empty string."""
    record = _valid_minimal_record()
    record["claim"] = bad_claim
    with pytest.raises(ValueError, match="Claim must be a non-empty string"):
        validate_claim_record(record)


@pytest.mark.parametrize("bad_topic", [123, [], {}, True])
def test_invalid_topic_type_raises(bad_topic):
    """Topic must be string or None."""
    record = _valid_minimal_record()
    record["topic"] = bad_topic
    with pytest.raises(ValueError, match="topic must be a string or None"):
        validate_claim_record(record)


# ==============================================================================
# 5. Label & Split Validation Tests
# ==============================================================================


@pytest.mark.parametrize(
    "invalid_label",
    ["pants-fire", "barely-true", "mostly-true", "supported", "refuted", "TRUE", "fake", ""],
)
def test_invalid_label_raises(invalid_label):
    """Labels outside the 5 VERITAS labels must raise ValueError."""
    record = _valid_minimal_record()
    record["label"] = invalid_label
    with pytest.raises(ValueError, match="Invalid VERITAS label"):
        validate_claim_record(record)


@pytest.mark.parametrize(
    "invalid_split",
    ["train_split", "val", "dev", "eval", "heldout", "TRAIN", ""],
)
def test_invalid_split_raises(invalid_split):
    """Splits outside the 4 canonical splits must raise ValueError."""
    record = _valid_minimal_record()
    record["split"] = invalid_split
    with pytest.raises(ValueError, match="Invalid split"):
        validate_claim_record(record)


# ==============================================================================
# 6. Date Validation Tests (claim_date and evidence date)
# ==============================================================================


@pytest.mark.parametrize(
    "valid_date",
    [
        "2024-03-15",
        "2024-02-29",  # Leap year
        "2000-02-29",  # Century leap year
        "1999-12-31",
        "2026-01-01",
    ],
)
def test_valid_iso_dates(valid_date):
    """Valid ISO YYYY-MM-DD dates must pass validation."""
    assert is_valid_iso_date(valid_date) is True
    record = _valid_minimal_record()
    record["claim_date"] = valid_date
    validate_claim_record(record)


@pytest.mark.parametrize(
    "invalid_date",
    [
        "2023-02-29",  # 2023 is not a leap year
        "2024-02-30",  # Feb 30 does not exist
        "2024-04-31",  # April has 30 days
        "2024-13-01",  # Month 13
        "2024-00-10",  # Month 00
        "2024-05-00",  # Day 00
        "2024/05/15",  # Slashes
        "05/15/2024",  # MM/DD/YYYY
        "15-05-2024",  # DD-MM-YYYY
        "May 15, 2024",
        "2024-5-15",   # Missing leading zero
        "2024-05-15T12:00:00",  # ISO with time
        "2024-05-15Z",
        "not-a-date",
        "",
        "   ",
        12345,
    ],
)
def test_invalid_claim_date_raises(invalid_date):
    """Malformed or invalid calendar claim_date must raise ValueError."""
    record = _valid_minimal_record()
    record["claim_date"] = invalid_date
    with pytest.raises(ValueError, match="claim_date must be an ISO YYYY-MM-DD string or None"):
        validate_claim_record(record)


# ==============================================================================
# 7. Evidence Validation Tests
# ==============================================================================


def test_evidence_must_be_list():
    """Evidence field must be a list."""
    record = _valid_minimal_record()
    record["evidence"] = "not_a_list"
    with pytest.raises(ValueError, match="Evidence must be a list"):
        validate_claim_record(record)


def test_evidence_item_must_be_dict():
    """Each item in evidence must be a dictionary."""
    record = _valid_minimal_record()
    record["evidence"] = ["not_a_dict"]
    with pytest.raises(ValueError, match="Evidence item 0 must be a dictionary"):
        validate_claim_record(record)


@pytest.mark.parametrize("ev_field", sorted(REQUIRED_EVIDENCE_FIELDS))
def test_missing_evidence_field_raises(ev_field):
    """Missing any required field in evidence item must raise ValueError."""
    ev = {
        "text": "Evidence snippet.",
        "url": "https://example.com",
        "date": "2024-01-01",
        "domain": "example.com",
    }
    del ev[ev_field]
    with pytest.raises(ValueError, match=f"Evidence item missing fields:.*'{ev_field}'"):
        validate_evidence_record(ev)


@pytest.mark.parametrize("bad_text", [None, 123, "", "   "])
def test_invalid_evidence_text_raises(bad_text):
    """Evidence text must be a non-empty string."""
    ev = {
        "text": bad_text,
        "url": None,
        "date": None,
        "domain": None,
    }
    with pytest.raises(ValueError, match="text must be a non-empty string"):
        validate_evidence_record(ev)


@pytest.mark.parametrize("bad_url", [123, [], {}])
def test_invalid_evidence_url_raises(bad_url):
    """Evidence url must be a string or None."""
    ev = {
        "text": "Valid snippet",
        "url": bad_url,
        "date": None,
        "domain": None,
    }
    with pytest.raises(ValueError, match="url must be a string or None"):
        validate_evidence_record(ev)


@pytest.mark.parametrize("bad_domain", [123, [], {}])
def test_invalid_evidence_domain_raises(bad_domain):
    """Evidence domain must be a string or None."""
    ev = {
        "text": "Valid snippet",
        "url": None,
        "date": None,
        "domain": bad_domain,
    }
    with pytest.raises(ValueError, match="domain must be a string or None"):
        validate_evidence_record(ev)


@pytest.mark.parametrize("bad_date", ["2024/01/01", "2023-02-29", "invalid", 20240101])
def test_invalid_evidence_date_raises(bad_date):
    """Evidence date must be valid ISO YYYY-MM-DD or None."""
    ev = {
        "text": "Valid snippet",
        "url": None,
        "date": bad_date,
        "domain": None,
    }
    with pytest.raises(ValueError, match="date must be an ISO YYYY-MM-DD string or None"):
        validate_evidence_record(ev)


# ==============================================================================
# 8. LIAR Dataset Compatibility Test
# ==============================================================================


def test_processed_liar_records_are_compatible_with_schema():
    """All 12,836 processed LIAR records must pass schema validation."""
    base = Path("data/processed/liar")
    total_checked = 0

    for split in ("train", "validation", "test"):
        path = base / f"{split}.jsonl"
        if not path.is_file():
            pytest.skip(f"Processed file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                rec = json.loads(line)
                validate_claim_record(rec)
                assert rec["split"] == split
                total_checked += 1

    assert total_checked == 12836
