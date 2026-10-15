"""VERITAS shared data contracts and claim record validation.

Defines ClaimRecord and EvidenceRecord structures conforming to the VERITAS
specification, and provides strict validation of required fields, data types,
taxonomic labels, splits, ISO date formatting, and optional metadata preservation.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, NotRequired, TypedDict

ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class EvidenceRecord(TypedDict):
    text: str
    url: str | None
    date: str | None
    domain: str | None


class ClaimRecord(TypedDict):
    id: str
    claim: str
    label: str
    claim_date: str | None
    topic: str | None
    split: str
    evidence: list[EvidenceRecord]
    metadata: NotRequired[dict[str, Any]]


VALID_LABELS: set[str] = {
    "True",
    "Partially True",
    "Misleading",
    "False",
    "Unverified",
}

VALID_SPLITS: set[str] = {
    "train",
    "validation",
    "test",
    "heldout_topic",
}

REQUIRED_CLAIM_FIELDS: set[str] = {
    "id",
    "claim",
    "label",
    "claim_date",
    "topic",
    "split",
    "evidence",
}

REQUIRED_EVIDENCE_FIELDS: set[str] = {
    "text",
    "url",
    "date",
    "domain",
}


def is_valid_iso_date(value: Any) -> bool:
    """Return True if value is an ISO YYYY-MM-DD string representing a valid calendar date."""
    if not isinstance(value, str):
        return False
    if not ISO_DATE_PATTERN.match(value):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def validate_evidence_record(evidence: dict[str, Any], index: int | None = None) -> None:
    """Raise ValueError if an evidence record violates the VERITAS evidence schema."""
    prefix = f"Evidence item {index}" if index is not None else "Evidence item"

    if not isinstance(evidence, dict):
        raise ValueError(f"{prefix} must be a dictionary.")

    missing_fields = REQUIRED_EVIDENCE_FIELDS - evidence.keys()
    if missing_fields:
        raise ValueError(f"{prefix} missing fields: {sorted(missing_fields)}")

    if not isinstance(evidence["text"], str) or not evidence["text"].strip():
        raise ValueError(f"{prefix} text must be a non-empty string.")

    if evidence["url"] is not None and not isinstance(evidence["url"], str):
        raise ValueError(
            f"{prefix} url must be a string or None, got {type(evidence['url']).__name__}"
        )

    if evidence["date"] is not None:
        if not is_valid_iso_date(evidence["date"]):
            raise ValueError(
                f"{prefix} date must be an ISO YYYY-MM-DD string or None, got {evidence['date']!r}"
            )

    if evidence["domain"] is not None and not isinstance(evidence["domain"], str):
        raise ValueError(
            f"{prefix} domain must be a string or None, got {type(evidence['domain']).__name__}"
        )


def validate_claim_record(record: dict[str, Any]) -> None:
    """Raise ValueError if a normalized claim record violates the VERITAS data contract."""
    if not isinstance(record, dict):
        raise ValueError("Claim record must be a dictionary.")

    missing = REQUIRED_CLAIM_FIELDS - record.keys()
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")

    if not isinstance(record["id"], str) or not record["id"].strip():
        raise ValueError("Claim id must be a non-empty string.")

    if not isinstance(record["claim"], str) or not record["claim"].strip():
        raise ValueError("Claim must be a non-empty string.")

    if record["label"] not in VALID_LABELS:
        raise ValueError(
            f"Invalid VERITAS label: {record['label']!r}. Must be one of {sorted(VALID_LABELS)}."
        )

    if record["claim_date"] is not None:
        if not is_valid_iso_date(record["claim_date"]):
            raise ValueError(
                f"claim_date must be an ISO YYYY-MM-DD string or None, got {record['claim_date']!r}"
            )

    if record["topic"] is not None and not isinstance(record["topic"], str):
        raise ValueError(
            f"topic must be a string or None, got {type(record['topic']).__name__}"
        )

    if record["split"] not in VALID_SPLITS:
        raise ValueError(
            f"Invalid split: {record['split']!r}. Must be one of {sorted(VALID_SPLITS)}."
        )

    if not isinstance(record["evidence"], list):
        raise ValueError("Evidence must be a list.")

    for index, evidence in enumerate(record["evidence"]):
        validate_evidence_record(evidence, index=index)

    if "metadata" in record and not isinstance(record["metadata"], dict):
        raise ValueError("metadata must be a dictionary when present.")
