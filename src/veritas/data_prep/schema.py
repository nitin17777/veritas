
from typing import Any, TypedDict


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


VALID_LABELS = {
    "True",
    "Partially True",
    "Misleading",
    "False",
    "Unverified",
}


def validate_claim_record(record: dict[str, Any]) -> None:
    """Raise ValueError if a normalized claim record is invalid."""
    required = {
        "id",
        "claim",
        "label",
        "claim_date",
        "topic",
        "split",
        "evidence",
    }

    missing = required - record.keys()
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")

    if not isinstance(record["id"], str) or not record["id"].strip():
        raise ValueError("Claim id must be a non-empty string.")

    if not isinstance(record["claim"], str) or not record["claim"].strip():
        raise ValueError("Claim must be a non-empty string.")

    if record["label"] not in VALID_LABELS:
        raise ValueError(f"Invalid VERITAS label: {record['label']!r}")

    if record["split"] not in {"train", "validation", "test", "heldout_topic"}:
        raise ValueError(f"Invalid split: {record['split']!r}")

    if not isinstance(record["evidence"], list):
        raise ValueError("Evidence must be a list.")

    for index, evidence in enumerate(record["evidence"]):
        if not isinstance(evidence, dict):
            raise ValueError(f"Evidence item {index} must be a dictionary.")

        required_evidence = {"text", "url", "date", "domain"}
        missing_evidence = required_evidence - evidence.keys()
        if missing_evidence:
            raise ValueError(
                f"Evidence item {index} missing fields: "
                f"{sorted(missing_evidence)}"
            )

        if not isinstance(evidence["text"], str):
            raise ValueError(f"Evidence item {index} text must be a string.")
