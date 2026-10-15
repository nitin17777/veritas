"""Custom / Recent fact-check dataset loader for VERITAS.

Supports ingesting recent (e.g. 2025-2026) fact-check feeds, PolitiFact dumps,
or Google Fact Check API records into the canonical ClaimRecord schema.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from veritas.data_prep.date_utils import normalize_date
from veritas.data_prep.label_map import map_label
from veritas.data_prep.schema import validate_claim_record
from veritas.data_prep.text_cleaning import clean_text

logger = logging.getLogger(__name__)

CUSTOM_SPLIT_FILES: dict[str, list[str]] = {
    "train": ["train.jsonl", "train.json"],
    "validation": ["valid.jsonl", "valid.json", "val.jsonl"],
    "test": ["test.jsonl", "test.json"],
}


@dataclass
class CustomIngestionSummary:
    """Summary metrics for custom recent dataset ingestion."""

    records_processed: int = 0
    valid_records: int = 0
    malformed_rows: int = 0
    label_distribution: dict[str, int] = field(default_factory=dict)
    original_label_distribution: dict[str, int] = field(default_factory=dict)
    unknown_labels: list[str] = field(default_factory=list)
    malformed_details: list[dict[str, Any]] = field(default_factory=list)


def parse_custom_item(
    item: dict[str, Any],
    split: str,
    label_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Convert a custom fact-check item into a VERITAS ClaimRecord."""
    if not isinstance(item, dict):
        raise ValueError("Custom item must be a dictionary.")

    for field_name in ("id", "claim", "label"):
        if field_name not in item:
            raise ValueError(f"Missing required field {field_name!r}.")

    claim_id = str(item["id"]).strip()
    claim_text = clean_text(item["claim"])
    raw_label = str(item["label"]).strip()

    if not claim_id:
        raise ValueError("Claim ID cannot be empty.")
    if not claim_text:
        raise ValueError("Claim statement cannot be empty.")

    mapped_label = map_label(raw_label, "custom", custom_mapping=label_map)

    # Date normalization
    claim_date: str | None = None
    if item.get("claim_date"):
        try:
            claim_date = normalize_date(item["claim_date"])
        except Exception:
            claim_date = None

    # Evidence normalization
    evidence_list: list[dict[str, Any]] = []
    raw_evidence = item.get("evidence") or []
    if isinstance(raw_evidence, list):
        for ev in raw_evidence:
            if not isinstance(ev, dict):
                continue
            ev_text = clean_text(ev.get("text"))
            if ev_text:
                ev_date: str | None = None
                if ev.get("date"):
                    try:
                        ev_date = normalize_date(ev.get("date"))
                    except Exception:
                        ev_date = None
                evidence_list.append(
                    {
                        "text": ev_text,
                        "url": ev.get("url"),
                        "date": ev_date,
                        "domain": ev.get("domain"),
                    }
                )

    record: dict[str, Any] = {
        "id": f"custom:{claim_id}" if not claim_id.startswith("custom:") else claim_id,
        "claim": claim_text,
        "label": mapped_label,
        "claim_date": claim_date,
        "topic": clean_text(item.get("topic")) or None,
        "split": split,
        "evidence": evidence_list,
        "metadata": item.get("metadata") or {
            "dataset": "custom_recent",
            "source_id": claim_id,
            "original_label": raw_label,
        },
    }

    validate_claim_record(record)
    return record


def load_custom_recent(
    raw_dir: str | Path = "data/raw/custom_recent",
    output_dir: str | Path = "data/processed/custom_recent",
    label_map: dict[str, str] | None = None,
    strict: bool = False,
) -> dict[str, int]:
    """Normalize custom recent dataset split files into canonical JSONL files."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}

    for split, candidates in CUSTOM_SPLIT_FILES.items():
        input_path: Path | None = None
        for candidate in candidates:
            cand_path = raw_dir / candidate
            if cand_path.is_file():
                input_path = cand_path
                break

        if input_path is None:
            logger.info("Custom recent split '%s' not found under %s. Skipping.", split, raw_dir)
            continue

        records: list[dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    item = json.loads(stripped)
                    rec = parse_custom_item(item, split=split, label_map=label_map)
                    records.append(rec)
                except Exception as exc:
                    if strict:
                        raise ValueError(f"{input_path}:{line_num}: {exc}") from exc

        output_path = output_dir / f"{split}.jsonl"
        with output_path.open("w", encoding="utf-8", newline="\n") as out_f:
            for rec in records:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        counts[split] = len(records)
        logger.info("Processed custom recent %s: %d records", split, len(records))

    return counts
