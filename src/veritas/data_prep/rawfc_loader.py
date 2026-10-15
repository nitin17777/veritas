"""RAWFC dataset loader for VERITAS fact verification.

Normalizes RAWFC (Snopes-sourced raw social media reports, Hu et al., COLING 2022)
into canonical ClaimRecord JSONL files.

NOTE ON LABEL MAPPING:
Per project requirements, the RAWFC label mapping is pending human review and is NOT
activated by default. Users must pass an approved label_map (e.g. PROPOSED_RAWFC_LABEL_MAP)
to activate ingestion.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from veritas.data_prep.date_utils import normalize_date
from veritas.data_prep.label_map import PROPOSED_RAWFC_LABEL_MAP, map_label
from veritas.data_prep.schema import validate_claim_record
from veritas.data_prep.text_cleaning import clean_text

logger = logging.getLogger(__name__)

RAWFC_SPLIT_FILES: dict[str, list[str]] = {
    "train": ["train.json", "train.jsonl"],
    "validation": ["val.json", "val.jsonl", "dev.json", "dev.jsonl"],
    "test": ["test.json", "test.jsonl"],
}


@dataclass
class RawfcIngestionSummary:
    """Summary metrics for RAWFC ingestion."""

    records_processed: int = 0
    valid_records: int = 0
    malformed_rows: int = 0
    label_distribution: dict[str, int] = field(default_factory=dict)
    original_label_distribution: dict[str, int] = field(default_factory=dict)
    unknown_labels: list[str] = field(default_factory=list)
    malformed_details: list[dict[str, Any]] = field(default_factory=list)


def parse_rawfc_item(
    item: dict[str, Any],
    split: str,
    label_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Convert a raw RAWFC dictionary into a VERITAS ClaimRecord."""
    if not isinstance(item, dict):
        raise ValueError("RAWFC item must be a dictionary.")

    for req in ("id", "claim", "label"):
        if req not in item:
            raise ValueError(f"Missing required RAWFC field {req!r}.")

    claim_id = str(item["id"]).strip()
    claim_text = clean_text(item["claim"])
    raw_label = str(item["label"]).strip()

    if not claim_id:
        raise ValueError("RAWFC claim ID cannot be empty.")
    if not claim_text:
        raise ValueError("RAWFC claim statement cannot be empty.")

    if label_map is None:
        raise ValueError(
            "RAWFC label mapping has not been activated. "
            "Please review the proposed mapping (PROPOSED_RAWFC_LABEL_MAP) "
            "and pass an approved label_map dictionary."
        )

    mapped_label = map_label(raw_label, "rawfc", custom_mapping=label_map)

    # Date normalization
    raw_date = item.get("date")
    claim_date: str | None = None
    if raw_date:
        try:
            claim_date = normalize_date(raw_date)
        except Exception:
            claim_date = None

    # Evidence extraction from background reports if present
    evidence_list: list[dict[str, Any]] = []
    raw_reports = item.get("reports") or item.get("evidence") or []
    if isinstance(raw_reports, list):
        for rep in raw_reports:
            if not isinstance(rep, dict):
                continue
            rep_text = clean_text(rep.get("text") or rep.get("content") or rep.get("snippet"))
            if rep_text:
                rep_date: str | None = None
                if rep.get("date"):
                    try:
                        rep_date = normalize_date(rep.get("date"))
                    except Exception:
                        rep_date = None
                evidence_list.append(
                    {
                        "text": rep_text,
                        "url": rep.get("url"),
                        "date": rep_date,
                        "domain": rep.get("domain"),
                    }
                )

    record: dict[str, Any] = {
        "id": f"rawfc:{claim_id}",
        "claim": claim_text,
        "label": mapped_label,
        "claim_date": claim_date,
        "topic": clean_text(item.get("topic")) or None,
        "split": split,
        "evidence": evidence_list,
        "metadata": {
            "dataset": "rawfc",
            "source_id": claim_id,
            "original_label": raw_label,
            "source_url": item.get("url"),
        },
    }

    validate_claim_record(record)
    return record


def load_rawfc_split(
    file_path: Path,
    split: str,
    label_map: dict[str, str] | None = None,
    strict: bool = False,
) -> tuple[list[dict[str, Any]], RawfcIngestionSummary]:
    """Parse a single RAWFC file (JSON array or JSONL) into normalized records."""
    records: list[dict[str, Any]] = []
    summary = RawfcIngestionSummary()
    label_counter: Counter[str] = Counter()
    orig_label_counter: Counter[str] = Counter()

    with file_path.open("r", encoding="utf-8") as f:
        content = f.read().strip()

    items: list[dict[str, Any]] = []
    if content.startswith("["):
        # JSON Array format
        try:
            items = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse RAWFC JSON array from {file_path}: {exc}") from exc
    else:
        # JSONL format
        for line in content.splitlines():
            line_str = line.strip()
            if line_str:
                items.append(json.loads(line_str))

    for idx, item in enumerate(items, start=1):
        summary.records_processed += 1
        raw_label = str(item.get("label", ""))
        orig_label_counter[raw_label] += 1

        try:
            record = parse_rawfc_item(item, split=split, label_map=label_map)
            records.append(record)
            label_counter[record["label"]] += 1
            summary.valid_records += 1
        except Exception as exc:
            summary.malformed_rows += 1
            err_msg = str(exc)
            if "Unknown label" in err_msg or "not been activated" in err_msg:
                summary.unknown_labels.append(err_msg)
            summary.malformed_details.append({"index": idx, "error": err_msg})
            if strict:
                raise ValueError(f"{file_path}:item {idx}: Malformed RAWFC item: {exc}") from exc

    summary.label_distribution = dict(label_counter)
    summary.original_label_distribution = dict(orig_label_counter)
    return records, summary


def load_rawfc(
    raw_dir: str | Path = "data/raw/rawfc",
    output_dir: str | Path = "data/processed/rawfc",
    label_map: dict[str, str] | None = None,
    strict: bool = False,
) -> dict[str, int]:
    """Normalize available RAWFC split files into canonical JSONL files."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}

    for split, candidates in RAWFC_SPLIT_FILES.items():
        input_path: Path | None = None
        for candidate in candidates:
            cand_path = raw_dir / candidate
            if cand_path.is_file():
                input_path = cand_path
                break

        if input_path is None:
            logger.info("RAWFC split '%s' not found under %s. Skipping.", split, raw_dir)
            continue

        records, summary = load_rawfc_split(
            input_path,
            split=split,
            label_map=label_map,
            strict=strict,
        )
        output_path = output_dir / f"{split}.jsonl"

        with output_path.open("w", encoding="utf-8", newline="\n") as out_f:
            for rec in records:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        counts[split] = len(records)
        logger.info(
            "Processed RAWFC %s: %d records (%d malformed)",
            split,
            len(records),
            summary.malformed_rows,
        )

    return counts
