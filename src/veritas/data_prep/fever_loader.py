"""FEVER dataset loader for VERITAS fact verification.

Normalizes FEVER (Fact Extraction and VERification) claims into canonical ClaimRecord
JSONL files. Maps Supported -> True, Refuted -> False, and Not Enough Info -> Unverified.
Preserves official train/dev/test splits, keeps missing dates as None, and stores
evidence pointers without inventing ungrounded evidence.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from veritas.data_prep.date_utils import normalize_date
from veritas.data_prep.label_map import map_label
from veritas.data_prep.schema import validate_claim_record
from veritas.data_prep.text_cleaning import clean_text

logger = logging.getLogger(__name__)

FEVER_SPLIT_CANDIDATES: dict[str, list[str]] = {
    "train": ["train.jsonl"],
    "validation": ["paper_dev.jsonl", "shared_task_dev.jsonl"],
    "test": ["paper_test.jsonl", "shared_task_test.jsonl"],
}


@dataclass
class FeverIngestionSummary:
    """Summary metrics for FEVER ingestion."""

    records_processed: int = 0
    valid_records: int = 0
    malformed_rows: int = 0
    label_distribution: dict[str, int] = field(default_factory=dict)
    original_label_distribution: dict[str, int] = field(default_factory=dict)
    unknown_labels: list[str] = field(default_factory=list)
    malformed_details: list[dict[str, Any]] = field(default_factory=list)


def parse_fever_item(item: dict[str, Any], split: str) -> dict[str, Any]:
    """Convert a raw FEVER dictionary into a VERITAS ClaimRecord."""
    if not isinstance(item, dict):
        raise ValueError("FEVER row must be a JSON object.")

    if "id" not in item:
        raise ValueError("Missing required FEVER field 'id'.")
    if "claim" not in item:
        raise ValueError("Missing required FEVER field 'claim'.")
    if "label" not in item:
        raise ValueError("Missing required FEVER field 'label'.")

    claim_id = str(item["id"]).strip()
    claim_text = clean_text(item["claim"])

    if not claim_id:
        raise ValueError("FEVER claim ID cannot be empty.")
    if not claim_text:
        raise ValueError("FEVER claim statement cannot be empty.")

    mapped_label = map_label(str(item["label"]), "fever")

    # In raw FEVER, evidence is a list of lists of annotation tuples [ann_id, ev_id, title, sent_id].
    # Never invent evidence text or URLs if they are not provided inline.
    evidence_list: list[dict[str, Any]] = []
    raw_evidence = item.get("evidence", [])

    # If evidence items are already structured dictionaries with text (e.g. from an enriched corpus):
    if isinstance(raw_evidence, list) and raw_evidence and isinstance(raw_evidence[0], dict):
        for ev in raw_evidence:
            ev_text = clean_text(ev.get("text"))
            if ev_text:
                evidence_list.append(
                    {
                        "text": ev_text,
                        "url": ev.get("url"),
                        "date": normalize_date(ev.get("date")),
                        "domain": ev.get("domain"),
                    }
                )

    record: dict[str, Any] = {
        "id": f"fever:{claim_id}",
        "claim": claim_text,
        "label": mapped_label,
        "claim_date": None,  # FEVER claims lack publication dates; never invent dates.
        "topic": None,
        "split": split,
        "evidence": evidence_list,
        "metadata": {
            "dataset": "fever",
            "source_id": claim_id,
            "verifiable": item.get("verifiable"),
            "raw_evidence": raw_evidence if not evidence_list else [],
        },
    }

    validate_claim_record(record)
    return record


def load_fever_split(
    file_path: Path,
    split: str,
    strict: bool = False,
) -> tuple[list[dict[str, Any]], FeverIngestionSummary]:
    """Parse a single FEVER JSONL split file into validated records."""
    records: list[dict[str, Any]] = []
    summary = FeverIngestionSummary()
    label_counter: Counter[str] = Counter()
    orig_label_counter: Counter[str] = Counter()

    with file_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            summary.records_processed += 1
            try:
                raw_item = json.loads(stripped)
                raw_label = str(raw_item.get("label", ""))
                orig_label_counter[raw_label] += 1

                record = parse_fever_item(raw_item, split=split)
                records.append(record)
                label_counter[record["label"]] += 1
                summary.valid_records += 1
            except Exception as exc:
                summary.malformed_rows += 1
                err_msg = str(exc)
                if "Unknown label" in err_msg:
                    summary.unknown_labels.append(err_msg)
                summary.malformed_details.append(
                    {"line": line_num, "error": err_msg}
                )
                if strict:
                    raise ValueError(
                        f"{file_path}:{line_num}: Malformed FEVER row: {exc}"
                    ) from exc

    summary.label_distribution = dict(label_counter)
    summary.original_label_distribution = dict(orig_label_counter)
    return records, summary


def load_fever(
    raw_dir: str | Path = "data/raw/fever",
    output_dir: str | Path = "data/processed/fever",
    strict: bool = False,
) -> dict[str, int]:
    """Normalize available FEVER split files into canonical JSONL files."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}

    for split, candidates in FEVER_SPLIT_CANDIDATES.items():
        input_path: Path | None = None
        for candidate in candidates:
            cand_path = raw_dir / candidate
            if cand_path.is_file():
                input_path = cand_path
                break

        if input_path is None:
            logger.info("FEVER split '%s' not found under %s. Skipping.", split, raw_dir)
            continue

        records, summary = load_fever_split(input_path, split=split, strict=strict)
        output_path = output_dir / f"{split}.jsonl"

        with output_path.open("w", encoding="utf-8", newline="\n") as out_f:
            for rec in records:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        counts[split] = len(records)
        logger.info(
            "Processed FEVER %s: %d records (%d malformed)",
            split,
            len(records),
            summary.malformed_rows,
        )

    return counts
