"""AVeriTeC dataset loader for VERITAS fact verification.

Normalizes AVeriTeC (real-world web claim verification, Schlichtkrull et al., NeurIPS 2023)
into canonical ClaimRecord JSONL files.

NOTE ON LABEL MAPPING:
Per project requirements, the AVeriTeC label vocabulary is inspected and proposed,
remaining pending human review before default activation. Users must pass an approved
label_map (e.g. PROPOSED_AVERITEC_LABEL_MAP) to activate ingestion.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from veritas.data_prep.date_utils import normalize_date
from veritas.data_prep.label_map import PROPOSED_AVERITEC_LABEL_MAP, map_label
from veritas.data_prep.schema import validate_claim_record
from veritas.data_prep.text_cleaning import clean_text

logger = logging.getLogger(__name__)

AVERITEC_SPLIT_FILES: dict[str, list[str]] = {
    "train": ["train.json", "train.jsonl"],
    "validation": ["dev.json", "dev.jsonl", "val.json", "val.jsonl"],
    "test": ["test.json", "test.jsonl"],
}


@dataclass
class AVeriTeCIngestionSummary:
    """Summary metrics for AVeriTeC ingestion."""

    records_processed: int = 0
    valid_records: int = 0
    malformed_rows: int = 0
    label_distribution: dict[str, int] = field(default_factory=dict)
    original_label_distribution: dict[str, int] = field(default_factory=dict)
    unknown_labels: list[str] = field(default_factory=list)
    malformed_details: list[dict[str, Any]] = field(default_factory=list)


def _extract_domain(url: str | None) -> str | None:
    """Safely extract network location (domain) from a URL string."""
    if not url:
        return None
    try:
        parsed = urllib.parse.urlparse(url)
        return parsed.netloc if parsed.netloc else None
    except Exception:
        return None


def parse_averitec_item(
    item: dict[str, Any],
    split: str,
    label_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Convert a raw AVeriTeC claim object into a canonical VERITAS ClaimRecord."""
    if not isinstance(item, dict):
        raise ValueError("AVeriTeC item must be a dictionary.")

    claim_id_raw = item.get("claim_id") or item.get("id")
    if claim_id_raw is None:
        raise ValueError("Missing required AVeriTeC field 'claim_id' or 'id'.")

    claim_raw = item.get("claim")
    if not claim_raw:
        raise ValueError("Missing required AVeriTeC field 'claim'.")

    label_raw = item.get("label")
    if not label_raw:
        raise ValueError("Missing required AVeriTeC field 'label'.")

    claim_id = str(claim_id_raw).strip()
    claim_text = clean_text(str(claim_raw))
    raw_label_str = str(label_raw).strip()

    if not claim_id:
        raise ValueError("AVeriTeC claim ID cannot be empty.")
    if not claim_text:
        raise ValueError("AVeriTeC claim text cannot be empty.")

    if label_map is None:
        raise ValueError(
            "AVeriTeC label mapping has not been activated. "
            "Please review the proposed mapping (PROPOSED_AVERITEC_LABEL_MAP) "
            "and pass an approved label_map dictionary."
        )

    mapped_label = map_label(raw_label_str, "averitec", custom_mapping=label_map)

    # Date normalization
    claim_date: str | None = None
    if item.get("claim_date"):
        try:
            claim_date = normalize_date(item["claim_date"])
        except Exception:
            claim_date = None

    # Evidence extraction from QA question-answer decomposition
    evidence_list: list[dict[str, Any]] = []
    questions = item.get("questions") or []
    if isinstance(questions, list):
        for q in questions:
            if not isinstance(q, dict):
                continue
            answers = q.get("answers") or []
            if isinstance(answers, list):
                for ans in answers:
                    if not isinstance(ans, dict):
                        continue
                    ans_text = clean_text(ans.get("answer"))
                    if ans_text:
                        source_url = ans.get("source_url")
                        ans_date: str | None = None
                        if ans.get("date"):
                            try:
                                ans_date = normalize_date(ans.get("date"))
                            except Exception:
                                ans_date = None
                        evidence_list.append(
                            {
                                "text": ans_text,
                                "url": source_url if source_url else None,
                                "date": ans_date,
                                "domain": _extract_domain(source_url),
                            }
                        )

    record: dict[str, Any] = {
        "id": f"averitec:{claim_id}",
        "claim": claim_text,
        "label": mapped_label,
        "claim_date": claim_date,
        "topic": clean_text(item.get("speaker")) or None,  # Speaker preserved or topic
        "split": split,
        "evidence": evidence_list,
        "metadata": {
            "dataset": "averitec",
            "source_id": claim_id,
            "speaker": clean_text(item.get("speaker")) or None,
            "original_label": raw_label_str,
            "question_count": len(questions) if isinstance(questions, list) else 0,
        },
    }

    validate_claim_record(record)
    return record


def load_averitec_split(
    file_path: Path,
    split: str,
    label_map: dict[str, str] | None = None,
    strict: bool = False,
) -> tuple[list[dict[str, Any]], AVeriTeCIngestionSummary]:
    """Parse a single AVeriTeC file (JSON array or JSONL) into normalized records."""
    records: list[dict[str, Any]] = []
    summary = AVeriTeCIngestionSummary()
    label_counter: Counter[str] = Counter()
    orig_label_counter: Counter[str] = Counter()

    with file_path.open("r", encoding="utf-8") as f:
        content = f.read().strip()

    items: list[dict[str, Any]] = []
    if content.startswith("["):
        try:
            items = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse AVeriTeC JSON array from {file_path}: {exc}") from exc
    else:
        for line in content.splitlines():
            line_str = line.strip()
            if line_str:
                items.append(json.loads(line_str))

    for idx, item in enumerate(items, start=1):
        summary.records_processed += 1
        raw_label = str(item.get("label", ""))
        orig_label_counter[raw_label] += 1

        try:
            record = parse_averitec_item(item, split=split, label_map=label_map)
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
                raise ValueError(f"{file_path}:item {idx}: Malformed AVeriTeC item: {exc}") from exc

    summary.label_distribution = dict(label_counter)
    summary.original_label_distribution = dict(orig_label_counter)
    return records, summary


def load_averitec(
    raw_dir: str | Path = "data/raw/averitec",
    output_dir: str | Path = "data/processed/averitec",
    label_map: dict[str, str] | None = None,
    strict: bool = False,
) -> dict[str, int]:
    """Normalize available AVeriTeC split files into canonical JSONL files."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}

    for split, candidates in AVERITEC_SPLIT_FILES.items():
        input_path: Path | None = None
        for candidate in candidates:
            cand_path = raw_dir / candidate
            if cand_path.is_file():
                input_path = cand_path
                break

        if input_path is None:
            logger.info("AVeriTeC split '%s' not found under %s. Skipping.", split, raw_dir)
            continue

        records, summary = load_averitec_split(
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
            "Processed AVeriTeC %s: %d records (%d malformed)",
            split,
            len(records),
            summary.malformed_rows,
        )

    return counts
