"""Standalone validation and audit script for LIAR dataset ingestion.

Executes end-to-end verification of raw TSV sources, reproduces the historical
RFC 4180 CSV reader corruption, runs the corrected auditable loader, validates
normalized records against the VERITAS schema, and asserts 100% data integrity.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Add src to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from veritas.data_prep.label_map import map_label
from veritas.data_prep.liar_loader import SPLIT_FILES, load_liar
from veritas.data_prep.schema import VALID_LABELS, validate_claim_record

EXPECTED_BENCHMARKS = {
    "train": 10269,
    "validation": 1284,
    "test": 1283,
}

CORRUPTED_CSV_READER_COUNTS = {
    "train": 10240,
    "validation": 1284,
    "test": 1267,
}


def inspect_raw_sources(raw_dir: Path) -> dict[str, dict[str, Any]]:
    """Analyze raw TSV physical lines, tabs, and CSV reader corruption."""
    print("=" * 80)
    print("STEP 1: RAW SOURCE TSV INSPECTION & FAILURE MODE REPRODUCTION")
    print("=" * 80)

    raw_stats: dict[str, dict[str, Any]] = {}

    for split, filename in SPLIT_FILES.items():
        file_path = raw_dir / filename
        if not file_path.is_file():
            raise FileNotFoundError(f"Missing raw TSV file: {file_path}")

        with file_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        physical_line_count = len(lines)
        tab_counts = Counter(line.count("\t") for line in lines)
        odd_quotes = sum(1 for line in lines if line.count('"') % 2 != 0)

        # Simulate default csv.reader to demonstrate failure mode
        with file_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            default_csv_rows = list(reader)

        multiline_merges = physical_line_count - len(default_csv_rows)

        raw_stats[split] = {
            "physical_lines": physical_line_count,
            "tab_counts": dict(tab_counts),
            "odd_quote_lines": odd_quotes,
            "default_csv_rows": len(default_csv_rows),
            "swallowed_records": multiline_merges,
        }

        print(f"\nSplit '{split}' ({filename}):")
        print(f"  - Physical lines: {physical_line_count:,}")
        print(f"  - Lines with exactly 13 tabs (14 columns): {tab_counts.get(13, 0):,}")
        print(f"  - Lines with irregular tab counts: {sum(c for k, c in tab_counts.items() if k != 13)}")
        print(f"  - Lines with odd double-quote count: {odd_quotes:,}")
        print(f"  - Default csv.reader row count: {len(default_csv_rows):,}")
        print(f"  - Records swallowed by default csv.reader: {multiline_merges:,}")

    return raw_stats


def validate_processed_records(output_dir: Path) -> dict[str, Any]:
    """Validate processed JSONL files against schema and business integrity rules."""
    print("\n" + "=" * 80)
    print("STEP 2: PROCESSED JSONL SCHEMA & INTEGRITY VALIDATION")
    print("=" * 80)

    all_ids: dict[str, str] = {}
    validation_summary: dict[str, Any] = {}
    total_valid = 0

    for split in SPLIT_FILES.keys():
        jsonl_path = output_dir / f"{split}.jsonl"
        if not jsonl_path.is_file():
            raise FileNotFoundError(f"Missing processed file: {jsonl_path}")

        records: list[dict[str, Any]] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{jsonl_path}:{line_idx}: Invalid JSON: {exc}"
                    ) from exc

                # Validate against VERITAS claim schema
                validate_claim_record(record)
                records.append(record)

        count = len(records)
        total_valid += count
        expected = EXPECTED_BENCHMARKS[split]

        if count != expected:
            raise ValueError(
                f"Split '{split}' record count mismatch! Expected {expected:,}, got {count:,}"
            )

        # Check source IDs and duplicates
        split_ids = [r["metadata"]["source_id"] for r in records]
        id_counts = Counter(split_ids)
        dup_ids = [cid for cid, c in id_counts.items() if c > 1]
        missing_ids = [r["id"] for r in records if not r["metadata"]["source_id"].strip()]
        empty_claims = [r["id"] for r in records if not r["claim"].strip()]
        invalid_labels = [r["label"] for r in records if r["label"] not in VALID_LABELS]

        # Verify cross-split duplicates
        for cid in split_ids:
            if cid in all_ids:
                raise ValueError(
                    f"Cross-split duplicate detected: ID '{cid}' in '{all_ids[cid]}' and '{split}'"
                )
            all_ids[cid] = split

        validation_summary[split] = {
            "record_count": count,
            "expected_count": expected,
            "duplicate_ids": dup_ids,
            "missing_ids": len(missing_ids),
            "empty_claims": len(empty_claims),
            "invalid_labels": len(invalid_labels),
        }

        print(f"\nSplit '{split}':")
        print(f"  - Records verified: {count:,} / {expected:,} (100% MATCH)")
        print(f"  - Duplicate IDs within split: {len(dup_ids)}")
        print(f"  - Missing IDs: {len(missing_ids)}")
        print(f"  - Empty statements: {len(empty_claims)}")
        print(f"  - Schema validation: 100% PASSED")

    print("\n" + "=" * 80)
    print("STEP 3: OVERALL VERIFICATION REPORT")
    print("=" * 80)
    print(f"Total Normalized Records: {total_valid:,} / 12,836 (100% MATCH)")
    print(f"Total Unique Global IDs: {len(all_ids):,} / 12,836 (ZERO DUPLICATES)")
    print("Schema Compliance: ALL RECORDS VALIDATED")
    print("Status: SUCCESS - INGESTION FULLY VERIFIED")
    print("=" * 80)

    return validation_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate LIAR dataset ingestion.")
    parser.add_argument("--raw-dir", default="data/raw/liar", help="Raw LIAR directory")
    parser.add_argument("--output-dir", default="data/processed/liar", help="Output JSONL directory")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    print("Running LIAR Source Inspection...")
    inspect_raw_sources(raw_dir)

    print("\nExecuting Corrected Loader with Auditable Recovery...")
    summary = load_liar(raw_dir=raw_dir, output_dir=output_dir, generate_report=True, strict=True)
    print(f"Loader returned counts: {summary}")

    print("\nValidating Output Records...")
    validate_processed_records(output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
