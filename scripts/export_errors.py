#!/usr/bin/env python3
"""Export incorrect VERITAS predictions to CSV for qualitative human error analysis.

Extracts all claims where the predicted label diverges from the ground-truth label,
attaching:
- Claim ID, statement text, and claim date.
- Gold and predicted labels with model confidence.
- Retrieved passages, domain credibility, scores, and dates.
- Faithfulness and abstention status.
- Verification timing and parsing status.
- Diagnostic preliminary suggestions.
- Blank or merged human annotation columns.

Guarantees:
- Leaves raw prediction records completely untouched.
- Retains existing human review annotations when re-running or updating exports.
- Produces spreadsheet-compatible CSV formatted for Excel, Google Sheets, or manual inspection.

Usage:
  python scripts/export_errors.py \\
      --gold data/processed/liar_test.jsonl \\
      --predictions results/experiments/overall_full_system/predictions.jsonl \\
      --output results/error_analysis/errors_for_review.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

# Ensure root in python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from veritas.evaluation.error_analysis import (
    ERROR_CATEGORIES,
    ErrorRecord,
    export_errors_to_csv,
    extract_prediction_errors,
    load_annotations_from_csv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("veritas.export_errors")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export prediction errors to CSV for manual qualitative annotation."
    )
    parser.add_argument(
        "--gold",
        type=str,
        required=True,
        help="Path to gold dataset JSONL file containing true labels and evidence.",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to prediction JSONL file output by VERITAS pipeline.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/error_analysis/errors_for_review.csv",
        help="Path where review CSV spreadsheet will be saved.",
    )
    parser.add_argument(
        "--existing-annotations",
        type=str,
        default=None,
        help="Optional path to previously annotated CSV. If omitted and --output exists, merges from --output.",
    )
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL file into list of dicts."""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    records: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                try:
                    records.append(json.loads(stripped))
                except Exception as exc:
                    logger.warning("Skipping invalid JSON line: %s", exc)
    return records


def main() -> None:
    args = parse_args()

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    # Determine annotation source for retention
    ann_source: Path | None = None
    if args.existing_annotations:
        ann_source = Path(args.existing_annotations)
        if not ann_source.is_absolute():
            ann_source = ROOT / ann_source
    elif out_path.exists():
        ann_source = out_path

    existing_annotations: dict[str, dict[str, Any]] = {}
    if ann_source and ann_source.exists():
        logger.info("Merging existing human annotations from %s", ann_source)
        existing_annotations = load_annotations_from_csv(ann_source)

    logger.info("Loading gold records from %s", args.gold)
    gold_records = load_jsonl(args.gold)

    logger.info("Loading predictions from %s", args.predictions)
    pred_records = load_jsonl(args.predictions)

    # Extract errors (raw predictions are NOT modified)
    errors = extract_prediction_errors(
        gold_records=gold_records,
        prediction_records=pred_records,
        existing_annotations=existing_annotations,
    )

    export_errors_to_csv(errors, out_path)

    reviewed = sum(1 for e in errors if e.annotated_category)
    print("\n" + "=" * 70)
    print("VERITAS ERROR EXPORT COMPLETE")
    print("=" * 70)
    print(f"Total Evaluated Claims : {len(pred_records)}")
    print(f"Total Errors Exported  : {len(errors)}")
    print(f"Human-Annotated Claims : {reviewed} (retained from previous review passes)")
    print(f"Pending Review Claims  : {len(errors) - reviewed}")
    print(f"Review CSV Saved To    : {out_path}")
    print("\nAllowed Annotation Categories:")
    for cat in ERROR_CATEGORIES:
        print(f"  - {cat}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
