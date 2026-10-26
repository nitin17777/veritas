#!/usr/bin/env python3
"""Summarize VERITAS error analysis findings and generate failure pattern reports.

Reads error review CSV (or raw gold + prediction files), calculates:
- Category counts and proportions (both provisional suggestions and human annotations).
- Label confusion breakdown for errors.
- Confidence distribution across error categories.
- Qualitative synthesis of recurring failure patterns with representative examples.

Outputs:
- Markdown failure pattern report (error_analysis_report.md).
- Machine-readable statistics JSON.
- Terminal overview table.

Usage:
  python scripts/summarize_errors.py \\
      --errors results/error_analysis/errors_for_review.csv \\
      --output-report results/error_analysis/error_analysis_report.md
"""

from __future__ import annotations

import argparse
import csv
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
    compute_error_statistics,
    extract_prediction_errors,
    generate_error_summary_report,
    load_annotations_from_csv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("veritas.summarize_errors")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize VERITAS prediction errors and generate qualitative failure pattern reports."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--errors",
        type=str,
        help="Path to review CSV spreadsheet (e.g. results/error_analysis/errors_for_review.csv).",
    )
    group.add_argument(
        "--predictions",
        type=str,
        help="Path to prediction JSONL file (used together with --gold).",
    )

    parser.add_argument(
        "--gold",
        type=str,
        help="Path to gold dataset JSONL file (required when using --predictions).",
    )
    parser.add_argument(
        "--total-claims",
        type=int,
        default=None,
        help="Total evaluated claims count across test set (used to compute overall error rate).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="dataset",
        help="Name of evaluated dataset.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="error_run",
        help="Experiment or evaluation run identifier.",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="results/error_analysis/error_analysis_report.md",
        help="Path where Markdown summary report will be saved.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="results/error_analysis/error_statistics.json",
        help="Path where machine-readable statistics JSON will be saved.",
    )
    return parser.parse_args()


def load_errors_from_csv(csv_path: Path) -> list[ErrorRecord]:
    """Parse ErrorRecords from review CSV spreadsheet."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Errors CSV file not found: {csv_path}")

    records: list[ErrorRecord] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = str(row.get("claim_id", "")).strip()
            if not cid:
                continue

            conf_raw = row.get("confidence", "0.0")
            try:
                conf = float(conf_raw) if conf_raw else 0.0
            except ValueError:
                conf = 0.0

            faith_raw = row.get("faithfulness_score")
            faith = float(faith_raw) if faith_raw else None

            attempts_raw = row.get("attempts", "1")
            attempts = int(attempts_raw) if attempts_raw and attempts_raw.isdigit() else 1

            ret_ids = [s.strip() for s in row.get("retrieved_evidence_ids", "").split(";") if s.strip()]
            cited_ids = [s.strip() for s in row.get("cited_evidence_ids", "").split(";") if s.strip()]
            gold_ids = [s.strip() for s in row.get("gold_evidence_ids", "").split(";") if s.strip()]

            rec = ErrorRecord(
                claim_id=cid,
                claim=row.get("claim", ""),
                gold_label=row.get("gold_label", ""),
                predicted_label=row.get("predicted_label", ""),
                confidence=conf,
                explanation=row.get("explanation", ""),
                retrieved_evidence_ids=ret_ids,
                retrieved_sources_summary=row.get("retrieved_sources", ""),
                cited_evidence_ids=cited_ids,
                gold_evidence_ids=gold_ids,
                claim_date=row.get("claim_date") or None,
                faithfulness_score=faith,
                abstained=row.get("abstained", "").lower() == "true",
                original_label=row.get("original_label") or None,
                parse_failed=row.get("parse_failed", "").lower() == "true",
                attempts=attempts,
                suggested_category=row.get("suggested_category", "Other / unknown"),
                suggestion_rationale=row.get("suggestion_rationale", ""),
                annotated_category=row.get("annotated_category", ""),
                annotator_id=row.get("annotator_id", ""),
                annotator_notes=row.get("annotator_notes", ""),
                verification_status=row.get("verification_status", "unreviewed"),
            )
            records.append(rec)
    return records


def main() -> None:
    args = parse_args()

    errors: list[ErrorRecord] = []
    total_claims = args.total_claims

    if args.errors:
        err_path = Path(args.errors)
        if not err_path.is_absolute():
            err_path = ROOT / err_path
        logger.info("Loading error records from %s", err_path)
        errors = load_errors_from_csv(err_path)
    else:
        if not args.gold:
            logger.error("--gold is required when using --predictions.")
            sys.exit(1)

        gold_path = Path(args.gold)
        if not gold_path.is_absolute():
            gold_path = ROOT / gold_path
        pred_path = Path(args.predictions)
        if not pred_path.is_absolute():
            pred_path = ROOT / pred_path

        gold_records = [json.loads(l) for l in gold_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        pred_records = [json.loads(l) for l in pred_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        total_claims = total_claims or len(pred_records)
        errors = extract_prediction_errors(gold_records, pred_records)

    report_md = generate_error_summary_report(
        errors=errors,
        total_evaluated_claims=total_claims,
        dataset_name=args.dataset_name,
        run_id=args.run_id,
    )

    stats = compute_error_statistics(errors, total_evaluated_claims=total_claims)

    rep_path = Path(args.output_report)
    if not rep_path.is_absolute():
        rep_path = ROOT / rep_path
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(report_md, encoding="utf-8")

    json_path = Path(args.output_json)
    if not json_path.is_absolute():
        json_path = ROOT / json_path
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\n" + "=" * 75)
    print("VERITAS ERROR ANALYSIS SUMMARY")
    print("=" * 75)
    print(f"Total Errors Analyzed : {len(errors)}")
    print(f"Human-Reviewed Errors : {stats['human_reviewed_errors']} ({stats['human_verified_errors']} verified)")
    print(f"\n{'Category':<45} | {'Suggested':<10} | {'Annotated':<10}")
    print("-" * 75)
    for cat in ERROR_CATEGORIES:
        sc = stats["suggested_counts"].get(cat, 0)
        ac = stats["annotated_counts"].get(cat, 0)
        print(f"{cat:<45} | {sc:<10} | {ac:<10}")

    print("\nTop Confusion Pairs (Gold -> Pred):")
    for pair, count in sorted(stats["confusion_breakdown"].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {pair:<30}: {count}")

    print(f"\nMarkdown Report Saved: {rep_path}")
    print(f"Statistics JSON Saved: {json_path}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
