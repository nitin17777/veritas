#!/usr/bin/env python
"""Reproducible Evaluation CLI for VERITAS P9.

Computes:
- Accuracy, Macro-F1, per-class Precision/Recall/F1, and Confusion Matrix.
- Retrieval metrics (Recall@k, nDCG@k, MRR@k) when gold evidence exists.
- System metrics: JSON parse failure rate, abstention rate, mean faithfulness, hallucination proxy.
- Latency percentiles (Average, P50, P90, P95, P99).
- Coverage-versus-accuracy curve and multi-run aggregation.

Usage:
    # Fast smoke-test on synthetic benchmark claims:
    python scripts/evaluate_veritas.py --smoke-test

    # Evaluate predictions against gold dataset:
    python scripts/evaluate_veritas.py \\
        --gold data/processed/liar/test.jsonl \\
        --predictions results/verdicts.jsonl \\
        --output-dir results/evaluation \\
        --dataset-name liar \\
        --model-version gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from veritas.config import load_config
from veritas.evaluation.metrics import aggregate_multi_run_metrics
from veritas.evaluation.report import (
    build_evaluation_report,
    format_console_report,
    save_report_artifacts,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("veritas.scripts.evaluate")


# ---------------------------------------------------------------------------
# Synthetic Benchmark Data for Smoke Testing
# ---------------------------------------------------------------------------


def generate_synthetic_evaluation_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate deterministic gold and prediction pairs covering all 5 labels."""
    golds = []
    preds = []

    labels_sequence = [
        ("True", "True", 0.94, True, 0.95),
        ("True", "True", 0.88, True, 0.90),
        ("True", "Partially True", 0.65, True, 0.70),
        ("Partially True", "Partially True", 0.78, True, 0.85),
        ("Partially True", "True", 0.55, False, 0.60),
        ("Misleading", "Misleading", 0.82, True, 0.80),
        ("Misleading", "False", 0.60, True, 0.75),
        ("False", "False", 0.96, True, 0.98),
        ("False", "False", 0.91, True, 0.92),
        ("False", "True", 0.40, False, 0.30),
        ("Unverified", "Unverified", 0.50, True, 0.0),
        ("Unverified", "Unverified", 0.50, True, 0.0),
    ]

    for idx, (gold_lbl, pred_lbl, conf, has_gold_ev, faith) in enumerate(labels_sequence, start=1):
        cid = f"claim_{idx:03d}"
        gold_ev = [{"passage_id": f"p_{idx}", "text": f"Evidence text for claim {idx}"}] if has_gold_ev else []
        retrieved_ev = [{"passage_id": f"p_{idx}"}, {"passage_id": f"other_{idx}"}]

        golds.append({
            "id": cid,
            "claim": f"Synthetic claim statement {idx}.",
            "label": gold_lbl,
            "evidence": gold_ev,
        })

        preds.append({
            "claim_id": cid,
            "claim": f"Synthetic claim statement {idx}.",
            "label": pred_lbl,
            "confidence": conf,
            "evidence_ids": [f"p_{idx}"],
            "explanation": f"Explanation for claim {idx}.",
            "retrieved_passages": retrieved_ev,
            "faithfulness_score": faith,
            "parse_failed": False,
            "stage_timings_s": {"retrieval": 0.002, "verify": 0.005},
        })

    return golds, preds


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VERITAS Verification Performance")
    parser.add_argument("--smoke-test", action="store_true", help="Run self-contained smoke test on synthetic data")
    parser.add_argument("--gold", type=str, default=None, help="Path to gold dataset JSONL file")
    parser.add_argument("--predictions", nargs="+", default=None, help="Path(s) to prediction JSONL file(s)")
    parser.add_argument("--config", type=str, default=None, help="Path to experiment config YAML file")
    parser.add_argument("--output-dir", type=str, default="results/evaluation", help="Directory to save report artifacts")
    parser.add_argument("--model-version", type=str, default="veritas-p9-model", help="Model version identifier")
    parser.add_argument("--dataset-name", type=str, default="benchmark", help="Dataset name")
    parser.add_argument("--dataset-version", type=str, default="v1.0", help="Dataset version")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used in experiment")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    cfg_dict = {}
    if args.config:
        cfg_dict = load_config(args.config)

    if args.smoke_test or args.gold is None or args.predictions is None:
        logger.info("Executing evaluation in smoke-test mode...")
        gold_records, pred_records = generate_synthetic_evaluation_data()
        report = build_evaluation_report(
            gold_records=gold_records,
            prediction_records=pred_records,
            config=cfg_dict,
            model_version=args.model_version,
            dataset_name="synthetic_smoke_test",
            dataset_version=args.dataset_version,
            seed=args.seed,
        )
    else:
        # Load gold dataset
        gold_path = Path(args.gold)
        if not gold_path.exists():
            logger.error("Gold file '%s' does not exist.", gold_path)
            sys.exit(1)

        gold_records = []
        with gold_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    gold_records.append(json.loads(line))

        # Check if single or multiple prediction files
        reports_list = []
        for p_str in args.predictions:
            pred_path = Path(p_str)
            if not pred_path.exists():
                logger.error("Prediction file '%s' does not exist.", pred_path)
                sys.exit(1)

            pred_records = []
            with pred_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        pred_records.append(json.loads(line))

            r = build_evaluation_report(
                gold_records=gold_records,
                prediction_records=pred_records,
                config=cfg_dict,
                model_version=args.model_version,
                dataset_name=args.dataset_name,
                dataset_version=args.dataset_version,
                seed=args.seed,
                run_id=pred_path.stem,
            )
            reports_list.append(r)

        report = reports_list[0]
        if len(reports_list) > 1:
            # Aggregate repeated runs
            scalars = [
                {
                    "accuracy": r.classification.get("accuracy"),
                    "macro_f1": r.classification.get("macro_f1"),
                    "abstention_rate": r.system_quality.get("abstention_rate"),
                    "mean_faithfulness": r.system_quality.get("mean_faithfulness"),
                }
                for r in reports_list
            ]
            report.multi_run_aggregate = aggregate_multi_run_metrics(scalars)

    # 1. Print formatted console report
    console_out = format_console_report(report)
    print("\n" + console_out + "\n")

    # 2. Save artifacts
    saved = save_report_artifacts(report, out_dir)
    logger.info("Evaluation artifacts successfully saved to '%s':", out_dir)
    for k, p in saved.items():
        logger.info("  - %s: %s", k, p)


if __name__ == "__main__":
    main()
