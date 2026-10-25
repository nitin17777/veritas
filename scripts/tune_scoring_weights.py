#!/usr/bin/env python
"""CLI script for validation-only grid search of VERITAS P4 scoring weights.

STRICT METHODOLOGICAL GUARD:
This script tunes weights exclusively on validation / dev sets. Tuning on test sets
is prohibited and will raise an error.

Usage
-----
# Run with built-in synthetic validation claims:
python scripts/tune_scoring_weights.py --synthetic-demo

# Run on a validation JSONL dataset:
python scripts/tune_scoring_weights.py \\
    --validation-file data/processed/val_claims.jsonl \\
    --split val \\
    --output results/best_scoring_weights.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure src/ is importable
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from veritas.retrieval.schema import Passage
from veritas.scoring.grid_search import ScoringGridSearch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("tune_scoring_weights")


def create_synthetic_validation_data() -> list[dict]:
    """Create synthetic validation queries with known gold evidence and varied domain/date metadata."""
    return [
        {
            "claim": "mRNA COVID-19 vaccines prevent severe hospitalization.",
            "claim_date": "2021-06-01",
            "claim_type": "health",
            "gold_passage_ids": ["p1"],
            "candidates": [
                Passage("p1", "Vaccines reduce hospitalizations.", date="2021-05-15", domain="cdc.gov"),
                Passage("p2", "Random historical statement.", date="1995-01-01", domain="unverified-blog.net"),
                Passage("p3", "Old report on general health.", date="2010-06-01", domain="who.int"),
            ],
        },
        {
            "claim": "Spacecraft landed on Mars in recent mission.",
            "claim_date": "2021-02-18",
            "claim_type": "science",
            "gold_passage_ids": ["p4"],
            "candidates": [
                Passage("p5", "Speculation about aliens on Mars.", date="2018-01-01", domain="tabloid.co.uk"),
                Passage("p4", "Perseverance rover successfully touches down on Mars.", date="2021-02-18", domain="apnews.com"),
                Passage("p6", "Weather report from Florida.", date="2021-02-20", domain="weather.com"),
            ],
        },
        {
            "claim": "Election results certified by officials.",
            "claim_date": "2020-11-15",
            "claim_type": "election",
            "gold_passage_ids": ["p7"],
            "candidates": [
                Passage("p8", "Unsubstantiated claims of ballot issues.", date="2020-11-04", domain="partisan-blog.com"),
                Passage("p7", "Election officials certify official state vote tallies.", date="2020-11-14", domain="reuters.com"),
                Passage("p9", "Sports scores from Sunday.", date="2020-11-15", domain="sports.com"),
            ],
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune VERITAS P4 scoring weights (alpha, beta, gamma, tau) on validation data."
    )
    parser.add_argument(
        "--validation-file",
        type=str,
        default=None,
        help="Path to JSONL file containing validation queries and candidates.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Split name (must be 'val', 'validation', or 'dev'; 'test' is prohibited).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="mrr@10",
        help="Evaluation metric ('mrr@10', 'recall@5').",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.1,
        help="Grid step for simplex weights (default: 0.1).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output JSON path for saving optimal weights.",
    )
    parser.add_argument(
        "--synthetic-demo",
        action="store_true",
        help="Run grid search using synthetic validation fixtures.",
    )

    args = parser.parse_args()

    # Refuse test split immediately
    if "test" in args.split.lower():
        print(f"ERROR: Split '{args.split}' is not permitted. Grid search is strictly validation-only.", file=sys.stderr)
        sys.exit(1)

    if args.synthetic_demo or not args.validation_file:
        logger.info("Using built-in synthetic validation dataset (3 claims)...")
        val_data = create_synthetic_validation_data()
    else:
        val_path = Path(args.validation_file)
        if not val_path.exists():
            print(f"ERROR: Validation file not found: {val_path}", file=sys.stderr)
            sys.exit(1)
        val_data = []
        with val_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    val_data.append(json.loads(line))

    grid_search = ScoringGridSearch(
        weight_step=args.step,
        metric=args.metric,
    )

    result = grid_search.run(validation_data=val_data, split_name=args.split)

    print("\n" + "=" * 60)
    print("VERITAS P4 Scoring Hyperparameter Grid Search Results")
    print("=" * 60)
    print(f"Validation Split        : {result.split_name}")
    print(f"Validation Queries      : {result.num_validation_queries}")
    print(f"Evaluated Metric        : {result.metric_name}")
    print(f"Pure Relevance Baseline : {result.baseline_pure_relevance_score:.4f}")
    print(f"Best Combined Score     : {result.best_metric_score:.4f}")
    print("-" * 60)
    print(f"Optimal alpha (relevance)  : {result.best_alpha:.2f}")
    print(f"Optimal beta (credibility) : {result.best_beta:.2f}")
    print(f"Optimal gamma (temporal)   : {result.best_gamma:.2f}")
    print(f"Optimal tau (decay days)   : {result.best_tau:.1f}")
    print("=" * 60)

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nSaved optimal configuration to {out_p}")


if __name__ == "__main__":
    main()
