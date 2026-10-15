"""VERITAS TF-IDF + Logistic Regression Training Script.

Trains a TF-IDF/LR baseline classifier on VERITAS ClaimRecord data.

Usage
-----
From the project root::

    python scripts/train_tfidf_lr.py
    python scripts/train_tfidf_lr.py --config configs/tfidf_lr.yaml
    python scripts/train_tfidf_lr.py --processed-dir data/processed/liar --dataset liar
    python scripts/train_tfidf_lr.py --config configs/tfidf_lr.yaml --skip-test

What it does
------------
1. Loads train, validation, and test JSONL files from ``processed_dir``.
2. Runs a C-sweep on the validation split (macro-F1 selection criterion).
3. Trains final model on train split with the best C.
4. Evaluates on validation and test splits (test is never used for selection).
5. Saves: fitted model, predictions JSONL, per-split metrics JSON, model card.

Outputs (inside the configured ``results_dir/{run_id}/``)
----------------------------------------------------------
- ``model.joblib``         — fitted sklearn Pipeline
- ``predictions_validation.jsonl``
- ``predictions_test.jsonl``
- ``metrics_validation.json``
- ``metrics_test.json``
- ``model_card.json``
- ``config.yaml``          — exact config snapshot (from start_run)
- ``run.log``              — full run log
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure src/ is importable when run as a script
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from veritas.config import ROOT_DIR, load_config
from veritas.models.evaluate import (
    compute_metrics,
    format_confusion_matrix,
    save_metrics,
    save_model_card,
    save_predictions,
)
from veritas.models.tfidf_lr import (
    label_distribution,
    load_jsonl_split,
    load_model,
    predict,
    records_to_texts_labels,
    save_model,
    select_best_C,
    train,
)
from veritas.run_utils import start_run

logger = logging.getLogger("veritas.train_tfidf_lr")


# ---------------------------------------------------------------------------
# Defaults — overridden by config file
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "dataset": "liar",
    "processed_dir": "data/processed/liar",
    "model_name": "tfidf_lr",
    "random_seed": 42,
    "tfidf": {
        "max_features": 100_000,
        "ngram_min": 1,
        "ngram_max": 2,
        "sublinear_tf": True,
    },
    "logistic_regression": {
        "C_values": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        "max_iter": 1000,
        "solver": "saga",
    },
}


def _resolve_cfg(config_path: str | None, cli_args: argparse.Namespace) -> dict:
    """Merge config file with CLI overrides. CLI wins."""
    cfg = dict(_DEFAULTS)

    if config_path:
        file_cfg = load_config(config_path)
        baseline_cfg = file_cfg.get("baseline_tfidf_lr", {})
        # Deep merge top-level keys
        for key, value in baseline_cfg.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key] = {**cfg[key], **value}
            else:
                cfg[key] = value

    # CLI overrides
    if cli_args.dataset:
        cfg["dataset"] = cli_args.dataset
    if cli_args.processed_dir:
        cfg["processed_dir"] = cli_args.processed_dir
    if cli_args.seed is not None:
        cfg["random_seed"] = cli_args.seed

    return cfg


def _run(
    cfg: dict,
    run_dir: Path,
    *,
    skip_test: bool = False,
) -> None:
    dataset = cfg["dataset"]
    processed_dir = ROOT_DIR / cfg["processed_dir"]
    seed = cfg["random_seed"]
    model_name = cfg["model_name"]

    tfidf_cfg = cfg.get("tfidf", _DEFAULTS["tfidf"])
    lr_cfg = cfg.get("logistic_regression", _DEFAULTS["logistic_regression"])

    pipeline_kwargs = {
        "max_features": tfidf_cfg.get("max_features", 100_000),
        "ngram_min": tfidf_cfg.get("ngram_min", 1),
        "ngram_max": tfidf_cfg.get("ngram_max", 2),
        "sublinear_tf": tfidf_cfg.get("sublinear_tf", True),
        "max_iter": lr_cfg.get("max_iter", 1000),
        "solver": lr_cfg.get("solver", "saga"),
    }
    C_values = [float(c) for c in lr_cfg.get("C_values", [0.01, 0.1, 0.5, 1.0, 5.0, 10.0])]

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    logger.info("Loading data from %s", processed_dir)
    train_records = load_jsonl_split(processed_dir, "train")
    val_records = load_jsonl_split(processed_dir, "validation")

    train_texts, train_labels = records_to_texts_labels(train_records)
    val_texts, val_labels = records_to_texts_labels(val_records)

    train_dist = label_distribution(train_labels)
    val_dist = label_distribution(val_labels)

    logger.info("Train label distribution: %s", train_dist)
    logger.info("Validation label distribution: %s", val_dist)

    # -----------------------------------------------------------------------
    # 2. C-sweep on validation
    # -----------------------------------------------------------------------
    logger.info("Starting C-sweep over %s on validation split ...", C_values)
    best_C, sweep_scores = select_best_C(
        train_texts, train_labels,
        val_texts, val_labels,
        C_values=C_values,
        pipeline_kwargs=pipeline_kwargs,
        random_state=seed,
    )

    # -----------------------------------------------------------------------
    # 3. Train final model on train split only (DO NOT include validation)
    # -----------------------------------------------------------------------
    logger.info("Training final model with C=%.4f on train split only", best_C)
    pipeline = train(
        train_texts,
        train_labels,
        C=best_C,
        random_state=seed,
        pipeline_kwargs=pipeline_kwargs,
    )

    # -----------------------------------------------------------------------
    # 4. Save model
    # -----------------------------------------------------------------------
    model_path = run_dir / "model.joblib"
    save_model(pipeline, model_path)

    # -----------------------------------------------------------------------
    # 5. Evaluate on validation
    # -----------------------------------------------------------------------
    logger.info("Evaluating on validation split ...")
    val_preds = predict(pipeline, val_records)
    val_gold = [p["gold_label"] for p in val_preds]
    val_pred = [p["predicted_label"] for p in val_preds]
    val_metrics = compute_metrics(val_gold, val_pred)

    save_predictions(val_preds, run_dir, split="validation", model_name=model_name,
                     model_config={"C": best_C, **pipeline_kwargs})
    save_metrics(val_metrics, run_dir, split="validation", model_name=model_name)
    print("\n" + format_confusion_matrix(val_metrics))

    # -----------------------------------------------------------------------
    # 6. Evaluate on test (never used for selection)
    # -----------------------------------------------------------------------
    test_metrics: dict = {}
    test_split_size = 0
    if not skip_test:
        try:
            test_records = load_jsonl_split(processed_dir, "test")
        except FileNotFoundError:
            logger.warning("No test.jsonl found — skipping test evaluation.")
        else:
            test_split_size = len(test_records)
            logger.info("Evaluating on test split (%d records) ...", test_split_size)
            test_preds = predict(pipeline, test_records)
            test_gold = [p["gold_label"] for p in test_preds]
            test_pred = [p["predicted_label"] for p in test_preds]
            test_metrics = compute_metrics(test_gold, test_pred)

            save_predictions(test_preds, run_dir, split="test", model_name=model_name,
                             model_config={"C": best_C, **pipeline_kwargs})
            save_metrics(test_metrics, run_dir, split="test", model_name=model_name)
            print("\n" + format_confusion_matrix(test_metrics))

    # -----------------------------------------------------------------------
    # 7. Save model card
    # -----------------------------------------------------------------------
    save_model_card(
        run_dir,
        model_name=model_name,
        dataset=dataset,
        train_split_size=len(train_records),
        val_split_size=len(val_records),
        test_split_size=test_split_size,
        best_C=best_C,
        C_sweep_results=sweep_scores,
        train_label_dist=train_dist,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        pipeline_kwargs={"C": best_C, **pipeline_kwargs},
        random_seed=seed,
    )

    logger.info("Run complete. Outputs in: %s", run_dir)
    print(f"\n[DONE] Run outputs saved to: {run_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train VERITAS TF-IDF + Logistic Regression baseline"
    )
    parser.add_argument(
        "--config",
        default="configs/tfidf_lr.yaml",
        help="Path to baseline config YAML (default: configs/tfidf_lr.yaml)",
    )
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="Path to processed JSONL directory (overrides config)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset identifier for logging (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip test-set evaluation (useful during development)",
    )
    args = parser.parse_args()

    # start_run sets up logging, creates results dir, snapshots config
    run_id, run_dir, _ = start_run(config_path=args.config)
    cfg = _resolve_cfg(args.config, args)

    logger.info("Dataset: %s | seed: %d", cfg["dataset"], cfg["random_seed"])

    _run(cfg, run_dir, skip_test=args.skip_test)
    return 0


if __name__ == "__main__":
    sys.exit(main())
