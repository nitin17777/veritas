"""VERITAS Transformer Baseline Training Script.

Trains a Hugging Face Transformer sequence classifier on VERITAS ClaimRecord data.

Usage
-----
Smoke test (fast sanity check on tiny slice)::

    python scripts/train_transformer.py --config configs/transformer.yaml --smoke-test

Full training run::

    python scripts/train_transformer.py --config configs/transformer.yaml

Resume from checkpoint::

    python scripts/train_transformer.py --config configs/transformer.yaml --resume-from-checkpoint results/<run_id>/checkpoints/checkpoint-500

What it does
------------
1. Loads train, validation, and test JSONL files from ``processed_dir``.
2. Applies class-weighting (balanced, inverse_freq, or none).
3. Configures Hugging Face TrainingArguments (GPU memory safe, fp16 auto-detected, checkpointing).
4. Trains only on the train split; uses validation split for model selection (best macro-F1).
5. Evaluates on validation and test splits (test split evaluated only once after model is frozen).
6. Saves: checkpoints, best model, predictions JSONL, per-split metrics JSON, model card, config snapshot.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure src/ is importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    set_seed,
)

from veritas.config import ROOT_DIR, load_config
from veritas.models.evaluate import (
    compute_metrics,
    format_confusion_matrix,
    save_metrics,
    save_model_card,
    save_predictions,
)
from veritas.models.tfidf_lr import (
    LABEL_ORDER,
    label_distribution,
    load_jsonl_split,
)
from veritas.models.transformer import (
    ID2LABEL,
    LABEL2ID,
    NUM_LABELS,
    WeightedTrainer,
    compute_class_weights,
    make_compute_metrics,
    predict,
    prepare_dataset,
)
from veritas.run_utils import start_run

logger = logging.getLogger("veritas.train_transformer")


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "dataset": "liar",
    "processed_dir": "data/processed/liar",
    "model_name_or_path": "distilroberta-base",
    "random_seed": 42,
    "max_seq_length": 128,
    "imbalance_strategy": "balanced",
    "training": {
        "learning_rate": 3.0e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "num_train_epochs": 3,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "logging_steps": 50,
        "fp16": False,
        "dataloader_num_workers": 0,
    },
    "smoke_test": {
        "max_train_samples": 32,
        "max_eval_samples": 16,
        "max_test_samples": 16,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "logging_steps": 2,
    },
}


def _resolve_cfg(config_path: str | None, cli_args: argparse.Namespace) -> dict[str, Any]:
    """Merge default config with config file and CLI overrides."""
    import copy
    cfg = copy.deepcopy(_DEFAULTS)

    if config_path:
        file_cfg = load_config(config_path)
        base_cfg = file_cfg.get("baseline_transformer", {})
        for k, v in base_cfg.items():
            if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
                cfg[k].update(v)
            else:
                cfg[k] = v

    # CLI overrides
    if cli_args.dataset:
        cfg["dataset"] = cli_args.dataset
    if cli_args.processed_dir:
        cfg["processed_dir"] = cli_args.processed_dir
    if cli_args.model_name:
        cfg["model_name_or_path"] = cli_args.model_name
    if cli_args.seed is not None:
        cfg["random_seed"] = cli_args.seed
    if cli_args.imbalance_strategy:
        cfg["imbalance_strategy"] = cli_args.imbalance_strategy
    if cli_args.max_seq_length is not None:
        cfg["max_seq_length"] = cli_args.max_seq_length
    if cli_args.batch_size is not None:
        cfg["training"]["per_device_train_batch_size"] = cli_args.batch_size
    if cli_args.epochs is not None:
        cfg["training"]["num_train_epochs"] = cli_args.epochs
    if cli_args.lr is not None:
        cfg["training"]["learning_rate"] = cli_args.lr
    if cli_args.fp16 is not None:
        cfg["training"]["fp16"] = cli_args.fp16

    return cfg


def _check_gpu_and_adjust_config(cfg: dict[str, Any]) -> None:
    """Detect GPU, log memory, and auto-enable fp16 if CUDA is available."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        logger.info("GPU detected: %s (Total VRAM: %.0f MiB)", gpu_name, vram_mb)

        # Auto-enable fp16 on CUDA if not explicitly disabled
        if not cfg["training"].get("fp16"):
            cfg["training"]["fp16"] = True
            logger.info("CUDA detected: auto-enabled fp16 mixed precision.")

        # If VRAM is tight (<= 4096 MiB), adjust batch size if necessary
        if vram_mb <= 4096:
            orig_batch = cfg["training"].get("per_device_train_batch_size", 16)
            if orig_batch > 16:
                cfg["training"]["per_device_train_batch_size"] = 16
                logger.info("Tight GPU memory (%.0f MiB): capped train batch size to 16", vram_mb)
    else:
        logger.info("No CUDA GPU detected. Training on CPU.")
        cfg["training"]["fp16"] = False


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------


def run_training(
    cfg: dict[str, Any],
    run_dir: Path,
    smoke_test: bool = False,
    skip_test: bool = False,
    resume_from_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Execute transformer training and evaluation pipeline."""
    seed = int(cfg["random_seed"])
    set_seed(seed)
    _check_gpu_and_adjust_config(cfg)

    dataset_name = cfg["dataset"]
    processed_dir = ROOT_DIR / cfg["processed_dir"] if not Path(cfg["processed_dir"]).is_absolute() else Path(cfg["processed_dir"])
    model_name_or_path = cfg["model_name_or_path"]
    max_seq_length = int(cfg["max_seq_length"])
    imbalance_strategy = cfg.get("imbalance_strategy", "balanced")

    # 1. Load Data
    logger.info("Loading dataset splits from %s ...", processed_dir)
    train_records = load_jsonl_split(processed_dir, "train")
    val_records = load_jsonl_split(processed_dir, "validation")
    test_records = []
    if not skip_test:
        try:
            test_records = load_jsonl_split(processed_dir, "test")
        except FileNotFoundError:
            logger.warning("No test.jsonl found in %s; skipping test.", processed_dir)

    # Apply smoke test limits if requested
    if smoke_test:
        smk = cfg.get("smoke_test", {})
        n_tr = smk.get("max_train_samples", 32)
        n_va = smk.get("max_eval_samples", 16)
        n_te = smk.get("max_test_samples", 16)
        train_records = train_records[:n_tr]
        val_records = val_records[:n_va]
        if test_records:
            test_records = test_records[:n_te]
        logger.info("[SMOKE TEST] Truncated splits to train=%d, val=%d, test=%d", len(train_records), len(val_records), len(test_records))

        cfg["training"]["num_train_epochs"] = smk.get("num_train_epochs", 1)
        cfg["training"]["per_device_train_batch_size"] = smk.get("per_device_train_batch_size", 8)
        cfg["training"]["per_device_eval_batch_size"] = smk.get("per_device_eval_batch_size", 8)
        cfg["training"]["logging_steps"] = smk.get("logging_steps", 2)

    train_labels = [r["label"] for r in train_records]
    train_dist = label_distribution(train_labels)
    logger.info("Train label distribution (%d records): %s", len(train_records), train_dist)

    # 2. Tokenizer & Model
    logger.info("Loading tokenizer and model: %s", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # 3. Class weights
    class_weights = compute_class_weights(
        train_labels,
        strategy=imbalance_strategy,
        num_classes=NUM_LABELS,
        label2id=LABEL2ID,
    )
    if class_weights is not None:
        logger.info("Class weights (%s): %s", imbalance_strategy, class_weights.tolist())

    # 4. Prepare Datasets
    train_dataset = prepare_dataset(
        train_records,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        label2id=LABEL2ID,
        has_labels=True,
    )
    val_dataset = prepare_dataset(
        val_records,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        label2id=LABEL2ID,
        has_labels=True,
    )

    # 5. Training Arguments
    checkpoints_dir = run_dir / "checkpoints"
    t_cfg = dict(cfg["training"])

    batch_size = int(t_cfg.get("per_device_train_batch_size", 16))
    eval_batch_size = int(t_cfg.get("per_device_eval_batch_size", 32))
    grad_accum = int(t_cfg.get("gradient_accumulation_steps", 1))
    num_epochs = float(t_cfg.get("num_train_epochs", 3))

    total_steps = max(1, math.ceil(len(train_dataset) / (batch_size * grad_accum)) * int(math.ceil(num_epochs)))
    warmup_steps = int(t_cfg.get("warmup_steps", 0))
    if warmup_steps == 0 and "warmup_ratio" in t_cfg:
        warmup_steps = int(float(t_cfg["warmup_ratio"]) * total_steps)

    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        learning_rate=float(t_cfg.get("learning_rate", 3e-5)),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        weight_decay=float(t_cfg.get("weight_decay", 0.01)),
        warmup_steps=warmup_steps,
        eval_strategy=str(t_cfg.get("eval_strategy", "epoch")),
        save_strategy=str(t_cfg.get("save_strategy", "epoch")),
        load_best_model_at_end=bool(t_cfg.get("load_best_model_at_end", True)),
        metric_for_best_model=str(t_cfg.get("metric_for_best_model", "macro_f1")),
        greater_is_better=bool(t_cfg.get("greater_is_better", True)),
        save_total_limit=int(t_cfg.get("save_total_limit", 2)),
        logging_steps=int(t_cfg.get("logging_steps", 50)),
        fp16=bool(t_cfg.get("fp16", False)),
        dataloader_num_workers=int(t_cfg.get("dataloader_num_workers", 0)),
        seed=seed,
        report_to="none",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(ID2LABEL),
        class_weights=class_weights,
    )

    # 6. Train (with checkpoint recovery if specified)
    logger.info("Starting training ...")
    resume_arg = None
    if resume_from_checkpoint:
        if resume_from_checkpoint.lower() == "true":
            resume_arg = True
        else:
            resume_arg = resume_from_checkpoint
        logger.info("Resuming from checkpoint: %s", resume_arg)

    train_result = trainer.train(resume_from_checkpoint=resume_arg)
    logger.info("Training finished. Global step: %d, Training loss: %.4f", train_result.global_step, train_result.training_loss)

    # Save best model and tokenizer
    final_model_dir = run_dir / "model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    logger.info("Saved best model and tokenizer to %s", final_model_dir)

    # 7. Evaluate on Validation Split
    logger.info("Evaluating best model on validation split ...")
    val_preds = predict(
        model=model,
        tokenizer=tokenizer,
        records=val_records,
        batch_size=int(t_cfg.get("per_device_eval_batch_size", 32)),
        max_seq_length=max_seq_length,
        model_name=f"transformer_{model_name_or_path}",
        model_config=cfg,
    )
    val_gold = [p["gold_label"] for p in val_preds]
    val_pred = [p["predicted_label"] for p in val_preds]
    val_metrics = compute_metrics(val_gold, val_pred)

    save_predictions(val_preds, run_dir, split="validation", model_name=f"transformer_{model_name_or_path}")
    save_metrics(val_metrics, run_dir, split="validation", model_name=f"transformer_{model_name_or_path}")

    print("\n--- Validation Split Confusion Matrix ---")
    print(format_confusion_matrix(val_metrics))

    # 8. Evaluate on Test Split (only once, never used for selection)
    test_metrics = {}
    if test_records:
        logger.info("Evaluating best model on test split (%d records) ...", len(test_records))
        test_preds = predict(
            model=model,
            tokenizer=tokenizer,
            records=test_records,
            batch_size=int(t_cfg.get("per_device_eval_batch_size", 32)),
            max_seq_length=max_seq_length,
            model_name=f"transformer_{model_name_or_path}",
            model_config=cfg,
        )
        test_gold = [p["gold_label"] for p in test_preds]
        test_pred = [p["predicted_label"] for p in test_preds]
        test_metrics = compute_metrics(test_gold, test_pred)

        save_predictions(test_preds, run_dir, split="test", model_name=f"transformer_{model_name_or_path}")
        save_metrics(test_metrics, run_dir, split="test", model_name=f"transformer_{model_name_or_path}")

        print("\n--- Test Split Confusion Matrix ---")
        print(format_confusion_matrix(test_metrics))

    # 9. Model Card
    save_model_card(
        run_dir,
        model_name=f"transformer_{model_name_or_path}",
        dataset=dataset_name,
        train_split_size=len(train_records),
        val_split_size=len(val_records),
        test_split_size=len(test_records),
        best_C=float("nan"),  # Not applicable for transformers
        C_sweep_results={},
        train_label_dist=train_dist,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        pipeline_kwargs=cfg,
        random_seed=seed,
        class_imbalance_strategy=f"WeightedTrainer with loss weighting (strategy='{imbalance_strategy}')",
        notes=(
            f"Baseline: Transformer sequence classifier ({model_name_or_path}). "
            f"Trained on train split with {imbalance_strategy} class weights. "
            f"Model selection tuned exclusively on validation split macro-F1. "
            f"Test split evaluated only after model training completed and settings frozen."
        ),
    )

    logger.info("Run finished. Outputs saved to %s", run_dir)
    print(f"\n[DONE] All outputs and checkpoints saved to: {run_dir}")
    return {"val_metrics": val_metrics, "test_metrics": test_metrics, "run_dir": run_dir}


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train VERITAS Transformer Baseline Classifier"
    )
    parser.add_argument(
        "--config",
        default="configs/transformer.yaml",
        help="Path to transformer config YAML (default: configs/transformer.yaml)",
    )
    parser.add_argument("--processed-dir", default=None, help="Processed data directory")
    parser.add_argument("--dataset", default=None, help="Dataset identifier")
    parser.add_argument("--model-name", default=None, help="Pretrained model identifier or path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=None, help="Train batch size")
    parser.add_argument("--epochs", type=float, default=None, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--imbalance-strategy", default=None, help="Class imbalance strategy (balanced, inverse_freq, none)")
    parser.add_argument("--max-seq-length", type=int, default=None, help="Max sequence length")
    parser.add_argument("--fp16", action="store_true", default=None, help="Enable fp16 mixed precision")
    parser.add_argument("--no-fp16", action="store_false", dest="fp16", help="Disable fp16")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Fast sanity check on small subset of data (1 epoch)",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip test set evaluation (useful during development)",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Path to checkpoint directory to resume training from",
    )
    args = parser.parse_args()

    run_id, run_dir, _ = start_run(config_path=args.config)
    cfg = _resolve_cfg(args.config, args)

    run_training(
        cfg=cfg,
        run_dir=run_dir,
        smoke_test=args.smoke_test,
        skip_test=args.skip_test,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
