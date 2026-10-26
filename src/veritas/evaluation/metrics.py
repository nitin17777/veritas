"""Classification, faithfulness, system latency, and aggregate metrics for VERITAS P9.

Requirements:
- Match predictions to gold records using stable claim IDs.
- Fail clearly on missing or duplicated claim IDs.
- Distinguish unavailable metrics (None) from zero-valued metrics (0.0).
- Multi-run aggregation reporting mean and spread.
- Never invent fabricated metrics.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

STANDARD_LABEL_ORDER = ["True", "Partially True", "Misleading", "False", "Unverified"]


class EvaluationDataError(Exception):
    """Raised when evaluation inputs fail data integrity checks."""


# ---------------------------------------------------------------------------
# ID Matching and Validation
# ---------------------------------------------------------------------------


def match_gold_and_predictions(
    gold_records: list[dict[str, Any]],
    prediction_records: list[dict[str, Any]],
    strict_match: bool = True,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Match gold records and prediction records by stable claim ID.

    Args:
        gold_records: List of ground-truth claim dictionaries.
        prediction_records: List of predicted verdict dictionaries.
        strict_match: If True, raises error if any ID is missing in either set.

    Returns:
        List of paired (gold_record, prediction_record) tuples in stable order.

    Raises:
        EvaluationDataError: On duplicate IDs or mismatched IDs.
    """
    # 1. Index gold records with duplicate detection
    gold_by_id: dict[str, dict[str, Any]] = {}
    for g in gold_records:
        cid = str(g.get("id") or g.get("claim_id") or "").strip()
        if not cid:
            raise EvaluationDataError("Gold record missing 'id' or 'claim_id' identifier.")
        if cid in gold_by_id:
            raise EvaluationDataError(f"Duplicate claim ID detected in gold records: '{cid}'.")
        gold_by_id[cid] = g

    # 2. Index prediction records with duplicate detection
    pred_by_id: dict[str, dict[str, Any]] = {}
    for p in prediction_records:
        cid = str(p.get("claim_id") or p.get("id") or "").strip()
        if not cid:
            raise EvaluationDataError("Prediction record missing 'claim_id' or 'id' identifier.")
        if cid in pred_by_id:
            raise EvaluationDataError(f"Duplicate claim ID detected in predictions: '{cid}'.")
        pred_by_id[cid] = p

    # 3. Check for set equivalence
    gold_ids = set(gold_by_id.keys())
    pred_ids = set(pred_by_id.keys())

    if strict_match:
        missing_in_pred = gold_ids - pred_ids
        missing_in_gold = pred_ids - gold_ids
        if missing_in_pred or missing_in_gold:
            details = []
            if missing_in_pred:
                sample_missing = sorted(list(missing_in_pred))[:5]
                details.append(f"{len(missing_in_pred)} gold IDs missing in predictions (e.g. {sample_missing})")
            if missing_in_gold:
                sample_extra = sorted(list(missing_in_gold))[:5]
                details.append(f"{len(missing_in_gold)} prediction IDs missing in gold (e.g. {sample_extra})")
            raise EvaluationDataError(f"Mismatched claim IDs between gold and predictions: {'; '.join(details)}.")

    # 4. Pair matched records preserving gold record order
    matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for cid in gold_by_id:
        if cid in pred_by_id:
            matched.append((gold_by_id[cid], pred_by_id[cid]))

    return matched


# ---------------------------------------------------------------------------
# Classification Metrics
# ---------------------------------------------------------------------------


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Compute Accuracy, Macro-F1, per-class metrics, and confusion matrix.

    Args:
        y_true: List of ground-truth string labels.
        y_pred: List of predicted string labels.
        labels: Label classes to include in matrix and per-class reports.

    Returns:
        Structured dictionary containing all classification metrics.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: {len(y_true)} gold labels vs {len(y_pred)} predictions.")

    if not y_true:
        return {
            "accuracy": None,
            "macro_f1": None,
            "macro_precision": None,
            "macro_recall": None,
            "per_class": {},
            "confusion_matrix": {"labels": labels or [], "matrix": []},
            "total_samples": 0,
        }

    target_labels = labels or STANDARD_LABEL_ORDER
    # Ensure any unexpected label is also accounted for
    all_seen = set(y_true) | set(y_pred)
    extra_labels = sorted(list(all_seen - set(target_labels)))
    full_label_order = target_labels + extra_labels

    label2idx = {lbl: i for i, lbl in enumerate(full_label_order)}
    num_classes = len(full_label_order)

    # Initialize confusion matrix [gold, pred]
    cm = [[0] * num_classes for _ in range(num_classes)]
    for yt, yp in zip(y_true, y_pred):
        if yt in label2idx and yp in label2idx:
            cm[label2idx[yt]][label2idx[yp]] += 1

    total_samples = len(y_true)
    correct_samples = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = round(correct_samples / total_samples, 4)

    # Per-class precision, recall, f1, support
    per_class: dict[str, dict[str, Any]] = {}
    f1_list: list[float] = []
    prec_list: list[float] = []
    rec_list: list[float] = []

    for i, lbl in enumerate(full_label_order):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(num_classes) if r != i)
        fn = sum(cm[i][c] for c in range(num_classes) if c != i)
        support = sum(cm[i])

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[lbl] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }

        # Include in macro average only if class has representation in gold or pred
        if support > 0 or (tp + fp) > 0:
            f1_list.append(f1)
            prec_list.append(prec)
            rec_list.append(rec)

    macro_f1 = round(sum(f1_list) / len(f1_list), 4) if f1_list else 0.0
    macro_prec = round(sum(prec_list) / len(prec_list), 4) if prec_list else 0.0
    macro_rec = round(sum(rec_list) / len(rec_list), 4) if rec_list else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": full_label_order,
            "matrix": cm,
        },
        "total_samples": total_samples,
    }


# ---------------------------------------------------------------------------
# Latency & Percentiles
# ---------------------------------------------------------------------------


def compute_latency_metrics(latencies: list[float]) -> dict[str, float | None]:
    """Compute average and percentile latency statistics.

    Args:
        latencies: List of latency values in seconds.

    Returns:
        Dictionary with avg, p50, p90, p95, p99, or None values if empty.
    """
    clean_vals = sorted([float(x) for x in latencies if x is not None and not math.isnan(x)])
    if not clean_vals:
        return {
            "avg_latency_s": None,
            "p50_latency_s": None,
            "p90_latency_s": None,
            "p95_latency_s": None,
            "p99_latency_s": None,
        }

    n = len(clean_vals)

    def percentile(p: float) -> float:
        idx = int(math.ceil(p * n)) - 1
        return clean_vals[max(0, min(idx, n - 1))]

    return {
        "avg_latency_s": round(sum(clean_vals) / n, 4),
        "p50_latency_s": round(percentile(0.50), 4),
        "p90_latency_s": round(percentile(0.90), 4),
        "p95_latency_s": round(percentile(0.95), 4),
        "p99_latency_s": round(percentile(0.99), 4),
    }


# ---------------------------------------------------------------------------
# Coverage vs. Accuracy Curve & System Metrics
# ---------------------------------------------------------------------------


def compute_coverage_accuracy_curve(
    matched_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    num_thresholds: int = 11,
) -> list[dict[str, Any]]:
    """Compute coverage-versus-accuracy tradeoff across confidence thresholds."""
    if not matched_pairs:
        return []

    thresholds = [round(i / (num_thresholds - 1), 2) for i in range(num_thresholds)]
    curve: list[dict[str, Any]] = []

    total = len(matched_pairs)

    for thresh in thresholds:
        retained = [
            (g, p)
            for g, p in matched_pairs
            if float(p.get("confidence", 0.0)) >= thresh
        ]

        cov = round(len(retained) / total, 4) if total > 0 else 0.0
        if retained:
            correct = sum(
                1 for g, p in retained
                if g.get("label") == p.get("label")
            )
            acc = round(correct / len(retained), 4)
        else:
            acc = 1.0  # Perfect accuracy when abstaining completely

        curve.append({
            "confidence_threshold": thresh,
            "coverage": cov,
            "accuracy": acc,
            "retained_count": len(retained),
        })

    return curve


def compute_system_quality_metrics(
    matched_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    """Compute abstention rate, parse failure rate, and hallucination rate proxy."""
    if not matched_pairs:
        return {
            "json_parse_failure_rate": None,
            "abstention_rate": None,
            "mean_faithfulness": None,
            "hallucination_rate_proxy": None,
        }

    total = len(matched_pairs)
    parse_failures = 0
    abstained_count = 0
    faithfulness_scores: list[float] = []
    hallucination_flags = 0

    for _, pred in matched_pairs:
        # 1. JSON parsing failure
        if pred.get("parse_failed") is True:
            parse_failures += 1

        # 2. Abstention
        meta = pred.get("model_metadata") or {}
        abst_info = meta.get("abstention") or {}
        is_abstained = (
            pred.get("abstained") is True
            or abst_info.get("abstained") is True
            or pred.get("label") == "Unverified"
        )
        if is_abstained:
            abstained_count += 1

        # 3. Faithfulness score
        faith_val = pred.get("faithfulness_score")
        if faith_val is None:
            faith_val = abst_info.get("faithfulness_score")
        if faith_val is not None and not math.isnan(float(faith_val)):
            faithfulness_scores.append(float(faith_val))

        # 4. Hallucination-rate proxy:
        # A claim is flagged as potentially hallucinated if:
        # a) Evidence IDs cited were not present in retrieved passages, OR
        # b) Faithfulness is very low (< 0.50) while claiming True/False
        cited_ids = set(pred.get("evidence_ids") or [])
        retrieved_passages = pred.get("retrieved_passages") or []
        retrieved_ids = {
            str(p.get("passage_id") or p.get("id") or "")
            for p in retrieved_passages
        }

        has_unsupported_citation = bool(cited_ids and not cited_ids.issubset(retrieved_ids))
        has_low_faithfulness = (faith_val is not None and float(faith_val) < 0.50 and pred.get("label") in {"True", "False"})

        if has_unsupported_citation or has_low_faithfulness:
            hallucination_flags += 1

    return {
        "json_parse_failure_rate": round(parse_failures / total, 4),
        "abstention_rate": round(abstained_count / total, 4),
        "mean_faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 4) if faithfulness_scores else None,
        "hallucination_rate_proxy": round(hallucination_flags / total, 4),
    }


# ---------------------------------------------------------------------------
# Multi-Run Aggregator
# ---------------------------------------------------------------------------


def aggregate_multi_run_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metric dictionaries across multiple repeated experiment runs.

    Computes mean and spread (std) across all shared numerical scalar metrics.
    """
    if not runs:
        return {}

    if len(runs) == 1:
        return {
            "num_runs": 1,
            "metrics": runs[0],
            "spread": {},
        }

    num_runs = len(runs)
    aggregated_scalars: dict[str, dict[str, float]] = {}

    # Extract all scalar keys from run 1
    sample = runs[0]
    scalar_keys = [
        k for k, v in sample.items()
        if isinstance(v, (int, float)) and v is not None and not isinstance(v, bool)
    ]

    for k in scalar_keys:
        vals = [float(r[k]) for r in runs if r.get(k) is not None and not math.isnan(float(r[k]))]
        if not vals:
            continue
        mean_val = sum(vals) / len(vals)
        variance = sum((x - mean_val) ** 2 for x in vals) / len(vals)
        std_val = math.sqrt(variance)

        aggregated_scalars[k] = {
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }

    return {
        "num_runs": num_runs,
        "aggregated_metrics": aggregated_scalars,
    }
