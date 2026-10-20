"""Information retrieval evaluation metrics for VERITAS.

Computes Recall@k and nDCG@k strictly when gold evidence is available.
If gold evidence annotations are missing for a claim, retrieval metrics
are omitted per the VERITAS specification.
"""

from __future__ import annotations

import math
from typing import Any


def compute_recall_at_k(
    retrieved_ids: list[str],
    gold_ids: set[str] | list[str],
    k: int,
) -> float | None:
    """Compute Recall@k for binary relevance.

    Args:
        retrieved_ids: Ordered list of retrieved passage IDs.
        gold_ids: Set or list of gold relevant passage IDs.
        k: Cut-off rank.

    Returns:
        Recall@k as a float between 0.0 and 1.0, or None if gold_ids is empty.
    """
    gold_set = set(gold_ids)
    if not gold_set:
        return None

    if k <= 0:
        return 0.0

    retrieved_top_k = retrieved_ids[:k]
    hits = len(set(retrieved_top_k) & gold_set)
    return hits / len(gold_set)


def compute_dcg_at_k(
    retrieved_ids: list[str],
    gold_ids: set[str] | list[str],
    k: int,
) -> float:
    """Compute Discounted Cumulative Gain at rank k."""
    gold_set = set(gold_ids)
    if not gold_set or k <= 0:
        return 0.0

    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in gold_set:
            dcg += 1.0 / math.log2(i + 1)
    return dcg


def compute_ndcg_at_k(
    retrieved_ids: list[str],
    gold_ids: set[str] | list[str],
    k: int,
) -> float | None:
    """Compute Normalized Discounted Cumulative Gain at rank k.

    Args:
        retrieved_ids: Ordered list of retrieved passage IDs.
        gold_ids: Set or list of gold relevant passage IDs.
        k: Cut-off rank.

    Returns:
        nDCG@k as a float between 0.0 and 1.0, or None if gold_ids is empty.
    """
    gold_set = set(gold_ids)
    if not gold_set:
        return None

    if k <= 0:
        return 0.0

    dcg = compute_dcg_at_k(retrieved_ids, gold_set, k)

    # Ideal DCG: all gold items ranked at the top up to min(k, |gold|)
    n_ideal = min(k, len(gold_set))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_ideal + 1))

    if idcg <= 0.0:
        return 0.0

    return dcg / idcg


def evaluate_retrieval_predictions(
    instances: list[dict[str, Any]],
    k_values: list[int] | None = None,
) -> dict[str, Any]:
    """Evaluate retrieval predictions across a collection of claims.

    Enforces the rule:
    - Never reports retrieval metrics for claims where gold evidence is missing.
    - Aggregates metrics strictly over claims with valid gold evidence annotations.

    Args:
        instances: List of dicts, each containing:
                   - 'claim_id': str
                   - 'retrieved_ids': list[str]
                   - 'gold_ids': list[str] or set[str]
        k_values: List of k thresholds to evaluate (default [1, 5, 10, 20, 50]).

    Returns:
        Evaluation dictionary with summary metrics and per-claim results.
    """
    if k_values is None:
        k_values = [1, 5, 10, 20, 50]

    evaluated_claims: list[dict[str, Any]] = []
    skipped_claims: list[str] = []

    accumulators: dict[str, list[float]] = {f"Recall@{k}": [] for k in k_values}
    accumulators.update({f"nDCG@{k}": [] for k in k_values})

    for item in instances:
        claim_id = str(item.get("claim_id", "unknown"))
        retrieved = list(item.get("retrieved_ids", []))
        raw_gold = item.get("gold_ids") or []
        gold_set = set(raw_gold) if isinstance(raw_gold, (list, set, tuple)) else set()

        if not gold_set:
            skipped_claims.append(claim_id)
            continue

        claim_metrics: dict[str, Any] = {
            "claim_id": claim_id,
            "gold_count": len(gold_set),
            "retrieved_count": len(retrieved),
        }

        for k in k_values:
            rec = compute_recall_at_k(retrieved, gold_set, k)
            ndcg = compute_ndcg_at_k(retrieved, gold_set, k)
            claim_metrics[f"Recall@{k}"] = rec
            claim_metrics[f"nDCG@{k}"] = ndcg
            if rec is not None:
                accumulators[f"Recall@{k}"].append(rec)
            if ndcg is not None:
                accumulators[f"nDCG@{k}"].append(ndcg)

        evaluated_claims.append(claim_metrics)

    summary: dict[str, Any] = {
        "total_claims": len(instances),
        "claims_with_gold_evidence": len(evaluated_claims),
        "claims_without_gold_evidence": len(skipped_claims),
        "metrics_reported": len(evaluated_claims) > 0,
    }

    if evaluated_claims:
        mean_metrics: dict[str, float] = {}
        for metric_name, values in accumulators.items():
            mean_metrics[metric_name] = sum(values) / len(values) if values else 0.0
        summary["mean_metrics"] = mean_metrics
        summary["per_claim_metrics"] = evaluated_claims
    else:
        summary["reason"] = (
            "No gold evidence annotations available in evaluated instances. "
            "Retrieval metrics are not reported per VERITAS specification."
        )

    return summary
