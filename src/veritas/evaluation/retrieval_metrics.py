"""Retrieval evaluation metrics for VERITAS P9 Evaluation.

Computes Recall@k, nDCG@k, and MRR@k strictly when gold evidence is available.
If gold evidence annotations are missing or empty for a claim, retrieval metrics
are returned as None (unavailable), distinguishing missing annotations from
zero-valued retrieval performance.
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

    # Ideal DCG: all gold items ranked at top up to min(k, |gold|)
    n_ideal = min(k, len(gold_set))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_ideal + 1))

    if idcg <= 0.0:
        return 0.0

    return dcg / idcg


def compute_mrr_at_k(
    retrieved_ids: list[str],
    gold_ids: set[str] | list[str],
    k: int,
) -> float | None:
    """Compute Mean Reciprocal Rank (MRR) at rank k.

    Returns reciprocal rank of first relevant passage within top-k,
    or 0.0 if not found, or None if gold_ids is empty.
    """
    gold_set = set(gold_ids)
    if not gold_set:
        return None

    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in gold_set:
            return 1.0 / i
    return 0.0


def evaluate_retrieval_batch(
    retrieval_records: list[dict[str, Any]],
    k_values: list[int] | None = None,
) -> dict[str, Any]:
    """Aggregate retrieval metrics over claims where gold evidence exists.

    Args:
        retrieval_records: List of dicts, each with:
            - 'retrieved_ids': list of retrieved passage IDs.
            - 'gold_ids': list/set of gold relevant passage IDs (or None/empty).
        k_values: Ranks to evaluate (default: [1, 3, 5, 10]).

    Returns:
        Dictionary of aggregated metrics, total claims, and eligible claims count.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    eligible_count = 0
    total_count = len(retrieval_records)
    metric_sums: dict[str, float] = {
        f"recall@{k}": 0.0 for k in k_values
    }
    metric_sums.update({f"ndcg@{k}": 0.0 for k in k_values})
    metric_sums.update({f"mrr@{k}": 0.0 for k in k_values})

    for item in retrieval_records:
        gold = item.get("gold_ids") or []
        retrieved = item.get("retrieved_ids") or []

        if not gold:
            # Skip claims without gold evidence annotations (e.g. LIAR dataset)
            continue

        eligible_count += 1
        for k in k_values:
            rec = compute_recall_at_k(retrieved, gold, k)
            if rec is not None:
                metric_sums[f"recall@{k}"] += rec

            ndcg = compute_ndcg_at_k(retrieved, gold, k)
            if ndcg is not None:
                metric_sums[f"ndcg@{k}"] += ndcg

            mrr = compute_mrr_at_k(retrieved, gold, k)
            if mrr is not None:
                metric_sums[f"mrr@{k}"] += mrr

    if eligible_count == 0:
        return {
            "eligible_claims": 0,
            "total_claims": total_count,
            "status": "unavailable",
            "message": "No gold evidence annotations available for retrieval evaluation.",
            "metrics": {k: None for k in metric_sums},
        }

    averaged_metrics = {
        m_name: round(val / eligible_count, 4)
        for m_name, val in metric_sums.items()
    }

    return {
        "eligible_claims": eligible_count,
        "total_claims": total_count,
        "status": "available",
        "metrics": averaged_metrics,
    }
