"""Tests for retrieval evaluation metrics (Recall@k, nDCG@k) and evaluation harness."""

import json
import math
from pathlib import Path

import pytest

from veritas.retrieval.evaluate import run_retrieval_evaluation
from veritas.retrieval.metrics import (
    compute_dcg_at_k,
    compute_ndcg_at_k,
    compute_recall_at_k,
    evaluate_retrieval_predictions,
)


def test_recall_at_k_calculation():
    gold = {"doc_1", "doc_2"}
    retrieved = ["doc_1", "doc_3", "doc_4", "doc_2", "doc_5"]

    # At k=1: hits=1 (doc_1) -> 1/2 = 0.5
    assert compute_recall_at_k(retrieved, gold, k=1) == 0.5

    # At k=3: hits=1 (doc_1) -> 1/2 = 0.5
    assert compute_recall_at_k(retrieved, gold, k=3) == 0.5

    # At k=4: hits=2 (doc_1, doc_2) -> 2/2 = 1.0
    assert compute_recall_at_k(retrieved, gold, k=4) == 1.0

    # At k=5: hits=2 -> 1.0
    assert compute_recall_at_k(retrieved, gold, k=5) == 1.0


def test_recall_at_k_missing_gold_returns_none():
    assert compute_recall_at_k(["doc_1", "doc_2"], set(), k=5) is None
    assert compute_recall_at_k(["doc_1", "doc_2"], [], k=5) is None


def test_ndcg_at_k_calculation():
    gold = {"doc_1", "doc_2"}

    # Perfect ranking: doc_1 (rank 1), doc_2 (rank 2)
    perfect_retrieved = ["doc_1", "doc_2", "doc_3"]
    assert compute_ndcg_at_k(perfect_retrieved, gold, k=2) == 1.0
    assert compute_ndcg_at_k(perfect_retrieved, gold, k=5) == 1.0

    # Sub-optimal ranking: doc_3 (rank 1), doc_1 (rank 2), doc_2 (rank 3)
    # DCG@3 = 0/log2(2) + 1/log2(3) + 1/log2(4) = 1/1.58496 + 1/2.0 = 0.63093 + 0.5 = 1.13093
    # IDCG@3 = 1/log2(2) + 1/log2(3) = 1.0 + 0.63093 = 1.63093
    # nDCG@3 = 1.13093 / 1.63093 ~= 0.6934
    subopt_retrieved = ["doc_3", "doc_1", "doc_2"]
    ndcg_3 = compute_ndcg_at_k(subopt_retrieved, gold, k=3)
    expected_ndcg_3 = (1.0 / math.log2(3) + 1.0 / math.log2(4)) / (1.0 / math.log2(2) + 1.0 / math.log2(3))
    assert abs(ndcg_3 - expected_ndcg_3) < 1e-4

    # Zero hits
    zero_retrieved = ["doc_x", "doc_y", "doc_z"]
    assert compute_ndcg_at_k(zero_retrieved, gold, k=3) == 0.0


def test_ndcg_at_k_missing_gold_returns_none():
    assert compute_ndcg_at_k(["doc_1"], set(), k=5) is None


def test_evaluate_predictions_skips_claims_missing_gold():
    """VERITAS requirement: Do not report retrieval metrics when gold evidence is missing."""
    instances = [
        # Claim with gold
        {
            "claim_id": "c_with_gold",
            "retrieved_ids": ["doc_1", "doc_2", "doc_3"],
            "gold_ids": ["doc_1"],
        },
        # Claim WITHOUT gold
        {
            "claim_id": "c_no_gold",
            "retrieved_ids": ["doc_4", "doc_5"],
            "gold_ids": [],
        },
    ]

    report = evaluate_retrieval_predictions(instances, k_values=[1, 5])

    assert report["total_claims"] == 2
    assert report["claims_with_gold_evidence"] == 1
    assert report["claims_without_gold_evidence"] == 1
    assert report["metrics_reported"] is True

    # Check that per-claim metrics only contains the claim with gold
    per_claim = report["per_claim_metrics"]
    assert len(per_claim) == 1
    assert per_claim[0]["claim_id"] == "c_with_gold"

    # Mean metrics are computed exclusively from c_with_gold
    assert report["mean_metrics"]["Recall@1"] == 1.0
    assert report["mean_metrics"]["nDCG@1"] == 1.0


def test_evaluate_predictions_all_claims_missing_gold():
    """When all claims lack gold evidence, report metrics_reported=False and give reason."""
    instances = [
        {"claim_id": "c1", "retrieved_ids": ["d1"], "gold_ids": []},
        {"claim_id": "c2", "retrieved_ids": ["d2"], "gold_ids": []},
    ]

    report = evaluate_retrieval_predictions(instances, k_values=[1, 5])
    assert report["total_claims"] == 2
    assert report["claims_with_gold_evidence"] == 0
    assert report["claims_without_gold_evidence"] == 2
    assert report["metrics_reported"] is False
    assert "reason" in report
    assert "mean_metrics" not in report


def test_run_retrieval_evaluation_persists_results_and_config(tmp_path: Path):
    out_dir = tmp_path / "results" / "eval_run"

    res = run_retrieval_evaluation(
        output_dir=out_dir,
        mode="hybrid",
        smoke_test=True,
    )

    assert (out_dir / "retrieval_metrics.json").is_file()
    assert (out_dir / "evaluation_config.json").is_file()

    with (out_dir / "retrieval_metrics.json").open("r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    with (out_dir / "evaluation_config.json").open("r", encoding="utf-8") as f:
        config_data = json.load(f)

    assert metrics_data["metrics_reported"] is True
    assert metrics_data["claims_with_gold_evidence"] == 2
    assert metrics_data["claims_without_gold_evidence"] == 1  # 1 claim lacked gold in smoke fixture

    assert config_data["mode"] == "hybrid"
    assert config_data["bm25_top_k"] == 50
    assert config_data["dense_top_k"] == 50
    assert config_data["rrf_k"] == 60
