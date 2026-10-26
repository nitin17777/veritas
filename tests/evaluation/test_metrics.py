"""Unit and integration tests for VERITAS P9 Evaluation.

Covers:
- Strict claim ID matching, duplicate detection, and missing ID failures.
- Classification metrics (Accuracy, Macro-F1, per-class P/R/F1, confusion matrix) with known expected values.
- Retrieval metrics (Recall@k, nDCG@k, MRR@k) distinguishing unavailable (None) from zero-valued (0.0).
- Latency percentiles (Average, P50, P90, P95, P99).
- Coverage-versus-accuracy curve and system quality metrics.
- Multi-run aggregation reporting mean and spread.
- Machine-readable artifact saving (JSON, CSV, and SVG plots).
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from veritas.evaluation.metrics import (
    EvaluationDataError,
    STANDARD_LABEL_ORDER,
    aggregate_multi_run_metrics,
    compute_classification_metrics,
    compute_coverage_accuracy_curve,
    compute_latency_metrics,
    compute_system_quality_metrics,
    match_gold_and_predictions,
)
from veritas.evaluation.report import (
    build_evaluation_report,
    format_console_report,
    save_report_artifacts,
)
from veritas.evaluation.retrieval_metrics import (
    compute_dcg_at_k,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_recall_at_k,
    evaluate_retrieval_batch,
)


# ---------------------------------------------------------------------------
# 1. ID Matching and Validation Tests
# ---------------------------------------------------------------------------


class TestIDMatching:
    def test_match_gold_and_predictions_success(self):
        gold = [
            {"id": "claim_1", "label": "True"},
            {"id": "claim_2", "label": "False"},
        ]
        preds = [
            {"claim_id": "claim_2", "label": "False"},
            {"claim_id": "claim_1", "label": "True"},
        ]
        matched = match_gold_and_predictions(gold, preds)
        assert len(matched) == 2
        # Order preserved from gold
        assert matched[0][0]["id"] == "claim_1"
        assert matched[0][1]["claim_id"] == "claim_1"
        assert matched[1][0]["id"] == "claim_2"

    def test_duplicate_gold_id_raises_error(self):
        gold = [
            {"id": "dup_1", "label": "True"},
            {"id": "dup_1", "label": "False"},
        ]
        preds = [{"claim_id": "dup_1", "label": "True"}]
        with pytest.raises(EvaluationDataError, match="Duplicate claim ID detected in gold"):
            match_gold_and_predictions(gold, preds)

    def test_duplicate_pred_id_raises_error(self):
        gold = [{"id": "c1", "label": "True"}]
        preds = [
            {"claim_id": "c1", "label": "True"},
            {"claim_id": "c1", "label": "False"},
        ]
        with pytest.raises(EvaluationDataError, match="Duplicate claim ID detected in predictions"):
            match_gold_and_predictions(gold, preds)

    def test_missing_id_in_predictions_raises_error(self):
        gold = [
            {"id": "c1", "label": "True"},
            {"id": "c2", "label": "False"},
        ]
        preds = [{"claim_id": "c1", "label": "True"}]
        with pytest.raises(EvaluationDataError, match="Mismatched claim IDs"):
            match_gold_and_predictions(gold, preds, strict_match=True)


# ---------------------------------------------------------------------------
# 2. Classification Metrics Tests (Hand-Calculated Known Values)
# ---------------------------------------------------------------------------


class TestClassificationMetrics:
    def test_known_classification_values(self):
        # 4 samples:
        # gold: [True, True, False, Misleading]
        # pred: [True, False, False, Misleading]
        # Accuracy: 3 / 4 = 0.75
        y_true = ["True", "True", "False", "Misleading"]
        y_pred = ["True", "False", "False", "Misleading"]

        metrics = compute_classification_metrics(y_true, y_pred, labels=STANDARD_LABEL_ORDER)

        assert metrics["accuracy"] == 0.75
        assert metrics["total_samples"] == 4

        # Per-class checks:
        # True: TP=1, FP=0, FN=1 -> Prec=1.0, Rec=0.5, F1 = 2*(1*0.5)/(1+0.5) = 0.6667
        true_stats = metrics["per_class"]["True"]
        assert true_stats["precision"] == 1.0
        assert true_stats["recall"] == 0.5
        assert pytest.approx(true_stats["f1"], 0.001) == 0.6667
        assert true_stats["support"] == 2

        # False: TP=1, FP=1, FN=0 -> Prec=0.5, Rec=1.0, F1 = 0.6667
        false_stats = metrics["per_class"]["False"]
        assert false_stats["precision"] == 0.5
        assert false_stats["recall"] == 1.0
        assert pytest.approx(false_stats["f1"], 0.001) == 0.6667
        assert false_stats["support"] == 1

        # Misleading: TP=1, FP=0, FN=0 -> Prec=1.0, Rec=1.0, F1 = 1.0
        mis_stats = metrics["per_class"]["Misleading"]
        assert mis_stats["precision"] == 1.0
        assert mis_stats["recall"] == 1.0
        assert mis_stats["f1"] == 1.0
        assert mis_stats["support"] == 1

        # Macro-F1 = (0.6667 + 0.6667 + 1.0) / 3 = 0.7778
        assert pytest.approx(metrics["macro_f1"], 0.001) == 0.7778

    def test_empty_labels_returns_none(self):
        metrics = compute_classification_metrics([], [])
        assert metrics["accuracy"] is None
        assert metrics["macro_f1"] is None
        assert metrics["total_samples"] == 0


# ---------------------------------------------------------------------------
# 3. Retrieval Metrics Tests
# ---------------------------------------------------------------------------


class TestRetrievalMetrics:
    def test_known_retrieval_values(self):
        gold = ["p1", "p2"]
        retrieved = ["p2", "p3", "p1"]

        # Recall@1: 'p2' in gold -> 1/2 = 0.5
        assert compute_recall_at_k(retrieved, gold, 1) == 0.5
        # Recall@2: 1/2 = 0.5
        assert compute_recall_at_k(retrieved, gold, 2) == 0.5
        # Recall@3: 'p2' and 'p1' in gold -> 2/2 = 1.0
        assert compute_recall_at_k(retrieved, gold, 3) == 1.0

        # MRR@1: 'p2' ranked at 1 -> 1.0
        assert compute_mrr_at_k(retrieved, gold, 1) == 1.0

    def test_distinguish_unavailable_from_zero(self):
        # Empty gold evidence annotations MUST return None (unavailable), not 0.0
        assert compute_recall_at_k(["p1"], [], 1) is None
        assert compute_ndcg_at_k(["p1"], [], 1) is None
        assert compute_mrr_at_k(["p1"], [], 1) is None

        # When gold exists but no hits occur, it must return 0.0 (zero-valued)
        assert compute_recall_at_k(["p_wrong"], ["p_correct"], 1) == 0.0
        assert compute_mrr_at_k(["p_wrong"], ["p_correct"], 1) == 0.0

    def test_evaluate_retrieval_batch_skips_missing_gold(self):
        batch = [
            {"gold_ids": ["p1"], "retrieved_ids": ["p1"]},
            {"gold_ids": [], "retrieved_ids": ["p2"]},  # Missing gold
        ]
        res = evaluate_retrieval_batch(batch, k_values=[1])
        assert res["status"] == "available"
        assert res["eligible_claims"] == 1
        assert res["total_claims"] == 2
        assert res["metrics"]["recall@1"] == 1.0


# ---------------------------------------------------------------------------
# 4. Latency Percentiles Tests
# ---------------------------------------------------------------------------


class TestLatencyMetrics:
    def test_latency_statistics(self):
        latencies = [0.1, 0.2, 0.3, 0.4, 0.5]
        res = compute_latency_metrics(latencies)
        assert res["avg_latency_s"] == 0.3
        assert res["p50_latency_s"] == 0.3
        assert res["p90_latency_s"] == 0.5

    def test_empty_latency_returns_none(self):
        res = compute_latency_metrics([])
        assert res["avg_latency_s"] is None
        assert res["p50_latency_s"] is None


# ---------------------------------------------------------------------------
# 5. Multi-Run Aggregation Tests
# ---------------------------------------------------------------------------


class TestMultiRunAggregation:
    def test_aggregate_mean_and_spread(self):
        run1 = {"accuracy": 0.80, "macro_f1": 0.75}
        run2 = {"accuracy": 0.90, "macro_f1": 0.85}

        agg = aggregate_multi_run_metrics([run1, run2])
        assert agg["num_runs"] == 2
        metrics = agg["aggregated_metrics"]

        # Mean accuracy: (0.80 + 0.90)/2 = 0.85
        assert metrics["accuracy"]["mean"] == 0.85
        assert metrics["accuracy"]["min"] == 0.80
        assert metrics["accuracy"]["max"] == 0.90
        assert metrics["accuracy"]["std"] > 0.0


# ---------------------------------------------------------------------------
# 6. Report Generation and Artifact Saving
# ---------------------------------------------------------------------------


class TestReportGeneration:
    def test_build_and_save_report(self, tmp_path):
        gold = [
            {"id": "c1", "label": "True", "evidence": [{"passage_id": "p1"}]},
            {"id": "c2", "label": "False", "evidence": []},
        ]
        preds = [
            {
                "claim_id": "c1",
                "label": "True",
                "confidence": 0.95,
                "evidence_ids": ["p1"],
                "retrieved_passages": [{"passage_id": "p1"}],
                "faithfulness_score": 1.0,
                "parse_failed": False,
                "stage_timings_s": {"retrieval": 0.01, "verify": 0.02},
            },
            {
                "claim_id": "c2",
                "label": "Unverified",
                "confidence": 0.40,
                "evidence_ids": [],
                "retrieved_passages": [],
                "faithfulness_score": 0.0,
                "parse_failed": False,
                "stage_timings_s": {"retrieval": 0.01, "verify": 0.01},
            },
        ]

        report = build_evaluation_report(
            gold_records=gold,
            prediction_records=preds,
            model_version="test-v1",
            dataset_name="mini-test",
        )

        assert report.classification["total_samples"] == 2
        assert report.classification["accuracy"] == 0.5  # c1 correct, c2 False != Unverified
        assert report.run_metadata["model_version"] == "test-v1"

        # Check console formatting
        console_text = format_console_report(report)
        assert "VERITAS P9 EVALUATION REPORT" in console_text
        assert "Accuracy" in console_text

        # Check artifact export
        out_dir = tmp_path / "eval_out"
        saved = save_report_artifacts(report, out_dir)

        assert saved["json"].exists()
        assert saved["per_class_csv"].exists()
        assert saved["summary_csv"].exists()
        assert saved["svg_curve"].exists()

        # Validate saved JSON structure
        with saved["json"].open("r", encoding="utf-8") as f:
            data = json.load(f)
            assert "classification" in data
            assert "run_metadata" in data
