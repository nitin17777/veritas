"""Tests for validation-only grid search in VERITAS P4.

Validates:
  1. Grid search optimizes weights on validation fixtures.
  2. Strict method guard prohibits test-set tuning.
  3. Metric evaluation (MRR, Recall) works correctly.
  4. Simplex grid generation covers (1.0, 0.0, 0.0) baseline.
"""

from __future__ import annotations

import pytest

from veritas.retrieval.schema import Passage
from veritas.scoring.grid_search import (
    ScoringGridSearch,
    compute_mrr_at_k,
    compute_recall_at_k,
    generate_simplex_grid,
)


@pytest.fixture
def synthetic_validation_set() -> list[dict]:
    """Validation dataset where gold evidence has high credibility and fresh dates."""
    return [
        {
            "claim": "COVID-19 vaccination reduces hospitalization rates.",
            "claim_date": "2021-06-01",
            "claim_type": "health",
            "gold_passage_ids": ["gold_1"],
            "candidates": [
                Passage("decoy_1", "Low credibility blog.", date="2020-01-01", domain="unverified-blog.org"),
                Passage("gold_1", "CDC report confirms effectiveness.", date="2021-05-20", domain="cdc.gov"),
                Passage("decoy_2", "Ancient historical snippet.", date="1990-01-01", domain="tabloid.co.uk"),
            ],
        },
        {
            "claim": "Perseverance rover lands on Mars.",
            "claim_date": "2021-02-18",
            "claim_type": "science",
            "gold_passage_ids": ["gold_2"],
            "candidates": [
                Passage("decoy_3", "General astronomy text.", date="2010-01-01", domain="random.net"),
                Passage("gold_2", "Reuters confirms rover landing.", date="2021-02-18", domain="reuters.com"),
            ],
        },
    ]


class TestGridSearchSimplex:

    def test_simplex_grid_generation(self):
        grid = generate_simplex_grid(step=0.2)
        assert len(grid) > 0
        assert (1.0, 0.0, 0.0) in grid

        # Sum of every triple must equal 1.0
        for a, b, g in grid:
            assert a >= 0.0 and b >= 0.0 and g >= 0.0
            assert round(a + b + g, 4) == 1.0


class TestMetricComputations:

    def test_compute_mrr_at_k(self):
        ranked = ["p1", "p2", "p3", "p4"]
        assert compute_mrr_at_k(ranked, {"p1"}, k=10) == 1.0
        assert compute_mrr_at_k(ranked, {"p2"}, k=10) == 0.5
        assert compute_mrr_at_k(ranked, {"p3"}, k=10) == 1.0 / 3.0
        assert compute_mrr_at_k(ranked, {"p99"}, k=10) == 0.0
        assert compute_mrr_at_k(ranked, set(), k=10) == 0.0

    def test_compute_recall_at_k(self):
        ranked = ["p1", "p2", "p3", "p4"]
        assert compute_recall_at_k(ranked, {"p2"}, k=2) == 1.0
        assert compute_recall_at_k(ranked, {"p3"}, k=2) == 0.0
        assert compute_recall_at_k(ranked, {"p3"}, k=3) == 1.0


class TestScoringGridSearchExecution:

    def test_grid_search_runs_and_identifies_best_weights(self, synthetic_validation_set):
        searcher = ScoringGridSearch(weight_step=0.2, tau_candidates=[30.0, 180.0])
        result = searcher.run(synthetic_validation_set, split_name="validation")

        assert result.num_validation_queries == 2
        assert result.split_name == "validation"
        assert result.best_metric_score >= result.baseline_pure_relevance_score
        assert len(result.all_trials) > 0

        d = result.to_dict()
        assert "best_weights" in d
        assert "best_tau" in d
        assert d["best_weights"]["alpha"] >= 0.0

    def test_strict_methodological_guard_prohibits_test_split(self, synthetic_validation_set):
        """Supplying split_name='test' must raise ValueError."""
        searcher = ScoringGridSearch()

        with pytest.raises(ValueError, match="TEST-SET TUNING FORBIDDEN"):
            searcher.run(synthetic_validation_set, split_name="test")

        with pytest.raises(ValueError, match="TEST-SET TUNING FORBIDDEN"):
            searcher.run(synthetic_validation_set, split_name="test_split")

        with pytest.raises(ValueError, match="Invalid split_name"):
            searcher.run(synthetic_validation_set, split_name="production_eval")

    def test_empty_validation_data_raises_value_error(self):
        searcher = ScoringGridSearch()
        with pytest.raises(ValueError, match="validation_data is empty"):
            searcher.run([], split_name="val")
