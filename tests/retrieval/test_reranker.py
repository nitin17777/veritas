"""Unit and integration tests for the VERITAS P4 relevance reranker.

IMPORTANT DISCLAIMER ON ACCURACY & PROBABILITIES:
-------------------------------------------------
Mocked unit tests passing here confirm that the reranking logic, sorting,
normalization, tie-breaking, schema conformance, and configuration recording
work as designed. They do NOT validate:
  1. That the reranker surfaces objectively accurate real-world facts.
  2. That normalized scores represent calibrated posterior probabilities.
Normalized scores represent monotonic relative ranking signals on [0.0, 1.0]
and must never be interpreted as calibrated probabilities.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from veritas.retrieval.fusion import FusedSearchResult
from veritas.retrieval.reranker import (
    CrossEncoderReranker,
    MockCrossEncoder,
    RerankResult,
    RerankerConfig,
    normalize_scores,
    rerank_candidates,
)
from veritas.retrieval.schema import Passage, SearchResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_passages() -> list[Passage]:
    return [
        Passage("p1", "The Earth completes an orbit around the Sun in approximately 365.25 days."),
        Passage("p2", "The Moon exhibits distinct phases as viewed from Earth over a 29.5-day cycle."),
        Passage("p3", "Mars has two small moons named Phobos and Deimos."),
        Passage("p4", "Jupiter is the largest planet in the Solar System by mass and volume."),
    ]


@pytest.fixture
def earth_claim() -> str:
    return "The Earth orbits the Sun once every year."


# ---------------------------------------------------------------------------
# Test Score Normalization Functions
# ---------------------------------------------------------------------------

class TestScoreNormalization:

    def test_sigmoid_monotonic_and_bounded(self):
        """Sigmoid must strictly map logits to (0.0, 1.0) monotonically."""
        logits = [-10.0, -2.0, 0.0, 2.0, 10.0]
        norm = normalize_scores(logits, method="sigmoid")

        assert len(norm) == len(logits)
        assert np.all(norm >= 0.0) and np.all(norm <= 1.0)
        # Monotonically increasing
        assert np.all(np.diff(norm) > 0)
        # Logit 0.0 must map to exactly 0.5
        assert np.isclose(norm[2], 0.5, atol=1e-5)

    def test_sigmoid_extreme_values_safe(self):
        """Extreme values must not raise FloatingPointError or overflow."""
        logits = [-10000.0, 10000.0]
        norm = normalize_scores(logits, method="sigmoid")
        assert np.isclose(norm[0], 0.0, atol=1e-6)
        assert np.isclose(norm[1], 1.0, atol=1e-6)

    def test_minmax_rescales_to_0_1(self):
        """MinMax must rescale minimum to 0.0 and maximum to 1.0."""
        logits = [2.0, 4.0, 10.0]
        norm = normalize_scores(logits, method="minmax")
        assert np.isclose(norm[0], 0.0, atol=1e-6)
        assert np.isclose(norm[-1], 1.0, atol=1e-6)
        assert 0.0 <= norm[1] <= 1.0

    def test_minmax_zero_variance_safe(self):
        """When all candidate scores are identical, minmax must return 1.0 without dividing by zero."""
        logits = [3.5, 3.5, 3.5]
        norm = normalize_scores(logits, method="minmax")
        assert np.all(norm == 1.0)

    def test_softmax_sums_to_one(self):
        """Softmax must map logits such that their sum equals 1.0."""
        logits = [1.0, 2.0, 3.0]
        norm = normalize_scores(logits, method="softmax")
        assert np.isclose(float(np.sum(norm)), 1.0, atol=1e-5)
        assert np.all(norm > 0.0)

    def test_none_normalization_clips_to_0_1(self):
        """None method clips raw scores to [0.0, 1.0]."""
        raw = [-0.5, 0.4, 1.5]
        norm = normalize_scores(raw, method="none")
        assert np.allclose(norm, [0.0, 0.4, 1.0])

    def test_empty_scores_returns_empty_array(self):
        norm = normalize_scores([], method="sigmoid")
        assert norm.size == 0

    def test_invalid_method_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_scores([1.0, 2.0], method="magic_unsupported")


# ---------------------------------------------------------------------------
# Test MockCrossEncoder
# ---------------------------------------------------------------------------

class TestMockCrossEncoder:

    def test_mock_canned_scores(self):
        mock = MockCrossEncoder(canned_scores=[1.5, -2.0, 3.0])
        pairs = [("q", "d1"), ("q", "d2"), ("q", "d3")]
        preds = mock.predict(pairs)
        assert np.allclose(preds, [1.5, -2.0, 3.0])
        assert len(mock.history) == 1
        assert mock.history[0]["num_pairs"] == 3

    def test_mock_score_map(self):
        mock = MockCrossEncoder(
            score_map={
                ("claim", "relevant text"): 5.0,
                ("claim", "irrelevant text"): -5.0,
            }
        )
        preds = mock.predict([("claim", "relevant text"), ("claim", "irrelevant text")])
        assert preds[0] == 5.0
        assert preds[1] == -5.0

    def test_mock_deterministic_hash_fallback(self):
        mock = MockCrossEncoder()
        p1 = [("claim", "passage 1")]
        preds1 = mock.predict(p1)
        preds2 = mock.predict(p1)
        assert preds1[0] == preds2[0]


# ---------------------------------------------------------------------------
# Test CrossEncoderReranker Core Functionality
# ---------------------------------------------------------------------------

class TestCrossEncoderReranker:

    def test_reranks_passages_descending_by_score(self, sample_passages, earth_claim):
        """Candidates must be ordered by relevance score in descending order."""
        # Assign highest score to p1 (orbital period), second to p2 (moon)
        scores = [4.5, 1.0, -1.0, -3.0]
        mock = MockCrossEncoder(canned_scores=scores)
        cfg = RerankerConfig(normalization_method="sigmoid")
        reranker = CrossEncoderReranker(config=cfg, model=mock)

        results = reranker.rerank(earth_claim, sample_passages)

        assert len(results) == 4
        # Ranks must be 1-indexed and sorted descending by score
        assert [r.rank for r in results] == [1, 2, 3, 4]
        assert [r.passage_id for r in results] == ["p1", "p2", "p3", "p4"]
        assert results[0].score > results[1].score > results[2].score > results[3].score
        assert all(0.0 <= r.score <= 1.0 for r in results)

    def test_raw_score_and_provenance_preserved(self, sample_passages, earth_claim):
        """Raw model scores and source passage objects must be preserved in RerankResult."""
        mock = MockCrossEncoder(canned_scores=[2.5, 0.0, -1.0, -2.0])
        reranker = CrossEncoderReranker(model=mock)

        results = reranker.rerank(earth_claim, sample_passages)

        top = results[0]
        assert top.passage_id == "p1"
        assert top.raw_score == 2.5
        assert top.passage is not None
        assert top.passage.passage_id == "p1"
        assert top.original_rank == 1

        d = top.to_dict()
        assert d["passage_id"] == "p1"
        assert "score" in d
        assert "raw_score" in d
        assert d["rank"] == 1

    def test_top_n_truncation(self, sample_passages, earth_claim):
        """Specifying top_n must clip output to the top N candidates with updated ranks."""
        mock = MockCrossEncoder(canned_scores=[4.0, 3.0, 2.0, 1.0])
        cfg = RerankerConfig(top_n=2)
        reranker = CrossEncoderReranker(config=cfg, model=mock)

        results = reranker.rerank(earth_claim, sample_passages)
        assert len(results) == 2
        assert [r.rank for r in results] == [1, 2]
        assert [r.passage_id for r in results] == ["p1", "p2"]

        # Call-time top_n override
        results_override = reranker.rerank(earth_claim, sample_passages, top_n=3)
        assert len(results_override) == 3
        assert [r.rank for r in results_override] == [1, 2, 3]

    def test_top_n_larger_than_candidate_count(self, sample_passages, earth_claim):
        mock = MockCrossEncoder(canned_scores=[1.0, 2.0, 3.0, 4.0])
        reranker = CrossEncoderReranker(model=mock)
        results = reranker.rerank(earth_claim, sample_passages, top_n=100)
        assert len(results) == 4

    def test_top_n_zero_or_negative_returns_empty(self, sample_passages, earth_claim):
        mock = MockCrossEncoder()
        reranker = CrossEncoderReranker(model=mock)
        assert reranker.rerank(earth_claim, sample_passages, top_n=0) == []


# ---------------------------------------------------------------------------
# Test Edge Cases: Empty Candidates & Ties
# ---------------------------------------------------------------------------

class TestEdgeCasesAndTies:

    def test_empty_candidates_returns_empty_without_calling_model(self, earth_claim):
        """Empty candidate input must return [] immediately without invoking model."""
        mock = MockCrossEncoder()
        reranker = CrossEncoderReranker(model=mock)

        results = reranker.rerank(earth_claim, [])
        assert results == []
        assert len(mock.history) == 0

    def test_empty_claim_returns_candidates_with_zero_scores(self, sample_passages):
        """Empty claim should safely return candidates with score=0.0 without model error."""
        mock = MockCrossEncoder()
        reranker = CrossEncoderReranker(model=mock)

        results = reranker.rerank("   ", sample_passages)
        assert len(results) == len(sample_passages)
        assert all(r.score == 0.0 for r in results)
        assert len(mock.history) == 0

    def test_ties_handled_with_stable_original_order(self, sample_passages, earth_claim):
        """When scores are tied, tie_breaker='original' preserves input candidate order."""
        # All passages receive identical score 2.0
        mock = MockCrossEncoder(canned_scores=[2.0, 2.0, 2.0, 2.0])
        cfg = RerankerConfig(tie_breaker="original")
        reranker = CrossEncoderReranker(config=cfg, model=mock)

        results = reranker.rerank(earth_claim, sample_passages)

        assert [r.passage_id for r in results] == ["p1", "p2", "p3", "p4"]
        assert [r.rank for r in results] == [1, 2, 3, 4]
        assert all(r.score == results[0].score for r in results)

    def test_ties_handled_with_passage_id_order(self, earth_claim):
        """When scores are tied, tie_breaker='passage_id' sorts ascending by passage ID."""
        passages = [
            Passage("gamma", "text gamma"),
            Passage("alpha", "text alpha"),
            Passage("beta", "text beta"),
        ]
        mock = MockCrossEncoder(canned_scores=[1.0, 1.0, 1.0])
        cfg = RerankerConfig(tie_breaker="passage_id")
        reranker = CrossEncoderReranker(config=cfg, model=mock)

        results = reranker.rerank(earth_claim, passages)

        assert [r.passage_id for r in results] == ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# Test Input Polymorphism
# ---------------------------------------------------------------------------

class TestInputPolymorphism:

    def test_supports_search_result_objects(self, sample_passages, earth_claim):
        """Supports SearchResult inputs from BM25 or Dense search."""
        search_results = [
            SearchResult(passage_id=p.passage_id, score=0.5, rank=i + 1, passage=p)
            for i, p in enumerate(sample_passages)
        ]
        mock = MockCrossEncoder(canned_scores=[3.0, 2.0, 1.0, 0.0])
        reranker = CrossEncoderReranker(model=mock)

        results = reranker.rerank(earth_claim, search_results)
        assert len(results) == 4
        assert results[0].passage_id == "p1"
        assert results[0].original_rank == 1

    def test_supports_fused_search_result_objects(self, sample_passages, earth_claim):
        """Supports FusedSearchResult inputs from hybrid retrieval."""
        fused = [
            FusedSearchResult(
                passage_id=p.passage_id,
                rank=i + 1,
                score=0.03,
                sources=["bm25", "dense"],
                passage=p,
            )
            for i, p in enumerate(sample_passages)
        ]
        mock = MockCrossEncoder(canned_scores=[5.0, 2.0, 1.0, -1.0])
        reranker = CrossEncoderReranker(model=mock)

        results = reranker.rerank(earth_claim, fused)
        assert len(results) == 4
        assert results[0].passage_id == "p1"

    def test_supports_dictionary_inputs(self, earth_claim):
        """Supports candidate dictionaries with passage_id and text."""
        dicts = [
            {"passage_id": "d1", "text": "Passage one text.", "rank": 1},
            {"id": "d2", "text": "Passage two text.", "rank": 2},
        ]
        mock = MockCrossEncoder(canned_scores=[2.0, 4.0])
        reranker = CrossEncoderReranker(model=mock)

        results = reranker.rerank(earth_claim, dicts)
        assert len(results) == 2
        # d2 had higher score (4.0 vs 2.0)
        assert results[0].passage_id == "d2"
        assert results[1].passage_id == "d1"


# ---------------------------------------------------------------------------
# Test Configuration & Metadata Recording
# ---------------------------------------------------------------------------

class TestRerankerConfigAndMetadata:

    def test_metadata_records_model_name_and_version(self):
        """Run metadata must record model name, version, normalization, device."""
        cfg = RerankerConfig(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            model_version="1.0.0",
            normalization_method="sigmoid",
            top_n=5,
        )
        reranker = CrossEncoderReranker(config=cfg, model=MockCrossEncoder())
        meta = reranker.get_metadata()

        assert meta["component"] == "P4_relevance_reranker"
        assert meta["model_name"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert meta["model_version"] == "1.0.0"
        assert meta["normalization_method"] == "sigmoid"
        assert meta["top_n"] == 5

    def test_config_to_dict_and_from_dict_roundtrip(self):
        cfg = RerankerConfig(
            model_name="test-model",
            model_version="2.1.0",
            batch_size=16,
            max_length=256,
            top_n=8,
            normalization_method="minmax",
            tie_breaker="passage_id",
        )
        d = cfg.to_dict()
        cfg2 = RerankerConfig.from_dict(d)
        assert cfg2.model_name == "test-model"
        assert cfg2.model_version == "2.1.0"
        assert cfg2.batch_size == 16
        assert cfg2.max_length == 256
        assert cfg2.top_n == 8
        assert cfg2.normalization_method == "minmax"
        assert cfg2.tie_breaker == "passage_id"

    def test_config_from_yaml(self, tmp_path):
        yaml_content = """
reranker:
  model_name: "custom-cross-encoder"
  model_version: "3.0.0"
  batch_size: 64
  top_n: 15
  normalization_method: "sigmoid"
"""
        cfg_file = tmp_path / "reranker.yaml"
        cfg_file.write_text(yaml_content)

        cfg = RerankerConfig.from_yaml(cfg_file)
        assert cfg.model_name == "custom-cross-encoder"
        assert cfg.model_version == "3.0.0"
        assert cfg.batch_size == 64
        assert cfg.top_n == 15

    def test_convenience_function_rerank_candidates(self, sample_passages, earth_claim):
        mock = MockCrossEncoder(canned_scores=[3.0, 2.0, 1.0, 0.0])
        results = rerank_candidates(
            claim=earth_claim,
            candidates=sample_passages,
            top_n=2,
            model=mock,
        )
        assert len(results) == 2
        assert results[0].passage_id == "p1"
        assert isinstance(results[0], RerankResult)


# ---------------------------------------------------------------------------
# Test Zero Weight Downloads & Calibration Disclaimer
# ---------------------------------------------------------------------------

class TestZeroDownloadAndCalibrationDisclaimer:

    def test_unit_tests_run_offline_with_zero_downloads(self, sample_passages, earth_claim):
        """All unit tests must use MockCrossEncoder and execute purely in-memory."""
        mock = MockCrossEncoder()
        reranker = CrossEncoderReranker(model=mock)
        results = reranker.rerank(earth_claim, sample_passages)
        assert len(results) == 4
        assert len(mock.history) == 1

    def test_calibration_disclaimer_documented(self):
        """The module docstring must explicitly state that scores are not calibrated probabilities."""
        import veritas.retrieval.reranker as r_mod

        doc = r_mod.__doc__
        assert "NOT calibrated" in doc
        assert "probabilities" in doc
