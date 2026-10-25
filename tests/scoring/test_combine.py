"""Tests for score combination module in VERITAS P4.

Validates:
  1. Combined score formula: combined = alpha*relevance + beta*credibility + gamma*temporal
  2. All component scores in [0.0, 1.0].
  3. Pure relevance ranking when beta=gamma=0.
  4. Ranking behavior changes when credibility or recency signals are active.
  5. Deterministic tie-breaking and top-N candidate truncation.
  6. Empty candidates handling.
"""

from __future__ import annotations

import pytest

from veritas.retrieval.schema import Passage
from veritas.scoring.combine import (
    CombinedScorer,
    CombinedScorerConfig,
    ScoreWeights,
    ScoredCandidate,
)
from veritas.scoring.credibility import CredibilityConfig, CredibilityScorer
from veritas.scoring.temporal import TemporalConfig, TemporalScorer


@pytest.fixture
def mock_candidates() -> list[Passage]:
    """Sample candidates with distinct relevance, credibility, and dates."""
    return [
        Passage(
            passage_id="p1",
            text="High relevance, low credibility (unverified blog), recent date",
            domain="tabloid.co.uk",
            date="2021-06-01",
        ),
        Passage(
            passage_id="p2",
            text="Moderate relevance, high credibility (reuters.com), very recent date",
            domain="reuters.com",
            date="2021-06-01",
        ),
        Passage(
            passage_id="p3",
            text="Moderate relevance, high credibility (apnews.com), very old date",
            domain="apnews.com",
            date="2010-01-01",
        ),
    ]


class TestScoreCombinationFormula:

    def test_combined_formula_exact_values(self):
        """Verifies combined = alpha*rel + beta*cred + gamma*temp with normalized weights."""
        cfg = CombinedScorerConfig(
            weights=ScoreWeights(alpha=0.5, beta=0.3, gamma=0.2, normalize_weights=True),
            temporal=TemporalConfig(default_tau=100.0),
        )
        scorer = CombinedScorer(config=cfg)

        # Candidate with known domain (reuters = 0.90) and exact same date (temp = 1.0)
        cand = [
            {"passage_id": "c1", "score": 0.80, "domain": "reuters.com", "date": "2021-06-01"}
        ]

        scored = scorer.score_candidates(cand, claim_date="2021-06-01")
        assert len(scored) == 1
        res = scored[0]

        assert res.relevance_score == 0.80
        assert res.credibility_score == 0.90
        assert res.temporal_score == 1.0

        expected_combined = 0.5 * 0.80 + 0.3 * 0.90 + 0.2 * 1.0
        assert round(res.combined_score, 4) == round(expected_combined, 4)
        assert 0.0 <= res.combined_score <= 1.0

    def test_all_component_scores_strictly_bounded(self, mock_candidates):
        """All component scores and combined score must be in [0.0, 1.0]."""
        scorer = CombinedScorer()
        results = scorer.score_candidates(mock_candidates, claim_date="2021-06-01")

        for r in results:
            assert 0.0 <= r.relevance_score <= 1.0
            assert 0.0 <= r.credibility_score <= 1.0
            assert 0.0 <= r.temporal_score <= 1.0
            assert 0.0 <= r.combined_score <= 1.0


class TestPureRelevanceMode:

    def test_pure_relevance_ranking_when_beta_gamma_zero(self):
        """When beta=gamma=0, combined score ranking is identical to relevance ranking."""
        candidates = [
            {"passage_id": "c_low", "score": 0.20, "domain": "reuters.com", "date": "2021-06-01"},
            {"passage_id": "c_high", "score": 0.95, "domain": "tabloid.co.uk", "date": "1990-01-01"},
            {"passage_id": "c_mid", "score": 0.60, "domain": "unknown.org", "date": None},
        ]

        cfg = CombinedScorerConfig(
            weights=ScoreWeights(alpha=1.0, beta=0.0, gamma=0.0, normalize_weights=True)
        )
        scorer = CombinedScorer(config=cfg)

        ranked = scorer.score_candidates(candidates, claim_date="2021-06-01")

        # Ranking must strictly follow relevance: c_high (0.95) > c_mid (0.60) > c_low (0.20)
        assert [r.passage_id for r in ranked] == ["c_high", "c_mid", "c_low"]
        assert ranked[0].combined_score == ranked[0].relevance_score == 0.95
        assert ranked[1].combined_score == ranked[1].relevance_score == 0.60
        assert ranked[2].combined_score == ranked[2].relevance_score == 0.20


class TestCredibilityAndTemporalRankingImpact:

    def test_credibility_promotes_trustworthy_candidate_over_tabloid(self):
        """When beta > 0, an established outlet can outrank a slightly higher-relevance tabloid."""
        candidates = [
            # High relevance but very low credibility tabloid (tier 4: 0.25)
            {"passage_id": "tabloid", "score": 0.70, "domain": "dailymail.co.uk", "date": "2021-06-01"},
            # Slightly lower relevance but tier 1 wire agency (reuters: 0.90)
            {"passage_id": "reuters", "score": 0.65, "domain": "reuters.com", "date": "2021-06-01"},
        ]

        # Weights: alpha=0.4, beta=0.6, gamma=0.0
        cfg = CombinedScorerConfig(
            weights=ScoreWeights(alpha=0.4, beta=0.6, gamma=0.0, normalize_weights=True)
        )
        scorer = CombinedScorer(config=cfg)
        ranked = scorer.score_candidates(candidates, claim_date="2021-06-01")

        # Tabloid: 0.4 * 0.70 + 0.6 * 0.25 = 0.28 + 0.15 = 0.43
        # Reuters: 0.4 * 0.65 + 0.6 * 0.90 = 0.26 + 0.54 = 0.80
        assert ranked[0].passage_id == "reuters"
        assert ranked[1].passage_id == "tabloid"

    def test_temporal_promotes_recent_candidate_over_stale_candidate(self):
        """When gamma > 0, recent evidence outranks stale evidence with equal relevance and domain."""
        candidates = [
            # 10 years old evidence
            {"passage_id": "stale", "score": 0.80, "domain": "cdc.gov", "date": "2011-06-01"},
            # Recent evidence (same day)
            {"passage_id": "fresh", "score": 0.80, "domain": "cdc.gov", "date": "2021-06-01"},
        ]

        cfg = CombinedScorerConfig(
            weights=ScoreWeights(alpha=0.5, beta=0.0, gamma=0.5, normalize_weights=True),
            temporal=TemporalConfig(default_tau=180.0),
        )
        scorer = CombinedScorer(config=cfg)
        ranked = scorer.score_candidates(candidates, claim_date="2021-06-01")

        assert ranked[0].passage_id == "fresh"
        assert ranked[1].passage_id == "stale"


class TestEdgeCasesAndConfig:

    def test_empty_candidates_returns_empty_list(self):
        scorer = CombinedScorer()
        assert scorer.score_candidates([]) == []

    def test_top_n_truncation(self, mock_candidates):
        scorer = CombinedScorer()
        results = scorer.score_candidates(mock_candidates, top_n=2)
        assert len(results) == 2
        assert [r.rank for r in results] == [1, 2]

    def test_load_from_yaml(self, tmp_path):
        yaml_content = """
scoring:
  weights:
    alpha: 0.7
    beta: 0.2
    gamma: 0.1
  temporal:
    default_tau: 90.0
    neutral_score: 0.5
"""
        p = tmp_path / "scoring.yaml"
        p.write_text(yaml_content)

        cfg = CombinedScorerConfig.from_yaml(p)
        assert cfg.weights.alpha == 0.7
        assert cfg.temporal.default_tau == 90.0
