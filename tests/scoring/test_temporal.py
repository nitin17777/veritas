"""Tests for temporal scoring module in VERITAS P4.

Validates date parsing, exponential decay: exp(-delta_days / tau),
score bounds [0.0, 1.0], missing date neutral fallback with logging,
and claim-type specific tau parameters.
"""

from __future__ import annotations

import datetime
import logging
import math
import pytest

from veritas.retrieval.schema import Passage
from veritas.scoring.temporal import (
    TemporalConfig,
    TemporalScorer,
    parse_iso_date,
)


class TestDateParsing:

    def test_parse_iso_date_strings(self):
        assert parse_iso_date("2021-06-15") == datetime.date(2021, 6, 15)
        assert parse_iso_date("2020-01-01T14:30:00Z") == datetime.date(2020, 1, 1)
        assert parse_iso_date("2019") == datetime.date(2019, 1, 1)
        assert parse_iso_date(datetime.date(2022, 5, 1)) == datetime.date(2022, 5, 1)
        assert parse_iso_date(datetime.datetime(2022, 5, 1, 12, 0)) == datetime.date(2022, 5, 1)

    def test_parse_invalid_or_empty_dates_returns_none(self):
        assert parse_iso_date(None) is None
        assert parse_iso_date("") is None
        assert parse_iso_date("not-a-date") is None
        assert parse_iso_date("2021-99-99") is None


class TestTemporalScorerDecay:

    @pytest.fixture
    def scorer(self) -> TemporalScorer:
        return TemporalScorer(TemporalConfig(default_tau=100.0, neutral_score=0.5))

    def test_zero_day_difference_gives_score_one(self, scorer: TemporalScorer):
        """When evidence and claim are published on the same day, score must be exactly 1.0."""
        score = scorer.score("2021-05-01", "2021-05-01")
        assert score == 1.0

    def test_exponential_decay_with_increasing_delta(self, scorer: TemporalScorer):
        """Larger day difference must monotonically decrease temporal score."""
        claim_date = "2021-05-01"
        s0 = scorer.score(claim_date, "2021-05-01")   # delta = 0
        s30 = scorer.score(claim_date, "2021-04-01")  # delta = 30 days
        s100 = scorer.score(claim_date, "2021-01-21") # delta = 100 days (~tau)
        s365 = scorer.score(claim_date, "2020-05-01") # delta = 365 days

        assert 1.0 == s0 > s30 > s100 > s365 > 0.0

        # At delta = tau = 100, score should be exp(-1) ~ 0.367879
        expected_at_tau = math.exp(-1.0)
        assert math.isclose(s100, expected_at_tau, abs_tol=1e-3)

    def test_symmetry_past_and_future_evidence(self, scorer: TemporalScorer):
        """Delta is absolute: evidence published 30 days before or after has equal score."""
        claim_date = "2021-05-15"
        before = scorer.score(claim_date, "2021-04-15")  # -30 days
        after = scorer.score(claim_date, "2021-06-14")   # +30 days
        assert before == after

    def test_score_bounds_always_in_zero_to_one(self, scorer: TemporalScorer):
        """Scores must strictly be within [0.0, 1.0]."""
        # Very old evidence (e.g. 100 years ago)
        score_ancient = scorer.score("2021-01-01", "1900-01-01")
        assert 0.0 <= score_ancient <= 1.0

    def test_score_passage_method(self, scorer: TemporalScorer):
        p = Passage("p1", "text", date="2021-05-01")
        score = scorer.score_passage(claim_date="2021-05-01", passage=p)
        assert score == 1.0


class TestMissingDatesAndLogging:

    def test_missing_claim_date_falls_back_to_neutral_score(self, caplog):
        scorer = TemporalScorer(TemporalConfig(neutral_score=0.45))
        with caplog.at_level(logging.DEBUG):
            score = scorer.score(None, "2021-05-01")
        assert score == 0.45
        assert "Missing or unparseable date" in caplog.text

    def test_missing_evidence_date_falls_back_to_neutral_score(self, caplog):
        scorer = TemporalScorer(TemporalConfig(neutral_score=0.50))
        with caplog.at_level(logging.DEBUG):
            score = scorer.score("2021-05-01", None)
        assert score == 0.50
        assert "Missing or unparseable date" in caplog.text

    def test_both_dates_missing(self, caplog):
        scorer = TemporalScorer(TemporalConfig(neutral_score=0.50))
        with caplog.at_level(logging.DEBUG):
            score = scorer.score(None, None)
        assert score == 0.50


class TestClaimTypeSpecificTau:

    def test_breaking_news_decays_faster_than_science(self):
        """Claim type 'breaking' (tau=14) must decay much faster than 'science' (tau=730)."""
        cfg = TemporalConfig(
            tau_by_claim_type={"breaking": 14.0, "science": 730.0}
        )
        scorer = TemporalScorer(cfg)

        claim_date = "2021-06-01"
        evidence_date = "2021-05-01"  # 31 days difference

        score_breaking = scorer.score(claim_date, evidence_date, claim_type="breaking")
        score_science = scorer.score(claim_date, evidence_date, claim_type="science")

        # For breaking news, 31 days is > 2 half-lives (exp(-31/14) ~ 0.109)
        # For science, 31 days is a tiny fraction of tau (exp(-31/730) ~ 0.958)
        assert score_breaking < score_science
        assert score_breaking < 0.20
        assert score_science > 0.90

    def test_unknown_claim_type_uses_default_tau(self):
        cfg = TemporalConfig(default_tau=100.0, tau_by_claim_type={"breaking": 14.0})
        scorer = TemporalScorer(cfg)

        s_unknown = scorer.score("2021-06-01", "2021-05-01", claim_type="unknown_category")
        s_default = scorer.score("2021-06-01", "2021-05-01", claim_type=None)
        assert s_unknown == s_default

    def test_invalid_tau_raises_value_error(self):
        with pytest.raises(ValueError):
            TemporalConfig(default_tau=-10.0)
        with pytest.raises(ValueError):
            TemporalConfig(tau_by_claim_type={"bad": 0.0})
