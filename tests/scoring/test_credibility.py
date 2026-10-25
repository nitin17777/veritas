"""Tests for credibility scoring module in VERITAS P4.

Validates domain extraction, tier table lookup, fallback for unknown domains,
score ranges in [0.0, 1.0], random mode reproducibility, and disclaimer documentation.
"""

from __future__ import annotations

import pytest

from veritas.retrieval.schema import Passage
from veritas.scoring.credibility import (
    CredibilityConfig,
    CredibilityScorer,
    extract_domain,
)


class TestDomainExtraction:

    @pytest.mark.parametrize(
        "url_or_domain,expected",
        [
            ("https://www.reuters.com/world/us/article-123", "reuters.com"),
            ("http://apnews.com/hub/politics", "apnews.com"),
            ("news.bbc.co.uk", "news.bbc.co.uk"),
            ("HTTPS://WWW.NYTIMES.COM/interactive/2021", "nytimes.com"),
            ("http://example.org:8080/path?query=1#frag", "example.org"),
            ("cdc.gov", "cdc.gov"),
            ("", ""),
            (None, ""),
            ("   ", ""),
        ],
    )
    def test_extract_domain_normalization(self, url_or_domain, expected):
        assert extract_domain(url_or_domain) == expected


class TestCredibilityScoring:

    @pytest.fixture
    def scorer(self) -> CredibilityScorer:
        return CredibilityScorer()

    def test_known_tier_1_domain(self, scorer: CredibilityScorer):
        # Reuters is in tier 1 (0.90)
        score = scorer.score_domain("reuters.com")
        assert score == 0.90
        assert 0.0 <= score <= 1.0

    def test_known_tier_2_domain(self, scorer: CredibilityScorer):
        # BBC is in tier 2 (0.75)
        score = scorer.score_domain("bbc.com")
        assert score == 0.75

    def test_subdomain_matches_parent(self, scorer: CredibilityScorer):
        # health.cdc.gov should match cdc.gov in tier 1 (0.90)
        score = scorer.score_domain("health.cdc.gov")
        assert score == 0.90

    def test_unknown_domain_returns_default(self, scorer: CredibilityScorer):
        # Unknown domain returns configured fallback (0.50)
        score = scorer.score_domain("totally-unknown-news-site.org")
        assert score == 0.50

    def test_empty_domain_returns_default(self, scorer: CredibilityScorer):
        assert scorer.score_domain("") == 0.50
        assert scorer.score_domain(None) == 0.50

    def test_score_passage_object(self, scorer: CredibilityScorer):
        p = Passage(passage_id="p1", text="Sample text", url="https://apnews.com/article/99")
        score = scorer.score_passage(p)
        assert score == 0.90


class TestRandomCredibilityAblation:

    def test_random_mode_reproducible_with_same_seed(self):
        """Random mode must produce identical scores for the same domain and seed."""
        cfg1 = CredibilityConfig(random_mode=True, random_seed=42)
        scorer1 = CredibilityScorer(cfg1)

        cfg2 = CredibilityConfig(random_mode=True, random_seed=42)
        scorer2 = CredibilityScorer(cfg2)

        domains = ["reuters.com", "unknown.org", "fakeblog.com", "cdc.gov"]
        for d in domains:
            s1 = scorer1.score_domain(d)
            s2 = scorer2.score_domain(d)
            assert s1 == s2
            assert 0.0 <= s1 <= 1.0

    def test_random_mode_varies_with_different_seeds(self):
        """Different seeds must produce different pseudo-random scores."""
        scorer_seed1 = CredibilityScorer(CredibilityConfig(random_mode=True, random_seed=42))
        scorer_seed2 = CredibilityScorer(CredibilityConfig(random_mode=True, random_seed=9999))

        scores1 = [scorer_seed1.score_domain(f"domain_{i}.com") for i in range(10)]
        scores2 = [scorer_seed2.score_domain(f"domain_{i}.com") for i in range(10)]

        assert scores1 != scores2


class TestCredibilityConfigAndDisclaimer:

    def test_custom_unknown_score(self):
        cfg = CredibilityConfig(unknown_domain_score=0.33)
        scorer = CredibilityScorer(cfg)
        assert scorer.score_domain("random.net") == 0.33

    def test_invalid_score_bounds_raises(self):
        with pytest.raises(ValueError):
            CredibilityConfig(unknown_domain_score=1.5)
        with pytest.raises(ValueError):
            CredibilityConfig(tier_table={"bad.com": -0.1})

    def test_disclaimer_documented(self):
        """Credibility module must explicitly document provisional status and limitations."""
        import veritas.scoring.credibility as c_mod
        doc = c_mod.__doc__
        assert "PROVISIONAL" in doc
        assert "NOT represent objective ground truth" in doc
        assert "limitations" in doc.lower()
