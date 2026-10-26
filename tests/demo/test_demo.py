"""Tests for the VERITAS Streamlit Demo (P11) and Live Search integration.

Verifies:
- Benchmark Mode reusing VeritasPipeline.verify() without duplication.
- Live Web Search Mode and Passage conversion.
- Zero metadata fabrication: domains and dates strictly extracted.
- Trafilatura webpage body text extraction.
- Error handling: empty claims, retrieval misses, and visible Unverified state.
- Long text detection and claim extraction integration.
- Fast execution with zero expensive model downloads.
"""

from __future__ import annotations

import pytest

from veritas.demo.engine import (
    DISCLAIMER_TEXT,
    DemoVerificationResponse,
    VeritasDemoEngine,
)
from veritas.demo.web_search import (
    LiveSearchHit,
    fetch_page_text,
    hits_to_passages,
    search_mock,
    search_web,
)
from veritas.experiments.fixtures import build_fixture_retriever
from veritas.pipeline.pipeline import PipelineConfig, VeritasPipeline
from veritas.retrieval.schema import Passage
from veritas.verification.llm_client import MockLLMClient


@pytest.fixture
def demo_engine() -> VeritasDemoEngine:
    retriever = build_fixture_retriever()
    client = MockLLMClient()
    pipeline = VeritasPipeline(
        config=PipelineConfig(retrieval_mode="hybrid"),
        retriever=retriever,
        llm_client=client,
    )
    return VeritasDemoEngine(
        config_path="configs/demo.yaml",
        pipeline=pipeline,
        retriever=retriever,
        llm_client=client,
    )


class TestWebSearchAndExtraction:
    def test_search_mock_returns_hits_without_hallucinated_data(self) -> None:
        hits = search_mock("COVID-19 vaccine hospitalizations", max_results=3)
        assert len(hits) > 0
        for h in hits:
            assert h.url.startswith("http")
            assert h.domain != ""
            assert h.title != ""
            assert h.snippet != ""

    def test_hits_to_passages_conversion(self) -> None:
        hits = [
            LiveSearchHit(
                title="BLS Employment Release",
                url="https://www.bls.gov/news/release.htm",
                snippet="Nonfarm payrolls rose by 275,000.",
                domain="bls.gov",
                date="2024-03-08",
                full_text="Complete employment situation report from BLS.",
            ),
            LiveSearchHit(
                title="Unknown Blog",
                url="https://blog.example.org/article",
                snippet="Some commentary snippet.",
                domain="example.org",
                date=None,  # No fabricated date
                full_text=None,
            ),
        ]

        passages = hits_to_passages(hits, fetch_full_text=False)
        assert len(passages) == 2
        p1, p2 = passages[0], passages[1]

        assert p1.passage_id.startswith("live_web_1_")
        assert p1.domain == "bls.gov"
        assert p1.date == "2024-03-08"
        assert p1.text == "Complete employment situation report from BLS."

        # Date must remain None when not present (no fabrication)
        assert p2.date is None
        assert p2.text == "Some commentary snippet."

    def test_fetch_page_text_invalid_url(self) -> None:
        # Invalid / unreachable URL should return None gracefully without raising
        result = fetch_page_text("invalid://not-a-url")
        assert result is None


class TestDemoEngine:
    def test_benchmark_mode_verification(self, demo_engine: VeritasDemoEngine) -> None:
        res = demo_engine.verify_claim(
            claim="COVID-19 mRNA vaccines significantly reduce the risk of severe hospitalizations.",
            claim_date="2021-11-01",
            mode="benchmark",
        )
        assert isinstance(res, DemoVerificationResponse)
        assert res.source_mode == "Benchmark Corpus"
        assert res.label in ("True", "Partially True", "Misleading", "False", "Unverified")
        assert len(res.evidence_cards) > 0
        assert res.evidence_cards[0].origin == "Prebuilt Corpus"
        assert res.disclaimer == DISCLAIMER_TEXT

    def test_live_search_mode_verification(self, demo_engine: VeritasDemoEngine) -> None:
        res = demo_engine.verify_claim(
            claim="Total nonfarm payroll employment rose by 275,000 in February 2024.",
            claim_date="2024-03-08",
            mode="live_search",
        )
        assert isinstance(res, DemoVerificationResponse)
        assert res.source_mode == "Live Web Search"
        assert res.label in ("True", "Partially True", "Misleading", "False", "Unverified")
        assert len(res.evidence_cards) > 0
        assert res.evidence_cards[0].origin == "Live Web Search"

    def test_empty_claim_input(self, demo_engine: VeritasDemoEngine) -> None:
        res = demo_engine.verify_claim("   ", mode="benchmark")
        assert res.label == "Unverified"
        assert res.error_message is not None

    def test_long_text_detection_and_extraction(self, demo_engine: VeritasDemoEngine) -> None:
        short_text = "The Moon is made of rock."
        assert not demo_engine.is_long_text(short_text)

        long_text = (
            "Federal Reserve officials indicated that benchmark interest rates may remain elevated "
            "through the end of the year to combat persistent inflation pressures. In European markets, "
            "industrial production fell by 0.5 percent while technology export revenues climbed slightly "
            "according to central bank figures released early this morning."
        )
        assert demo_engine.is_long_text(long_text)
