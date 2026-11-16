"""VERITAS Streamlit Demo Engine.

Orchestrates:
- Claim verification via existing VeritasPipeline.verify() without code duplication.
- Benchmark Mode (pre-built offline corpus / fixtures) vs. Live Web Search Mode.
- Long text / article claim extraction via ClaimExtractor (P5).
- Formatting structured UI evidence cards with clear provenance origin.
- Handling empty input, unavailable models, and abstention fallbacks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
import time
from typing import Any

from veritas.config import ROOT_DIR, load_config
from veritas.demo.web_search import hits_to_passages, search_web
from veritas.experiments.fixtures import build_fixture_retriever
from veritas.extraction.extract import ClaimExtractor, ExtractedClaim
from veritas.pipeline.pipeline import PipelineConfig, Verdict, VeritasPipeline
from veritas.retrieval.corpus import PassageStore
from veritas.retrieval.fusion import FusedSearchResult
from veritas.retrieval.hybrid import HybridRetriever
from veritas.retrieval.schema import Passage
from veritas.scoring.credibility import extract_domain
from veritas.verification.llm_client import BaseLLMClient, MockLLMClient

logger = logging.getLogger("veritas.demo.engine")

DISCLAIMER_TEXT = "VERITAS assists human judgment and does not guarantee factual correctness."


@dataclass
class DemoEvidenceCard:
    """Evidence passage details formatted for UI presentation."""

    passage_id: str
    domain: str
    date: str | None
    snippet: str
    url: str | None
    score: float
    origin: str  # 'Prebuilt Corpus' or 'Live Web Search'
    is_cited: bool = False
    credibility_tier: str = "Unspecified"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DemoVerificationResponse:
    """Rich verification output for Streamlit UI rendering."""

    claim: str
    label: str
    confidence: float
    explanation: str
    evidence_cards: list[DemoEvidenceCard]
    source_mode: str  # 'Benchmark Corpus' | 'Live Web Search'
    abstained: bool = False
    original_label: str | None = None
    abstention_reasons: list[str] = field(default_factory=list)
    faithfulness_score: float | None = None
    parse_failed: bool = False
    stage_timings: dict[str, float] = field(default_factory=dict)
    extracted_claims: list[dict[str, Any]] = field(default_factory=list)
    disclaimer: str = DISCLAIMER_TEXT
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "evidence_cards": [c.to_dict() for c in self.evidence_cards],
        }


class VeritasDemoEngine:
    """Manages fact verification sessions for the Streamlit demo application."""

    def __init__(
        self,
        config_path: str | Path = "configs/demo.yaml",
        pipeline: VeritasPipeline | None = None,
        retriever: HybridRetriever | None = None,
        llm_client: BaseLLMClient | None = None,
    ) -> None:
        p = Path(config_path)
        if not p.is_absolute():
            p = ROOT_DIR / p
        self.raw_config = load_config(p) if p.exists() else {}
        self.demo_cfg = self.raw_config.get("demo", {})

        # Build or receive pipeline
        if pipeline is not None:
            self.pipeline = pipeline
        else:
            # Flatten demo sections so PipelineConfig can find benchmark/scoring/faithfulness
            pipe_dict = {**self.demo_cfg, **self.raw_config}
            pipe_cfg = PipelineConfig.from_dict(pipe_dict)
            eff_retriever = retriever
            if eff_retriever is None:
                # Check if default BM25 directory exists and has passages; if not, fallback to fixture retriever
                bm25_path = ROOT_DIR / pipe_cfg.bm25_index_dir
                passages_file = bm25_path / "passages.jsonl"
                if not bm25_path.is_dir() or not passages_file.exists() or passages_file.stat().st_size == 0:
                    eff_retriever = build_fixture_retriever()

            self.pipeline = VeritasPipeline(
                config=pipe_cfg,
                retriever=eff_retriever,
                llm_client=llm_client,
            )

        # Optional claim extractor for long text inputs (P5)
        self.claim_extractor: ClaimExtractor | None = None
        if self.demo_cfg.get("claim_extraction", {}).get("enabled", True):
            try:
                self.claim_extractor = ClaimExtractor(llm_client=llm_client)
            except Exception as exc:
                logger.warning("ClaimExtractor initialization skipped: %s", exc)

    def is_long_text(self, text: str) -> bool:
        """Determine whether input text is an article/post requiring claim extraction."""
        threshold = int(self.demo_cfg.get("claim_extraction", {}).get("word_threshold", 30))
        words = text.strip().split()
        return len(words) > threshold

    def extract_claims(self, text: str) -> list[ExtractedClaim]:
        """Extract check-worthy claims from long input text."""
        if not self.claim_extractor or not text.strip():
            return []
        try:
            res = self.claim_extractor.extract(text)
            return res.claims
        except Exception as exc:
            logger.warning("Claim extraction failed: %s", exc)
            return []

    def _make_evidence_cards(
        self,
        retrieved_passages: list[dict[str, Any]],
        cited_ids: list[str],
        origin: str,
    ) -> list[DemoEvidenceCard]:
        """Format retrieved passages into UI evidence cards."""
        cards: list[DemoEvidenceCard] = []
        cited_set = set(str(cid).strip() for cid in cited_ids)

        for p in retrieved_passages:
            pid = str(p.get("passage_id") or p.get("id") or "")
            dom = p.get("domain") or extract_domain(p.get("url") or "") or "Unknown Source"
            snippet = p.get("text_snippet") or p.get("text") or ""
            score = float(p.get("score") or 0.0)
            url = p.get("url")
            date = p.get("date")

            cards.append(
                DemoEvidenceCard(
                    passage_id=pid,
                    domain=dom,
                    date=str(date) if date else None,
                    snippet=snippet.strip(),
                    url=str(url) if url else None,
                    score=round(score, 4),
                    origin=origin,
                    is_cited=(pid in cited_set),
                )
            )
        return cards

    def verify_claim(
        self,
        claim: str,
        claim_date: str | None = None,
        mode: str = "benchmark",  # 'benchmark' | 'live_search'
        search_query: str | None = None,
    ) -> DemoVerificationResponse:
        """Verify an individual factual statement.

        Reuses `self.pipeline.verify()` directly, adhering strictly to the architecture.

        Args:
            claim: Statement to verify.
            claim_date: Optional temporal ISO-8601 date.
            mode: 'benchmark' (local corpus) or 'live_search' (web retrieval).
            search_query: Optional custom search query for live search.

        Returns:
            DemoVerificationResponse.
        """
        clean_claim = str(claim).strip()
        if not clean_claim:
            return DemoVerificationResponse(
                claim="",
                label="Unverified",
                confidence=0.0,
                explanation="No claim text was provided for verification.",
                evidence_cards=[],
                source_mode="None",
                error_message="Please enter a factual claim, article, or post statement.",
            )

        # -------------------------------------------------------------
        # Mode 1: Benchmark Mode (Uses VeritasPipeline.verify() directly)
        # -------------------------------------------------------------
        if mode.lower() == "benchmark":
            verdict: Verdict = self.pipeline.verify(
                claim=clean_claim,
                claim_date=claim_date,
            )

            cards = self._make_evidence_cards(
                retrieved_passages=verdict.retrieved_passages,
                cited_ids=verdict.evidence_ids,
                origin="Prebuilt Corpus",
            )

            meta = verdict.model_metadata
            abst_info = meta.get("abstention", {})
            abstained = bool(abst_info.get("abstained") or (verdict.label == "Unverified" and meta.get("original_label")))

            return DemoVerificationResponse(
                claim=clean_claim,
                label=verdict.label,
                confidence=verdict.confidence,
                explanation=verdict.explanation,
                evidence_cards=cards,
                source_mode="Benchmark Corpus",
                abstained=abstained,
                original_label=abst_info.get("original_label") or meta.get("original_label"),
                abstention_reasons=abst_info.get("abstention_reasons", []),
                faithfulness_score=abst_info.get("faithfulness_score"),
                parse_failed=verdict.parse_failed,
                stage_timings=verdict.stage_timings_s,
            )

        # -------------------------------------------------------------
        # Mode 2: Live Web Search Mode
        # -------------------------------------------------------------
        stage_timings: dict[str, float] = {}
        t0 = time.perf_counter()

        # Step 2a: Live web retrieval
        live_cfg = self.demo_cfg.get("live_search", {})
        query_to_run = search_query.strip() if search_query else clean_claim
        hits = search_web(
            query=query_to_run,
            max_results=int(live_cfg.get("max_results", 5)),
            timeout=float(live_cfg.get("timeout_seconds", 10.0)),
            provider=str(live_cfg.get("provider", "duckduckgo")),
        )
        stage_timings["live_web_retrieval"] = round(time.perf_counter() - t0, 4)

        # Step 2b: Web extraction & conversion to Passages
        t0 = time.perf_counter()
        extract_cfg = self.demo_cfg.get("web_extraction", {})
        passages: list[Passage] = hits_to_passages(
            hits=hits,
            fetch_full_text=bool(extract_cfg.get("enabled", True)),
            timeout=float(extract_cfg.get("timeout_seconds", 8.0)),
            max_chars=int(extract_cfg.get("max_text_chars", 2500)),
        )
        stage_timings["webpage_extraction"] = round(time.perf_counter() - t0, 4)

        if not passages:
            return DemoVerificationResponse(
                claim=clean_claim,
                label="Unverified",
                confidence=0.0,
                explanation="Live web search returned no accessible evidence passages.",
                evidence_cards=[],
                source_mode="Live Web Search",
                abstained=True,
                abstention_reasons=["Live search query yielded 0 reachable web evidence documents."],
                stage_timings=stage_timings,
            )

        # Step 2c: Score & rank live passages using pipeline ranking
        t0 = time.perf_counter()
        candidates: list[FusedSearchResult] = [
            FusedSearchResult(
                passage_id=p.passage_id,
                score=1.0 / (idx + 1),  # rank-based initial relevance score
                rank=idx + 1,
                passage=p,
                sources=["live_web_search"],
            )
            for idx, p in enumerate(passages)
        ]

        ranked = self.pipeline._rank_evidence(candidates, claim_date=claim_date)
        stage_timings["evidence_ranking"] = round(time.perf_counter() - t0, 4)

        # Step 2d: Verify via EvidenceConstrainedVerifier
        t0 = time.perf_counter()
        vresult = self.pipeline._verifier.verify(
            claim=clean_claim,
            evidence=ranked,
            claim_date=claim_date,
        )
        stage_timings["verification"] = round(time.perf_counter() - t0, 4)

        retrieved_records = self.pipeline._make_retrieved_passage_records(ranked)
        cards = self._make_evidence_cards(
            retrieved_passages=retrieved_records,
            cited_ids=vresult.evidence_ids,
            origin="Live Web Search",
        )

        # Step 2e: Abstention Gate (if active)
        abstained = False
        final_label = vresult.label
        final_conf = vresult.confidence
        orig_label = None
        abst_reasons: list[str] = []
        faith_score = None

        if self.pipeline.abstention_gate is not None:
            t0 = time.perf_counter()
            try:
                # Construct synthetic verdict for gate
                v_dummy = Verdict(
                    claim_id="live_claim",
                    claim=clean_claim,
                    claim_date=claim_date,
                    label=vresult.label,
                    confidence=vresult.confidence,
                    evidence_ids=vresult.evidence_ids,
                    explanation=vresult.explanation,
                    parse_failed=vresult.parse_failed,
                    retrieved_passages=retrieved_records,
                    queries_used=[query_to_run],
                    stage_timings_s=stage_timings,
                    model_metadata={},
                )
                updated_v, audit = self.pipeline.abstention_gate.evaluate_verdict(v_dummy)
                final_label = updated_v.label
                final_conf = updated_v.confidence
                abstained = audit.abstained
                orig_label = audit.original_label
                abst_reasons = audit.abstention_reasons
                faith_score = audit.faithfulness_score
            except Exception as exc:
                logger.warning("Abstention gate evaluation failed: %s", exc)
            stage_timings["faithfulness_abstention"] = round(time.perf_counter() - t0, 4)

        return DemoVerificationResponse(
            claim=clean_claim,
            label=final_label,
            confidence=final_conf,
            explanation=vresult.explanation,
            evidence_cards=cards,
            source_mode="Live Web Search",
            abstained=abstained,
            original_label=orig_label,
            abstention_reasons=abst_reasons,
            faithfulness_score=faith_score,
            parse_failed=vresult.parse_failed,
            stage_timings=stage_timings,
        )
