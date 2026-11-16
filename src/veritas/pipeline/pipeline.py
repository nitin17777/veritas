"""VERITAS P8 Pipeline Orchestrator.

Integrates retrieval (BM25 / Dense / Hybrid), evidence ranking, and the
evidence-constrained LLM verifier into a single typed ``verify()`` call.

Design notes
------------
* **Ranking strategy**: The pipeline uses the retrieval system's own score
  (RRF fusion score for hybrid, BM25 score or cosine similarity for single-
  backend modes) to rank evidence.  No separate re-ranker module exists yet;
  a clearly documented placeholder ``_rank_evidence()`` is provided so a
  future cross-encoder re-ranker can be dropped in without touching the rest
  of the orchestrator.  The current strategy is "retrieval-score-as-evidence-
  weight": passages are kept in retrieval rank order and the
  ``min_relevance_score`` gate inside ``EvidenceConstrainedVerifier`` acts as
  a relevance threshold.

* **No existing modules were changed** – only new files were added.

* **Query generation**: The claim string is used directly as the retrieval
  query.  A placeholder ``_generate_queries()`` method is provided for a
  future query-decomposition / expansion step.

* **Replay**: Every ``Verdict`` is saved as a self-contained JSONL record that
  contains the claim, claim_date, retrieved passage IDs with scores, all
  stage timings, the raw LLM output, and the final label. Re-running the
  orchestrator on those saved inputs recreates an identical prediction (given
  the same model weights / API responses).

Pipeline stages
---------------
1. generate_queries  – derive search queries from the claim (currently: identity)
2. retrieve          – run HybridRetriever (BM25 / Dense / Hybrid)
3. rank_evidence     – order and gate candidate passages (see note above)
4. verify            – call EvidenceConstrainedVerifier
5. return Verdict    – typed result with full provenance
"""

from __future__ import annotations

import datetime
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from veritas.config import ROOT_DIR, load_config
from veritas.retrieval.bm25 import BM25Index
from veritas.retrieval.corpus import PassageStore, create_synthetic_corpus
from veritas.retrieval.dense import DenseIndex, MockEmbedder, DenseEncoder
from veritas.retrieval.fusion import FusedSearchResult
from veritas.retrieval.hybrid import HybridRetriever
from veritas.retrieval.schema import Passage, SearchResult
from veritas.verification.llm_client import BaseLLMClient, TraceRecorder, create_llm_client
from veritas.verification.parse import VerificationResult
from veritas.verification.verify import EvidenceConstrainedVerifier, VerifierConfig
from veritas.scoring.combine import CombinedScorer, CombinedScorerConfig
from veritas.faithfulness.abstain import AbstentionGate, AbstentionConfig

logger = logging.getLogger("veritas.pipeline")


# ---------------------------------------------------------------------------
# Typed output
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    """Fully traceable fact-verification verdict returned by the pipeline.

    Attributes:
        claim_id:          Caller-supplied claim identifier (generated if absent).
        claim:             The original claim string.
        claim_date:        Optional ISO date of the claim.
        label:             VERITAS 5-class label (True / Partially True /
                           Misleading / False / Unverified).
        confidence:        Float in [0.0, 1.0].
        evidence_ids:      Passage IDs cited by the verifier.
        explanation:       Grounded textual reasoning from the LLM.
        parse_failed:      True when the LLM output failed schema validation
                           after one retry and the pipeline abstained.
        retrieved_passages: Ordered list of passage-level provenance dicts
                           (passage_id, score, rank, sources, text_snippet).
        queries_used:      Queries issued to the retrieval layer.
        stage_timings_s:   Wall-clock seconds for each named stage.
        model_metadata:    Model name, backend, token usage, etc.
        raw_llm_output:    Unprocessed string from the LLM for auditability.
        run_id:            Identifier of the parent batch run (if applicable).
        pipeline_version:  Hard-coded schema version for forward compatibility.
    """

    claim_id: str
    claim: str
    claim_date: str | None
    label: str
    confidence: float
    evidence_ids: list[str]
    explanation: str
    parse_failed: bool
    retrieved_passages: list[dict[str, Any]]
    queries_used: list[str]
    stage_timings_s: dict[str, float]
    model_metadata: dict[str, Any]
    raw_llm_output: str | None = None
    run_id: str | None = None
    pipeline_version: str = "p8.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Verdict:
        return cls(
            claim_id=str(data.get("claim_id", "")),
            claim=str(data.get("claim", "")),
            claim_date=data.get("claim_date"),
            label=str(data.get("label", "Unverified")),
            confidence=float(data.get("confidence", 0.0)),
            evidence_ids=list(data.get("evidence_ids", [])),
            explanation=str(data.get("explanation", "")),
            parse_failed=bool(data.get("parse_failed", False)),
            retrieved_passages=list(data.get("retrieved_passages", [])),
            queries_used=list(data.get("queries_used", [])),
            stage_timings_s=dict(data.get("stage_timings_s", {})),
            model_metadata=dict(data.get("model_metadata", {})),
            raw_llm_output=data.get("raw_llm_output"),
            run_id=data.get("run_id"),
            pipeline_version=str(data.get("pipeline_version", "p8.0")),
        )


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Unified configuration for the VERITAS pipeline orchestrator.

    Attributes:
        retrieval_mode:   "bm25", "dense", or "hybrid".
        retrieval_top_k:  Maximum candidates to pass to the verifier.
        bm25_index_dir:   Path to a pre-built BM25 index.
        dense_index_dir:  Path to a pre-built Dense index.
        bm25_top_k:       BM25 candidate pool size.
        dense_top_k:      Dense candidate pool size.
        rrf_k:            Reciprocal rank fusion smoothing constant.
        verification:     Dict forwarded directly to ``VerifierConfig.from_dict``.
        results_dir:      Root output directory for verdicts and traces.
        save_verdicts:    Whether to append each Verdict to JSONL.
    """

    retrieval_mode: str = "bm25"
    retrieval_top_k: int = 10
    bm25_index_dir: str = "data/indexes/bm25_default"
    dense_index_dir: str = "data/indexes/dense_default"
    bm25_top_k: int = 50
    dense_top_k: int = 50
    rrf_k: int = 60
    verification: dict[str, Any] = field(default_factory=dict)
    scoring: dict[str, Any] = field(default_factory=dict)
    faithfulness: dict[str, Any] = field(default_factory=dict)
    results_dir: str = "results"
    save_verdicts: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PipelineConfig:
        """Build PipelineConfig from a raw YAML-loaded mapping.

        Handles both flat configurations and those with nested ``retrieval``,
        ``verification``, ``scoring``, and ``faithfulness`` sections.
        """
        r = raw.get("retrieval", {})
        v = raw.get("verification", {})
        p = raw.get("pipeline", {})
        s = raw.get("scoring", {})
        f = raw.get("faithfulness", raw.get("abstention", {}))
        paths = raw.get("paths", {})
        return cls(
            retrieval_mode=str(
                p.get("retrieval_mode", r.get("mode", "bm25"))
            ),
            retrieval_top_k=int(p.get("retrieval_top_k", 10)),
            bm25_index_dir=str(
                r.get("bm25", {}).get("index_dir", "data/indexes/bm25_default")
            ),
            dense_index_dir=str(
                r.get("dense", {}).get("index_dir", "data/indexes/dense_default")
            ),
            bm25_top_k=int(r.get("bm25", {}).get("bm25_top_k", 50)),
            dense_top_k=int(r.get("dense", {}).get("dense_top_k", 50)),
            rrf_k=int(r.get("hybrid", {}).get("rrf_constant", 60)),
            verification=dict(v),
            scoring=dict(s),
            faithfulness=dict(f),
            results_dir=str(paths.get("results_dir", p.get("results_dir", "results"))),
            save_verdicts=bool(p.get("save_verdicts", True)),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        raw = load_config(path)
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Main orchestrator class
# ---------------------------------------------------------------------------

class VeritasPipeline:
    """End-to-end VERITAS fact-verification pipeline.

    Wires retrieval → ranking → verification into a single ``verify()`` call.

    Parameters
    ----------
    config:
        ``PipelineConfig`` instance or mapping; also accepts a path string.
    retriever:
        Pre-built ``HybridRetriever``.  If *None* the pipeline will attempt to
        load indexes from disk using paths in ``config``.  When no index
        exists on disk (e.g. in tests), callers must inject an in-memory
        retriever.
    llm_client:
        Pre-built ``BaseLLMClient``.  If *None* a client is created from
        ``config.verification``.
    combined_scorer:
        Optional pre-built ``CombinedScorer`` for evidence re-ranking.
    abstention_gate:
        Optional pre-built ``AbstentionGate`` for NLI-based faithfulness & abstention.
    """

    def __init__(
        self,
        config: PipelineConfig | dict[str, Any] | str | Path | None = None,
        retriever: HybridRetriever | None = None,
        llm_client: BaseLLMClient | None = None,
        combined_scorer: CombinedScorer | None = None,
        abstention_gate: AbstentionGate | None = None,
    ) -> None:
        # Resolve config
        if config is None:
            self.config = PipelineConfig()
        elif isinstance(config, PipelineConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = PipelineConfig.from_dict(config)
        else:
            self.config = PipelineConfig.from_yaml(config)

        # Resolve retriever
        if retriever is not None:
            self.retriever = retriever
        elif self.config.retrieval_mode.lower() in ("none", "llm_only", "disabled"):
            self.retriever = None
        else:
            self.retriever = self._load_retriever()

        # Resolve verifier
        verifier_cfg = VerifierConfig.from_dict({"verification": self.config.verification})
        self._verifier = EvidenceConstrainedVerifier(
            config=verifier_cfg,
            client=llm_client,
        )

        # Resolve scoring
        if combined_scorer is not None:
            self.combined_scorer = combined_scorer
        elif self.config.scoring:
            try:
                self.combined_scorer = CombinedScorer(self.config.scoring)
            except Exception as exc:
                logger.warning("Could not initialize CombinedScorer: %s", exc)
                self.combined_scorer = None
        else:
            self.combined_scorer = None

        # Resolve abstention gate
        if abstention_gate is not None:
            self.abstention_gate = abstention_gate
        elif self.config.faithfulness:
            try:
                self.abstention_gate = AbstentionGate(self.config.faithfulness)
            except Exception as exc:
                logger.warning("Could not initialize AbstentionGate: %s", exc)
                self.abstention_gate = None
        else:
            self.abstention_gate = None

        # Verdict output path
        results_root = Path(self.config.results_dir)
        if not results_root.is_absolute():
            results_root = ROOT_DIR / results_root
        self._verdicts_path = results_root / "verdicts.jsonl"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_retriever(self) -> HybridRetriever:
        """Attempt to load BM25 / Dense indexes from disk.

        Returns an empty-index ``HybridRetriever`` when no index directory
        exists (safe fallback; retrieval simply returns no results).
        """
        mode = self.config.retrieval_mode

        bm25_dir = Path(self.config.bm25_index_dir)
        if not bm25_dir.is_absolute():
            bm25_dir = ROOT_DIR / bm25_dir

        dense_dir = Path(self.config.dense_index_dir)
        if not dense_dir.is_absolute():
            dense_dir = ROOT_DIR / dense_dir

        bm25_index: BM25Index | None = None
        if mode in ("bm25", "hybrid") and bm25_dir.is_dir():
            try:
                bm25_index = BM25Index.load(bm25_dir)
                logger.info("Loaded BM25 index from %s (%d passages)",
                            bm25_dir, len(bm25_index.passage_store))
            except Exception as exc:
                logger.warning("Could not load BM25 index from %s: %s", bm25_dir, exc)

        dense_index: DenseIndex | None = None
        if mode in ("dense", "hybrid") and dense_dir.is_dir():
            try:
                dense_index = DenseIndex.load(dense_dir)
                logger.info("Loaded Dense index from %s", dense_dir)
            except Exception as exc:
                logger.warning("Could not load Dense index from %s: %s", dense_dir, exc)

        if bm25_index is None and dense_index is None:
            logger.warning(
                "No retrieval index found for mode='%s'. "
                "Retrieval will return empty results for all queries. "
                "Build an index first with 'python scripts/build_bm25_index.py'.",
                mode,
            )

        return HybridRetriever(
            bm25_index=bm25_index,
            dense_index=dense_index,
            mode=mode,
            bm25_top_k=self.config.bm25_top_k,
            dense_top_k=self.config.dense_top_k,
            rrf_k=self.config.rrf_k,
            top_k=self.config.retrieval_top_k,
        )

    def _generate_queries(self, claim: str, claim_date: str | None) -> list[str]:
        """Derive retrieval queries from the claim.

        Current strategy (v1): use the claim verbatim as the single query.
        A future query-decomposition or expansion step can be inserted here
        without changing any other part of the orchestrator.

        NOTE: This is a documented placeholder, NOT a finished implementation.
        Multi-hop claims and temporally grounded queries are not yet handled.
        """
        queries = [claim.strip()]
        logger.debug("Generated %d query/queries for claim: %r", len(queries), claim[:80])
        return queries

    def _retrieve(self, queries: list[str]) -> list[FusedSearchResult]:
        """Run the configured retrieval backend and return ranked candidates."""
        if self.config.retrieval_mode.lower() in ("none", "llm_only", "disabled") or self.retriever is None:
            return []
        try:
            return self.retriever.retrieve(queries, top_k=self.config.retrieval_top_k)
        except RuntimeError as exc:
            # e.g. BM25 index missing but mode='bm25'
            logger.warning("Retrieval failed: %s. Returning empty results.", exc)
            return []

    def _rank_evidence(
        self,
        candidates: list[FusedSearchResult],
        claim_date: str | None = None,
    ) -> list[FusedSearchResult]:
        """Rank and gate evidence candidates before verification."""
        if not candidates or self.combined_scorer is None:
            return candidates

        try:
            scored = self.combined_scorer.score_candidates(candidates, claim_date=claim_date)
            reordered: list[FusedSearchResult] = []
            for sc in scored:
                fused = FusedSearchResult(
                    passage_id=sc.passage_id,
                    score=float(sc.combined_score),
                    rank=int(sc.rank),
                    passage=sc.passage,
                    sources=["combined_scorer"],
                )
                reordered.append(fused)
            return reordered
        except Exception as exc:
            logger.warning("Combined scoring failed (%s). Preserving original ranking.", exc)
            return candidates

    def _make_retrieved_passage_records(
        self,
        candidates: list[FusedSearchResult],
    ) -> list[dict[str, Any]]:
        """Serialise candidate passages into provenance dicts for the Verdict."""
        records: list[dict[str, Any]] = []
        for item in candidates:
            passage = item.passage
            records.append(
                {
                    "passage_id": item.passage_id,
                    "rank": item.rank,
                    "score": round(item.score, 6),
                    "sources": list(item.sources),
                    "text": passage.text if passage and passage.text else "",
                    "text_snippet": (
                        passage.text[:200] if passage and passage.text else ""
                    ),
                    "url": passage.url if passage else None,
                    "date": passage.date if passage else None,
                    "domain": passage.domain if passage else None,
                }
            )
        return records

    def _save_verdict(self, verdict: Verdict) -> None:
        """Append a Verdict to the run's JSONL file."""
        if not self.config.save_verdicts:
            return
        try:
            self._verdicts_path.parent.mkdir(parents=True, exist_ok=True)
            with self._verdicts_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(verdict.to_dict(), ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Failed to persist verdict for claim_id=%s: %s",
                           verdict.claim_id, exc)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def verify(
        self,
        claim: str,
        claim_date: str | None = None,
        claim_id: str | None = None,
        run_id: str | None = None,
    ) -> Verdict:
        """Verify a single claim end-to-end.

        Parameters
        ----------
        claim:
            Natural-language factual statement.
        claim_date:
            Optional ISO-8601 date (YYYY-MM-DD) grounding the claim temporally.
        claim_id:
            Caller-supplied identifier; a UUID is generated if omitted.
        run_id:
            Optional batch run identifier for grouping verdicts.

        Returns
        -------
        Verdict
            Fully typed and traceable verdict record.
        """
        effective_id = str(claim_id).strip() if claim_id else str(uuid.uuid4())
        stage_timings: dict[str, float] = {}

        # ---- Stage 1: Query generation --------------------------------
        t0 = time.perf_counter()
        queries = self._generate_queries(claim, claim_date)
        stage_timings["query_generation"] = round(time.perf_counter() - t0, 4)

        # ---- Stage 2: Retrieval ----------------------------------------
        t0 = time.perf_counter()
        candidates = self._retrieve(queries)
        stage_timings["retrieval"] = round(time.perf_counter() - t0, 4)
        logger.info(
            "claim_id=%s | retrieved %d candidates in %.3fs",
            effective_id, len(candidates), stage_timings["retrieval"],
        )

        # ---- Stage 3: Evidence ranking ---------------------------------
        t0 = time.perf_counter()
        ranked = self._rank_evidence(candidates, claim_date=claim_date)
        stage_timings["evidence_ranking"] = round(time.perf_counter() - t0, 4)

        retrieved_records = self._make_retrieved_passage_records(ranked)

        # ---- Stage 4: Verification -------------------------------------
        t0 = time.perf_counter()
        try:
            vresult: VerificationResult = self._verifier.verify(
                claim=claim,
                evidence=ranked,
                claim_date=claim_date,
            )
        except Exception as exc:
            logger.error(
                "Verification raised unexpected exception for claim_id=%s: %s",
                effective_id, exc,
            )
            vresult = VerificationResult(
                label="Unverified",
                confidence=0.0,
                evidence_ids=[],
                explanation=f"Verification failed with an internal error: {exc}",
                parse_failed=True,
            )
        stage_timings["verification"] = round(time.perf_counter() - t0, 4)

        # ---- Stage 5: Assemble Verdict --------------------------------
        model_meta: dict[str, Any] = {
            "backend": self.config.verification.get("backend", "mock"),
            "model": self.config.verification.get("model", ""),
            "attempts": vresult.attempts,
            **(vresult.metadata or {}),
        }

        verdict = Verdict(
            claim_id=effective_id,
            claim=claim,
            claim_date=claim_date,
            label=vresult.label,
            confidence=vresult.confidence,
            evidence_ids=vresult.evidence_ids,
            explanation=vresult.explanation,
            parse_failed=vresult.parse_failed,
            retrieved_passages=retrieved_records,
            queries_used=queries,
            stage_timings_s=stage_timings,
            model_metadata=model_meta,
            raw_llm_output=vresult.raw_response,
            run_id=run_id,
        )

        # ---- Stage 6: Faithfulness & Abstention Gate (if configured) ---
        if self.abstention_gate is not None:
            t0 = time.perf_counter()
            try:
                verdict, _ = self.abstention_gate.evaluate_verdict(verdict)
            except Exception as exc:
                logger.warning("Abstention gate evaluation failed: %s", exc)
            stage_timings["faithfulness_abstention"] = round(time.perf_counter() - t0, 4)
            verdict.stage_timings_s["faithfulness_abstention"] = stage_timings["faithfulness_abstention"]

        self._save_verdict(verdict)
        logger.info(
            "claim_id=%s | label=%s | confidence=%.2f | parse_failed=%s",
            effective_id, verdict.label, verdict.confidence, verdict.parse_failed,
        )
        return verdict


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def verify(
    claim: str,
    claim_date: str | None = None,
    cfg: PipelineConfig | dict[str, Any] | str | Path | None = None,
    retriever: HybridRetriever | None = None,
    llm_client: BaseLLMClient | None = None,
    combined_scorer: CombinedScorer | None = None,
    abstention_gate: AbstentionGate | None = None,
    claim_id: str | None = None,
) -> Verdict:
    """Convenience wrapper: construct a one-shot ``VeritasPipeline`` and verify.

    This is the primary entry point documented in the P8 specification:

        ``verify(claim, claim_date, cfg) -> Verdict``

    For batch use or when you need to share an index across many claims,
    construct ``VeritasPipeline`` directly.
    """
    pipeline = VeritasPipeline(
        config=cfg,
        retriever=retriever,
        llm_client=llm_client,
        combined_scorer=combined_scorer,
        abstention_gate=abstention_gate,
    )
    return pipeline.verify(claim=claim, claim_date=claim_date, claim_id=claim_id)
