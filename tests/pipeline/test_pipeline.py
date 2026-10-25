"""Integration tests for the VERITAS P8 pipeline orchestrator.

These tests exercise the full retrieval → ranking → verification flow with:
  * An in-memory BM25 index (no disk I/O, no large model downloads).
  * A ``MockLLMClient`` (no paid API calls).

IMPORTANT DISCLAIMER
--------------------
Mocked tests passing here does NOT validate that:
  * The pipeline produces correct verdicts on real claims.
  * The retrieval system finds the right evidence for real queries.
  * The LLM produces valid verdicts when given real evidence.

Real end-to-end validation requires:
  1. A populated BM25 / Dense index built from actual evidence corpora.
  2. A capable LLM backend (API or local model).
  3. Evaluation against a held-out labelled dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from veritas.pipeline.batch import BatchRunner, _load_completed_ids, run_batch
from veritas.pipeline.pipeline import (
    PipelineConfig,
    Verdict,
    VeritasPipeline,
    verify,
)
from veritas.retrieval.bm25 import BM25Index
from veritas.retrieval.corpus import PassageStore
from veritas.retrieval.fusion import FusedSearchResult
from veritas.retrieval.hybrid import HybridRetriever
from veritas.retrieval.schema import Passage
from veritas.verification.llm_client import MockLLMClient


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FIVE_CLAIMS = [
    {
        "id": "c1",
        "claim": "Vaccination significantly reduces the risk of severe COVID-19 hospitalizations.",
        "claim_date": "2021-11-01",
        "expected_label": "True",
    },
    {
        "id": "c2",
        "claim": "The Bureau of Labor Statistics reported nonfarm payrolls rose by 275,000 in February 2024.",
        "claim_date": "2024-03-08",
        "expected_label": "True",
    },
    {
        "id": "c3",
        "claim": "The Moon is made of cheese.",
        "claim_date": None,
        "expected_label": "False",
    },
    {
        "id": "c4",
        "claim": "NASA's James Webb Space Telescope detected methane on exoplanet K2-18b.",
        "claim_date": "2023-09-11",
        "expected_label": "True",
    },
    {
        "id": "c5",
        "claim": "Mount Everest is shorter than K2.",
        "claim_date": None,
        "expected_label": "False",
    },
]

CORPUS_PASSAGES = [
    Passage(
        "p_vax",
        "Randomised controlled trials and real-world data show COVID-19 vaccines "
        "reduce severe illness and hospitalisation by 70–95 %.",
        domain="nejm.org",
    ),
    Passage(
        "p_bls",
        "The Bureau of Labor Statistics reported that nonfarm payroll employment "
        "increased by 275,000 in February 2024, and the unemployment rate edged "
        "up to 3.9 percent.",
        domain="bls.gov",
    ),
    Passage(
        "p_moon",
        "The Moon is a rocky, cratered body with no atmosphere; its surface is "
        "composed of regolith, basalt, and anorthosite – not dairy products.",
        domain="nasa.gov",
    ),
    Passage(
        "p_jwst",
        "NASA announced that the James Webb Space Telescope identified "
        "carbon-dioxide and methane in the atmosphere of exoplanet K2-18b, "
        "consistent with a water-ocean world.",
        domain="nasa.gov",
    ),
    Passage(
        "p_everest",
        "Mount Everest, at 8,849 m above sea level, is the highest mountain on "
        "Earth. K2 stands at 8,611 m – 238 m shorter.",
        domain="nationalgeographic.com",
    ),
]

CANNED_RESPONSES = [
    {
        "label": "True",
        "confidence": 0.96,
        "evidence_ids": ["p_vax"],
        "explanation": "p_vax confirms vaccine efficacy against severe hospitalisation.",
    },
    {
        "label": "True",
        "confidence": 0.99,
        "evidence_ids": ["p_bls"],
        "explanation": "p_bls directly confirms the BLS payroll figure.",
    },
    {
        "label": "False",
        "confidence": 0.98,
        "evidence_ids": ["p_moon"],
        "explanation": "p_moon states the Moon is rocky, directly refuting the cheese claim.",
    },
    {
        "label": "True",
        "confidence": 0.95,
        "evidence_ids": ["p_jwst"],
        "explanation": "p_jwst confirms JWST detected methane on K2-18b.",
    },
    {
        "label": "False",
        "confidence": 0.97,
        "evidence_ids": ["p_everest"],
        "explanation": "p_everest states Everest is 8,849 m and K2 is 8,611 m, "
                       "so Everest is taller, contradicting the claim.",
    },
]


@pytest.fixture
def in_memory_pipeline() -> VeritasPipeline:
    """Build a fully in-memory pipeline using BM25 + MockLLMClient."""
    store = PassageStore(CORPUS_PASSAGES)
    bm25_idx = BM25Index.build(store)
    retriever = HybridRetriever(
        bm25_index=bm25_idx,
        mode="bm25",
        bm25_top_k=5,
        top_k=5,
    )
    client = MockLLMClient(canned_responses=list(CANNED_RESPONSES))
    cfg = PipelineConfig(
        retrieval_mode="bm25",
        retrieval_top_k=5,
        save_verdicts=False,
        verification={"backend": "mock", "save_traces": False},
    )
    return VeritasPipeline(config=cfg, retriever=retriever, llm_client=client)


# ---------------------------------------------------------------------------
# Integration test: 5 synthetic claims
# ---------------------------------------------------------------------------

class TestFiveClaimIntegration:
    """End-to-end integration test across five synthetic claims.

    See the disclaimer at the top of this module – mocked LLM responses mean
    these tests validate orchestration correctness (routing, typing, provenance
    recording) rather than real fact-checking accuracy.
    """

    def test_five_claims_return_verdicts(self, in_memory_pipeline):
        """All five claims must yield a Verdict with a valid VERITAS label."""
        from veritas.data_prep.schema import VALID_LABELS

        for item in FIVE_CLAIMS:
            v = in_memory_pipeline.verify(
                claim=item["claim"],
                claim_date=item["claim_date"],
                claim_id=item["id"],
            )
            assert isinstance(v, Verdict)
            assert v.label in VALID_LABELS, f"Bad label {v.label!r} for {item['id']}"
            assert 0.0 <= v.confidence <= 1.0
            assert isinstance(v.evidence_ids, list)
            assert isinstance(v.explanation, str) and v.explanation
            assert v.claim_id == item["id"]

    def test_five_claims_expected_labels(self, in_memory_pipeline):
        """Mocked responses yield the expected labels for each synthetic claim."""
        for item in FIVE_CLAIMS:
            v = in_memory_pipeline.verify(
                claim=item["claim"],
                claim_date=item["claim_date"],
                claim_id=item["id"],
            )
            assert v.label == item["expected_label"], (
                f"claim_id={item['id']}: expected {item['expected_label']!r}, "
                f"got {v.label!r}"
            )

    def test_stage_timings_recorded(self, in_memory_pipeline):
        """Every stage must record a non-negative timing."""
        v = in_memory_pipeline.verify(
            claim=FIVE_CLAIMS[0]["claim"],
            claim_date=FIVE_CLAIMS[0]["claim_date"],
            claim_id=FIVE_CLAIMS[0]["id"],
        )
        for stage in ("query_generation", "retrieval", "evidence_ranking", "verification"):
            assert stage in v.stage_timings_s, f"Missing timing for stage: {stage}"
            assert v.stage_timings_s[stage] >= 0.0

    def test_retrieved_passages_provenance(self, in_memory_pipeline):
        """Retrieved passage records must carry passage_id, rank, score, and snippet."""
        v = in_memory_pipeline.verify(
            claim=FIVE_CLAIMS[0]["claim"],
            claim_date=None,
            claim_id="prov_test",
        )
        assert len(v.retrieved_passages) > 0
        for rec in v.retrieved_passages:
            assert "passage_id" in rec
            assert "rank" in rec and isinstance(rec["rank"], int)
            assert "score" in rec and isinstance(rec["score"], float)
            assert "text_snippet" in rec

    def test_queries_used_recorded(self, in_memory_pipeline):
        """The pipeline must record which query strings were issued."""
        v = in_memory_pipeline.verify(claim="Vaccination is effective.", claim_date=None)
        assert isinstance(v.queries_used, list)
        assert len(v.queries_used) >= 1
        assert v.queries_used[0] == "Vaccination is effective."

    def test_model_metadata_recorded(self, in_memory_pipeline):
        """model_metadata must include at minimum backend and attempts."""
        v = in_memory_pipeline.verify(claim=FIVE_CLAIMS[2]["claim"], claim_date=None)
        assert "backend" in v.model_metadata
        assert "attempts" in v.model_metadata

    def test_pipeline_version_field(self, in_memory_pipeline):
        v = in_memory_pipeline.verify(claim="Test claim", claim_date=None)
        assert v.pipeline_version == "p8.0"

    def test_verdict_round_trips_through_json(self, in_memory_pipeline):
        """Verdict.to_dict() / from_dict() must be lossless."""
        v = in_memory_pipeline.verify(
            claim=FIVE_CLAIMS[3]["claim"],
            claim_date=FIVE_CLAIMS[3]["claim_date"],
        )
        restored = Verdict.from_dict(v.to_dict())
        assert restored.label == v.label
        assert restored.confidence == v.confidence
        assert restored.evidence_ids == v.evidence_ids
        assert restored.claim == v.claim
        assert restored.claim_date == v.claim_date


# ---------------------------------------------------------------------------
# Empty-retrieval safety
# ---------------------------------------------------------------------------

class TestEmptyRetrievalSafety:

    def test_empty_retrieval_returns_unverified(self):
        """When the retrieval index is empty, the pipeline must return Unverified
        without calling the LLM (MockLLMClient call count stays 0)."""
        empty_store = PassageStore([])
        empty_idx = BM25Index.build(empty_store)
        retriever = HybridRetriever(bm25_index=empty_idx, mode="bm25", bm25_top_k=5, top_k=5)
        client = MockLLMClient()
        cfg = PipelineConfig(
            retrieval_mode="bm25",
            retrieval_top_k=5,
            save_verdicts=False,
            verification={"backend": "mock", "save_traces": False},
        )
        pipeline = VeritasPipeline(config=cfg, retriever=retriever, llm_client=client)
        v = pipeline.verify(claim="Some factual claim.")
        assert v.label == "Unverified"
        assert v.confidence == 0.0
        assert v.parse_failed is False
        assert len(client.history) == 0, "LLM must NOT be called when evidence is empty"

    def test_relevance_threshold_gates_llm_call(self):
        """When all retrieved scores are below min_relevance_score, the LLM is
        not called and the pipeline returns Unverified."""
        # BM25 scores for unrelated query will be very low (or zero)
        store = PassageStore([Passage("p_unrelated", "Completely unrelated passage text here.")])
        bm25_idx = BM25Index.build(store)
        retriever = HybridRetriever(bm25_index=bm25_idx, mode="bm25", bm25_top_k=5, top_k=5)
        client = MockLLMClient()
        cfg = PipelineConfig(
            retrieval_mode="bm25",
            retrieval_top_k=5,
            save_verdicts=False,
            verification={
                "backend": "mock",
                "save_traces": False,
                "min_relevance_score": 999.0,  # unreachably high threshold
            },
        )
        pipeline = VeritasPipeline(config=cfg, retriever=retriever, llm_client=client)
        v = pipeline.verify(claim="Totally different topic claim.")
        assert v.label == "Unverified"
        assert len(client.history) == 0


# ---------------------------------------------------------------------------
# Model failure safety
# ---------------------------------------------------------------------------

class TestModelFailureSafety:

    def test_llm_exception_returns_unverified_parse_failed(self):
        """If the LLM raises an unexpected exception, the pipeline must return a
        safe Unverified verdict (parse_failed=True) rather than propagating.

        Uses a stub retriever so BM25 IDF=0 in single-doc corpora doesn't hide
        the LLM code-path.
        """
        evidence_passage = Passage("p1", "vaccination reduces severe hospitalizations")
        fused = FusedSearchResult(
            passage_id="p1", rank=1, score=1.0, sources=["bm25"], passage=evidence_passage
        )

        class StubRetriever(HybridRetriever):
            def retrieve(self, queries, **kw):
                return [fused]

        retriever = StubRetriever(mode="bm25")
        client = MockLLMClient(
            canned_responses=[
                RuntimeError("Simulated LLM network timeout"),
                RuntimeError("Simulated LLM network timeout (retry)"),
            ]
        )
        cfg = PipelineConfig(
            retrieval_mode="bm25",
            retrieval_top_k=5,
            save_verdicts=False,
            verification={"backend": "mock", "save_traces": False},
        )
        pipeline = VeritasPipeline(config=cfg, retriever=retriever, llm_client=client)
        v = pipeline.verify(claim="vaccination reduces severe hospitalizations")
        assert v.label == "Unverified"
        assert v.parse_failed is True

    def test_malformed_llm_output_retry_then_parse_failed(self):
        """Two consecutive malformed LLM responses must result in parse_failed=True.

        Uses a stub retriever so evidence is guaranteed and the LLM path is exercised.
        """
        evidence_passage = Passage("p1", "climate data shows rising global temperatures")
        fused = FusedSearchResult(
            passage_id="p1", rank=1, score=1.0, sources=["bm25"], passage=evidence_passage
        )

        class StubRetriever(HybridRetriever):
            def retrieve(self, queries, **kw):
                return [fused]

        retriever = StubRetriever(mode="bm25")
        client = MockLLMClient(canned_responses=["not json at all", "also not json"])
        cfg = PipelineConfig(
            retrieval_mode="bm25",
            retrieval_top_k=5,
            save_verdicts=False,
            verification={"backend": "mock", "save_traces": False},
        )
        pipeline = VeritasPipeline(config=cfg, retriever=retriever, llm_client=client)
        v = pipeline.verify(claim="climate data shows rising global temperatures")
        assert v.label == "Unverified"
        assert v.parse_failed is True
        assert v.model_metadata["attempts"] == 2
        assert len(client.history) == 2


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------

class TestPipelineConfig:

    def test_from_dict_defaults(self):
        cfg = PipelineConfig.from_dict({})
        assert cfg.retrieval_mode == "bm25"
        assert cfg.retrieval_top_k == 10
        assert cfg.save_verdicts is True

    def test_from_dict_nested(self):
        raw = {
            "pipeline": {"retrieval_mode": "hybrid", "retrieval_top_k": 5},
            "retrieval": {
                "mode": "hybrid",
                "bm25": {"index_dir": "/tmp/bm25", "bm25_top_k": 20},
                "dense": {"index_dir": "/tmp/dense", "dense_top_k": 20},
                "hybrid": {"rrf_constant": 30},
            },
            "verification": {"backend": "api", "model": "gpt-4o"},
        }
        cfg = PipelineConfig.from_dict(raw)
        assert cfg.retrieval_mode == "hybrid"
        assert cfg.retrieval_top_k == 5
        assert cfg.bm25_index_dir == "/tmp/bm25"
        assert cfg.rrf_k == 30
        assert cfg.verification["backend"] == "api"

    def test_from_yaml(self, tmp_path):
        yaml_content = (
            "pipeline:\n"
            "  retrieval_mode: dense\n"
            "  retrieval_top_k: 3\n"
            "retrieval:\n"
            "  dense:\n"
            "    index_dir: /some/path\n"
        )
        p = tmp_path / "test.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        cfg = PipelineConfig.from_yaml(p)
        assert cfg.retrieval_mode == "dense"
        assert cfg.retrieval_top_k == 3

    def test_to_dict_round_trip(self):
        cfg = PipelineConfig(retrieval_mode="hybrid", retrieval_top_k=7, rrf_k=30)
        d = cfg.to_dict()
        assert d["retrieval_mode"] == "hybrid"
        assert d["retrieval_top_k"] == 7
        assert d["rrf_k"] == 30


# ---------------------------------------------------------------------------
# Verdict serialisation
# ---------------------------------------------------------------------------

class TestVerdictSerialization:

    def _make_verdict(self, **overrides) -> Verdict:
        defaults = dict(
            claim_id="v_test",
            claim="Test claim.",
            claim_date="2024-01-01",
            label="True",
            confidence=0.9,
            evidence_ids=["p1"],
            explanation="p1 confirms the claim.",
            parse_failed=False,
            retrieved_passages=[{"passage_id": "p1", "rank": 1, "score": 0.5, "text_snippet": "Text."}],
            queries_used=["Test claim."],
            stage_timings_s={"retrieval": 0.01, "verification": 0.05},
            model_metadata={"backend": "mock"},
        )
        defaults.update(overrides)
        return Verdict(**defaults)

    def test_to_dict_contains_all_fields(self):
        v = self._make_verdict()
        d = v.to_dict()
        for key in ("claim_id", "claim", "label", "confidence", "evidence_ids",
                    "explanation", "parse_failed", "retrieved_passages",
                    "queries_used", "stage_timings_s", "model_metadata",
                    "pipeline_version"):
            assert key in d, f"Missing key: {key}"

    def test_from_dict_round_trip(self):
        v = self._make_verdict()
        restored = Verdict.from_dict(v.to_dict())
        assert restored.label == "True"
        assert restored.confidence == 0.9
        assert restored.evidence_ids == ["p1"]
        assert restored.pipeline_version == "p8.0"

    def test_json_serialisable(self):
        v = self._make_verdict()
        serialised = json.dumps(v.to_dict())
        parsed = json.loads(serialised)
        assert parsed["label"] == "True"


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

class TestBatchRunner:

    def _build_pipeline(self, responses: list) -> VeritasPipeline:
        store = PassageStore([
            Passage("p1", "Evidence for claim A."),
            Passage("p2", "Evidence for claim B."),
            Passage("p3", "Evidence for claim C."),
        ])
        idx = BM25Index.build(store)
        retriever = HybridRetriever(bm25_index=idx, mode="bm25", bm25_top_k=3, top_k=3)
        client = MockLLMClient(canned_responses=responses)
        cfg = PipelineConfig(
            retrieval_mode="bm25",
            retrieval_top_k=3,
            save_verdicts=False,
            verification={"backend": "mock", "save_traces": False},
        )
        return VeritasPipeline(config=cfg, retriever=retriever, llm_client=client)

    def _write_input(self, path: Path, records: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    def test_batch_processes_all_claims(self, tmp_path):
        input_path = tmp_path / "claims.jsonl"
        output_path = tmp_path / "verdicts.jsonl"
        claims = [
            {"id": "c1", "claim": "Claim A."},
            {"id": "c2", "claim": "Claim B."},
            {"id": "c3", "claim": "Claim C."},
        ]
        self._write_input(input_path, claims)

        responses = [
            {"label": "True", "confidence": 0.9, "evidence_ids": ["p1"], "explanation": "p1 supports."},
            {"label": "False", "confidence": 0.8, "evidence_ids": ["p2"], "explanation": "p2 refutes."},
            {"label": "Unverified", "confidence": 0.3, "evidence_ids": [], "explanation": "Insufficient."},
        ]
        pipeline = self._build_pipeline(responses)
        runner = BatchRunner(pipeline=pipeline, output_path=output_path, run_id="test_run")
        ckpt = runner.run(input_path=input_path, show_progress=False)

        assert ckpt.done == 3
        assert ckpt.failed == 0
        assert output_path.is_file()

        lines = [json.loads(l) for l in output_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 3
        labels = {l["claim_id"]: l["label"] for l in lines}
        assert labels["c1"] == "True"
        assert labels["c2"] == "False"
        assert labels["c3"] == "Unverified"

    def test_batch_resume_skips_completed(self, tmp_path):
        """If the output file already contains c1, a resumed run must skip it."""
        input_path = tmp_path / "claims.jsonl"
        output_path = tmp_path / "verdicts.jsonl"
        claims = [
            {"id": "c1", "claim": "Claim A."},
            {"id": "c2", "claim": "Claim B."},
        ]
        self._write_input(input_path, claims)

        # Pre-populate output with c1 (simulating a prior partial run)
        existing = Verdict(
            claim_id="c1", claim="Claim A.", claim_date=None, label="True",
            confidence=0.9, evidence_ids=["p1"], explanation="Done.",
            parse_failed=False, retrieved_passages=[], queries_used=["Claim A."],
            stage_timings_s={}, model_metadata={},
        )
        with output_path.open("w") as f:
            f.write(json.dumps(existing.to_dict()) + "\n")

        # Only c2 needs a mocked response
        responses = [
            {"label": "False", "confidence": 0.85, "evidence_ids": ["p2"], "explanation": "p2 refutes."},
        ]
        pipeline = self._build_pipeline(responses)
        runner = BatchRunner(pipeline=pipeline, output_path=output_path, run_id="resume_run")
        ckpt = runner.run(input_path=input_path, show_progress=False)

        assert ckpt.done == 1   # only c2 was newly processed
        assert ckpt.skipped == 1  # c1 was skipped

        lines = [json.loads(l) for l in output_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 2  # c1 (pre-existing) + c2 (new)
        ids = {l["claim_id"] for l in lines}
        assert ids == {"c1", "c2"}

    def test_load_completed_ids(self, tmp_path):
        output = tmp_path / "v.jsonl"
        with output.open("w") as f:
            f.write(json.dumps({"claim_id": "x1"}) + "\n")
            f.write(json.dumps({"claim_id": "x2"}) + "\n")
        ids = _load_completed_ids(output)
        assert ids == {"x1", "x2"}

    def test_load_completed_ids_missing_file(self, tmp_path):
        ids = _load_completed_ids(tmp_path / "nonexistent.jsonl")
        assert ids == set()

    def test_checkpoint_saved(self, tmp_path):
        input_path = tmp_path / "claims.jsonl"
        output_path = tmp_path / "verdicts.jsonl"
        claims = [{"id": "ck1", "claim": "Claim one."}]
        self._write_input(input_path, claims)

        responses = [
            {"label": "True", "confidence": 0.9, "evidence_ids": ["p1"], "explanation": "Confirmed."},
        ]
        pipeline = self._build_pipeline(responses)
        runner = BatchRunner(pipeline=pipeline, output_path=output_path, run_id="ckpt_test")
        ckpt = runner.run(input_path=input_path, show_progress=False)

        ckpt_path = output_path.with_suffix(".ckpt.json")
        assert ckpt_path.is_file()
        data = json.loads(ckpt_path.read_text())
        assert data["run_id"] == "ckpt_test"
        assert data["done"] == 1
        assert data["finished_at"] is not None


# ---------------------------------------------------------------------------
# Module-level verify() convenience function
# ---------------------------------------------------------------------------

class TestModuleLevelVerify:

    def test_verify_convenience_function(self):
        """Uses a stub retriever to inject evidence directly so BM25's IDF=0
        behaviour in single-document corpora doesn't hide the LLM path."""
        evidence_passage = Passage("p1", "Earth orbits the Sun every year")
        fused = FusedSearchResult(
            passage_id="p1", rank=1, score=1.0, sources=["bm25"], passage=evidence_passage
        )

        class StubRetriever(HybridRetriever):
            def retrieve(self, queries, **kw):
                return [fused]

        retriever = StubRetriever(mode="bm25")
        client = MockLLMClient(canned_responses=[
            {"label": "True", "confidence": 0.88, "evidence_ids": ["p1"], "explanation": "p1 confirms."},
        ])
        cfg = PipelineConfig(
            retrieval_mode="bm25",
            save_verdicts=False,
            verification={"backend": "mock", "save_traces": False},
        )
        v = verify(
            claim="Earth orbits the Sun every year",
            claim_date=None,
            cfg=cfg,
            retriever=retriever,
            llm_client=client,
        )
        assert isinstance(v, Verdict)
        assert v.label == "True"
