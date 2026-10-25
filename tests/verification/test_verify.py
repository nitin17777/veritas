"""End-to-end unit tests for P6 EvidenceConstrainedVerifier."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from veritas.retrieval.schema import Passage, SearchResult
from veritas.verification.llm_client import MockLLMClient, TraceRecorder
from veritas.verification.verify import (
    EvidenceConstrainedVerifier,
    VerifierConfig,
    normalize_evidence_item,
)


def test_normalize_evidence_item_dict():
    item = {"id": "doc1", "text": "Evidence body", "score": 0.85, "date": "2022-01-01"}
    norm = normalize_evidence_item(item)
    assert norm["id"] == "doc1"
    assert norm["text"] == "Evidence body"
    assert norm["score"] == 0.85
    assert norm["date"] == "2022-01-01"


def test_normalize_evidence_item_passage():
    p = Passage(passage_id="p123", text="Passage text", domain="example.org")
    norm = normalize_evidence_item(p)
    assert norm["id"] == "p123"
    assert norm["text"] == "Passage text"
    assert norm["domain"] == "example.org"


def test_normalize_evidence_item_search_result():
    p = Passage(passage_id="res_1", text="Search result passage text")
    sr = SearchResult(passage_id="res_1", score=0.912, rank=1, passage=p)
    norm = normalize_evidence_item(sr)
    assert norm["id"] == "res_1"
    assert norm["text"] == "Search result passage text"
    assert norm["score"] == 0.912


def test_verify_empty_evidence_no_llm_call():
    """Requirement 8: Return Unverified without invoking the LLM if evidence is empty."""
    mock_client = MockLLMClient()
    verifier = EvidenceConstrainedVerifier(
        config=VerifierConfig(backend="mock", save_traces=False),
        client=mock_client,
    )

    res = verifier.verify(claim="Some claim", evidence=[])

    assert res.label == "Unverified"
    assert res.confidence == 0.0
    assert res.evidence_ids == []
    assert res.parse_failed is False
    assert "No evidence passages supplied" in res.explanation
    # LLM MUST NOT have been called
    assert len(mock_client.history) == 0


def test_verify_below_relevance_threshold_no_llm_call():
    """Requirement 8: Return Unverified without LLM call if all scores are below threshold."""
    mock_client = MockLLMClient()
    verifier = EvidenceConstrainedVerifier(
        config=VerifierConfig(
            backend="mock",
            min_relevance_score=0.5,
            save_traces=False,
        ),
        client=mock_client,
    )

    evidence = [
        {"id": "doc_low_1", "text": "Low score text 1", "score": 0.2},
        {"id": "doc_low_2", "text": "Low score text 2", "score": 0.35},
    ]

    res = verifier.verify(claim="Some claim", evidence=evidence)

    assert res.label == "Unverified"
    assert res.confidence == 0.0
    assert res.parse_failed is False
    assert "minimum relevance score threshold" in res.explanation
    # LLM MUST NOT have been called
    assert len(mock_client.history) == 0


def test_verify_successful_first_attempt():
    """Happy path: LLM returns valid JSON on attempt 1."""
    mock_response = {
        "label": "True",
        "confidence": 0.95,
        "evidence_ids": ["doc_1"],
        "explanation": "Document doc_1 directly confirms the statement with official statistics.",
    }
    mock_client = MockLLMClient(canned_responses=[mock_response])

    verifier = EvidenceConstrainedVerifier(
        config=VerifierConfig(backend="mock", save_traces=False),
        client=mock_client,
    )

    evidence = [
        {"id": "doc_1", "text": "Official report: unemployment reached 3.5%."},
        {"id": "doc_2", "text": "Other unrelated text."},
    ]

    res = verifier.verify(claim="Unemployment reached 3.5%.", evidence=evidence)

    assert res.label == "True"
    assert res.confidence == 0.95
    assert res.evidence_ids == ["doc_1"]
    assert res.explanation == mock_response["explanation"]
    assert res.parse_failed is False
    assert res.attempts == 1
    assert len(mock_client.history) == 1


def test_verify_retry_succeeds_after_malformed_first_response():
    """Requirement 7: Retry invalid output once. Succeeded on attempt 2."""
    malformed_first = "```json\n{ bad json syntax here ...\n```"
    valid_second = {
        "label": "False",
        "confidence": 0.88,
        "evidence_ids": ["doc_1"],
        "explanation": "Document doc_1 refutes the assertion directly.",
    }
    mock_client = MockLLMClient(canned_responses=[malformed_first, valid_second])

    verifier = EvidenceConstrainedVerifier(
        config=VerifierConfig(backend="mock", save_traces=False),
        client=mock_client,
    )

    evidence = [{"id": "doc_1", "text": "The claim is completely contradicted by records."}]

    res = verifier.verify(claim="False statement", evidence=evidence)

    assert res.label == "False"
    assert res.confidence == 0.88
    assert res.evidence_ids == ["doc_1"]
    assert res.parse_failed is False
    assert res.attempts == 2
    assert len(mock_client.history) == 2


def test_verify_retry_fails_both_attempts():
    """Requirement 7: If retry fails, return Unverified with parse_failed=True."""
    mock_client = MockLLMClient(
        canned_responses=[
            "Invalid response 1",
            "Invalid response 2",
        ]
    )

    verifier = EvidenceConstrainedVerifier(
        config=VerifierConfig(backend="mock", save_traces=False),
        client=mock_client,
    )

    evidence = [{"id": "doc_1", "text": "Some evidence text."}]

    res = verifier.verify(claim="Test statement", evidence=evidence)

    assert res.label == "Unverified"
    assert res.confidence == 0.0
    assert res.evidence_ids == []
    assert res.parse_failed is True
    assert res.attempts == 2
    assert "schema validation failed after retry" in res.explanation
    assert len(mock_client.history) == 2


def test_verify_rejects_unsupplied_evidence_ids():
    """Requirement 5: Reject evidence IDs that were not supplied."""
    # Attempt 1 returns an unsupplied ID; attempt 2 corrects it
    resp_with_hallucinated_id = {
        "label": "True",
        "confidence": 0.9,
        "evidence_ids": ["doc_1", "invented_id_999"],
        "explanation": "Supported according to documents.",
    }
    corrected_resp = {
        "label": "True",
        "confidence": 0.9,
        "evidence_ids": ["doc_1"],
        "explanation": "Supported according to document doc_1.",
    }
    mock_client = MockLLMClient(canned_responses=[resp_with_hallucinated_id, corrected_resp])

    verifier = EvidenceConstrainedVerifier(
        config=VerifierConfig(backend="mock", save_traces=False),
        client=mock_client,
    )

    evidence = [{"id": "doc_1", "text": "Genuine evidence text."}]

    res = verifier.verify(claim="Test statement", evidence=evidence)

    assert res.label == "True"
    assert res.evidence_ids == ["doc_1"]
    assert res.parse_failed is False
    assert res.attempts == 2


def test_verify_saves_raw_traces_to_disk(tmp_path):
    """Requirement 11: Save raw model outputs for later analysis and replay."""
    mock_resp = {
        "label": "Misleading",
        "confidence": 0.72,
        "evidence_ids": ["doc_m"],
        "explanation": "The claim omits critical context.",
    }
    mock_client = MockLLMClient(canned_responses=[mock_resp])

    recorder = TraceRecorder(trace_dir=tmp_path, enabled=True)
    verifier = EvidenceConstrainedVerifier(
        config=VerifierConfig(
            backend="mock",
            save_traces=True,
            trace_dir=str(tmp_path),
        ),
        client=mock_client,
        trace_recorder=recorder,
    )

    evidence = [{"id": "doc_m", "text": "Contextual evidence text."}]
    res = verifier.verify(claim="Misleading claim", evidence=evidence, claim_date="2024-06-01")

    assert res.label == "Misleading"

    # Verify trace file exists and contains valid JSON
    trace_files = list(tmp_path.glob("verification_traces_*.jsonl"))
    assert len(trace_files) >= 1
    lines = trace_files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    trace_record = json.loads(lines[0])
    assert trace_record["claim"] == "Misleading claim"
    assert trace_record["claim_date"] == "2024-06-01"
    assert trace_record["supplied_evidence_ids"] == ["doc_m"]
    assert trace_record["parsed_result"]["label"] == "Misleading"


def test_verify_batch():
    mock_client = MockLLMClient(
        canned_responses=[
            {"label": "True", "confidence": 0.9, "evidence_ids": ["e1"], "explanation": "Fact 1."},
            {"label": "False", "confidence": 0.8, "evidence_ids": ["e2"], "explanation": "Fact 2."},
        ]
    )
    verifier = EvidenceConstrainedVerifier(
        config=VerifierConfig(backend="mock", save_traces=False),
        client=mock_client,
    )

    items = [
        {"claim": "Claim 1", "evidence": [{"id": "e1", "text": "Evidence 1"}]},
        {"claim": "Claim 2", "evidence": [{"id": "e2", "text": "Evidence 2"}]},
    ]

    results = verifier.verify_batch(items)
    assert len(results) == 2
    assert results[0].label == "True"
    assert results[1].label == "False"
