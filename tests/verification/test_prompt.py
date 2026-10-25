"""Unit tests for prompt formatting and templates in VERITAS verification."""

from __future__ import annotations

import json
import pytest

from veritas.data_prep.schema import VALID_LABELS
from veritas.verification.prompt import (
    DEFAULT_LABEL_DEFINITIONS,
    build_retry_prompt,
    build_system_prompt,
    build_verification_prompt,
    format_evidence_block,
    format_label_definitions,
)


def test_format_label_definitions():
    formatted = format_label_definitions()
    for label in VALID_LABELS:
        assert f'- "{label}":' in formatted
        assert len(DEFAULT_LABEL_DEFINITIONS[label]) > 0


def test_build_system_prompt():
    system_prompt = build_system_prompt()

    # Closed-world constraint
    assert "EVIDENCE-CONSTRAINED ONLY" in system_prompt
    assert "rely EXCLUSIVELY on the provided evidence" in system_prompt
    assert "Do NOT use external knowledge" in system_prompt

    # Citation requirement
    assert "CITATION REQUIREMENT" in system_prompt
    assert "exclusively from the supplied evidence passages" in system_prompt

    # When to return Unverified
    assert "WHEN TO RETURN 'Unverified'" in system_prompt
    assert "abstain" in system_prompt.lower()

    # JSON only requirement
    assert "JSON-ONLY OUTPUT" in system_prompt

    # All 5 labels
    for label in VALID_LABELS:
        assert label in system_prompt


def test_format_evidence_block_empty():
    assert format_evidence_block([]) == "[No evidence passages supplied]"


def test_format_evidence_block_with_metadata():
    items = [
        {
            "id": "doc_1",
            "text": "The Eiffel Tower was completed in 1889.",
            "date": "2021-05-01",
            "domain": "britannica.com",
            "score": 0.8912,
        },
        {
            "id": "doc_2",
            "text": "It stands in Paris, France.",
        },
    ]
    block = format_evidence_block(items)
    assert "[1] Evidence ID: doc_1 (Date: 2021-05-01, Source: britannica.com, Retrieval Score: 0.8912)" in block
    assert "Passage: The Eiffel Tower was completed in 1889." in block
    assert "[2] Evidence ID: doc_2" in block
    assert "Passage: It stands in Paris, France." in block


def test_build_verification_prompt_with_date():
    claim = "Unemployment dropped to 3.5% in April 2023."
    date = "2023-05-02"
    evidence = [
        {"id": "ev_101", "text": "Bureau of Labor Statistics reported 3.5% unemployment rate for April 2023."},
        {"id": "ev_102", "text": "The economy added 253,000 jobs in April."},
    ]

    prompt = build_verification_prompt(claim, evidence, claim_date=date)

    assert f'CLAIM TO VERIFY:\n"{claim}"\nClaim Date: {date}' in prompt
    assert "Evidence ID: ev_101" in prompt
    assert "Evidence ID: ev_102" in prompt
    assert '["ev_101", "ev_102"]' in prompt
    assert "TASK INSTRUCTIONS:" in prompt
    assert "Select one of the 5 allowed labels" in prompt


def test_build_verification_prompt_without_date():
    claim = "The Moon orbits the Earth."
    evidence = [{"id": "moon_1", "text": "The Moon is Earth's only natural satellite."}]

    prompt = build_verification_prompt(claim, evidence, claim_date=None)
    assert "Claim Date:" not in prompt
    assert claim in prompt
    assert "moon_1" in prompt


def test_build_retry_prompt():
    orig_prompt = "ORIGINAL_TASK_CONTENT"
    malformed_resp = "{ bad json "
    error_msg = "Failed to decode JSON from model response"
    valid_ids = ["p1", "p2"]

    retry = build_retry_prompt(orig_prompt, malformed_resp, error_msg, valid_ids)

    assert "Your previous response was INVALID" in retry
    assert error_msg in retry
    assert malformed_resp in retry
    assert '["p1", "p2"]' in retry
    assert orig_prompt in retry
    assert "Allowed labels are strictly:" in retry
