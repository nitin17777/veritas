"""Unit tests for JSON extraction, code fence stripping, and schema validation."""

from __future__ import annotations

import json
import pytest

from veritas.verification.parse import (
    VerificationParseError,
    VerificationResult,
    VerificationValidationError,
    normalize_label,
    parse_and_validate_response,
    parse_raw_json,
    strip_code_fences,
    validate_confidence,
    validate_evidence_ids,
    validate_explanation,
    validate_verification_dict,
)


def test_strip_code_fences_plain():
    text = '{"label": "True", "confidence": 0.95}'
    assert strip_code_fences(text) == text


def test_strip_code_fences_markdown_json():
    text = '```json\n{"label": "True", "confidence": 0.95}\n```'
    assert strip_code_fences(text) == '{"label": "True", "confidence": 0.95}'


def test_strip_code_fences_with_surrounding_text():
    text = 'Here is your verification output:\n```json\n{"label": "False", "confidence": 0.8}\n```\nHope that helps!'
    assert strip_code_fences(text) == '{"label": "False", "confidence": 0.8}'


def test_strip_code_fences_unclosed():
    text = '```json\n{"label": "True", "confidence": 0.95}'
    assert strip_code_fences(text) == '{"label": "True", "confidence": 0.95}'


def test_parse_raw_json_valid():
    raw = '{"label": "True", "confidence": 0.9, "evidence_ids": ["p1"], "explanation": "Direct match."}'
    parsed = parse_raw_json(raw)
    assert parsed["label"] == "True"
    assert parsed["confidence"] == 0.9
    assert parsed["evidence_ids"] == ["p1"]


def test_parse_raw_json_with_trailing_commas():
    raw = '{\n  "label": "Partially True",\n  "confidence": 0.7,\n  "evidence_ids": ["p1",],\n  "explanation": "Mixed fact.",\n}'
    parsed = parse_raw_json(raw)
    assert parsed["label"] == "Partially True"
    assert parsed["confidence"] == 0.7


def test_parse_raw_json_empty_or_invalid():
    with pytest.raises(VerificationParseError):
        parse_raw_json("")

    with pytest.raises(VerificationParseError):
        parse_raw_json("   \n\t  ")

    with pytest.raises(VerificationParseError):
        parse_raw_json("Sorry, I cannot verify this claim.")


def test_normalize_label_valid():
    assert normalize_label("True") == "True"
    assert normalize_label("true") == "True"
    assert normalize_label("Partially True") == "Partially True"
    assert normalize_label("partially_true") == "Partially True"
    assert normalize_label("Misleading") == "Misleading"
    assert normalize_label("misleading") == "Misleading"
    assert normalize_label("False") == "False"
    assert normalize_label("false") == "False"
    assert normalize_label("Unverified") == "Unverified"
    assert normalize_label("unverified") == "Unverified"
    assert normalize_label("not enough info") == "Unverified"


def test_normalize_label_invalid():
    with pytest.raises(VerificationValidationError, match="Invalid label 'Pants on Fire'"):
        normalize_label("Pants on Fire")

    with pytest.raises(VerificationValidationError):
        normalize_label(123)


def test_validate_confidence_valid():
    assert validate_confidence(1.0) == 1.0
    assert validate_confidence(0.0) == 0.0
    assert validate_confidence(0.75) == 0.75
    assert validate_confidence("0.85") == 0.85


def test_validate_confidence_invalid():
    with pytest.raises(VerificationValidationError, match="outside valid interval"):
        validate_confidence(-0.1)

    with pytest.raises(VerificationValidationError, match="outside valid interval"):
        validate_confidence(1.05)

    with pytest.raises(VerificationValidationError, match="numeric"):
        validate_confidence("high")

    with pytest.raises(VerificationValidationError, match="numeric"):
        validate_confidence(True)  # bool is instance of int in Python, should be rejected


def test_validate_evidence_ids_valid():
    supplied = ["doc1", "doc2", "doc3"]
    assert validate_evidence_ids(["doc1", "doc3"], supplied) == ["doc1", "doc3"]
    assert validate_evidence_ids([], supplied) == []
    # Deduplicates
    assert validate_evidence_ids(["doc2", "doc2"], supplied) == ["doc2"]


def test_validate_evidence_ids_reject_unsupplied():
    supplied = ["doc1", "doc2"]
    with pytest.raises(VerificationValidationError, match="Rejected evidence IDs not present in supplied evidence"):
        validate_evidence_ids(["doc1", "fake_doc_999"], supplied)


def test_validate_explanation():
    assert validate_explanation("Valid reason.") == "Valid reason."
    with pytest.raises(VerificationValidationError):
        validate_explanation("")
    with pytest.raises(VerificationValidationError):
        validate_explanation("   ")
    with pytest.raises(VerificationValidationError):
        validate_explanation(None)


def test_validate_verification_dict_full():
    data = {
        "label": "True",
        "confidence": 0.98,
        "evidence_ids": ["doc_1"],
        "explanation": "Evidence doc_1 confirms the claim statement.",
    }
    res = validate_verification_dict(data, supplied_evidence_ids=["doc_1", "doc_2"])
    assert isinstance(res, VerificationResult)
    assert res.label == "True"
    assert res.confidence == 0.98
    assert res.evidence_ids == ["doc_1"]
    assert res.explanation == "Evidence doc_1 confirms the claim statement."
    assert res.parse_failed is False


def test_validate_verification_dict_missing_field():
    data = {
        "label": "True",
        # missing confidence
        "evidence_ids": ["doc_1"],
        "explanation": "Missing confidence.",
    }
    with pytest.raises(VerificationValidationError, match="Missing required field: 'confidence'"):
        validate_verification_dict(data, supplied_evidence_ids=["doc_1"])


def test_parse_and_validate_response_success():
    raw = """
    ```json
    {
      "label": "False",
      "confidence": 0.9,
      "evidence_ids": ["e1"],
      "explanation": "Document e1 directly refutes the statement."
    }
    ```
    """
    res, err = parse_and_validate_response(raw, supplied_evidence_ids=["e1", "e2"])
    assert err is None
    assert res is not None
    assert res.label == "False"
    assert res.confidence == 0.9
    assert res.evidence_ids == ["e1"]
    assert res.parse_failed is False


def test_parse_and_validate_response_failure():
    raw = "This is not JSON at all."
    res, err = parse_and_validate_response(raw, supplied_evidence_ids=["e1"])
    assert res is None
    assert err is not None
    assert "Failed to decode JSON" in err
