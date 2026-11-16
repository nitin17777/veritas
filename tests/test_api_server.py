"""Unit tests for VERITAS REST API server endpoints (app/api_server.py)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from app.api_server import (
    determine_stance,
    enrich_verification_response,
    get_credibility_tier,
    get_engine,
    load_history,
    normalize_verdict,
)


def test_normalize_verdict() -> None:
    assert normalize_verdict("True") == "TRUE"
    assert normalize_verdict("mostly_true") == "TRUE"
    assert normalize_verdict("Half-True") == "PARTIALLY TRUE"
    assert normalize_verdict("Misleading") == "MISLEADING"
    assert normalize_verdict("False") == "FALSE"
    assert normalize_verdict("pants on fire") == "FALSE"
    assert normalize_verdict("Unverified") == "UNVERIFIED"
    assert normalize_verdict("unknown") == "UNVERIFIED"


def test_get_credibility_tier() -> None:
    assert get_credibility_tier("cdc.gov", 0.95) == "Primary source"
    assert get_credibility_tier("reuters.com", 0.90) == "Tier 1: Wire Agency / Major Institution"
    assert get_credibility_tier("news.example.com", 0.75) == "Tier 2: Established News"
    assert get_credibility_tier("blog.example.com", 0.55) == "Tier 3: Mainstream / General"
    assert get_credibility_tier("sketchy.info", 0.30) == "Tier 4: Low Editorial Standard"


def test_determine_stance() -> None:
    assert determine_stance("TRUE", is_cited=True, rank=1) == "Supports claim"
    assert determine_stance("FALSE", is_cited=True, rank=1) == "Contradicts claim"
    assert determine_stance("MISLEADING", is_cited=True, rank=1) == "Contradicts claim"
    assert determine_stance("MISLEADING", is_cited=False, rank=4) == "Provides context"
    assert determine_stance("UNVERIFIED", is_cited=False, rank=1) == "Provides context"


def test_load_history() -> None:
    history = load_history()
    assert isinstance(history, list)
    assert len(history) > 0
    first = history[0]
    assert "claim" in first
    assert "verdict" in first
    assert "confidence" in first
    assert "result" in first


def test_verify_and_enrich() -> None:
    engine = get_engine("mock")
    claim = "COVID-19 mRNA vaccines significantly reduce the risk of severe hospitalizations."
    res = engine.verify_claim(claim=claim, mode="benchmark")
    enriched = enrich_verification_response(res)

    assert enriched["claim"] == claim
    assert enriched["verdict"] == "TRUE"
    assert enriched["confidence"] >= 80
    assert len(enriched["evidence_cards"]) > 0
    assert enriched["is_supported"] is True
    assert enriched["nli_status"] == "Passed"
    assert "relevance" in enriched["evidence_scoring"]
    assert "combined" in enriched["evidence_scoring"]
