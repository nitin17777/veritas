"""Unit and integration tests for VERITAS P5 Claim Extraction and Query Generation.

All tests use MockLLMClient to guarantee zero live search and zero API costs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest

from veritas.extraction.extract import (
    ClaimExtractor,
    ExtractedClaim,
    ExtractionConfig,
    ExtractionResult,
    ExtractionValidationError,
    extract_claims,
)
from veritas.extraction.prompt import (
    DEFAULT_PROMPT_VERSION,
    build_retry_prompt,
    build_system_prompt,
    build_user_prompt,
)
from veritas.verification.llm_client import MockLLMClient


# ---------------------------------------------------------------------------
# Fixtures and Sample Data
# ---------------------------------------------------------------------------

SAMPLE_LONG_ARTICLE = (
    "On October 12, 2023, the Department of Energy announced a $3.5 billion investment "
    "across 44 states to strengthen the national electrical grid. Secretary Jennifer Granholm "
    "called the program the largest direct investment in grid infrastructure in modern history. "
    "In her speech, Granholm stated that power outages cost the U.S. economy $150 billion annually. "
    "Personally, I find the entire initiative to be thoroughly inspiring and wonderful. "
    "She further emphasized that the projects would create over 10,000 union jobs by 2026."
)

SAMPLE_VALID_PAYLOAD = {
    "claims": [
        {
            "claim": "The Department of Energy announced a $3.5 billion investment across 44 states for grid infrastructure on October 12, 2023.",
            "queries": [
                "Department of Energy 3.5 billion grid investment October 12 2023",
                "national electrical grid federal funding 44 states DOE",
            ],
            "context": "On October 12, 2023, the Department of Energy announced a $3.5 billion investment across 44 states",
            "reason": "Specific date, federal department, dollar figure ($3.5B), and state count (44).",
        },
        {
            "claim": "Power outages cost the United States economy approximately $150 billion annually according to the Department of Energy.",
            "queries": [
                "US economy annual cost power outages 150 billion",
                "Department of Energy annual electrical outage economic loss study",
                "Jennifer Granholm power outages 150 billion dollar cost",
            ],
            "context": "Granholm stated that power outages cost the U.S. economy $150 billion annually.",
            "reason": "Verifiable empirical statistic attributed to an official public statement.",
        },
        {
            "claim": "The Department of Energy grid initiative aims to create over 10,000 union jobs by 2026.",
            "queries": [
                "Department of Energy grid initiative 10000 union jobs 2026",
                "federal grid modern infrastructure projected employment jobs 2026",
            ],
            "context": "She further emphasized that the projects would create over 10,000 union jobs by 2026.",
            "reason": "Empirical employment target with specific date (2026) and job category.",
        },
    ]
}


# ---------------------------------------------------------------------------
# Core Extraction and Schema Tests
# ---------------------------------------------------------------------------


class TestClaimExtractionCore:
    def test_extract_valid_claims_success(self):
        """Test standard extraction returning 3 check-worthy claims with 2-3 queries each."""
        client = MockLLMClient(canned_responses=[json.dumps(SAMPLE_VALID_PAYLOAD)])
        config = ExtractionConfig(max_claims=5, model_name="test-llm")
        extractor = ClaimExtractor(config=config, llm_client=client)

        result = extractor.extract(SAMPLE_LONG_ARTICLE)

        assert isinstance(result, ExtractionResult)
        assert not result.parse_failed
        assert result.attempts == 1
        assert result.total_claims == 3
        assert len(result.claims) == 3
        assert result.model == "test-llm"
        assert result.error is None

        # Verify claims preservation and query bounds
        for claim_item in result.claims:
            assert isinstance(claim_item, ExtractedClaim)
            assert len(claim_item.claim) > 10
            assert 2 <= len(claim_item.queries) <= 3
            for q in claim_item.queries:
                assert isinstance(q, str) and len(q) > 0

    def test_extract_empty_or_whitespace_input(self):
        """Empty or whitespace text should return 0 claims immediately without calling LLM."""
        client = MockLLMClient()
        extractor = ClaimExtractor(llm_client=client)

        for empty_text in ["", "   ", "\n\t  \n"]:
            res = extractor.extract(empty_text)
            assert res.total_claims == 0
            assert res.claims == []
            assert res.attempts == 0
            assert not res.parse_failed
        # LLM client history must be empty (0 calls)
        assert len(client.history) == 0

    def test_extract_markdown_code_fences_stripped(self):
        """LLM response wrapped in ```json ... ``` code fences must parse cleanly."""
        fenced_payload = f"```json\n{json.dumps(SAMPLE_VALID_PAYLOAD)}\n```"
        client = MockLLMClient(canned_responses=[fenced_payload])
        extractor = ClaimExtractor(llm_client=client)

        res = extractor.extract(SAMPLE_LONG_ARTICLE)
        assert not res.parse_failed
        assert res.total_claims == 3

    def test_extract_direct_array_response(self):
        """LLM response formatted directly as JSON array [...] must be parsed."""
        array_payload = SAMPLE_VALID_PAYLOAD["claims"]
        client = MockLLMClient(canned_responses=[json.dumps(array_payload)])
        extractor = ClaimExtractor(llm_client=client)

        res = extractor.extract(SAMPLE_LONG_ARTICLE)
        assert not res.parse_failed
        assert res.total_claims == 3


# ---------------------------------------------------------------------------
# Check-Worthiness and Constraints Tests
# ---------------------------------------------------------------------------


class TestCheckWorthinessAndConstraints:
    def test_maximum_claims_capped_to_five(self):
        """When model returns more claims than max_claims (e.g. 6), result is safely capped."""
        many_claims = {
            "claims": [
                {
                    "claim": f"Factual statement number {i} with specific verifiable details.",
                    "queries": [f"query {i} a", f"query {i} b"],
                }
                for i in range(1, 8)
            ]
        }
        client = MockLLMClient(canned_responses=[json.dumps(many_claims)])
        config = ExtractionConfig(max_claims=5)
        extractor = ClaimExtractor(config=config, llm_client=client)

        res = extractor.extract(SAMPLE_LONG_ARTICLE)
        assert res.total_claims == 5
        assert len(res.claims) == 5

    def test_reject_pure_opinion_or_non_checkworthy(self):
        """Text containing no verifiable claims returns empty claim list."""
        empty_claims_payload = {"claims": []}
        client = MockLLMClient(canned_responses=[json.dumps(empty_claims_payload)])
        extractor = ClaimExtractor(llm_client=client)

        opinion_text = "I believe love is the greatest mystery in the universe and beauty is subjective."
        res = extractor.extract(opinion_text)

        assert not res.parse_failed
        assert res.total_claims == 0
        assert res.claims == []

    def test_preserves_numbers_dates_and_names(self):
        """Claims preserve essential factual tokens: numbers, dates, and proper nouns."""
        client = MockLLMClient(canned_responses=[json.dumps(SAMPLE_VALID_PAYLOAD)])
        extractor = ClaimExtractor(llm_client=client)

        res = extractor.extract(SAMPLE_LONG_ARTICLE)
        combined_claims = " ".join([c.claim for c in res.claims])

        # Essential factual details from article must appear in extracted claims
        assert "$3.5 billion" in combined_claims or "3.5" in combined_claims
        assert "October 12, 2023" in combined_claims or "2023" in combined_claims
        assert "Department of Energy" in combined_claims
        assert "150 billion" in combined_claims

    def test_pronoun_resolution_preserves_reference(self):
        """Pronoun 'She' was resolved to the unambiguous referent 'Department of Energy' or 'Granholm'."""
        client = MockLLMClient(canned_responses=[json.dumps(SAMPLE_VALID_PAYLOAD)])
        extractor = ClaimExtractor(llm_client=client)

        res = extractor.extract(SAMPLE_LONG_ARTICLE)
        # None of the atomic claims should start with an ambiguous dangling 'She'
        for c in res.claims:
            assert not c.claim.startswith("She ")
            assert not c.claim.startswith("He ")
            assert not c.claim.startswith("It ")


# ---------------------------------------------------------------------------
# Schema Validation and Self-Correction Retry Tests
# ---------------------------------------------------------------------------


class TestSchemaValidationAndRetry:
    def test_retry_on_malformed_json_syntax(self):
        """Malformed JSON on attempt 1 triggers automatic retry; valid JSON on attempt 2 succeeds."""
        bad_json = "{ claims: [ this is completely broken json } "
        good_json = json.dumps(SAMPLE_VALID_PAYLOAD)

        client = MockLLMClient(canned_responses=[bad_json, good_json])
        extractor = ClaimExtractor(llm_client=client)

        res = extractor.extract(SAMPLE_LONG_ARTICLE)

        assert res.attempts == 2
        assert not res.parse_failed
        assert res.total_claims == 3
        # Check that retry prompt was generated
        assert len(client.history) == 2
        retry_call = client.history[1]
        assert "Your previous response failed validation" in retry_call["prompt"]

    def test_retry_on_invalid_query_count(self):
        """Claim with fewer than 2 queries fails validation and triggers retry."""
        invalid_query_payload = {
            "claims": [
                {
                    "claim": "Valid factual statement with sufficient detail.",
                    "queries": ["only one query"],  # Invalid: requires at least 2 queries
                }
            ]
        }
        good_json = json.dumps(SAMPLE_VALID_PAYLOAD)

        client = MockLLMClient(canned_responses=[json.dumps(invalid_query_payload), good_json])
        extractor = ClaimExtractor(llm_client=client)

        res = extractor.extract(SAMPLE_LONG_ARTICLE)
        assert res.attempts == 2
        assert not res.parse_failed
        assert res.total_claims == 3

    def test_unrecoverable_failure_after_retry(self):
        """When both attempt 1 and retry fail, result records parse_failed=True without crashing."""
        broken_1 = "Sorry, I cannot produce JSON."
        broken_2 = "{ still invalid json"

        client = MockLLMClient(canned_responses=[broken_1, broken_2])
        extractor = ClaimExtractor(llm_client=client)

        res = extractor.extract(SAMPLE_LONG_ARTICLE)

        assert res.attempts == 2
        assert res.parse_failed is True
        assert res.total_claims == 0
        assert res.claims == []
        assert res.error is not None
        assert "Retry failed" in res.error


# ---------------------------------------------------------------------------
# Configurability and Helpers Tests
# ---------------------------------------------------------------------------


class TestConfigurabilityAndHelpers:
    def test_config_from_dict_and_overrides(self):
        """ExtractionConfig can be instantiated from dictionary with custom parameters."""
        cfg_dict = {
            "model_name": "custom-gpt-model",
            "temperature": 0.3,
            "max_claims": 4,
            "min_queries": 2,
            "max_queries": 3,
            "prompt_version": "v2.0-beta",
        }
        config = ExtractionConfig.from_dict(cfg_dict)
        assert config.model_name == "custom-gpt-model"
        assert config.temperature == 0.3
        assert config.max_claims == 4
        assert config.prompt_version == "v2.0-beta"

    def test_convenience_function_extract_claims(self):
        """Top-level extract_claims() helper operates correctly with overrides."""
        client = MockLLMClient(canned_responses=[json.dumps(SAMPLE_VALID_PAYLOAD)])

        res = extract_claims(
            text=SAMPLE_LONG_ARTICLE,
            llm_client=client,
            max_claims=3,
            temperature=0.0,
        )

        assert isinstance(res, ExtractionResult)
        assert res.total_claims == 3
        assert res.temperature == 0.0

    def test_prompt_builders(self):
        """Prompt builders correctly format max_claims and include source text."""
        sys_p = build_system_prompt(max_claims=4, prompt_version="v1.0")
        assert "up to 4 distinct, check-worthy factual claims" in sys_p
        assert "2 to 3 targeted search queries" in sys_p

        usr_p = build_user_prompt(text="Source text here", max_claims=4)
        assert "--- SOURCE TEXT ---" in usr_p
        assert "Source text here" in usr_p

        retry_p = build_retry_prompt(error_message="Missing queries list", raw_output="{}", max_claims=4)
        assert "Missing queries list" in retry_p
        assert "up to 4 claims" in retry_p

    def test_serialization(self):
        """ExtractionResult and ExtractedClaim serialize to dictionaries without error."""
        claim = ExtractedClaim(
            claim_id="c1",
            claim="Factual claim text.",
            queries=["query 1", "query 2"],
            context="source context",
            reason="check-worthy metric",
        )
        claim_dict = claim.to_dict()
        assert claim_dict["claim_id"] == "c1"
        assert claim_dict["queries"] == ["query 1", "query 2"]

        result = ExtractionResult(
            claims=[claim],
            source_text="source",
            total_claims=1,
            model="mock-model",
            prompt_version="v1.0",
            temperature=0.0,
        )
        res_dict = result.to_dict()
        assert res_dict["total_claims"] == 1
        assert len(res_dict["claims"]) == 1
        assert json.dumps(res_dict)  # Ensures valid JSON serializable
