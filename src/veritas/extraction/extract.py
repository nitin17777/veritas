"""VERITAS P5 Claim Extraction and Query Generation Module.

Extracts check-worthy factual claims from long text inputs (articles, social media posts,
transcripts, or reports) and generates 2 to 3 targeted retrieval search queries per claim.

Key design principles:
- Check-worthiness: extracts only objectively verifiable factual statements;
  rejects subjective opinions, pure speculation, and unverifiable commentary.
- Preserves names, numbers, dates, statistics, and essential factual relationships.
- Resolves pronouns only when references are clear and unambiguous in source text.
- Validates structured JSON schema with automatic single-retry self-correction.
- Configurable model name, prompt version, temperature, and maximum claims.
- Zero network/search side effects in unit testing via mock client injection.
- Benchmark claims (already atomic) can bypass this extraction layer.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from veritas.config import ROOT_DIR, load_config
from veritas.extraction.prompt import (
    DEFAULT_PROMPT_VERSION,
    build_retry_prompt,
    build_system_prompt,
    build_user_prompt,
)
from veritas.verification.llm_client import (
    BaseLLMClient,
    LLMResponse,
    MockLLMClient,
    create_llm_client,
)

logger = logging.getLogger("veritas.extraction.extract")


class ExtractionError(Exception):
    """Base exception for claim extraction failures."""


class ExtractionValidationError(ExtractionError):
    """Raised when extracted claims fail schema or semantic validation."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ExtractedClaim:
    """An individual check-worthy factual claim extracted from source text.

    Attributes:
        claim_id: Unique identifier for this extracted claim (e.g., 'claim_1').
        claim: Self-contained, atomic factual claim string.
        queries: 2 to 3 focused search queries for retrieval indexing.
        context: Relevant excerpt or quote from the source text.
        reason: Justification of why the statement is check-worthy.
        metadata: Optional provenance or extraction details.
    """

    claim_id: str
    claim: str
    queries: list[str]
    context: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class ExtractionConfig:
    """Configuration for claim extraction and query generation.

    Attributes:
        model_name: Name or path of the language model to use.
        backend: Backend provider ('mock', 'api', 'local').
        prompt_version: Version identifier for prompt templates.
        temperature: Generation temperature (0.0 for deterministic).
        max_claims: Maximum number of check-worthy claims to extract (default: 5).
        min_queries: Minimum search queries required per claim (default: 2).
        max_queries: Maximum search queries allowed per claim (default: 3).
        max_tokens: Maximum tokens for LLM generation.
        timeout_seconds: Request timeout for API calls.
        api_key: Optional API key for remote backends.
        api_base_url: Optional base URL for API endpoints.
        save_traces: Whether to log raw model traces to disk.
        trace_dir: Output path for trace logging.
    """

    model_name: str = "mock-model"
    backend: str = "mock"
    prompt_version: str = DEFAULT_PROMPT_VERSION
    temperature: float = 0.0
    max_claims: int = 5
    min_queries: int = 2
    max_queries: int = 3
    max_tokens: int = 1536
    timeout_seconds: float = 30.0
    api_key: str | None = None
    api_base_url: str | None = None
    save_traces: bool = False
    trace_dir: str | Path | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtractionConfig:
        """Create ExtractionConfig from a dictionary with validation."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> ExtractionConfig:
        """Load ExtractionConfig from a YAML file."""
        cfg_dict = load_config(yaml_path)
        extraction_dict = cfg_dict.get("extraction", cfg_dict)
        return cls.from_dict(extraction_dict)


@dataclass
class ExtractionResult:
    """Result of claim extraction on a source text.

    Attributes:
        claims: List of extracted check-worthy claims with queries.
        source_text: The original input text.
        total_claims: Number of claims extracted.
        model: Model identifier used for extraction.
        prompt_version: Version of the prompt template used.
        temperature: Generation temperature used.
        attempts: Number of LLM generation attempts (1 for initial, 2 if retried).
        raw_response: Raw model output string from the final attempt.
        parse_failed: True if the model output could not be parsed after retries.
        error: Diagnostic error message if extraction failed or aborted.
        metadata: Additional diagnostic info (timing, token usage, etc.).
    """

    claims: list[ExtractedClaim]
    source_text: str
    total_claims: int
    model: str
    prompt_version: str
    temperature: float
    attempts: int = 1
    raw_response: str | None = None
    parse_failed: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "claims": [c.to_dict() for c in self.claims],
            "source_text": self.source_text,
            "total_claims": self.total_claims,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "temperature": self.temperature,
            "attempts": self.attempts,
            "raw_response": self.raw_response,
            "parse_failed": self.parse_failed,
            "error": self.error,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Extraction Engine
# ---------------------------------------------------------------------------


class ClaimExtractor:
    """Orchestrates LLM-based claim extraction, query generation, and validation.

    Enforces:
    - Closed-loop schema validation with automatic single retry.
    - Check-worthiness filtering (rejecting non-verifiable opinions).
    - Preservation of names, numbers, dates, and core factual semantics.
    - Bounded query generation (2 to 3 search queries per claim).
    - Conservative pronoun resolution (only when antecedent is clear).
    """

    def __init__(
        self,
        config: ExtractionConfig | dict[str, Any] | None = None,
        llm_client: BaseLLMClient | None = None,
    ) -> None:
        if isinstance(config, dict):
            self.config = ExtractionConfig.from_dict(config)
        elif isinstance(config, ExtractionConfig):
            self.config = config
        else:
            self.config = ExtractionConfig()

        if llm_client is not None:
            self.client = llm_client
        else:
            self.client = create_llm_client(
                backend=self.config.backend,
                model_name=self.config.model_name,
                api_key=self.config.api_key,
                api_base_url=self.config.api_base_url,
                timeout=self.config.timeout_seconds,
            )

    def extract(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> ExtractionResult:
        """Extract check-worthy claims and queries from input text.

        Args:
            text: Source text (article, speech, social media post).
            metadata: Optional additional metadata to store in result.

        Returns:
            ExtractionResult containing extracted claims or error diagnostics.
        """
        source = text.strip() if text else ""
        meta = dict(metadata or {})

        # Guard: Empty or whitespace-only input
        if not source:
            logger.info("Empty input text provided for extraction; returning 0 claims.")
            return ExtractionResult(
                claims=[],
                source_text=source,
                total_claims=0,
                model=self.config.model_name,
                prompt_version=self.config.prompt_version,
                temperature=self.config.temperature,
                attempts=0,
                raw_response=None,
                parse_failed=False,
                error="Input text is empty or whitespace.",
                metadata=meta,
            )

        # Build initial prompts
        system_prompt = build_system_prompt(
            max_claims=self.config.max_claims,
            prompt_version=self.config.prompt_version,
        )
        user_prompt = build_user_prompt(
            text=source,
            max_claims=self.config.max_claims,
        )

        # Attempt 1: Initial model call
        attempt = 1
        raw_response: str | None = None
        last_error: str | None = None

        try:
            resp = self.client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            raw_response = resp.content
            claims = self._parse_and_validate(raw_response, source)

            logger.info(
                "Successfully extracted %d claims on attempt 1",
                len(claims),
            )
            return ExtractionResult(
                claims=claims,
                source_text=source,
                total_claims=len(claims),
                model=self.config.model_name,
                prompt_version=self.config.prompt_version,
                temperature=self.config.temperature,
                attempts=1,
                raw_response=raw_response,
                parse_failed=False,
                metadata=meta,
            )

        except (json.JSONDecodeError, ExtractionValidationError, ValueError) as exc:
            last_error = str(exc)
            logger.warning(
                "Extraction attempt 1 failed validation: %s. Initiating single retry...",
                last_error,
            )

        # Attempt 2: Corrective retry
        attempt = 2
        try:
            retry_prompt = build_retry_prompt(
                error_message=last_error or "Invalid JSON response",
                raw_output=raw_response or "",
                max_claims=self.config.max_claims,
            )
            resp = self.client.generate(
                prompt=retry_prompt,
                system_prompt=system_prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            raw_response = resp.content
            claims = self._parse_and_validate(raw_response, source)

            logger.info(
                "Successfully extracted %d claims on retry (attempt 2)",
                len(claims),
            )
            return ExtractionResult(
                claims=claims,
                source_text=source,
                total_claims=len(claims),
                model=self.config.model_name,
                prompt_version=self.config.prompt_version,
                temperature=self.config.temperature,
                attempts=2,
                raw_response=raw_response,
                parse_failed=False,
                metadata=meta,
            )

        except Exception as retry_exc:
            final_error = f"Attempt 1 failed ({last_error}); Retry failed ({retry_exc})"
            logger.error("Extraction failed after retry: %s", final_error)
            return ExtractionResult(
                claims=[],
                source_text=source,
                total_claims=0,
                model=self.config.model_name,
                prompt_version=self.config.prompt_version,
                temperature=self.config.temperature,
                attempts=2,
                raw_response=raw_response,
                parse_failed=True,
                error=final_error,
                metadata=meta,
            )

    # -----------------------------------------------------------------------
    # Parsing and Validation Helpers
    # -----------------------------------------------------------------------

    def _strip_markdown_code_fences(self, text: str) -> str:
        """Strip markdown ```json code fences from LLM responses."""
        s = text.strip()
        # Strip ```json ... ``` or ``` ... ```
        if s.startswith("```"):
            lines = s.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            s = "\n".join(lines).strip()
        return s

    def _extract_first_json_object(self, text: str) -> str:
        """Locate outermost JSON object {...} or array [...] in string."""
        s = self._strip_markdown_code_fences(text)
        first_brace = s.find("{")
        first_bracket = s.find("[")

        if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
            last_bracket = s.rfind("]")
            if last_bracket != -1 and last_bracket > first_bracket:
                return s[first_bracket : last_bracket + 1].strip()

        if first_brace != -1:
            last_brace = s.rfind("}")
            if last_brace != -1 and last_brace > first_brace:
                return s[first_brace : last_brace + 1].strip()

        return s

    def _parse_and_validate(self, raw_text: str, source_text: str) -> list[ExtractedClaim]:
        """Parse raw LLM response into validated ExtractedClaim objects."""
        if not raw_text or not raw_text.strip():
            raise ExtractionValidationError("Model returned empty or whitespace completion.")

        cleaned = self._extract_first_json_object(raw_text)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ExtractionValidationError(f"Invalid JSON syntax: {exc}") from exc

        # Handle parsed root structure: either {"claims": [...]} or direct list [...]
        raw_items: list[Any] = []
        if isinstance(parsed, dict):
            if "claims" in parsed:
                raw_items = parsed["claims"]
            elif "extracted_claims" in parsed:
                raw_items = parsed["extracted_claims"]
            else:
                raise ExtractionValidationError(
                    "JSON object must contain a 'claims' array field."
                )
        elif isinstance(parsed, list):
            raw_items = parsed
        else:
            raise ExtractionValidationError(
                f"Expected JSON object with 'claims' or array, got {type(parsed).__name__}."
            )

        if not isinstance(raw_items, list):
            raise ExtractionValidationError(
                f"'claims' field must be a list, got {type(raw_items).__name__}."
            )

        # Enforce maximum claims constraint
        if len(raw_items) > self.config.max_claims:
            logger.warning(
                "Model returned %d claims; capping to configured maximum of %d.",
                len(raw_items),
                self.config.max_claims,
            )
            raw_items = raw_items[: self.config.max_claims]

        validated_claims: list[ExtractedClaim] = []
        for idx, item in enumerate(raw_items, start=1):
            claim_obj = self._validate_single_claim(item, idx, source_text)
            if claim_obj is not None:
                validated_claims.append(claim_obj)

        return validated_claims

    def _validate_single_claim(
        self,
        item: Any,
        index: int,
        source_text: str,
    ) -> ExtractedClaim | None:
        """Validate a single claim item against schema constraints."""
        if not isinstance(item, dict):
            raise ExtractionValidationError(
                f"Claim item #{index} must be a dictionary object, got {type(item).__name__}."
            )

        # Claim statement text
        raw_claim = item.get("claim")
        if not isinstance(raw_claim, str) or not raw_claim.strip():
            raise ExtractionValidationError(
                f"Claim item #{index} is missing a non-empty 'claim' string."
            )
        claim_str = raw_claim.strip()

        # Reject claims that are excessively short or trivial
        if len(claim_str.split()) < 3:
            raise ExtractionValidationError(
                f"Claim #{index} ('{claim_str}') is too short to constitute a check-worthy claim."
            )

        # Queries validation (must be list of 2 or 3 strings)
        raw_queries = item.get("queries")
        if not isinstance(raw_queries, list):
            raise ExtractionValidationError(
                f"Claim #{index} is missing a 'queries' list."
            )

        clean_queries = [
            q.strip()
            for q in raw_queries
            if isinstance(q, str) and q.strip()
        ]

        if len(clean_queries) < self.config.min_queries:
            raise ExtractionValidationError(
                f"Claim #{index} has {len(clean_queries)} valid queries; "
                f"minimum required is {self.config.min_queries}."
            )

        if len(clean_queries) > self.config.max_queries:
            # Safely cap to max_queries
            clean_queries = clean_queries[: self.config.max_queries]

        context = item.get("context")
        if context is not None and not isinstance(context, str):
            context = str(context)

        reason = item.get("reason")
        if reason is not None and not isinstance(reason, str):
            reason = str(reason)

        claim_id = item.get("claim_id") or f"extracted_claim_{index}"

        return ExtractedClaim(
            claim_id=str(claim_id),
            claim=claim_str,
            queries=clean_queries,
            context=context.strip() if context else None,
            reason=reason.strip() if reason else None,
        )


# ---------------------------------------------------------------------------
# Convenience Function
# ---------------------------------------------------------------------------


def extract_claims(
    text: str,
    config: ExtractionConfig | dict[str, Any] | None = None,
    llm_client: BaseLLMClient | None = None,
    max_claims: int | None = None,
    temperature: float | None = None,
) -> ExtractionResult:
    """Extract check-worthy factual claims from long text.

    Args:
        text: Source text (article, post, speech).
        config: ExtractionConfig or dictionary of configuration parameters.
        llm_client: Optional custom LLM client instance (useful for testing).
        max_claims: Optional override for max_claims limit.
        temperature: Optional override for generation temperature.

    Returns:
        ExtractionResult containing extracted claims and retrieval queries.
    """
    if config is None:
        cfg = ExtractionConfig()
    elif isinstance(config, dict):
        cfg = ExtractionConfig.from_dict(config)
    else:
        cfg = config

    if max_claims is not None:
        cfg.max_claims = max_claims
    if temperature is not None:
        cfg.temperature = temperature

    extractor = ClaimExtractor(config=cfg, llm_client=llm_client)
    return extractor.extract(text)
