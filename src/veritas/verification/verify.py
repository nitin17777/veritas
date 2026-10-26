"""Evidence-constrained LLM verifier (P6) for VERITAS.

Verifies claims strictly against supplied evidence passages, adhering to:
- Evidence-only closed-world constraints.
- Explicit 5-class VERITAS labels.
- Structured JSON output parsing.
- Single-retry mechanism for invalid outputs.
- No-op abstention when retrieval yields insufficient evidence.
- Full trace persistence for reproducibility and replay.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
from typing import Any

from veritas.config import ROOT_DIR
from veritas.data_prep.schema import VALID_LABELS
from veritas.verification.llm_client import (
    BaseLLMClient,
    TraceRecorder,
    create_llm_client,
)
from veritas.verification.parse import (
    VerificationResult,
    parse_and_validate_response,
)
from veritas.verification.prompt import (
    DEFAULT_LABEL_DEFINITIONS,
    build_free_verification_prompt,
    build_retry_prompt,
    build_system_prompt,
    build_verification_prompt,
)

logger = logging.getLogger("veritas.verification.verify")


@dataclass
class VerifierConfig:
    """Configuration options for the evidence-constrained verifier."""

    backend: str = "mock"  # "mock", "api", "local"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 512
    abstain_label: str = "Unverified"
    min_relevance_score: float | None = None
    max_evidence_passages: int = 10
    save_traces: bool = True
    trace_dir: str = "results/verification_traces"
    label_definitions: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_LABEL_DEFINITIONS)
    )
    prompt_mode: str = "evidence_constrained"  # "evidence_constrained" or "free"
    allow_no_evidence: bool = False
    api_base: str = "https://api.openai.com/v1"
    api_key_env_var: str = "OPENAI_API_KEY"
    api_key: str | None = None
    provider: str = "openai"
    timeout: float = 60.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerifierConfig:
        """Create VerifierConfig from dictionary, handling nested verification section if present."""
        v_data = data.get("verification", data) if "verification" in data else data
        return cls(
            backend=str(v_data.get("backend", v_data.get("model_backend", "mock"))),
            model=str(v_data.get("model", v_data.get("model_name", "gpt-4o-mini"))),
            temperature=float(v_data.get("temperature", 0.0)),
            max_tokens=int(v_data.get("max_tokens", 512)),
            abstain_label=str(v_data.get("abstain_label", "Unverified")),
            min_relevance_score=(
                float(v_data["min_relevance_score"])
                if v_data.get("min_relevance_score") is not None
                else None
            ),
            max_evidence_passages=int(v_data.get("max_evidence_passages", 10)),
            save_traces=bool(v_data.get("save_traces", True)),
            trace_dir=str(v_data.get("trace_dir", "results/verification_traces")),
            label_definitions=dict(
                v_data.get("label_definitions", DEFAULT_LABEL_DEFINITIONS)
            ),
            prompt_mode=str(v_data.get("prompt_mode", "evidence_constrained")),
            allow_no_evidence=bool(v_data.get("allow_no_evidence", False)),
            api_base=str(v_data.get("api_base", "https://api.openai.com/v1")),
            api_key_env_var=str(v_data.get("api_key_env_var", "OPENAI_API_KEY")),
            api_key=v_data.get("api_key"),
            provider=str(v_data.get("provider", "openai")),
            timeout=float(v_data.get("timeout", 60.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_evidence_item(item: Any) -> dict[str, Any]:
    """Normalize a Passage, SearchResult, FusedSearchResult, or dict into a standard dict."""
    if hasattr(item, "passage") and getattr(item, "passage") is not None:
        # SearchResult or FusedSearchResult with nested passage
        passage = getattr(item, "passage")
        pid = getattr(item, "passage_id", None) or getattr(passage, "passage_id", None) or getattr(passage, "id", None)
        text = getattr(passage, "text", "")
        score = getattr(item, "score", None)
        date = getattr(passage, "date", None) or getattr(passage, "publication_date", None)
        domain = getattr(passage, "domain", None)
        url = getattr(passage, "url", None)
    elif hasattr(item, "passage_id") or hasattr(item, "id") or hasattr(item, "text"):
        pid = getattr(item, "passage_id", None) or getattr(item, "id", None)
        text = getattr(item, "text", "")
        score = getattr(item, "score", None)
        date = getattr(item, "date", None) or getattr(item, "publication_date", None)
        domain = getattr(item, "domain", None)
        url = getattr(item, "url", None)
    elif isinstance(item, dict):
        pid = item.get("passage_id") or item.get("id")
        text = item.get("text", "")
        score = item.get("score")
        date = item.get("date") or item.get("publication_date")
        domain = item.get("domain")
        url = item.get("url")
    else:
        pid = str(hash(str(item)))
        text = str(item)
        score = None
        date = None
        domain = None
        url = None

    return {
        "id": str(pid).strip() if pid else "",
        "text": str(text).strip() if text else "",
        "score": float(score) if score is not None else None,
        "date": str(date).strip() if date else None,
        "domain": str(domain).strip() if domain else None,
        "url": str(url).strip() if url else None,
    }


class EvidenceConstrainedVerifier:
    """VERITAS Phase 6 Evidence-Constrained LLM Verifier."""

    def __init__(
        self,
        config: VerifierConfig | dict[str, Any] | None = None,
        client: BaseLLMClient | None = None,
        trace_recorder: TraceRecorder | None = None,
    ) -> None:
        if config is None:
            self.config = VerifierConfig()
        elif isinstance(config, VerifierConfig):
            self.config = config
        else:
            self.config = VerifierConfig.from_dict(config)

        # Initialize LLM client
        if client is not None:
            self.client = client
        else:
            self.client = create_llm_client(
                backend=self.config.backend,
                model_name=self.config.model,
                config=self.config.to_dict(),
            )

        # Initialize trace recorder
        if trace_recorder is not None:
            self.trace_recorder = trace_recorder
        else:
            trace_path = Path(self.config.trace_dir)
            if not trace_path.is_absolute():
                trace_path = ROOT_DIR / trace_path
            self.trace_recorder = TraceRecorder(
                trace_dir=trace_path,
                enabled=self.config.save_traces,
            )

        self.system_prompt = build_system_prompt(self.config.label_definitions)

    def _filter_and_check_evidence(
        self,
        evidence: list[Any] | None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Validate and filter evidence items according to relevance thresholds.

        Requirement 8: If retrieval returns no sufficiently relevant evidence,
        return Unverified without invoking the LLM.
        """
        if not evidence:
            return [], "No evidence passages supplied."

        normalized: list[dict[str, Any]] = []
        for raw_item in evidence:
            norm = normalize_evidence_item(raw_item)
            if norm["id"] and norm["text"]:
                normalized.append(norm)

        if not normalized:
            return [], "Supplied evidence passages contained no valid text or identifiers."

        # Cap at max_evidence_passages
        if self.config.max_evidence_passages > 0:
            normalized = normalized[: self.config.max_evidence_passages]

        # Check relevance score threshold if specified
        if self.config.min_relevance_score is not None:
            threshold = self.config.min_relevance_score
            has_scores = any(item.get("score") is not None for item in normalized)
            if has_scores:
                filtered = [
                    item
                    for item in normalized
                    if item.get("score") is not None and item["score"] >= threshold
                ]
                if not filtered:
                    return [], (
                        f"No retrieved passages met the minimum relevance score threshold "
                        f"({threshold}). Highest score was "
                        f"{max((item['score'] for item in normalized if item.get('score') is not None), default=0.0):.4f}."
                    )
                normalized = filtered

        return normalized, None

    def verify(
        self,
        claim: str,
        evidence: list[Any] | None = None,
        claim_date: str | None = None,
    ) -> VerificationResult:
        """Verify a single claim against retrieved evidence passages.

        Args:
            claim: Factual statement to evaluate.
            evidence: List of Passage, SearchResult, or dict instances.
            claim_date: Optional temporal context (e.g. ISO date "2020-04-15").

        Returns:
            Structured VerificationResult.
        """
        claim_clean = str(claim).strip() if claim else ""
        if not claim_clean:
            return VerificationResult(
                label=self.config.abstain_label,
                confidence=0.0,
                evidence_ids=[],
                explanation="Claim statement was empty or missing.",
                parse_failed=False,
            )

        # 1. Relevance and evidence check (Requirement 8)
        valid_evidence, reject_reason = self._filter_and_check_evidence(evidence)
        can_proceed_without_evidence = self.config.allow_no_evidence or (self.config.prompt_mode == "free")

        if reject_reason is not None and not can_proceed_without_evidence:
            logger.info("Verification abstained without LLM call: %s", reject_reason)
            return VerificationResult(
                label=self.config.abstain_label,
                confidence=0.0,
                evidence_ids=[],
                explanation=reject_reason,
                parse_failed=False,
            )

        supplied_ids = [item["id"] for item in valid_evidence]
        is_free_prompt = self.config.prompt_mode == "free"

        # 2. Build initial user prompt
        if is_free_prompt:
            user_prompt = build_free_verification_prompt(
                claim=claim_clean,
                evidence_items=valid_evidence if valid_evidence else None,
                claim_date=claim_date,
                label_definitions=self.config.label_definitions,
            )
            supplied_ids_check = None  # Free prompt does not constrain citations to closed set
        else:
            user_prompt = build_verification_prompt(
                claim=claim_clean,
                evidence_items=valid_evidence,
                claim_date=claim_date,
                label_definitions=self.config.label_definitions,
            )
            supplied_ids_check = supplied_ids

        # 3. Attempt 1
        logger.debug("Executing verification Attempt 1 for claim: %s", claim_clean[:60])
        response1 = self.client.generate(
            prompt=user_prompt,
            system_prompt=self.system_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        result1, err1 = parse_and_validate_response(
            response1.content,
            supplied_evidence_ids=supplied_ids_check,
            allowed_labels=VALID_LABELS,
        )

        if result1 is not None:
            # Attempt 1 succeeded!
            result1.attempts = 1
            result1.metadata["model"] = response1.model
            result1.metadata["latency_seconds"] = response1.latency_seconds
            self.trace_recorder.record(
                prompt=user_prompt,
                system_prompt=self.system_prompt,
                response=response1,
                claim=claim_clean,
                claim_date=claim_date,
                supplied_evidence_ids=supplied_ids,
                attempt=1,
                parsed_result=result1.to_dict(),
            )
            return result1

        # Attempt 1 failed validation, record trace and prepare retry
        logger.warning(
            "Verification Attempt 1 failed validation: %s. Initiating single retry...",
            err1,
        )
        self.trace_recorder.record(
            prompt=user_prompt,
            system_prompt=self.system_prompt,
            response=response1,
            claim=claim_clean,
            claim_date=claim_date,
            supplied_evidence_ids=supplied_ids_check,
            attempt=1,
            error=err1,
        )

        # 4. Attempt 2: Retry once with corrective prompt (Requirement 7)
        retry_prompt = build_retry_prompt(
            original_prompt=user_prompt,
            raw_response=response1.content,
            error_message=err1 or "Invalid response format.",
            supplied_evidence_ids=supplied_ids_check,
        )

        response2 = self.client.generate(
            prompt=retry_prompt,
            system_prompt=self.system_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        result2, err2 = parse_and_validate_response(
            response2.content,
            supplied_evidence_ids=supplied_ids_check,
            allowed_labels=VALID_LABELS,
        )

        if result2 is not None:
            # Attempt 2 retry succeeded!
            logger.info("Verification succeeded on retry Attempt 2.")
            result2.attempts = 2
            result2.metadata["model"] = response2.model
            result2.metadata["latency_seconds"] = response2.latency_seconds
            self.trace_recorder.record(
                prompt=retry_prompt,
                system_prompt=self.system_prompt,
                response=response2,
                claim=claim_clean,
                claim_date=claim_date,
                supplied_evidence_ids=supplied_ids,
                attempt=2,
                parsed_result=result2.to_dict(),
            )
            return result2

        # 5. Retry failed: Fallback to Unverified with parse_failed=True
        logger.error("Verification Attempt 2 failed retry validation: %s", err2)
        self.trace_recorder.record(
            prompt=retry_prompt,
            system_prompt=self.system_prompt,
            response=response2,
            claim=claim_clean,
            claim_date=claim_date,
            supplied_evidence_ids=supplied_ids,
            attempt=2,
            error=err2,
        )

        return VerificationResult(
            label=self.config.abstain_label,
            confidence=0.0,
            evidence_ids=[],
            explanation=f"Model output schema validation failed after retry. Error: {err2}",
            parse_failed=True,
            raw_response=response2.content,
            attempts=2,
            metadata={
                "error_attempt_1": err1,
                "error_attempt_2": err2,
                "model": response2.model,
            },
        )

    def verify_batch(
        self,
        items: list[dict[str, Any]],
    ) -> list[VerificationResult]:
        """Verify a collection of claims with their retrieved evidence.

        Args:
            items: List of dictionaries with 'claim', 'evidence', and optional 'claim_date'.

        Returns:
            List of VerificationResult objects.
        """
        results: list[VerificationResult] = []
        for item in items:
            claim = item.get("claim", "")
            evidence = item.get("evidence", [])
            claim_date = item.get("claim_date")
            res = self.verify(claim=claim, evidence=evidence, claim_date=claim_date)
            results.append(res)
        return results
