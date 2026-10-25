"""VERITAS P6 Evidence-Constrained LLM Verifier package."""

from veritas.verification.llm_client import (
    APILLMClient,
    BaseLLMClient,
    LLMResponse,
    LocalLLMClient,
    MockLLMClient,
    TraceRecorder,
    create_llm_client,
    mask_secret,
)
from veritas.verification.parse import (
    VerificationParseError,
    VerificationResult,
    VerificationValidationError,
    parse_and_validate_response,
    strip_code_fences,
)
from veritas.verification.prompt import (
    DEFAULT_LABEL_DEFINITIONS,
    build_retry_prompt,
    build_system_prompt,
    build_verification_prompt,
    format_evidence_block,
    format_label_definitions,
)
from veritas.verification.verify import (
    EvidenceConstrainedVerifier,
    VerifierConfig,
    normalize_evidence_item,
)

__all__ = [
    "EvidenceConstrainedVerifier",
    "VerifierConfig",
    "VerificationResult",
    "VerificationParseError",
    "VerificationValidationError",
    "BaseLLMClient",
    "MockLLMClient",
    "APILLMClient",
    "LocalLLMClient",
    "LLMResponse",
    "TraceRecorder",
    "create_llm_client",
    "mask_secret",
    "parse_and_validate_response",
    "strip_code_fences",
    "build_system_prompt",
    "build_verification_prompt",
    "build_retry_prompt",
    "format_evidence_block",
    "format_label_definitions",
    "DEFAULT_LABEL_DEFINITIONS",
    "normalize_evidence_item",
]
