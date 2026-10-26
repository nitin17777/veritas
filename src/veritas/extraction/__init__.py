"""VERITAS P5 Claim Extraction and Query Generation package."""

from veritas.extraction.extract import (
    ClaimExtractor,
    ExtractedClaim,
    ExtractionConfig,
    ExtractionError,
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

__all__ = [
    "ClaimExtractor",
    "ExtractedClaim",
    "ExtractionConfig",
    "ExtractionError",
    "ExtractionResult",
    "ExtractionValidationError",
    "extract_claims",
    "DEFAULT_PROMPT_VERSION",
    "build_system_prompt",
    "build_user_prompt",
    "build_retry_prompt",
]
