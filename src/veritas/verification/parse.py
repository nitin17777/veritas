"""Parsing and schema validation for evidence-constrained verification outputs.

Handles:
- Stripping markdown code fences (```json, ```).
- Extracting and robustly parsing JSON objects.
- Normalizing and validating VERITAS 5-class labels.
- Validating confidence intervals in [0.0, 1.0].
- Strictly validating evidence IDs against supplied candidate IDs.
- Validating non-empty explanations.
- Returning structured VerificationResult or informative error diagnostics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any

from veritas.data_prep.schema import VALID_LABELS


class VerificationParseError(Exception):
    """Raised when the LLM response cannot be parsed as JSON."""


class VerificationValidationError(Exception):
    """Raised when the parsed JSON fails VERITAS schema validation."""


@dataclass
class VerificationResult:
    """Standardized output of the VERITAS evidence verifier.

    Attributes:
        label: One of the 5 VERITAS labels: True, Partially True, Misleading, False, Unverified.
        confidence: Numeric certainty in the interval [0.0, 1.0].
        evidence_ids: List of validated evidence IDs supporting the verdict.
        explanation: Grounded textual explanation citing supplied evidence.
        parse_failed: True if the model output could not be parsed or validated after retries.
        raw_response: The original string completion from the LLM.
        attempts: Number of model generation attempts executed.
        metadata: Additional diagnostic info (timestamps, tokens, latency, etc.).
    """

    label: str
    confidence: float
    evidence_ids: list[str]
    explanation: str
    parse_failed: bool = False
    raw_response: str | None = None
    attempts: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to serializable dictionary."""
        return asdict(self)


# Mapping for common case/formatting variations of valid labels
LABEL_NORMALIZATION: dict[str, str] = {
    "true": "True",
    "partially true": "Partially True",
    "partially_true": "Partially True",
    "partially-true": "Partially True",
    "misleading": "Misleading",
    "false": "False",
    "unverified": "Unverified",
    "nei": "Unverified",
    "not enough info": "Unverified",
}


def strip_code_fences(text: str) -> str:
    """Strip markdown code block fences and isolate the JSON payload.

    Handles:
    - ```json ... ```
    - ``` ... ```
    - Preceding and trailing conversational noise.
    """
    if not text:
        return ""

    cleaned = text.strip()

    # If wrapped in code fence regex
    fence_pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    matches = fence_pattern.findall(cleaned)
    if matches:
        # Take the first matched block containing braces
        for match in matches:
            if "{" in match and "}" in match:
                return match.strip()
        return matches[0].strip()

    # If code fence is opened but not closed (e.g. truncated)
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    return cleaned


def extract_outer_json_object(text: str) -> str:
    """Extract substring spanning the first '{' and the last '}'."""
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1].strip()
    return text.strip()


def sanitize_json_text(text: str) -> str:
    """Attempt basic sanitization for common LLM JSON syntax issues."""
    # Remove trailing commas before } or ]
    sanitized = re.sub(r",\s*(\}|\])", r"\1", text)
    return sanitized


def parse_raw_json(raw_text: str) -> dict[str, Any]:
    """Parse raw LLM response text into a dictionary.

    Args:
        raw_text: Raw string completion from the model.

    Returns:
        Parsed dictionary.

    Raises:
        VerificationParseError: If no valid JSON object could be extracted.
    """
    if not raw_text or not raw_text.strip():
        raise VerificationParseError("Model returned empty response.")

    stripped = strip_code_fences(raw_text)

    # 1. Direct parse attempt
    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # 2. Extract balanced outer braces
    extracted = extract_outer_json_object(stripped)
    try:
        data = json.loads(extracted)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # 3. Sanitize trailing commas and retry
    sanitized = sanitize_json_text(extracted)
    try:
        data = json.loads(sanitized)
        if isinstance(data, dict):
            return data
    except Exception as exc:
        raise VerificationParseError(
            f"Failed to decode JSON from model response: {exc}. Response snippet: {raw_text[:200]!r}"
        ) from exc

    raise VerificationParseError(f"Expected a JSON object (dict), got {type(data).__name__}.")


def normalize_label(label_raw: Any, allowed_labels: set[str] | None = None) -> str:
    """Normalize label string and validate against allowed VERITAS labels."""
    allowed = allowed_labels or VALID_LABELS
    if not isinstance(label_raw, str):
        raise VerificationValidationError(
            f"Field 'label' must be a string, got {type(label_raw).__name__}."
        )

    norm_key = label_raw.strip().lower()
    canonical = LABEL_NORMALIZATION.get(norm_key, label_raw.strip())

    if canonical not in allowed:
        raise VerificationValidationError(
            f"Invalid label {label_raw!r}. Must be one of: {sorted(allowed)}."
        )

    return canonical


def validate_confidence(confidence_raw: Any) -> float:
    """Validate that confidence is a float in the range [0.0, 1.0]."""
    if confidence_raw is None or isinstance(confidence_raw, bool):
        raise VerificationValidationError(
            f"Field 'confidence' must be a numeric value, got {type(confidence_raw).__name__}."
        )

    try:
        val = float(confidence_raw)
    except (ValueError, TypeError) as exc:
        raise VerificationValidationError(
            f"Field 'confidence' must be a numeric value in [0.0, 1.0], cannot convert {confidence_raw!r} to float."
        ) from exc

    if not (0.0 <= val <= 1.0):
        raise VerificationValidationError(
            f"Field 'confidence' value {val} is outside valid interval [0.0, 1.0]."
        )

    return val


def validate_evidence_ids(
    evidence_ids_raw: Any,
    supplied_evidence_ids: set[str] | list[str],
) -> list[str]:
    """Validate that evidence_ids is a list and contains ONLY supplied IDs.

    Requirement 5: Reject evidence IDs that were not supplied.
    """
    if evidence_ids_raw is None:
        return []

    if not isinstance(evidence_ids_raw, list):
        raise VerificationValidationError(
            f"Field 'evidence_ids' must be a list of strings, got {type(evidence_ids_raw).__name__}."
        )

    supplied_set = {str(eid).strip() for eid in supplied_evidence_ids}
    cleaned_ids: list[str] = []

    unsupplied: list[str] = []
    for item in evidence_ids_raw:
        eid = str(item).strip()
        if not eid:
            continue
        if eid not in supplied_set:
            unsupplied.append(eid)
        else:
            cleaned_ids.append(eid)

    if unsupplied:
        raise VerificationValidationError(
            f"Rejected evidence IDs not present in supplied evidence: {unsupplied}. "
            f"Supplied IDs were: {sorted(supplied_set)}."
        )

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for eid in cleaned_ids:
        if eid not in seen:
            seen.add(eid)
            deduped.append(eid)

    return deduped


def validate_explanation(explanation_raw: Any) -> str:
    """Validate that explanation is a non-empty string."""
    if not isinstance(explanation_raw, str) or not explanation_raw.strip():
        raise VerificationValidationError(
            "Field 'explanation' must be a non-empty string explaining the decision."
        )
    return explanation_raw.strip()


def validate_verification_dict(
    data: dict[str, Any],
    supplied_evidence_ids: set[str] | list[str],
    allowed_labels: set[str] | None = None,
) -> VerificationResult:
    """Validate raw parsed dictionary against VERITAS verification requirements.

    Args:
        data: Parsed dictionary from LLM JSON.
        supplied_evidence_ids: Set or list of evidence IDs supplied in prompt.
        allowed_labels: Set of valid labels (defaults to VALID_LABELS).

    Returns:
        Structured VerificationResult.

    Raises:
        VerificationValidationError: If any requirement is violated.
    """
    for required_field in ("label", "confidence", "explanation"):
        if required_field not in data:
            raise VerificationValidationError(f"Missing required field: {required_field!r}")

    label = normalize_label(data.get("label"), allowed_labels=allowed_labels)
    confidence = validate_confidence(data.get("confidence"))
    evidence_ids = validate_evidence_ids(
        data.get("evidence_ids", []),
        supplied_evidence_ids=supplied_evidence_ids,
    )
    explanation = validate_explanation(data.get("explanation"))

    return VerificationResult(
        label=label,
        confidence=confidence,
        evidence_ids=evidence_ids,
        explanation=explanation,
        parse_failed=False,
    )


def parse_and_validate_response(
    raw_text: str,
    supplied_evidence_ids: set[str] | list[str],
    allowed_labels: set[str] | None = None,
) -> tuple[VerificationResult | None, str | None]:
    """Parse raw model output and validate against schema.

    Returns:
        (VerificationResult, None) if successful, or (None, error_message) if invalid.
    """
    try:
        parsed_dict = parse_raw_json(raw_text)
        result = validate_verification_dict(
            parsed_dict,
            supplied_evidence_ids=supplied_evidence_ids,
            allowed_labels=allowed_labels,
        )
        result.raw_response = raw_text
        return result, None
    except (VerificationParseError, VerificationValidationError) as exc:
        return None, str(exc)
    except Exception as exc:
        return None, f"Unexpected error while parsing response: {exc}"
