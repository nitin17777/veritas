"""Prompt templates and builders for evidence-constrained fact verification.

Defines:
- Standard VERITAS 5-class label definitions.
- Strict closed-world evidence constraint instructions.
- JSON-only response contracts.
- System, user, and retry prompt builders.
"""

from __future__ import annotations

import json
from typing import Any

from veritas.data_prep.schema import VALID_LABELS

# Standard VERITAS 5-class label definitions
DEFAULT_LABEL_DEFINITIONS: dict[str, str] = {
    "True": (
        "The claim is directly and fully supported by trustworthy evidence. "
        "All key assertions, factual details, figures, dates, and essential context "
        "are substantiated by the provided evidence, with no material inaccuracies."
    ),
    "Partially True": (
        "Core elements of the claim are supported by the evidence, but key aspects "
        "are incomplete, exaggerated, missing important nuance or context, or accompanied "
        "by minor inaccuracies."
    ),
    "Misleading": (
        "The claim contains selective, out-of-context, or distorted facts that give "
        "a false, deceptive, or unwarranted impression, even if isolated individual statements "
        "may be technically real or derived from actual events."
    ),
    "False": (
        "The central assertion of the claim is directly contradicted or refuted by "
        "trustworthy evidence; the substantive statement did not happen, is factually incorrect, "
        "or is fabricated."
    ),
    "Unverified": (
        "The supplied evidence is insufficient, absent, conflicting, or inconclusive to verify "
        "or refute the claim. CRITICAL: If the supplied evidence does not directly address "
        "the specific claim, or lacks sufficient factual detail to establish truth or falsity, "
        "you MUST return Unverified rather than guessing or extrapolating."
    ),
}

SYSTEM_PROMPT_TEMPLATE = """You are VERITAS, an evidence-constrained artificial intelligence fact verification system.
Your mission is to evaluate the factual truthfulness of a given claim strictly and solely on the basis of the provided evidence passages.

CRITICAL RULES AND CONSTRAINTS:
1. EVIDENCE-CONSTRAINED ONLY: You MUST rely EXCLUSIVELY on the provided evidence passages. Do NOT use external knowledge, personal beliefs, or unstated world assumptions. If a fact is not stated in the supplied evidence, treat it as unknown.
2. CITATION REQUIREMENT: Any evidence ID cited in `evidence_ids` MUST be chosen exclusively from the supplied evidence passages. Never invent, hallucinate, or reference non-existent evidence IDs.
3. WHEN TO RETURN 'Unverified':
   - If the provided evidence does not mention or address the central subject of the claim.
   - If the evidence provides ambiguous, vague, or inconclusive information.
   - If the evidence contains contradictory assertions from equally weighted sources without resolution.
   - When in doubt, you MUST abstain by returning 'Unverified'.
4. CONFIDENCE SCORE: Output a floating-point confidence score between 0.0 and 1.0 representing how strongly the supplied evidence justifies your verdict.
5. JSON-ONLY OUTPUT: You must respond ONLY with a single valid JSON object. Do not include markdown code fences (like ```json), do not include preambles, and do not include postscripts or conversational text.

ALLOWED LABELS AND DEFINITIONS:
{label_definitions_formatted}

JSON OUTPUT SCHEMA:
{{
  "label": "True" | "Partially True" | "Misleading" | "False" | "Unverified",
  "confidence": 0.0 - 1.0,
  "evidence_ids": ["id_of_used_evidence", ...],
  "explanation": "A concise explanation justifying the label and citing specific facts from the supplied evidence."
}}
"""


def format_label_definitions(label_definitions: dict[str, str] | None = None) -> str:
    """Format label definitions into a structured list for prompts."""
    definitions = label_definitions or DEFAULT_LABEL_DEFINITIONS
    lines = []
    # Ensure ordered presentation following standard VERITAS taxonomy
    for label in ["True", "Partially True", "Misleading", "False", "Unverified"]:
        desc = definitions.get(label, DEFAULT_LABEL_DEFINITIONS.get(label, ""))
        lines.append(f'- "{label}": {desc}')
    return "\n".join(lines)


def build_system_prompt(label_definitions: dict[str, str] | None = None) -> str:
    """Build the system prompt with strict evidence-constrained instructions."""
    formatted_labels = format_label_definitions(label_definitions)
    return SYSTEM_PROMPT_TEMPLATE.format(label_definitions_formatted=formatted_labels).strip()


def format_evidence_block(evidence_items: list[dict[str, Any]]) -> str:
    """Format a list of evidence passages into a numbered reference block."""
    if not evidence_items:
        return "[No evidence passages supplied]"

    blocks: list[str] = []
    for idx, item in enumerate(evidence_items, start=1):
        pid = str(item.get("id") or item.get("passage_id") or f"passage_{idx}").strip()
        text = str(item.get("text") or "").strip()
        date = item.get("date") or item.get("publication_date")
        domain = item.get("domain")
        score = item.get("score")

        meta_parts = []
        if date:
            meta_parts.append(f"Date: {date}")
        if domain:
            meta_parts.append(f"Source: {domain}")
        if score is not None:
            try:
                meta_parts.append(f"Retrieval Score: {float(score):.4f}")
            except (ValueError, TypeError):
                pass

        meta_line = f" ({', '.join(meta_parts)})" if meta_parts else ""
        blocks.append(f"[{idx}] Evidence ID: {pid}{meta_line}\nPassage: {text}")

    return "\n\n".join(blocks)


def build_verification_prompt(
    claim: str,
    evidence_items: list[dict[str, Any]],
    claim_date: str | None = None,
    label_definitions: dict[str, str] | None = None,
) -> str:
    """Construct the user prompt for verifying a claim against evidence passages.

    Args:
        claim: Statement to verify.
        evidence_items: List of dictionaries or Passage-like objects with 'id' and 'text'.
        claim_date: Optional temporal context of the claim (ISO date string).
        label_definitions: Optional custom label definitions.

    Returns:
        Rendered user prompt string.
    """
    claim_text = claim.strip()
    evidence_text = format_evidence_block(evidence_items)

    supplied_ids = [
        str(item.get("id") or item.get("passage_id") or "").strip()
        for item in evidence_items
        if (item.get("id") or item.get("passage_id"))
    ]
    ids_str = json.dumps(supplied_ids)

    date_context = f"\nClaim Date: {claim_date.strip()}" if claim_date and claim_date.strip() else ""

    prompt = f"""CLAIM TO VERIFY:
"{claim_text}"{date_context}

SUPPLIED EVIDENCE PASSAGES:
{evidence_text}

VALID EVIDENCE IDS:
{ids_str}

TASK INSTRUCTIONS:
1. Analyze the claim strictly using only the supplied evidence passages above.
2. Select one of the 5 allowed labels: "True", "Partially True", "Misleading", "False", "Unverified".
   If the evidence does not provide sufficient factual basis to confirm or refute the claim, you MUST choose "Unverified".
3. Provide a confidence score in the closed interval [0.0, 1.0].
4. In `evidence_ids`, list ONLY IDs from the VALID EVIDENCE IDS list above that you used to justify your decision.
   DO NOT include any ID that was not provided above.
5. In `explanation`, provide a clear, concise factual explanation referencing the evidence.
6. Return ONLY the raw JSON object with keys "label", "confidence", "evidence_ids", and "explanation". Do not add any conversational text or markdown code fences."""

    return prompt.strip()


def build_free_verification_prompt(
    claim: str,
    evidence_items: list[dict[str, Any]] | None = None,
    claim_date: str | None = None,
    label_definitions: dict[str, str] | None = None,
) -> str:
    """Construct an unconstrained user prompt (free prompt) for claim verification.

    Unlike the evidence-constrained prompt, this prompt allows the model to leverage
    its general world knowledge, without enforcing a closed-world evidence constraint.

    Args:
        claim: Statement to verify.
        evidence_items: Optional evidence passages.
        claim_date: Optional temporal context (ISO date string).
        label_definitions: Optional custom label definitions.

    Returns:
        Rendered prompt string for free/unconstrained verification.
    """
    claim_text = claim.strip()
    date_context = f"\nClaim Date: {claim_date.strip()}" if claim_date and claim_date.strip() else ""

    evidence_section = ""
    if evidence_items:
        evidence_text = format_evidence_block(evidence_items)
        evidence_section = f"\n\nBACKGROUND EVIDENCE (FOR REFERENCE ONLY):\n{evidence_text}\n"

    prompt = f"""CLAIM TO VERIFY:
"{claim_text}"{date_context}{evidence_section}

TASK INSTRUCTIONS:
1. Assess the truthfulness of the claim based on your general knowledge and any reference material above.
2. Select one of the 5 allowed labels: "True", "Partially True", "Misleading", "False", "Unverified".
3. Provide a confidence score in the closed interval [0.0, 1.0].
4. In `evidence_ids`, list any evidence IDs used, or an empty list [] if based on parametric knowledge.
5. In `explanation`, provide a clear, concise explanation justifying your verdict.
6. Return ONLY the raw JSON object with keys "label", "confidence", "evidence_ids", and "explanation". Do not add any conversational text or markdown code fences."""

    return prompt.strip()



def build_retry_prompt(
    original_prompt: str,
    raw_response: str,
    error_message: str,
    supplied_evidence_ids: list[str] | None = None,
) -> str:
    """Construct a retry prompt when previous model output failed schema validation.

    Args:
        original_prompt: The initial prompt submitted to the model.
        raw_response: The previous malformed response from the model.
        error_message: Detailed explanation of why validation failed.
        supplied_evidence_ids: List of allowed evidence IDs.

    Returns:
        Corrective retry prompt string.
    """
    allowed_labels_str = ", ".join(f'"{lbl}"' for lbl in sorted(VALID_LABELS))
    ids_str = json.dumps(supplied_evidence_ids or [])

    return f"""Your previous response was INVALID and failed verification schema validation.

VALIDATION ERROR:
{error_message}

PREVIOUS RESPONSE:
{raw_response}

REQUIRED FORMAT CORRECTION:
- You must return ONLY a single valid JSON object.
- Allowed labels are strictly: {allowed_labels_str}.
- "confidence" must be a float between 0.0 and 1.0.
- "evidence_ids" must be a JSON array containing ONLY IDs from: {ids_str}.
- "explanation" must be a non-empty string.
- No markdown code fences, no extra text outside the JSON.

ORIGINAL TASK:
{original_prompt}

Please output the corrected JSON object now:""".strip()
