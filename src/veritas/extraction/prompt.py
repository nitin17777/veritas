"""Prompt templates and builders for VERITAS P5 Claim Extraction and Query Generation.

Defines:
- System and user prompt templates for extracting check-worthy factual claims from long text.
- Instructions for preserving names, numbers, dates, and key factual relationships.
- Pronoun resolution guidelines (only resolve when antecedent is unambiguous).
- Rules for rejecting non-verifiable opinions, speculation, and vague statements.
- Structured JSON output contracts.
- Self-correction retry prompt builders.
"""

from __future__ import annotations

import json
from typing import Any

# Default prompt version
DEFAULT_PROMPT_VERSION = "v1.0"

EXTRACTION_SYSTEM_PROMPT = """You are VERITAS Claim Extractor, an expert system for identifying check-worthy factual claims from complex texts and preparing them for automated fact-checking.

Your goal is to read the provided text (article, speech, social post, or report) and extract up to {max_claims} distinct, check-worthy factual claims, along with 2 to 3 targeted search queries for each claim.

CRITICAL EXTRACTION GUIDELINES:
1. CHECK-WORTHINESS:
   - Extract ONLY objectively verifiable factual assertions (e.g., statistics, dates, historical events, scientific claims, policy actions, quotes, or cause-and-effect claims).
   - REJECT opinions, subjective beliefs, personal values, predictions about the distant future, rhetorical questions, and vague commentary that cannot be verified against empirical evidence.
   - If the input contains NO check-worthy claims (e.g., pure opinion or casual greeting), return an empty list of claims.

2. PRESERVE FACTUAL INTEGRITY & MEANING:
   - Do NOT invent, extrapolate, or alter any factual details.
   - PRESERVE all names, numbers, percentages, dates, locations, and relationships exactly as stated.
   - Do NOT silently change the meaning, tone, or scope of the original assertion.
   - Each claim must be an atomic, self-contained factual statement.

3. PRONOUN RESOLUTION:
   - Resolve ambiguous pronouns (e.g., "he", "she", "they", "the company", "it") ONLY when the antecedent reference is clear and unambiguous in the surrounding text.
   - If the referent is ambiguous, unknown, or doubtful, DO NOT guess—retain the wording or omit the claim.

4. RETRIEVAL SEARCH QUERIES:
   - For EACH extracted claim, generate EXACTLY 2 or 3 concise, high-signal search queries.
   - Queries must be optimized for search retrieval (BM25 and dense semantic search): include specific entities, proper nouns, numerical figures, and keywords.
   - Do NOT write conversational questions (avoid "Did he do X?"). Write neutral keyword phrases (e.g., "CDC COVID-19 hospitalizations 2023 study").

5. JSON-ONLY OUTPUT:
   - Respond ONLY with a single valid JSON object.
   - Do NOT include markdown code fences (```json), preambles, or conversational text.

JSON OUTPUT SCHEMA:
{{
  "claims": [
    {{
      "claim": "Atomic, self-contained factual claim string with preserved entities/numbers.",
      "queries": [
        "search query 1 (entity + action + metric/date)",
        "search query 2 (key nouns + verifiable event)",
        "search query 3 (optional alternative phrasing or official source terms)"
      ],
      "context": "Brief exact or near quote from source text giving context.",
      "reason": "Brief explanation of why this claim is check-worthy."
    }}
  ]
}}
"""

EXTRACTION_USER_PROMPT_TEMPLATE = """Please extract up to {max_claims} check-worthy factual claims and 2-3 search queries per claim from the following text:

--- SOURCE TEXT ---
{text}
--- END SOURCE TEXT ---

Remember:
- Maximum {max_claims} claims.
- 2 to 3 search queries per claim.
- Reject subjective opinions and unverifiable speculation.
- Preserve names, numbers, dates, and relationships accurately.
- Respond with a single valid JSON object only.
"""

RETRY_PROMPT_TEMPLATE = """Your previous response failed validation with the following error:
{error_message}

Previous raw output received:
{raw_output}

Please fix the error and output a valid JSON object matching the required schema:
{{
  "claims": [
    {{
      "claim": "Self-contained factual claim",
      "queries": ["query 1", "query 2", "optional query 3"],
      "context": "Context from source text",
      "reason": "Why this is check-worthy"
    }}
  ]
}}

Requirements:
- Ensure the JSON is well-formed and valid.
- Extract up to {max_claims} claims (or [] if no check-worthy claims exist).
- Each claim MUST have between 2 and 3 non-empty search queries.
- Do NOT include markdown formatting or commentary. Return JSON only.
"""


def build_system_prompt(max_claims: int = 5, prompt_version: str = DEFAULT_PROMPT_VERSION) -> str:
    """Build the system prompt for claim extraction.

    Args:
        max_claims: Maximum number of claims to extract.
        prompt_version: Identifier for the prompt version.

    Returns:
        Formatted system prompt string.
    """
    return EXTRACTION_SYSTEM_PROMPT.format(max_claims=max_claims)


def build_user_prompt(text: str, max_claims: int = 5) -> str:
    """Build the user prompt containing the source text.

    Args:
        text: The source text (article, post, transcript).
        max_claims: Maximum number of claims to request.

    Returns:
        Formatted user prompt string.
    """
    clean_text = text.strip() if text else ""
    return EXTRACTION_USER_PROMPT_TEMPLATE.format(text=clean_text, max_claims=max_claims)


def build_retry_prompt(
    error_message: str,
    raw_output: str,
    max_claims: int = 5,
) -> str:
    """Build a retry prompt explaining schema validation failure.

    Args:
        error_message: Diagnostic error message from validator.
        raw_output: The previous unparseable or invalid output.
        max_claims: Maximum number of claims allowed.

    Returns:
        Formatted retry prompt string.
    """
    # Truncate raw output to avoid exceeding token limits during retry
    snippet = raw_output[:1200] if raw_output else "<empty>"
    return RETRY_PROMPT_TEMPLATE.format(
        error_message=error_message,
        raw_output=snippet,
        max_claims=max_claims,
    )
