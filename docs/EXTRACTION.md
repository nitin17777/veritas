# VERITAS Phase 5 (P5): Claim Extraction & Retrieval Query Generation

## 1. Overview

Phase 5 implements the **Claim Extraction and Query Generation** module for VERITAS. While benchmark datasets (such as LIAR or FEVER) provide atomic, single-sentence claims that bypass extraction, real-world demo inputs often consist of full news articles, speeches, or social media posts containing mixtures of facts, opinions, and background commentary.

P5 bridges this gap by extracting up to 5 check-worthy factual claims and generating 2 to 3 targeted search queries per claim for downstream retrieval.

---

## 2. Architecture & Delivered Files

| Component | File Location | Purpose |
| :--- | :--- | :--- |
| **Extraction Engine** | [src/veritas/extraction/extract.py](file:///e:/veritas/src/veritas/extraction/extract.py) | `ClaimExtractor`, `ExtractionConfig`, `ExtractedClaim`, `ExtractionResult`, and `extract_claims()`. |
| **Prompt Templates** | [src/veritas/extraction/prompt.py](file:///e:/veritas/src/veritas/extraction/prompt.py) | Closed-world extraction instructions, pronoun resolution rules, and retry prompt builder. |
| **Package Exports** | [src/veritas/extraction/__init__.py](file:///e:/veritas/src/veritas/extraction/__init__.py) | Clean public API export. |
| **Configuration** | [configs/extraction.yaml](file:///e:/veritas/configs/extraction.yaml) | Configurable model, prompt version, temperature, and claim bounds. |
| **Unit Test Suite** | [tests/extraction/test_extract.py](file:///e:/veritas/tests/extraction/test_extract.py) | 15 deterministic unit tests with mocked LLM responses (no live network/search calls). |
| **Usage Example** | [examples/claim_extraction_usage.py](file:///e:/veritas/examples/claim_extraction_usage.py) | Demonstrates extracting claims and queries from a sample multi-paragraph article. |

---

## 3. Extraction Guidelines & Constraints

1. **Check-Worthiness**:
   - Extracts only verifiable factual statements (statistics, dates, policy actions, scientific assertions, historical facts).
   - Rejects subjective opinions, speculations, and non-falsifiable commentary.
2. **Factual Integrity & Meaning Preservation**:
   - Preserves exact names, numbers, percentages, dates, and relationships.
   - Does not silently alter or embellish meaning.
3. **Pronoun Resolution**:
   - Resolves ambiguous pronouns ("he", "she", "it") only when the antecedent is clearly established in the source text.
4. **Retrieval Queries**:
   - Generates 2 to 3 concise, high-signal search queries per claim optimized for BM25 and dense vector search.
5. **Self-Correction & Schema Validation**:
   - Validates JSON output structure; if malformed or schema-invalid, automatically executes a single structured retry before failing gracefully.

---

## 4. Usage Example

```python
from veritas.extraction.extract import extract_claims

article_text = (
    "On October 12, 2023, the Department of Energy announced a $3.5 billion investment "
    "across 44 states to strengthen the national electrical grid. In my personal opinion, "
    "this is wonderful. Power outages cost the U.S. economy $150 billion annually."
)

result = extract_claims(article_text)
for claim in result.claims:
    print(claim.claim)
    print("Queries:", claim.queries)
```
