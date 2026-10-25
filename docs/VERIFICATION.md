# VERITAS Phase 6 (P6): Evidence-Constrained LLM Verifier

## 1. Overview

Phase 6 implements the **evidence-constrained LLM verifier** for VERITAS. Unlike open-domain chatbots or unconstrained models, the verifier operates under a **closed-world assumption**: verdicts must be strictly substantiated and cited using only the retrieved evidence passages provided in the prompt.

The verifier produces structured verdicts conforming to the standard 5-class VERITAS taxonomy:
- `True`
- `Partially True`
- `Misleading`
- `False`
- `Unverified`

---

## 2. Architecture & Delivered Modules

| Module | Location | Purpose |
| :--- | :--- | :--- |
| **Prompt Engineering** | [src/veritas/verification/prompt.py](file:///e:/veritas/src/veritas/verification/prompt.py) | Closed-world system prompts, user prompt construction with temporal context (`claim_date`), formatted evidence blocks, and structured retry prompts. |
| **Parser & Validator** | [src/veritas/verification/parse.py](file:///e:/veritas/src/veritas/verification/parse.py) | Code-fence stripping (```json), JSON repair/sanitization, 5-class label normalization, confidence validation in $[0.0, 1.0]$, citation validation rejecting unsupplied IDs, and explanation validation. |
| **LLM Client Interface** | [src/veritas/verification/llm_client.py](file:///e:/veritas/src/veritas/verification/llm_client.py) | Common abstract client interface (`BaseLLMClient`), `MockLLMClient` for deterministic unit testing, `LocalLLMClient` for lazy Hugging Face Transformers inference, `APILLMClient` (OpenAI/Anthropic) using standard-library `urllib`, API secret hygiene, and `TraceRecorder` for raw JSONL output logging. |
| **Verification Engine** | [src/veritas/verification/verify.py](file:///e:/veritas/src/veritas/verification/verify.py) | `EvidenceConstrainedVerifier` orchestrator: relevance filtering gate, single-retry correction workflow, fallback to `Unverified` (`parse_failed=True`), and batch evaluation. |
| **Configuration** | [configs/verification.yaml](file:///e:/veritas/configs/verification.yaml) | Full verification parameters, backend selection, temperature, tokens, relevance threshold, trace paths, and label definitions. Also mirrored in [configs/default.yaml](file:///e:/veritas/configs/default.yaml). |
| **Test Suite** | [tests/verification/](file:///e:/veritas/tests/verification/) | 47 unit and integration tests using mocked responses covering all edge cases without making paid API calls or downloading large models. |

---

## 3. Label Taxonomy & Guidelines

| Label | Definition |
| :--- | :--- |
| **`True`** | The claim is directly and fully supported by trustworthy evidence. All key assertions, figures, dates, and essential context are accurate with no critical omissions. |
| **`Partially True`** | Core elements of the claim are supported, but key aspects are incomplete, exaggerated, missing important nuance or context, or accompanied by minor inaccuracies. |
| **`Misleading`** | The claim contains selective, out-of-context, or distorted facts that convey a false or deceptive impression, even if isolated individual statements are technically real. |
| **`False`** | The central assertion is directly contradicted or refuted by trustworthy evidence; the substantive statement did not happen, is incorrect, or is fabricated. |
| **`Unverified`** | The supplied evidence is insufficient, absent, conflicting, or inconclusive to verify or refute the claim. |

### When to Return `Unverified`:
1. **Empty/Insufficient Evidence**: The retrieval pipeline returns no passages or all passages fall below `min_relevance_score`. In this case, the verifier returns `Unverified` immediately **without calling the LLM**.
2. **Off-Topic / Missing Facts**: The evidence passages do not mention or address the specific subject or entity of the claim.
3. **Ambiguity / Conflict**: The evidence contains unresolved contradictions across sources.
4. **Parsing / Schema Failure**: The LLM output fails JSON decoding or validation after a retry attempt (`parse_failed=True`).

---

## 4. Contract Specification

### Inputs
- `claim`: Factual statement string to verify.
- `evidence`: List of retrieved passages (`Passage`, `SearchResult`, `FusedSearchResult`, or dictionary).
- `claim_date`: Optional ISO date string (e.g., `"2023-04-15"`) to ground temporal statements.
- `config`: `VerifierConfig` or mapping.

### Outputs (`VerificationResult`)
```json
{
  "label": "True",
  "confidence": 0.95,
  "evidence_ids": ["passage_102"],
  "explanation": "Bureau of Labor Statistics data in passage_102 confirms the 3.5% figure.",
  "parse_failed": false,
  "attempts": 1
}
```

---

## 5. Security & Traceability

1. **Secret Hygiene**: API keys are never hardcoded. Keys are loaded strictly from environment variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `VERITAS_API_KEY`). Headers and logs automatically mask secrets (e.g., `sk-...4a1b`).
2. **Trace Preservation**: All raw prompts, completions, latency, attempts, and error states are recorded to `results/verification_traces/verification_traces_<date>.jsonl` for offline analysis, audit, and replay.
