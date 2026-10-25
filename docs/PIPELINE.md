# VERITAS Phase 8 (P8): Pipeline Orchestrator

## 1. Overview

Phase 8 implements the **end-to-end pipeline orchestrator** for VERITAS. It seamlessly integrates the retrieval system (BM25, Dense, or Hybrid RRF) and the evidence-constrained LLM verifier into a unified, reproducible, and traceable verification workflow.

The pipeline exposes a clean programmatic entry point:
```python
verify(claim: str, claim_date: str | None = None, cfg: PipelineConfig | dict | None = None) -> Verdict
```

---

## 2. Architecture & Pipeline Stages

```
                +---------------------------------+
                |        Input Claim              |
                |  (claim, claim_date, claim_id)  |
                +---------------------------------+
                                |
                                v
               [Stage 1: Query Generation]
                (Identity / Query Expansion)
                                |
                                v
               [Stage 2: Candidate Retrieval]
                (HybridRetriever: BM25 / Dense / RRF)
                                |
                                v
               [Stage 3: Evidence Ranking]
                (Pass-through / Cross-Encoder Gate)
                                |
                                v
               [Stage 4: Verification]
                (EvidenceConstrainedVerifier)
                - Closed-world verification prompt
                - JSON extraction & validation
                - Automatic single retry on error
                - Short-circuit on empty evidence
                                |
                                v
                    +-----------------------+
                    |    Verdict Object     |
                    | (Typed JSONL Record)  |
                    +-----------------------+
```

### Stage Details

1. **`generate_queries(claim)`**: Generates search queries from the claim. Currently an identity pass-through, providing a documented extension point for query decomposition, entity-focused queries, or LLM-based sub-claim generation.
2. **`retrieve(queries)`**: Invokes `HybridRetriever` using the configured retrieval mode (`bm25`, `dense`, or `hybrid`) with Reciprocal Rank Fusion (RRF).
3. **`rank_evidence(candidates)`**: Orders and selects top evidence. Currently preserves the ranked order from retrieval while filtering out passages with empty text; serves as the integration hook for future cross-encoder re-ranking.
4. **`verify(claim, evidence, claim_date)`**: Invokes `EvidenceConstrainedVerifier`. Safely handles empty retrieval or low relevance without invoking the LLM, and gracefully catches model errors into `Unverified` verdicts with `parse_failed=True`.

---

## 3. Delivered Components

| Component | Location | Description |
| :--- | :--- | :--- |
| **Pipeline Orchestrator** | [src/veritas/pipeline/pipeline.py](file:///e:/veritas/src/veritas/pipeline/pipeline.py) | `VeritasPipeline`, `PipelineConfig`, `Verdict`, and convenience `verify()` function. |
| **Batch Runner & Resume** | [src/veritas/pipeline/batch.py](file:///e:/veritas/src/veritas/pipeline/batch.py) | `BatchRunner` with crash recovery: reads already-completed claim IDs from output JSONL and maintains a `.ckpt.json` status file. |
| **Batch CLI Script** | [scripts/run_pipeline_batch.py](file:///e:/veritas/scripts/run_pipeline_batch.py) | CLI utility supporting `--input`, `--output`, `--config`, and `--smoke-test` flags. |
| **Configuration** | [configs/pipeline.yaml](file:///e:/veritas/configs/pipeline.yaml) | Full orchestrator configuration including retrieval modes, verifier settings, and output paths. |
| **Integration Test Suite** | [tests/pipeline/test_pipeline.py](file:///e:/veritas/tests/pipeline/test_pipeline.py) | 25 integration tests verifying 5 synthetic claims, failure resilience, batch crash-resume, and serialization. |

---

## 4. Typed Output Specification (`Verdict`)

Every run returns a `Verdict` dataclass that serializes to JSON/JSONL for complete auditability:

```python
@dataclass
class Verdict:
    claim_id: str
    claim: str
    claim_date: str | None
    label: str               # "True" | "Partially True" | "Misleading" | "False" | "Unverified"
    confidence: float        # Float in [0.0, 1.0]
    evidence_ids: list[str]  # Passage IDs cited by the LLM
    explanation: str         # Grounded reasoning
    parse_failed: bool       # True if LLM output failed schema after retry
    retrieved_passages: list[dict[str, Any]]  # Full provenance for all retrieved passages
    queries_used: list[str]  # Exact queries executed against index
    stage_timings_s: dict[str, float]  # Timings for query_gen, retrieval, ranking, verify
    model_metadata: dict[str, Any]     # Backend, model name, attempts, tokens
    raw_llm_output: str | None         # Unprocessed LLM response string
    run_id: str | None                 # Batch run identifier
    pipeline_version: str = "p8.0"     # Schema version
```

---

## 5. Batch Execution & Crash-Resume

The batch runner supports resuming interrupted or failed runs:
1. When starting, it inspects the target output JSONL file.
2. Any `claim_id` already present in the output file is recorded as completed.
3. The runner skips completed claims on subsequent executions.
4. A checkpoint file `<output_path>.ckpt.json` is updated atomically with run metadata, timestamp, total count, and completed count.

### Running via CLI

```bash
# Run the built-in smoke test
python scripts/run_pipeline_batch.py --smoke-test

# Run a batch over input claims with custom configuration
python scripts/run_pipeline_batch.py \
  --input data/claims.jsonl \
  --output data/verdicts.jsonl \
  --config configs/pipeline.yaml \
  --run-id my_batch_run
```

---

## 6. Important Validation Disclaimer

> [!IMPORTANT]
> The passing unit and integration tests (including the 5 synthetic claims test) use deterministic `MockLLMClient` and in-memory indexes. Passing tests validate pipeline mechanics, schema compliance, error-handling paths, stage timings, and serialization.
>
> They do **NOT** validate that:
> - The pipeline produces accurate verdicts on real-world claims.
> - The retrieval stage surfaces the most relevant evidence from real corpora.
> - The prompt induces optimal reasoning across edge cases with live LLMs.
>
> Real end-to-end factuality requires validation against a held-out benchmark (e.g. PolitiFact, Snopes, FEVER) with live models and full corpus indexes.
