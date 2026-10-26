# VERITAS Phase 9 (P9): Comprehensive Evaluation Framework

## 1. Overview & Evaluation Principles

Phase 9 implements the unified, auditable **Evaluation Framework** for VERITAS. It establishes a rigorous scientific evaluation protocol across claim classification, evidence retrieval, explanation faithfulness, abstention policies, and system latency.

### Scientific Integrity Principles:
1. **Never Tune on Test Data**: The test split is held out and evaluated strictly once. All hyperparameter tuning (e.g. classification thresholds, regularisation $C$, RRF weights) is locked on training/validation splits.
2. **Stable Claim ID Matching**: Predictions are strictly paired with ground truth using unique claim IDs (`claim_id` / `id`). Any missing or duplicated IDs raise explicit data integrity errors.
3. **Distinguish Unavailable from Zero**: When gold evidence annotations are missing (such as in statement-only benchmarks like LIAR), retrieval metrics are reported as `None` / `unavailable`, rather than erroneously reporting `0.0`.
4. **No Fabricated Metrics**: Missing predictions or non-existent ground truth fields are never backfilled with invented values.
5. **Full Run Provenance**: Every evaluation records the complete configuration dictionary, model version, random seed, dataset version, and execution timestamp.

---

## 2. Delivered Modules & Architecture

| Module | Location | Description |
| :--- | :--- | :--- |
| **Classification & System Metrics** | [src/veritas/evaluation/metrics.py](file:///e:/veritas/src/veritas/evaluation/metrics.py) | Accuracy, Macro-F1, per-class P/R/F1, confusion matrix, latency percentiles, abstention rate, and multi-run aggregation. |
| **Retrieval Evaluation** | [src/veritas/evaluation/retrieval_metrics.py](file:///e:/veritas/src/veritas/evaluation/retrieval_metrics.py) | Recall@k, nDCG@k, and MRR@k strictly over claims with annotated gold evidence. |
| **Reporting & Artifact Exporter** | [src/veritas/evaluation/report.py](file:///e:/veritas/src/veritas/evaluation/report.py) | `EvaluationReport`, console ASCII formatting, JSON/CSV exports, and standalone SVG plot generator. |
| **Reproducible Evaluation CLI** | [scripts/evaluate_veritas.py](file:///e:/veritas/scripts/evaluate_veritas.py) | Command-line evaluation utility supporting single and multi-run aggregations. |
| **Unit Test Suite** | [tests/evaluation/test_metrics.py](file:///e:/veritas/tests/evaluation/test_metrics.py) | 13 deterministic tests covering exact hand-calculated metrics, ID validation, and artifact generation. |

---

## 3. Metrics Specification

### 3.1 Classification Metrics
- **Accuracy**: Overall fraction of correct predictions across all 5 VERITAS classes.
- **Macro-F1**: Unweighted mean of F1 scores across classes with non-zero support.
- **Per-Class Breakdown**: Precision, Recall, F1, and Support for each of the 5 labels: `True`, `Partially True`, `Misleading`, `False`, and `Unverified`.
- **Confusion Matrix**: Full $5 \times 5$ contingency table (Gold vs. Predicted).

### 3.2 Retrieval Metrics (When Gold Evidence Annotations Exist)
- **Recall@k**: Fraction of gold relevant passages recovered in top-$k$ candidates ($k \in [1, 3, 5, 10]$).
- **nDCG@k**: Normalized Discounted Cumulative Gain accounting for rank positions.
- **MRR@k**: Mean Reciprocal Rank of the first relevant passage.

### 3.3 Faithfulness & System Quality Metrics
- **JSON Parse Failure Rate**: Proportion of model completions that failed schema parsing after retry.
- **Abstention Rate**: Proportion of claims downgraded to `Unverified` due to low confidence or low explanation faithfulness.
- **Mean Faithfulness**: Average sentence-level NLI entailment score across explanation sentences.
- **Hallucination Rate Proxy**: Proportion of predictions citing non-existent passage IDs or exhibiting contradicted explanation reasoning.

### 3.4 Latency Percentiles
- **Average, P50 (Median), P90, P95, and P99** wall-clock execution time in seconds.

---

## 4. Reproducible Evaluation Commands

### Quick Smoke Test (Synthetic Claims):
```bash
python scripts/evaluate_veritas.py --smoke-test
```

### Full Benchmark Evaluation:
```bash
python scripts/evaluate_veritas.py \
    --gold data/processed/liar/test.jsonl \
    --predictions results/verdicts.jsonl \
    --config configs/pipeline.yaml \
    --output-dir results/evaluation \
    --dataset-name liar \
    --model-version gpt-4o-mini
```

### Multi-Run Aggregation (Repeated Seeds):
```bash
python scripts/evaluate_veritas.py \
    --gold data/processed/liar/test.jsonl \
    --predictions results/run_seed42.jsonl results/run_seed43.jsonl results/run_seed44.jsonl \
    --output-dir results/evaluation_multirun
```
