# VERITAS Research Experiment Suite

This document describes the empirical evaluation framework, research questions (RQs), factor-isolation guarantees, configuration architecture, and execution procedures for the **VERITAS** evidence-grounded fact verification system.

---

## 1. Research Questions & Hypotheses

| RQ | Research Question | Independent Variable | Compared Configurations | Primary Metrics |
|:---|:---|:---|:---|:---|
| **RQ1** | **LLM-Only vs. RAG** | Retrieval augmentation | Zero-evidence parametric verification vs. BM25 retrieval augmentation | Macro-F1, Accuracy, Hallucination Rate |
| **RQ2** | **Retrieval Backend** | Retrieval mechanism | BM25 (lexical) vs. Dense (semantic embeddings) vs. Hybrid (RRF fusion) | Recall@5, nDCG@5, Downstream Macro-F1 |
| **RQ3** | **Source Credibility** | Credibility weighting | Domain tier weights ($\beta=0.25$) vs. No credibility ($\beta=0.0$) vs. Random credibility | Accuracy, Macro-F1 on contested claims |
| **RQ4** | **Temporal Decay** | Temporal alignment | Recency decay $\exp(-\Delta t / \tau)$ ($\gamma=0.15$) vs. No temporal scoring ($\gamma=0.0$) | Macro-F1 on time-sensitive claims |
| **RQ5** | **Prompt Grounding** | Verifier prompt constraints | Strict closed-world evidence constraints vs. Free/unconstrained prompt | Citation precision, Parse failure rate |
| **RQ6** | **Dataset Transfer** | Evaluation domain | In-domain (LIAR test) vs. Held-out dataset/topic (FEVER test) | Macro-F1 degradation, Out-of-domain Abstention |
| **RQ7** | **Abstention Policy** | NLI safety gate | Active abstention downgrade vs. Flag-only / Unfiltered verdicts | Coverage vs. Accuracy curve, Abstention rate |
| **Overall**| **System Benchmark**| Pipeline architecture | Baseline LLM-only vs. Baseline BM25 vs. Full VERITAS pipeline | End-to-end Macro-F1, Latency p95, Faithfulness |

---

## 2. Factor Isolation & Scientific Validity Guarantees

A core requirement of VERITAS experimental methodology is ensuring that any ablation or comparative experiment differs **strictly in the intended independent variable**. 

To eliminate confounding variations:
1. **Automated Isolation Validation**: [`validate_factor_isolation()`](file:///e:/veritas/src/veritas/experiments/config.py) compares flattened configuration dictionaries between baseline and variant configs, asserting that only the targeted factor differs. Accidental changes to temperature, model backends, seed, or index parameters are immediately flagged.
2. **Deterministic Seeds**: All experiment configurations fix `seed: 42` to guarantee reproducible pseudo-random sampling.
3. **No Tuning on Test**: Configurations enforce `split: test` or `split: val`. The runner strictly rejects `split: train` to prevent evaluation data leakage.
4. **Index & Cache Reuse**: Pre-built indexes and embeddings are reused across experiments to ensure identical retrieval candidates across downstream ablations.

---

## 3. Directory Layout

```
configs/experiments/
├── base.yaml                         # Shared base defaults
├── rq1_llm_only.yaml                 # RQ1: Zero-retrieval LLM baseline
├── rq1_rag.yaml                      # RQ1: Standard RAG
├── rq2_bm25.yaml                     # RQ2: Lexical BM25 retrieval
├── rq2_dense.yaml                    # RQ2: Dense embedding retrieval
├── rq2_hybrid.yaml                   # RQ2: Hybrid RRF fusion
├── rq3_credibility.yaml              # RQ3: Credibility tiers enabled
├── rq3_no_credibility.yaml           # RQ3: No credibility weighting
├── rq3_random_credibility.yaml       # RQ3: Random credibility baseline
├── rq4_temporal.yaml                 # RQ4: Temporal decay enabled
├── rq4_no_temporal.yaml              # RQ4: Temporal decay disabled
├── rq5_evidence_constrained.yaml     # RQ5: Closed-world prompt
├── rq5_free_prompt.yaml              # RQ5: Free unconstrained prompt
├── rq6_in_domain.yaml                # RQ6: In-domain test distribution
├── rq6_held_out.yaml                 # RQ6: Held-out domain distribution
├── rq7_abstention.yaml               # RQ7: Abstention gate enabled
├── rq7_no_abstention.yaml            # RQ7: Flag-only / no abstention
├── overall_baseline_llm_only.yaml    # Overall: Baseline LLM-only
├── overall_baseline_bm25_only.yaml   # Overall: Baseline BM25
└── overall_full_system.yaml          # Overall: Full VERITAS system
```

---

## 4. Execution Procedures & CLI

The experiment runner CLI supports executing individual experiments, entire research questions, or the full research suite with automatic crash-recovery and summary reporting:

### Running an Individual Experiment
```bash
python scripts/run_experiments.py --config configs/experiments/rq1_rag.yaml --smoke-test
```

### Running All Variants for a Research Question
```bash
# Run RQ1 (LLM-only vs. RAG)
python scripts/run_experiments.py --rq rq1 --smoke-test

# Run RQ2 (BM25 vs. Dense vs. Hybrid)
python scripts/run_experiments.py --rq rq2 --smoke-test

# Run RQ7 (Abstention Gate behavior)
python scripts/run_experiments.py --rq rq7 --smoke-test
```

### Running the Entire Research Suite
```bash
python scripts/run_experiments.py --all --smoke-test
```

### Regenerating Summary Tables & Plots from Existing Results
```bash
python scripts/run_experiments.py --summary
```

---

## 5. Crash Resumption & Audit Trail

The experiment runner is engineered for resilience during long-running batch jobs:
1. **Crash Resumption**: On startup, [`ExperimentRunner`](file:///e:/veritas/src/veritas/experiments/runner.py) inspects `results/experiments/<name>/predictions.jsonl` and loads all previously evaluated `claim_id`s. Any interrupted run resumes from the exact claim where it stopped without repeating completed calls. To force a clean re-run, supply `--no-resume`.
2. **Exclusion Logging**: Claims skipped due to formatting anomalies or missing text are recorded in `excluded_claims.jsonl` with timestamps and rejection rationales.
3. **Failed Calls Audit**: Model parse failures or internal errors are recorded in `failed_calls.jsonl` alongside the raw LLM output.
4. **Result Artifacts**:
   - `predictions.jsonl`: Full provenance for every verdict (timings, queries, retrieved passages, model metadata).
   - `report.json`: Machine-readable P9 evaluation report.
   - `summary_metrics.csv` & `per_class_metrics.csv`: Standardized tabular metrics.
   - `coverage_vs_accuracy.svg`: Standalone SVG visualization of confidence-coverage trade-offs.
   - `results/experiments/summary_table.md`: Comprehensive Markdown summary table comparing all executed experiments.
   - `results/experiments/comparison_plot.svg`: Comparative bar chart across all research questions.
