# VERITAS: Comprehensive Experimental Evaluation & Results

> **Research Integrity Notice & Provenance Disclosure**  
> This document compiles all empirical results, baseline evaluations, metric reports, and experimental findings from the VERITAS fact-verification project. Every number reported here is grounded in actual executed test artifacts present in the repository (`results/`). Experiments configured but not yet executed at full dataset scale with a production LLM are explicitly designated as **[PENDING]** with exact reproduction instructions. No metric is fabricated or hypothetically approximated.

---

## Executive Summary of Results

| Experiment / System | Dataset & Split | Sample Size ($n$) | Accuracy | Macro-F1 | Precision | Recall | Status & Source |
|---|---|---|---|---|---|---|---|
| **TF-IDF + Logistic Regression** | LIAR (Test) | 1,283 | **0.3663** | **0.3391** | 0.3389 | 0.3394 | **COMPLETED** (`results/20261015T191254Z_ef7e781d`) |
| **TF-IDF + Logistic Regression** | LIAR (Validation) | 1,284 | **0.3847** | **0.3635** | 0.3645 | 0.3633 | **COMPLETED** (`results/20261015T191254Z_ef7e781d`) |
| **DistilRoBERTa Classifier** | LIAR (Test) | 16 | 0.2500* | 0.1000* | 0.0625* | 0.2500* | Smoke-test only (`20261015T192753Z_104c11a7`); CPU constrained |
| **LLM-Only (Direct Prompt)** | LIAR (Test) | 1,283 | [PENDING] | [PENDING] | [PENDING] | [PENDING] | Scripted (`configs/experiments/rq1_llm_only.yaml`) |
| **Standard RAG + LLM** | LIAR (Test) | 1,283 | [PENDING] | [PENDING] | [PENDING] | [PENDING] | Scripted (`configs/experiments/rq1_rag.yaml`) |
| **Full VERITAS Pipeline** | LIAR (Test) | 1,283 | [PENDING] | [PENDING] | [PENDING] | [PENDING] | Scripted (`configs/experiments/overall_full_system.yaml`) |

*\*Note on DistilRoBERTa: The smoke-test run utilized only 32 training and 16 test instances on CPU for pipeline sanity checking; it collapsed to predicting the majority class and is not representative of full-scale transformer capability.*

---

# 8. Experimental Evaluation

## 8.1 Experimental Setup

### 8.1.1 Benchmark Datasets and Ingestion Integrity

The primary dataset evaluated in this project is **LIAR** (Wang, ACL 2017), containing 12,836 real-world political claims extracted from PolitiFact. Each claim was annotated by professional fact-checkers under a 6-point truth rating system: *pants-fire*, *false*, *barely-true*, *half-true*, *mostly-true*, and *true*.

VERITAS maps these six fine-grained grades into its canonical verdict schema via a deterministic, verified transformation table:

**Table 8.1 — LIAR to VERITAS Label Mapping**

| Raw LIAR Label | VERITAS Canonical Label | Rationale / Grounding |
|---|---|---|
| `true` | **True** | Factual statement verified by primary sources. |
| `mostly-true` | **True** | Factually accurate with minor, non-distorting omission. |
| `half-true` | **Partially True** | Contains accurate elements but omits vital context. |
| `barely-true` | **Misleading** | True in technical detail but completely misleading in framing. |
| `false` | **False** | Substantively incorrect or contradicted by evidence. |
| `pants-fire` | **False** | Egregiously false fabrication or conspiracy. |

#### Ingestion and Split Verification
During standard CSV parsing, standard parsers dropped 45 physical records due to unescaped double quotes inside statements. VERITAS deployed a dedicated robust TSV parser with quote-boundary recovery (`repair_report.md`), successfully recovering 100% of records:

**Table 8.2 — Verified Dataset Split Statistics**

| Split | Total Records | True | Partially True | Misleading | False | Unverified |
|---|---|---|---|---|---|---|
| **Train** | 10,269 | 3,649 (35.5%) | 2,123 (20.7%) | 1,657 (16.1%) | 2,840 (27.7%) | 0 (0.0%) |
| **Validation** | 1,284 | 420 (32.7%) | 248 (19.3%) | 237 (18.5%) | 379 (29.5%) | 0 (0.0%) |
| **Test** | 1,283 | 460 (35.9%) | 267 (20.8%) | 214 (16.7%) | 342 (26.7%) | 0 (0.0%) |
| **Total** | **12,836** | **4,529 (35.3%)** | **2,638 (20.6%)** | **2,108 (16.4%)** | **3,561 (27.7%)** | **0** |

*Note: In the benchmark ground-truth corpus, zero claims are labeled `Unverified`. The `Unverified` verdict is produced dynamically by the VERITAS system when the NLI Faithfulness Gate abstains due to insufficient or ungrounded evidence.*

### 8.1.2 Leakage Prevention and Evaluation Protocol

To prevent data contamination:
1. **Strict Partitioning**: The 1,283 test samples were locked during development. All feature engineering and hyperparameter sweeps were conducted strictly on `train` and validated on `validation`.
2. **Index Isolation**: Retrieval indexes used during claim testing are built without ground-truth verdict labels or annotator rationale strings.
3. **Deterministic Seeding**: Global random seed `42` is fixed across all tokenizers, splitters, cross-encoders, and classifiers.

### 8.1.3 Hardware and Execution Environment

- **OS**: Microsoft Windows 11 Enterprise (64-bit)
- **Runtime**: Python 3.11.0, PyTorch CPU, Hugging Face `transformers` 4.40+, `sentence-transformers` 3.0+, `faiss-cpu` 1.8.0, `scikit-learn` 1.4+.
- **Execution Target**: CPU workstation (no CUDA GPU active during automated runs).

### 8.1.4 Pipeline Parameters and Scoring Weights

The VERITAS retrieval and ranking framework uses the following verified configurations:
- **BM25 Lexical Retrieval**: Okapi BM25 ($k_1 = 1.5, b = 0.75$), token chunk size 150, overlap 20 tokens.
- **Dense Semantic Retrieval**: `BAAI/bge-small-en-v1.5` embeddings, L2 normalized, FAISS inner-product vector indexing.
- **Reciprocal Rank Fusion (RRF)**: Constant $k = 60$, fusing top-50 BM25 and top-50 dense candidate passages.
- **Cross-Encoder Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`, sequence length 512 tokens, sigmoid score normalization.
- **Combined Scoring Formula**:
  $$\text{Score}_{\text{combined}} = 0.60 \cdot \text{Relevance} + 0.25 \cdot \text{Credibility} + 0.15 \cdot \text{Temporal}$$
- **Source Credibility Tiers**:
  - Tier 1 (0.90): Wire services, international fact-checkers (Reuters, AP, Snopes, PolitiFact, WHO, CDC, Nature).
  - Tier 2 (0.75): Established national/international news with editorial corrections policies (BBC, NYT, WSJ, Guardian).
  - Tier 3 (0.50): Mainstream aggregators and regional press (USA Today, Politico, Time).
  - Tier 4 (0.25): Tabloids, unvetted blogs, opinion commentary.
  - Neutral Prior (0.50): Unrecognized domains.
- **Temporal Decay**: Exponential decay $\exp(-\Delta t / \tau)$ with default $\tau = 180$ days (adjusted by claim domain: Breaking = 14d, Politics = 90d, Health = 365d, Science = 730d).

---

## 8.2 Classification Performance

### 8.2.1 Primary Metric: Macro-F1

Given the 2.1× imbalance between the plurality class (*True*, 35.9%) and the minority class (*Misleading*, 16.7%), standard classification accuracy is prone to majority-class bias. **Macro-F1** (unweighted mean of class F1 scores) serves as the primary metric because it awards equal importance to the detection of nuanced *Partially True* and *Misleading* claims.

### 8.2.2 Full Scale Baseline: TF-IDF + Logistic Regression

- **Run ID**: `20261015T191254Z_ef7e781d`
- **Model**: Word + Bigram TF-IDF (100,000 capped features, sublinear TF) + SAGA Logistic Regression.
- **Hyperparameter Tuning**: Regularization parameter $C \in \{0.01, 0.1, 0.5, 1.0, 5.0, 10.0\}$ swept against validation split Macro-F1:
  - $C=0.01$: $0.3488$
  - $C=0.10$: $0.3451$
  - $C=0.50$: $0.3526$
  - **$C=1.00$: $0.3635$ (Optimal, locked for test)**
  - $C=5.00$: $0.3455$
  - $C=10.00$: $0.3395$

**Table 8.3 — Full Test Evaluation Metrics ($n = 1,283$)**

| Metric | Score |
|---|---|
| **Accuracy** | **0.3663** (36.63%) |
| **Macro Precision** | **0.3389** |
| **Macro Recall** | **0.3394** |
| **Macro-F1** | **0.3391** |

**Table 8.4 — Per-Class Test Performance (TF-IDF + LR)**

| Verdict Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **True** | 0.4806 | 0.4587 | **0.4694** | 460 |
| **Partially True** | 0.2481 | 0.2472 | **0.2477** | 267 |
| **Misleading** | 0.2242 | 0.2336 | **0.2288** | 214 |
| **False** | 0.4028 | 0.4181 | **0.4103** | 342 |
| **Unverified** | 0.0000 | 0.0000 | 0.0000 | 0 |

**Table 8.5 — Test Confusion Matrix (Rows: Ground Truth, Columns: Predicted)**

| Ground Truth \ Predicted | True | Partially True | Misleading | False |
|---|---|---|---|---|
| **True** ($n=460$) | **211** | 94 | 68 | 87 |
| **Partially True** ($n=267$) | 99 | **66** | 42 | 60 |
| **Misleading** ($n=214$) | 53 | 46 | **50** | 65 |
| **False** ($n=342$) | 76 | 60 | 63 | **143** |

#### Analysis
The lexical model performs best at identifying unambiguous endpoints (*True* F1: 0.4694, *False* F1: 0.4103), but struggles severely on nuanced intermediate categories (*Misleading* F1: 0.2288, *Partially True* F1: 0.2477). This confirms that surface lexical cues cannot reliably distinguish partial truth or misleading context from outright falsehood without external retrieved evidence.

---

## 8.3 Information Retrieval Performance

Information retrieval metrics assess the quality of the multi-stage search subsystem in surfacing evidence passages that verify or refute submitted statements.

**Table 8.6 — Retrieval Architecture Comparison [Controlled Evaluation]**

| Subsystem Mode | Recall@5 | Recall@10 | Precision@5 | MRR@10 | nDCG@10 | Status |
|---|---|---|---|---|---|---|
| **BM25 Only** | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | Evaluates lexical match |
| **Dense (BGE-small) Only** | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | Evaluates semantic embedding |
| **Hybrid (BM25 + Dense)** | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | Evaluates candidate pooling |
| **Hybrid + RRF ($k=60$)** | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | Evaluates reciprocal rank fusion |
| **Hybrid + RRF + Reranker** | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | Evaluates cross-encoder scoring |

*Methodological Note: LIAR statements do not include full-text evidence document annotations. Complete retrieval evaluation requires benchmarking against the FEVER dataset knowledge base, currently planned as the primary task upon external corpus ingestion.*

---

## 8.4 Baseline Model Comparison

To contextualize the full VERITAS architecture, four baseline paradigms are compared:

**Table 8.7 — Model Comparison across Approaches on LIAR Benchmark**

| Model Architecture | Accuracy | Macro-F1 | Modality & Hardware | Status |
|---|---|---|---|---|
| **TF-IDF + Logistic Regression** | **0.3663** | **0.3391** | Surface text only (CPU) | **Completed** ($n=1,283$) |
| **DistilRoBERTa Sequence Classifier** | [PENDING] | [PENDING] | Transformer (Fine-tuned on LIAR) | Pending GPU run (Smoke test: $n=16$) |
| **LLM-Only Direct Verification** | [PENDING] | [PENDING] | Parametric zero-shot | Scripted (`rq1_llm_only.yaml`) |
| **Standard RAG + LLM** | [PENDING] | [PENDING] | Retrieval + unconstrained LLM | Scripted (`rq1_rag.yaml`) |
| **Full VERITAS Pipeline** | [PENDING] | [PENDING] | Hybrid RAG + Scoring + NLI Gate | Scripted (`overall_full_system.yaml`) |

---

## 8.5 Ablation Study

The VERITAS ablation suite isolates each component to measure its individual contribution to overall verification accuracy, faithfulness, and abstention behavior.

**Table 8.8 — Ablation Suite Architecture & Experimental Factor Table**

| Run ID | Factor Under Investigation | Systematic Modification | Status |
|---|---|---|---|
| `overall_full_system` | Full VERITAS Architecture | All stages active ($\alpha=0.60, \beta=0.25, \gamma=0.15$, NLI gate active) | Scripted |
| `rq2_bm25` | Lexical Retrieval Only | Disable dense semantic vectors and RRF | Scripted |
| `rq2_dense` | Dense Retrieval Only | Disable BM25 lexical retriever | Scripted |
| `rq2_hybrid` | Hybrid without Cross-Encoder | Disable cross-encoder reranking | Scripted |
| `rq3_no_credibility` | Credibility Scoring Removed | Set credibility weight $\beta = 0.0$ | Scripted |
| `rq3_random_credibility` | Credibility Scoring Randomized | Assign uniform random domain trust scores | Scripted |
| `rq4_no_temporal` | Temporal Decay Removed | Set temporal weight $\gamma = 0.0$ | Scripted |
| `rq5_free_prompt` | Unconstrained Prompting | Remove evidence citation constraints from prompt | Scripted |
| `rq7_no_abstention` | Faithfulness Gate Disabled | Disable NLI abstention downgrade | Scripted |

---

## 8.6 Evidence Faithfulness & Abstention Analysis

### 8.6.1 NLI-Grounded Faithfulness
VERITAS introduces an automated safety mechanism:
1. Every generated explanation sentence is treated as a hypothesis.
2. The retrieved passages are treated as premises.
3. A cross-encoder NLI model (`cross-encoder/nli-deberta-v3-small`) predicts entailment, contradiction, or neutral relation.
4. **Epistemic Distinction**: NLI measures *premise-hypothesis entailment* (whether the explanation accurately reflects the cited text), not ground-truth factual reality. If retrieved sources contain errors, NLI will still confirm entailment.

### 8.6.2 Abstention Thresholds & Coverage-versus-Accuracy
If model confidence $< 0.60$ or explanation faithfulness $< 0.70$, the pipeline downgrades the claim to **Unverified**.

**Table 8.9 — Faithfulness and Abstention Behavior ($n = 10$, Smoke Test Verification)**

| Metric | Measured Value | Operational Meaning |
|---|---|---|
| **JSON Schema Parse Failure Rate** | **0.0%** | Zero schema non-compliance occurrences |
| **Active Abstention Downgrade** | Tested | Pipeline successfully triggers `Unverified` fallback |
| **Selective Accuracy Sweep Tool** | Verified | `evaluate_abstention_thresholds.py` generates AUARC curves |

---

## 8.7 Computational Latency & Efficiency

Latency instrumentation measures the timing overhead of each stage in the end-to-end pipeline:

**Table 8.10 — Subsystem Processing Overhead (CPU Workstation Baseline)**

| Processing Stage | Mean Latency | Median (P50) | 95th Percentile (P95) |
|---|---|---|---|
| **Query Extraction & Preprocessing** | < 1 ms | < 1 ms | < 1 ms |
| **BM25 Lexical Retrieval** | 0.8 ms | 0.7 ms | 1.2 ms |
| **Combined Evidence Scoring ($\alpha, \beta, \gamma$)** | 0.4 ms | 0.3 ms | 0.6 ms |
| **Response Schema Formatting & JSON Validation** | 0.7 ms | 0.6 ms | 1.1 ms |
| **Orchestration Overhead (Excluding LLM generation)** | **1.9 – 2.5 ms** | **1.4 – 2.3 ms** | **3.6 – 5.8 ms** |

*Finding: Non-LLM stages contribute negligible latency (< 6 ms at P95). End-to-end user latency in production will be entirely dominated by LLM response generation.*

---

## 8.8 Error Analysis & Failure Taxonomy

VERITAS defines six canonical failure modes (`configs/error_categories.yaml`):

**Table 8.11 — Diagnostic Error Taxonomy**

| Error Category | Root Cause & Diagnostic Signal | Recommended Mitigation |
|---|---|---|
| **Retrieval Miss** | Gold passages absent from top-$k$ candidate list | Expand corpus coverage; query decomposition |
| **Credibility Distortion** | Low-reputation source outranks authoritative evidence | Calibrate tier weights ($\beta$); review domain table |
| **Temporal Misalignment** | High lexical match on obsolete or outdated historical facts | Tune decay constant ($\tau$); date-conditioned filtering |
| **LLM Misinterpretation** | Evidence is present but LLM fails logical deduction or negation | Few-shot reasoning exemplars; upgrade verifier model |
| **Label Ambiguity / Noise** | Benchmark gold annotation is disputed or inherently subjective | Data auditing; isolate subjective political rhetoric |
| **Schema Parse Failure** | Output formatting breaks JSON contract | Constrained decoding; retry prompt escalation |

---

## 8.9 Reproducibility & Step-by-Step Execution Guide

To reproduce every completed result and run remaining experiments:

### 1. Reproduce TF-IDF + Logistic Regression Baseline
```powershell
python scripts/train_tfidf_lr.py --config configs/tfidf_lr.yaml
```
Output: Generates exact results in `results/<RUN_ID>/` matching Macro-F1 = 0.3391.

### 2. Run Comprehensive Unit & Integration Tests
```powershell
python -m pytest tests/demo/test_demo.py tests/evaluation/ -v
```

### 3. Run Production Full-System Verification Experiments
Once an external API key (e.g. OpenAI / Anthropic) is provided:
```powershell
# Set API Key in environment
$env:OPENAI_API_KEY = "your-api-key-here"

# Execute Full VERITAS benchmark run
python scripts/run_experiments.py --config configs/experiments/overall_full_system.yaml
```
