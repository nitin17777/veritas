# VERITAS - Evidence-Grounded RAG + LLM Fact Verification

> **V**erification through **E**vidence **R**etrieval, **I**nterpretation, **T**emporal **A**nalysis, and **S**ource-credibility scoring.

VERITAS is a modular research system for automated fake-news and claim verification. It combines **BM25 / dense / hybrid retrieval**, **credibility and temporal scoring**, **NLI-based faithfulness checking**, and an **LLM verification step** into a single, reproducible pipeline — plus an interactive Streamlit demo and a full experimental suite covering seven research questions.

---

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
4. [Quickstart](#quickstart)
5. [Configuration](#configuration)
6. [Research Questions and Experiments](#research-questions-and-experiments)
7. [Key Scripts](#key-scripts)
8. [Testing](#testing)
9. [Limitations and Disclaimers](#limitations-and-disclaimers)
10. [Reproducibility](#reproducibility)

---

## Features

| Module | What it does |
|--------|-------------|
| **Claim extraction** (`extraction`) | Detects check-worthy factual claims from long text and generates search queries |
| **Retrieval** (`retrieval`) | BM25 (rank-bm25), dense (FAISS + sentence-transformers), or hybrid (RRF fusion) |
| **Credibility scoring** (`scoring`) | Tier-based source-trust weighting from `configs/credibility_tiers.yaml` |
| **Temporal scoring** (`scoring`) | Exponential decay on evidence age (configurable tau) |
| **Verification** (`verification`) | LLM-based verdict generation; supports mock, OpenAI, and local backends |
| **Faithfulness gate** (`faithfulness`) | NLI entailment check; abstains to *Unverified* when evidence support is insufficient |
| **Evaluation** (`evaluation`) | Accuracy, Macro-F1, per-class P/R/F1, confusion matrix, Recall@k, nDCG@k, latency percentiles |
| **Error analysis** | CSV export of misclassified claims with evidence, categories, and human-annotation columns |
| **Streamlit demo** (`app/`) | Interactive UI — Benchmark Corpus mode and Live Web Search mode |
| **Experiment runner** | Runs all 20 YAML-configured experiments; saves predictions, metrics, and traces |

---

## Project Structure

```
veritas/
├── app/
│   └── streamlit_app.py          # Streamlit interactive demo
├── configs/
│   ├── experiments/              # 20 YAML configs (RQ1–RQ7 + baselines)
│   ├── credibility_tiers.yaml    # Source-domain trust tiers
│   ├── default.yaml              # Default pipeline config
│   ├── faithfulness.yaml         # Abstention gate settings
│   ├── pipeline.yaml             # Full pipeline settings
│   ├── scoring.yaml
│   └── verification.yaml
├── data/
│   ├── indexes/                  # Built BM25 / FAISS indexes (generated)
│   └── processed/                # Preprocessed LIAR / FEVER JSONL files (generated)
├── docs/                         # Module-level documentation
├── results/
│   ├── experiments/              # Per-experiment reports and verdicts
│   ├── evaluation/               # Aggregate CSV metrics and plots
│   └── error_analysis/           # errors_for_review.csv (generated)
├── scripts/
│   ├── build_bm25_index.py
│   ├── build_dense_index.py
│   ├── evaluate_veritas.py
│   ├── export_errors.py
│   ├── run_experiments.py
│   ├── run_pipeline_batch.py
│   ├── summarize_errors.py
│   ├── train_tfidf_lr.py
│   └── train_transformer.py
├── src/veritas/
│   ├── data_prep/                # Dataset loaders, label mapping, date utils
│   ├── demo/                     # Demo engine + live web search
│   ├── evaluation/               # Metrics and report generation
│   ├── experiments/              # Runner, config loader, fixtures
│   ├── extraction/               # Claim extraction and query generation
│   ├── faithfulness/             # NLI entailment and abstention gate
│   ├── models/                   # TF-IDF/LR and transformer wrappers
│   ├── pipeline/                 # VeritasPipeline.verify() orchestrator
│   ├── retrieval/                # BM25, dense, hybrid retrievers
│   ├── scoring/                  # Credibility + temporal scoring
│   └── verification/             # Prompt builder, LLM backend adapters
├── tests/                        # Full pytest suite (584 tests)
├── pyproject.toml
└── requirements.txt
```

---

## Requirements

- Python >= 3.10
- PyTorch (any recent stable release)
- transformers >= 4.40
- sentence-transformers >= 3.0.0
- faiss-cpu >= 1.8.0
- rank-bm25 >= 0.2.2
- scikit-learn >= 1.4
- streamlit >= 1.66
- PyYAML 6.0.3

> GPU is optional but recommended for the dense retriever and NLI model. CPU fallback is fully supported.

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/nitin17777/veritas.git
cd veritas

# Create and activate a virtual environment
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
pip install streamlit trafilatura       # demo extras
pip install -e .                        # install veritas package (editable)
```

### 2. Prepare data

VERITAS expects datasets preprocessed into JSONL under `data/processed/`.

```bash
# Convert raw LIAR TSV files
python scripts/validate_liar.py \
    --train  data/raw/liar/train.tsv \
    --valid  data/raw/liar/valid.tsv \
    --test   data/raw/liar/test.tsv \
    --output data/processed/

# Build retrieval indexes
python scripts/build_bm25_index.py \
    --corpus data/processed/liar_test.jsonl \
    --output data/indexes/bm25_default

python scripts/build_dense_index.py \
    --corpus data/processed/liar_test.jsonl \
    --output data/indexes/dense_default \
    --model  sentence-transformers/all-MiniLM-L6-v2
```

Download **LIAR**: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip  
Download **FEVER**: https://fever.ai/dataset/fever.html

### 3. Run the Modern VERITAS Web Application (Figma-Engineered React UI)

The new frontend is a responsive React application built with TypeScript, Tailwind CSS, and Vite, served by a Starlette REST API:

```bash
# Start the unified server (serves both REST API and compiled React UI):
python -m uvicorn app.api_server:app --port 8000 --host 0.0.0.0

# Open http://localhost:8000 in your browser.
```

To run the frontend in development mode with hot-reloading:
```bash
# In one terminal, start the API backend:
python -m uvicorn app.api_server:app --port 8000

# In a second terminal, start Vite dev server:
cd frontend
npm install
npm run dev

# Open the Vite URL shown (e.g. http://localhost:8443)
```

### 3b. Legacy Streamlit demo (optional)

```bash
# Windows
streamlit run app\streamlit_app.py --server.port 8501

# macOS / Linux
streamlit run app/streamlit_app.py --server.port 8501
```

Open **http://localhost:8501** in your browser.

Two modes are available via the sidebar:
- **Benchmark Corpus** — verifies a claim against the pre-built local index.
- **Live Web Search** — retrieves live evidence via DuckDuckGo + Trafilatura. No API key required.

> If port 8501 is busy, change `--server.port` to any free port (e.g. `8502`).

### 4. Run the experiment suite

```bash
# Run all 20 experiments
python scripts/run_experiments.py \
    --config-dir configs/experiments/ \
    --results-dir results/experiments/

# Run one experiment
python scripts/run_experiments.py \
    --config configs/experiments/rq1_rag.yaml
```

Results are saved under `results/experiments/<name>/`. A cross-experiment summary is written to `results/experiments/summary_table.csv`.

### 5. Evaluate predictions

```bash
python scripts/evaluate_veritas.py \
    --predictions results/experiments/overall_full_system/verdicts.jsonl \
    --gold        data/processed/liar_test.jsonl \
    --output-dir  results/evaluation/
```

Outputs: `evaluation_report.json`, `summary_metrics.csv`, `per_class_metrics.csv`, `coverage_vs_accuracy.svg`.

### 6. Run error analysis

```bash
# Export misclassified claims
python scripts/export_errors.py \
    --predictions results/experiments/overall_full_system/verdicts.jsonl \
    --gold        data/processed/liar_test.jsonl \
    --output      results/error_analysis/errors_for_review.csv

# After human annotation, generate a failure-pattern report
python scripts/summarize_errors.py \
    --errors  results/error_analysis/errors_for_review.csv \
    --output  results/error_analysis/summary_report.md
```

---

## Configuration

All behaviour is driven by YAML files - no hard-coded paths or secrets.

| File | Purpose |
|------|---------|
| `configs/default.yaml` | Top-level pipeline defaults |
| `configs/pipeline.yaml` | Retrieval, scoring, verification, faithfulness sub-configs |
| `configs/verification.yaml` | LLM backend (`mock`, `openai`, `local`), model name, prompt mode |
| `configs/faithfulness.yaml` | NLI model, entailment threshold, abstention mode |
| `configs/scoring.yaml` | alpha/beta/gamma weight blending, credibility, temporal tau |
| `configs/credibility_tiers.yaml` | Domain to trust-score mapping |
| `configs/experiments/*.yaml` | One file per experiment; only the comparison factor differs |

**To use a real LLM backend**, edit `configs/verification.yaml`:

```yaml
verification:
  backend: openai          # or "local"
  model: gpt-4o-mini
  temperature: 0.0
  max_tokens: 512
```

Set your API key as an environment variable (never in the YAML):

```bash
# Linux / macOS
export OPENAI_API_KEY=sk-...

# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
```

---

## Research Questions and Experiments

| RQ | Configs | Factor compared |
|----|---------|----------------|
| **RQ1** — LLM-only vs. RAG | `rq1_llm_only`, `rq1_rag` | Retrieval on / off |
| **RQ2** — Retrieval type | `rq2_bm25`, `rq2_dense`, `rq2_hybrid` | BM25 / dense / hybrid |
| **RQ3** — Credibility scoring | `rq3_credibility`, `rq3_no_credibility`, `rq3_random_credibility` | Trust tiers on / off / random |
| **RQ4** — Temporal scoring | `rq4_temporal`, `rq4_no_temporal` | Temporal decay on / off |
| **RQ5** — Prompt style | `rq5_evidence_constrained`, `rq5_free_prompt` | Evidence-anchored vs. free prompt |
| **RQ6** — Generalisation | `rq6_in_domain` (LIAR), `rq6_held_out` (FEVER) | In-domain vs. held-out dataset |
| **RQ7** — Abstention | `rq7_abstention`, `rq7_no_abstention` | Faithfulness gate on / off |
| **Overall** | `overall_full_system`, `overall_baseline_bm25_only`, `overall_baseline_llm_only` | Full system vs. baselines |

Each config pair differs in **exactly one factor**; all other settings are inherited from `configs/base.yaml`.

---

## Key Scripts

| Script | Description |
|--------|-------------|
| `scripts/validate_liar.py` | Validates and converts raw LIAR TSV to clean JSONL |
| `scripts/build_bm25_index.py` | Builds a rank-bm25 index from a JSONL corpus |
| `scripts/build_dense_index.py` | Builds a FAISS index with sentence-transformer embeddings |
| `scripts/run_experiments.py` | Runs all or selected YAML-configured experiments |
| `scripts/run_pipeline_batch.py` | Verifies a batch of claims against a custom corpus |
| `scripts/evaluate_veritas.py` | Computes the full VERITAS metric suite |
| `scripts/evaluate_abstention_thresholds.py` | Sweeps confidence thresholds and outputs coverage-vs-accuracy curve |
| `scripts/export_errors.py` | Exports misclassified claims to an annotatable CSV |
| `scripts/summarize_errors.py` | Reads annotated CSV and writes a failure-pattern report |
| `scripts/train_tfidf_lr.py` | Trains a TF-IDF + Logistic Regression baseline |
| `scripts/train_transformer.py` | Fine-tunes a transformer classifier |
| `scripts/tune_scoring_weights.py` | Grid-searches alpha/beta/gamma blending weights on a validation split |

---

## Testing

```bash
# Run the full test suite
pytest

# Verbose output
pytest -v

# Run a specific module
pytest tests/evaluation/
```

All 584 tests pass. Tests use small synthetic predictions with known expected metrics — no real LLM calls or network access required.

---

## Limitations and Disclaimers

1. **Mock backend** — The default experiment configs use `backend: mock`. No real LLM is called. Swap to a real backend before interpreting classification metrics.
2. **Smoke-test scale** — Published smoke-test results use `max_claims: 20`. These are not representative of full-dataset performance.
3. **NLI faithfulness proxy** — The faithfulness score is an NLI entailment heuristic, not a ground-truth measure of factual accuracy.
4. **LIAR label mapping** — LIAR uses six truth grades; VERITAS maps them to five labels. See `src/veritas/data_prep/` for the exact mapping.
5. **No training on the test split** — The test JSONL is used exclusively for evaluation. Use the validation split for any tuning.
6. **Live search mode** — Web evidence is retrieved at query time. Freshness and availability depend on external providers. No fabricated metadata is ever inserted.
7. **VERITAS assists human judgment and does not guarantee factual correctness.**

---

## Reproducibility

Every saved result includes full provenance metadata (run ID, timestamp, model version, dataset name, dataset version, seed). To reproduce a saved run exactly:

```bash
python scripts/run_experiments.py \
    --config configs/experiments/rq1_rag.yaml \
    --seed 42
```

Results are deterministic when the same virtual environment, package versions, seed (default: 42), and backend are used.

---

> *"VERITAS assists human judgment and does not guarantee factual correctness."*
