# VERITAS Error Analysis & Failure Pattern Workflow

This guide details the qualitative error-analysis methodology, taxonomy, export tooling, manual annotation protocol, and summarization workflow for the **VERITAS** evidence-grounded fact-verification system.

---

## 1. Objectives and Core Principles

Quantitative metrics (Accuracy, Macro-F1, Recall@k) show *how often* the system errs, but not *why*. The VERITAS error-analysis workflow bridges this gap by enabling systematic, reproducible triage of incorrect predictions.

### Methodological Principles:
1. **Raw Predictions are Immutable**: Error extraction reads prediction JSONL files as strictly read-only inputs. The original predictions, retrieved passage IDs, timestamps, and model outputs are never altered.
2. **Automated Suggestions are Not Ground Truth**: The automated suggester computes diagnostic heuristics to accelerate manual triage. These suggestions are labeled as `suggested_category` and are **never** treated as authoritative ground truth without human inspection.
3. **Annotation Persistence**: Human reviewer annotations (`annotated_category`, `annotator_id`, `annotator_notes`, `verification_status`) are retained and merged across repeated export runs, preventing rework when re-evaluating batches.
4. **Actionable Taxonomy**: Errors are partitioned into discrete, engineering-actionable categories corresponding to specific pipeline stages (retrieval, scoring, temporal alignment, LLM reasoning, annotation noise, or parsing).

---

## 2. Canonical Error Taxonomy

VERITAS defines seven canonical error categories specified in [`configs/error_categories.yaml`](file:///e:/veritas/configs/error_categories.yaml):

| Error Category | Pipeline Stage | Diagnostic Criteria | Actionable Engineering Remedy |
|:---|:---:|:---|:---|
| **Retrieval miss** | Retrieval (BM25 / Dense / RRF) | Gold evidence passages omitted from retrieved candidate pool, or zero candidates returned. | Improve query decomposition, add synonym expansion, fine-tune dense embedder, or expand corpus coverage. |
| **Incorrect credibility ranking** | Scoring & Ranking | Relevant authoritative passages (Tier 1/2) retrieved but down-ranked below low-credibility or unknown domains. | Calibrate credibility weight ($\beta$), update domain tier tables, or renormalize component weights. |
| **Stale or temporally inappropriate evidence** | Temporal Scoring | Retrieved evidence was valid when published but outdated relative to claim date, leading to temporal misalignment. | Calibrate decay parameter ($\tau$) by claim domain, enforce temporal filtering or date-conditioned queries. |
| **LLM misinterpretation** | Verifier Reasoning | Retrieved evidence contained necessary facts, but LLM misread, hallucinated, misunderstood negation, or reasoned falsely. | Refine system prompt instructions, add few-shot exemplars of nuanced reasoning, or upgrade reasoning backend. |
| **Label noise or ambiguous annotation** | Ground Truth Dataset | Benchmark gold label itself is disputed, outdated, ambiguous, or factually wrong according to authoritative consensus. | Audit benchmark dataset, flag noisy samples, or exclude from clean evaluation splits. |
| **Parsing failure** | Schema & Format Parser | Model output failed structured JSON schema validation after automated retry, forcing fallback to `Unverified`. | Strengthen JSON instruction enforcement, lower temperature, or enhance retry prompt template. |
| **Other / unknown** | Residual / Edge Cases | Anomaly not attributable to the categories above; requires exploratory human investigation. | Document in annotator notes for qualitative review. |

---

## 3. Workflow & Usage Instructions

```
  ┌────────────────────────────────────────────────────────┐
  │ 1. Run Pipeline / Experiment                           │
  │    Produces results/experiments/<name>/predictions.jsonl│
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ 2. Export Errors to Review Spreadsheet                 │
  │    python scripts/export_errors.py                     │
  │    Produces results/error_analysis/errors_for_review.csv│
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ 3. Human Qualitative Review & Annotation               │
  │    Annotators open CSV in Excel / Google Sheets        │
  │    Fill 'annotated_category', 'annotator_notes', etc.   │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ 4. Summarize Failure Patterns & Generate Report        │
  │    python scripts/summarize_errors.py                  │
  │    Produces results/error_analysis/error_analysis_report.md
  └────────────────────────────────────────────────────────┘
```

### Step 1: Export Incorrect Predictions to CSV
To export all prediction errors into a review spreadsheet:

```bash
python scripts/export_errors.py \
    --gold data/processed/liar_test.jsonl \
    --predictions results/experiments/overall_full_system/predictions.jsonl \
    --output results/error_analysis/errors_for_review.csv
```

If you re-run the export after a new prediction run, existing annotations are automatically preserved:
```bash
python scripts/export_errors.py \
    --gold data/processed/liar_test.jsonl \
    --predictions results/experiments/overall_full_system/predictions.jsonl \
    --output results/error_analysis/errors_for_review.csv \
    --existing-annotations results/error_analysis/errors_for_review.csv
```

### Step 2: Human Manual Review
Open `results/error_analysis/errors_for_review.csv` in Excel, Google Sheets, or your preferred spreadsheet tool.

Each row contains:
- `claim_id`, `claim`, `claim_date`
- `gold_label`, `predicted_label`, `confidence`
- `explanation`: Model's generated reasoning
- `retrieved_evidence_ids`, `retrieved_sources`: Passages, publication dates, and credibility scores
- `suggested_category`: Provisional heuristic suggestion (for triage guidance only)
- `suggestion_rationale`: Explanation of why the heuristic suggested this category

**Fields for the reviewer to fill:**
1. `annotated_category`: Choose strictly one of the 7 allowed categories:
   - `Retrieval miss`
   - `Incorrect credibility ranking`
   - `Stale or temporally inappropriate evidence`
   - `LLM misinterpretation`
   - `Label noise or ambiguous annotation`
   - `Parsing failure`
   - `Other / unknown`
2. `annotator_id`: Your username or identifier (e.g. `reviewer_alice`).
3. `annotator_notes`: Brief explanation justifying the diagnosis.
4. `verification_status`: Set to `verified` once confident, or `disputed` if uncertain.

### Step 3: Generate Qualitative Failure Pattern Report
Compile the reviewed spreadsheet into a comprehensive Markdown analysis report:

```bash
python scripts/summarize_errors.py \
    --errors results/error_analysis/errors_for_review.csv \
    --output-report results/error_analysis/error_analysis_report.md \
    --output-json results/error_analysis/error_statistics.json
```

The generated report includes:
- Category distribution table (counts and proportions for both suggested and human-annotated errors).
- Label confusion breakdown (Gold &rarr; Predicted label patterns).
- Error distribution by confidence bin (flagging high-confidence / overconfident errors).
- Concrete representative examples per category with full factual evidence and reasoning analysis.

---

## 4. Key Architecture Files

- [`src/veritas/evaluation/error_analysis.py`](file:///e:/veritas/src/veritas/evaluation/error_analysis.py): Core extraction, heuristic inference, annotation merging, statistics calculation, and report generation.
- [`configs/error_categories.yaml`](file:///e:/veritas/configs/error_categories.yaml): Taxonomy definitions, diagnostic rules, and annotation field schemas.
- [`scripts/export_errors.py`](file:///e:/veritas/scripts/export_errors.py): CLI tool for generating spreadsheet CSVs with annotation retention.
- [`scripts/summarize_errors.py`](file:///e:/veritas/scripts/summarize_errors.py): CLI tool for computing category statistics and compiling Markdown reports.
- [`tests/evaluation/test_error_analysis.py`](file:///e:/veritas/tests/evaluation/test_error_analysis.py): Comprehensive unit and integration test suite.
