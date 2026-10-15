# VERITAS Data Source Manifest & Ingestion Guide

This document records the source URL, licensing terms, actual file format, split sizes, field schemas, original label distributions, and proposed label mappings for all supported datasets in the VERITAS fact verification system.

---

## Supported Datasets Summary

| Dataset | Format | Total Records | Official Splits | License | Mapping Status |
|---|---|---:|---|---|---|
| **LIAR / LIAR-RAW** | TSV (14 cols) | 12,836 | Train: 10,269 / Valid: 1,284 / Test: 1,283 | CC BY 4.0 | **Activated** |
| **FEVER** | JSONL | 185,445 | Train: 145,449 / Dev: 19,998 / Test: 19,998 | CC BY-SA 3.0 | **Activated** |
| **RAWFC** | JSON / JSONL | 1,543 | Train: 1,234 / Val: 154 / Test: 155 | Academic / Research | **Proposed (Pending Review)** |
| **AVeriTeC** | JSON | 4,100 | Train: 3,280 / Dev: 410 / Test: 410 | CC BY 4.0 | **Proposed (Pending Review)** |
| **Custom Recent** | JSONL | Flexible | Train / Validation / Test | Internal / CC BY-NC 4.0 | **Activated** |

---

## Detailed Dataset Specifications

### 1. LIAR / LIAR-RAW
* **Paper:** William Yang Wang. *"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection.* ACL 2017.
* **Source URL:** https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
* **File Format:** TSV without quotation encapsulation. Fields are strictly tab-delimited, preserving literal quotes in statement text.
* **Expected Row Counts:**
  * `train.tsv`: 10,269 records
  * `valid.tsv`: 1,284 records
  * `test.tsv`: 1,283 records
  * Total: 12,836 records
* **Fields (14 columns):**
  1. `id`: Claim identifier (e.g. `2635.json`)
  2. `label`: Veracity rating
  3. `statement`: PolitiFact claim text
  4. `subject`: Topic tags
  5. `speaker`: Speaker name
  6. `speaker_job`: Job title
  7. `state_info`: State or jurisdiction
  8. `party_affiliation`: Political party
  9. `barely_true_counts`: Count of historical barely true ratings
  10. `false_counts`: Count of historical false ratings
  11. `half_true_counts`: Count of historical half true ratings
  12. `mostly_true_counts`: Count of historical mostly true ratings
  13. `pants_fire_counts`: Count of historical pants on fire ratings
  14. `context`: Venue or medium of the statement
* **Original Label Distribution:**
  * Train: `half-true`: 2,123, `false`: 1,998, `mostly-true`: 1,966, `true`: 1,683, `barely-true`: 1,657, `pants-fire`: 842.
* **VERITAS Label Mapping (Activated):**
  * `pants-fire` $\to$ `False`
  * `false` $\to$ `False`
  * `barely-true` $\to$ `Misleading`
  * `half-true` $\to$ `Partially True`
  * `mostly-true` $\to$ `True`
  * `true` $\to$ `True`

---

### 2. FEVER (Fact Extraction and VERification)
* **Paper:** James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. *FEVER: a large-scale dataset for Fact Extraction and VERification.* NAACL-HLT 2018.
* **Source URL:** https://fever.ai/resources.html
  * Claims: `https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl`
  * Dev: `https://s3-eu-west-1.amazonaws.com/fever.public/paper_dev.jsonl`
  * Test: `https://s3-eu-west-1.amazonaws.com/fever.public/paper_test.jsonl`
* **License:** CC BY-SA 3.0 / Wikipedia Terms of Use
* **File Format:** JSONL (each row is a JSON dictionary)
* **Expected Row Counts:**
  * `train.jsonl`: 145,449 claims
  * `paper_dev.jsonl`: 19,998 claims
  * `paper_test.jsonl`: 19,998 claims
  * Total: 185,445 claims
* **Fields:**
  * `id`: Integer claim ID
  * `claim`: Sentence claim
  * `label`: String rating (`SUPPORTS`, `REFUTES`, `NOT ENOUGH INFO`)
  * `verifiable`: `VERIFIABLE` or `NOT VERIFIABLE`
  * `evidence`: Nested list of annotation sets pointing to `[annotation_id, evidence_id, wikipedia_title, sentence_id]`
* **Original Label Distribution (Train):**
  * `SUPPORTS`: 80,035 (55.0%)
  * `NOT ENOUGH INFO`: 35,639 (24.5%)
  * `REFUTES`: 29,775 (20.5%)
* **VERITAS Label Mapping (Activated):**
  * `SUPPORTS` / `Supported` $\to$ `True`
  * `REFUTES` / `Refuted` $\to$ `False`
  * `NOT ENOUGH INFO` / `Not Enough Info` $\to$ `Unverified`
* **Grounding Principles:**
  * Claims lack historical publication dates $\to$ `claim_date` is `None` (never invent dates).
  * Evidence text is external (Wikipedia dump) $\to$ `evidence` list is `[]`, and structured pointers are preserved under `metadata.raw_evidence`.

---

### 3. RAWFC (Snopes-Sourced Raw Social Media Reports)
* **Paper:** Zhiwei Hu, et al. *A Coarse-to-fine Cascaded Evidence-Distillation Neural Network for Explainable Fake News Detection.* COLING 2022.
* **Source URL:** https://github.com/Nicozwy/CofCED / https://opendatalab.com/OpenDataLab/RAWFC
* **License:** Research & Academic Use (Snopes copyright terms apply to underlying claims)
* **File Format:** JSON array or JSONL with claim details and background reports.
* **Expected Row Counts:**
  * `train.json`: 1,234
  * `val.json`: 154
  * `test.json`: 155
  * Total: 1,543
* **Fields:**
  * `id`: String identifier (e.g. `snopes_001`)
  * `claim`: Claim text
  * `label`: Snopes rating string
  * `date`: Fact-check date string
  * `topic`: Topic category
  * `reports`: List of background documents containing `text`, `url`, `date`, `domain`
* **Original Snopes Label Vocabulary:**
  * `true`, `mostly true`, `mixture` / `half true`, `mostly false`, `false`, `unproven`, `undetermined`
* **Proposed Label Mapping for Human Review (PENDING REVIEW):**
  * `true` $\to$ `True`
  * `mostly true` $\to$ `True`
  * `mixture` / `half true` / `half` $\to$ `Partially True`
  * `mostly false` $\to$ `Misleading`
  * `false` $\to$ `False`
  * `unproven` / `undetermined` $\to$ `Unverified`
* **Review Decision Needed:**
  * *Question for team review:* Should Snopes `mixture` be mapped to `Partially True` or `Misleading`? The proposed default is `Partially True` because mixture denotes statements containing elements of both truth and falsehood.

---

### 4. AVeriTeC (Real-World Web Claim Verification)
* **Paper:** Michael Schlichtkrull, et al. *AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web.* NeurIPS 2023.
* **Source URL:** https://github.com/MichSchli/AVeriTeC
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
* **File Format:** JSON with claim objects and QA search decomposition.
* **Expected Row Counts:**
  * `train.json`: 3,280
  * `dev.json`: 410
  * `test.json`: 410
  * Total: 4,100
* **Fields:**
  * `claim_id`: Unique integer/string ID
  * `claim`: Extracted real-world statement
  * `label`: Veracity label
  * `claim_date`: Utterance/fact-check date
  * `speaker`: Claim speaker/source
  * `questions`: List of decomposition questions, each containing `answers` with `answer` text, `source_url`, and `date`.
* **Original Label Vocabulary (Exact 4 Classes):**
  1. `Supported`
  2. `Refuted`
  3. `Conflicting Evidence/Cherry-picking`
  4. `Not Enough Evidence`
* **Original Label Distribution (Train):**
  * `Supported`: 1,312 (40%)
  * `Refuted`: 1,312 (40%)
  * `Conflicting Evidence/Cherry-picking`: 328 (10%)
  * `Not Enough Evidence`: 328 (10%)
* **Proposed Label Mapping for Human Review (PENDING REVIEW):**
  * `Supported` $\to$ `True`
  * `Refuted` $\to$ `False`
  * `Conflicting Evidence/Cherry-picking` $\to$ `Misleading`
  * `Not Enough Evidence` $\to$ `Unverified`
* **Review Decision Needed:**
  * *Question for team review:* We propose mapping `Conflicting Evidence/Cherry-picking` to `Misleading` because cherry-picking specifically constitutes intentional omission of contradictory facts to mislead, which matches VERITAS's definition of `Misleading`. An alternative option is `Partially True`.

---

### 5. Custom Recent Dataset (2025–2026 Feed)
* **Source:** Internal curated feed / PolitiFact recent dumps / Google Fact Check Tools API
* **License:** Internal / CC BY-NC 4.0
* **File Format:** JSONL conforming directly to ClaimRecord.
* **Fields:** `id`, `claim`, `label`, `claim_date`, `topic`, `evidence`, `metadata`.
* **VERITAS Labels:** Native 5-class labels (`True`, `Partially True`, `Misleading`, `False`, `Unverified`).

---

## Usage Guide & Commands

### 1. Inspect Data Source Manifest
```pwsh
python -m veritas.data_prep.prepare --manifest
```

### 2. Ingest Configured Datasets
```pwsh
python -m veritas.data_prep.prepare --config configs/default.yaml
```

### 3. Ingest Specific Datasets
```pwsh
python -m veritas.data_prep.prepare --datasets liar fever
```

### 4. Evaluate Proposed Mappings for RAWFC and AVeriTeC
```pwsh
python -m veritas.data_prep.prepare --datasets rawfc averitec --approve-proposed-mappings
```

### 5. Run the Ingestion Test Suite
```pwsh
pytest tests/data_prep -v
```
