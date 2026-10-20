# VERITAS Evidence Retrieval & BM25 Indexing (P2)

This document describes the corpus preparation pipeline, document chunking strategy, BM25 retrieval index, caching protocol, and command-line usage for VERITAS.

---

## 1. Overview

VERITAS grounds fact verification in retrieved evidence passages. Phase 2 (P2) establishes the local lexical retrieval baseline using **BM25Okapi** (`rank_bm25`).

Key design constraints enforced by the pipeline:
- **Evidence Extraction**: Passages are constructed strictly from available dataset evidence records (`record["evidence"]`).
- **Veracity Label Grounding**: A claim's veracity label (`record["label"]`) is **never** treated as evidence text.
- **No Fabricated Evidence**: Missing evidence (e.g. LIAR dataset) yields 0 passages; fabricated passages are never generated.
- **Document Chunking**: Sliding-window chunking targeting **100–200 words** with **20-word overlap** preserves local context across split boundaries.
- **Metadata Preservation**: Passage ID, passage text, source URL, ISO publication date, and domain are strictly preserved across serialization.
- **Consistent Tokenization**: Both indexing and querying share identical, lowercase, punctuation-normalized tokenization.
- **Index Caching & Determinism**: Indexes compute SHA-256 fingerprints over ordered passages and configurations; existing indexes are reused when fingerprints match.
- **Exact Index Alignment**: Passage ordering in the `PassageStore` maps 1:1 with BM25 internal document vectors.

---

## 2. Passage Data Contract

Each passage stored in the retrieval index conforms to the [Passage](file:///e:/veritas/src/veritas/retrieval/schema.py) contract:

| Field | Type | Description |
|---|---|---|
| `passage_id` / `id` | `str` | Unique passage identifier (e.g. `rawfc:snopes_001#e0#c0`) |
| `text` | `str` | Normalized body text of the evidence chunk |
| `url` | `str \| None` | Source web URL from which evidence was obtained |
| `date` / `publication_date` | `str \| None` | ISO 8601 publication date string (`YYYY-MM-DD`) |
| `domain` | `str \| None` | Fully-qualified domain (e.g. `snopes.com`, `who.int`) |
| `metadata` | `dict[str, Any]` | Provenance tracking (`source_claim_id`, `evidence_index`, etc.) |

---

## 3. Document Chunking Strategy

Implemented in [veritas.retrieval.chunking](file:///e:/veritas/src/veritas/retrieval/chunking.py):

- **Target Word Size**: 150 words (targeting approximately 100–200 words).
- **Max Single-Passage Threshold**: 200 words. Source texts $\le 200$ words remain intact as a single passage.
- **Sliding Overlap**: 20 words between consecutive chunks.
- **Trailing Remainder Absorption**: If fewer than `min_words` (20 words) remain at document end, they are absorbed into the final chunk if total words $\le 200$, preventing tiny orphan fragments.

---

## 4. Evidence Availability by Dataset

| Dataset | Evidence Availability | Ingestion Handling |
|---|---|---|
| **RAWFC** | Full raw web reports (`text`, `url`, `date`, `domain`) | Chunks reports into passages, extracts domains from URLs. |
| **AVeriTeC** | QA search pairs with answers, URLs, and scraping dates | Chunks QA answers into passages with date and domain tracking. |
| **Custom / Recent** | Flexible JSONL with `evidence` items | Extracted and chunked per contract. |
| **LIAR** | None (claim statements and speaker history only) | Ingests 0 passages. **Never** fabricates artificial passages. |
| **FEVER** | Wikipedia title/sentence pointers | Pointers stored in metadata; inline passages added if text provided. |

---

## 5. Configuration

Configured via [configs/retrieval_bm25.yaml](file:///e:/veritas/configs/retrieval_bm25.yaml) or [configs/default.yaml](file:///e:/veritas/configs/default.yaml):

```yaml
project:
  name: VERITAS
  seed: 42

retrieval:
  backend: bm25
  index_dir: data/indexes/bm25_default
  bm25_top_k: 50
  k1: 1.5
  b: 0.75
  chunk_size: 150
  chunk_overlap: 20
  datasets:
    - rawfc
    - averitec
    - custom_recent
```

---

## 6. How to Build and Query the Index

### 6.1 Building from Processed Datasets

Run using the default YAML configuration:
```bash
python scripts/build_bm25_index.py --config configs/retrieval_bm25.yaml
```

Or run via module CLI:
```bash
python -m veritas.retrieval.build_index --processed-dir data/processed --datasets rawfc custom_recent
```

### 6.2 Smoke-Test Mode (Fast Pipeline Validation)

Build a small synthetic index and immediately execute a test query:
```bash
python scripts/build_bm25_index.py --smoke-test --query "vaccination hospitalizations"
```

### 6.3 Force Rebuilding

Force re-indexing even when cached metadata matches:
```bash
python scripts/build_bm25_index.py --config configs/retrieval_bm25.yaml --force-rebuild
```

---

## 7. Python API Usage

```python
from veritas.retrieval import BM25Index, PassageStore, build_corpus_from_processed

# 1. Build or ingest corpus
passage_store, summary = build_corpus_from_processed(
    processed_dir="data/processed",
    datasets=["rawfc", "custom_recent"],
)

# 2. Build or load cached BM25 index
index = BM25Index.build_or_load(
    passage_store=passage_store,
    index_dir="data/indexes/bm25_default",
    k1=1.5,
    b=0.75,
)

# 3. Retrieve evidence passages
results = index.search("unemployment rate in February", top_k=5)
for res in results:
    print(f"Rank {res.rank} [score={res.score:.3f}]: {res.passage_id}")
    print(f"URL: {res.passage.url} | Date: {res.passage.publication_date}")
    print(f"Text snippet: {res.passage.text[:120]}...\n")
```

---

## 8. Dense Retrieval with Sentence-Transformers & FAISS

VERITAS implements dense semantic retrieval alongside lexical BM25 retrieval using `sentence-transformers` and `faiss.IndexFlatIP`.

### 8.1 Key Properties
- **Exact Cosine Similarity**: Embeddings are $L_2$-normalized (`normalize_embeddings=True`) and indexed using `faiss.IndexFlatIP`. For unit vectors, inner product equals cosine similarity.
- **Configurable Encoders**: Supports any HuggingFace embedding checkpoint (default: `BAAI/bge-small-en-v1.5`, `intfloat/e5-small-v2`, `sentence-transformers/all-MiniLM-L6-v2`).
- **Query Prefix Resolution**: Automatically attaches required prefixes:
  - BGE models: `"Represent this sentence for searching relevant passages: "`
  - E5 models: `"query: "` (queries) and `"passage: "` (documents)
  - Other models: no prefix (`""`)
- **Strict Positional Alignment**: Writes `passage_ids.json` as a separate ordered list mapping row index $i$ in FAISS directly to `PassageStore[i]`.
- **CPU & CUDA Fallback**: Automatically leverages CUDA mixed precision when available, with clean CPU fallback.
- **Fixture-Based Test Mode**: Provides `MockEmbedder` for instantaneous deterministic tests with zero model weight downloads.

### 8.2 Configuration

Configured via [configs/retrieval_dense.yaml](file:///e:/veritas/configs/retrieval_dense.yaml):

```yaml
retrieval:
  backend: dense
  index_dir: data/indexes/dense_default
  model_name_or_path: "BAAI/bge-small-en-v1.5"
  device: "auto"
  batch_size: 32
  normalize_embeddings: true
  index_type: "IndexFlatIP"
  dense_top_k: 50
  datasets:
    - rawfc
    - averitec
    - custom_recent
```

### 8.3 Building Dense Indexes

Build index using the YAML configuration:
```bash
python scripts/build_dense_index.py --config configs/retrieval_dense.yaml
```

Run smoke-test mode with query:
```bash
python scripts/build_dense_index.py --smoke-test --query "vaccination hospitalizations"
```

---

## 9. Test Suite

Run the full retrieval test suite:
```bash
pytest tests/retrieval/ -v
```

