# VERITAS Baselines

This document describes the baseline models implemented in VERITAS, their usage, and experimental results.

---

## 1. TF-IDF + Logistic Regression Baseline (`tfidf_lr`)

### 1.1 Overview

The `tfidf_lr` model serves as the first baseline classifier in VERITAS. It establishes an empirical lower bound for fact-checking performance using surface textual cues in claim statements without external evidence retrieval, speaker metadata, or pre-trained neural representations.

Key properties:
- **Feature Extraction**: Word-level TF-IDF vectorizer extracting unigrams and bigrams (`ngram_range=(1, 2)`), sublinear TF scaling (`1 + log(tf)`), Unicode accent stripping, and lowercase normalization. Vocabulary size is capped at 100,000 features.
- **Classifier**: Logistic Regression with `solver="saga"`, L2 regularisation, and hardcoded `class_weight="balanced"` to counteract class imbalance.
- **Multi-class Strategy**: Multinomial loss handled natively by the `saga` solver across all 5 VERITAS labels (`True`, `Partially True`, `Misleading`, `False`, `Unverified`).
- **Reproducibility**: All random operations are seeded with `seed=42`.

### 1.2 Model Selection Protocol

To maintain scientific integrity:
1. **Train Split Only**: The model is fitted only on the official training split (`data/processed/<dataset>/train.jsonl`).
2. **Validation Tuning**: Inverse regularisation parameter $C \in \{0.01, 0.1, 0.5, 1.0, 5.0, 10.0\}$ is tuned exclusively using macro-F1 on `validation.jsonl`.
3. **No Test Contamination**: The test split (`test.jsonl`) is strictly evaluated only once after the best $C$ has been locked from the validation sweep. The test split is never used for hyperparameter tuning.

### 1.3 Configuration

The baseline is configured via [configs/tfidf_lr.yaml](file:///e:/veritas/configs/tfidf_lr.yaml):

```yaml
paths:
  results_dir: results

logging:
  level: INFO

baseline_tfidf_lr:
  dataset: liar
  processed_dir: data/processed/liar
  model_name: tfidf_lr

  random_seed: 42

  tfidf:
    max_features: 100000
    ngram_min: 1
    ngram_max: 2
    sublinear_tf: true

  logistic_regression:
    C_values: [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    max_iter: 1000
    solver: saga
```

### 1.4 How to Run

#### Prerequisites
Prepare the processed dataset first (e.g., LIAR):
```bash
python -m veritas.data_prep.prepare --datasets liar
```

#### Training and Evaluation
Run using the default YAML configuration:
```bash
python scripts/train_tfidf_lr.py --config configs/tfidf_lr.yaml
```

CLI overrides are also supported:
```bash
python scripts/train_tfidf_lr.py --processed-dir data/processed/liar --dataset liar --seed 42
```

To skip test evaluation during rapid development:
```bash
python scripts/train_tfidf_lr.py --config configs/tfidf_lr.yaml --skip-test
```

### 1.5 Outputs

Each run creates a timestamped output directory under `results/<RUN_ID>/` containing:
- `model.joblib`: Serialized scikit-learn Pipeline (vectorizer + classifier).
- `predictions_validation.jsonl`: Predictions for validation records with probability distribution across all labels.
- `predictions_test.jsonl`: Predictions for test records with probability distribution across all labels.
- `metrics_validation.json`: Validation evaluation metrics (accuracy, macro-F1, macro-P/R, per-class metrics, confusion matrix).
- `metrics_test.json`: Final test evaluation metrics.
- `model_card.json`: Full provenance card including training label distribution, C-sweep results, hyperparameters, and git/timestamp metadata.
- `config.yaml`: Exact snapshot of configuration used.
- `run.log`: Formatted logging stream.

### 1.6 Empirical Results on LIAR Benchmark

Results obtained using fixed seed `42`, 10,269 train records, 1,284 validation records, and 1,283 test records:

- **Validation Macro-F1 by C**:
  - $C=0.01$: $0.3488$
  - $C=0.10$: $0.3451$
  - $C=0.50$: $0.3526$
  - **$C=1.00$: $0.3635$ (Selected)**
  - $C=5.00$: $0.3455$
  - $C=10.00$: $0.3395$

- **Validation Split ($n=1284$)**:
  - Accuracy: $0.3847$
  - Macro-F1: $0.3635$
  - Macro-Precision: $0.3645$
  - Macro-Recall: $0.3633$

- **Test Split ($n=1283$)**:
  - Accuracy: $0.3663$
  - Macro-F1: $0.3391$
  - Macro-Precision: $0.3389$
  - Macro-Recall: $0.3394$

#### Per-Class Breakdown (Test Split)

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| True | 0.4806 | 0.4587 | 0.4694 | 460 |
| Partially True | 0.2481 | 0.2472 | 0.2477 | 267 |
| Misleading | 0.2242 | 0.2336 | 0.2288 | 214 |
| False | 0.4028 | 0.4181 | 0.4103 | 342 |

*(Note: Unverified has 0 support in LIAR as LIAR labels map to True, Partially True, Misleading, and False).*

#### Test Confusion Matrix (Rows = Gold, Columns = Predicted)

```
                True    Partially True  Misleading      False
True            211     94              68              87
Partially True  99      66              42              60
Misleading      53      46              50              65
False           76      60              63              143
```

---

## 2. Baselines Test Suite

Run the full models test suite:
```bash
pytest tests/models/ -v
```

---

## 3. Transformer Classification Baseline (`transformer`)

### 3.1 Overview

The transformer baseline implements deep contextual sequence classification using pre-trained Hugging Face encoders (default: `distilroberta-base`, also supporting `microsoft/deberta-v3-small`, `roberta-base`, etc.).

Key properties:
- **Pretrained Encoders**: Configurable via `model_name_or_path` in YAML or `--model-name` in CLI.
- **5-Label Contract**: Maps directly to the VERITAS standard labels (`True`, `Partially True`, `Misleading`, `False`, `Unverified`).
- **Class Imbalance Handling**: Supports `balanced`, `inverse_freq`, or `none` strategies via a custom `WeightedTrainer` that weights cross-entropy loss by class frequency.
- **GPU Memory Safety & Optimization**:
  - Automatically detects CUDA availability.
  - Automatically enables `fp16` mixed-precision when running on CUDA.
  - Dynamically detects available VRAM and caps batch size on memory-constrained GPUs (such as 4GB GPUs like NVIDIA RTX 2050).
  - Supports configurable `gradient_accumulation_steps` and `max_seq_length`.
- **Checkpoint Recovery**: Full support for resuming interrupted training runs via `--resume-from-checkpoint <path>`.
- **Smoke-Test Mode**: Fast sanity check flag (`--smoke-test`) that runs 1 epoch on a tiny dataset slice (32 train, 16 val, 16 test) to verify end-to-end forward/backward passes and output serialization.

### 3.2 Configuration

The model is configured via [configs/transformer.yaml](file:///e:/veritas/configs/transformer.yaml):

```yaml
paths:
  results_dir: results

logging:
  level: INFO

baseline_transformer:
  dataset: liar
  processed_dir: data/processed/liar
  model_name_or_path: "distilroberta-base"
  random_seed: 42

  max_seq_length: 128
  imbalance_strategy: "balanced"

  training:
    learning_rate: 3.0e-5
    per_device_train_batch_size: 16
    per_device_eval_batch_size: 32
    gradient_accumulation_steps: 1
    num_train_epochs: 3
    weight_decay: 0.01
    warmup_ratio: 0.1
    eval_strategy: "epoch"
    save_strategy: "epoch"
    load_best_model_at_end: true
    metric_for_best_model: "macro_f1"
    greater_is_better: true
    save_total_limit: 2
    logging_steps: 50
    fp16: false
    dataloader_num_workers: 0

  smoke_test:
    max_train_samples: 32
    max_eval_samples: 16
    max_test_samples: 16
    num_train_epochs: 1
    per_device_train_batch_size: 8
    per_device_eval_batch_size: 8
    logging_steps: 2
```

### 3.3 How to Run

#### Smoke Test (Fast Sanity Check)
```bash
python scripts/train_transformer.py --config configs/transformer.yaml --smoke-test
```

#### Full Training Run
```bash
python scripts/train_transformer.py --config configs/transformer.yaml
```

#### Custom Hyperparameters / Pretrained Model
```bash
python scripts/train_transformer.py --model-name microsoft/deberta-v3-small --epochs 3 --batch-size 16 --lr 2e-5
```

#### Resuming From Checkpoint
```bash
python scripts/train_transformer.py --config configs/transformer.yaml --resume-from-checkpoint results/<RUN_ID>/checkpoints/checkpoint-500
```

