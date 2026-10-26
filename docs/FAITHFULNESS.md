# VERITAS Phase 7 (P7): Faithfulness Checking & Abstention Gate

## 1. Overview & Scientific Disclaimer

Phase 7 implements the **Faithfulness Checking and Abstention Gate** for VERITAS. Even when an LLM produces a verdict, its textual explanation may contain ungrounded assumptions, unsupported extrapolations, or subtle hallucinations that are not substantiated by the cited evidence passages.

> [!WARNING]
> ### Critical Scientific Disclaimer on NLI & Hallucination Detection
> Natural Language Inference (NLI) evaluates premise-hypothesis semantic entailment based on empirical neural representations.
> NLI scores **DO NOT constitute objective proof of real-world truth**, nor do they provide an infallible or complete guarantee of hallucination detection.
> NLI models are prone to both:
> - **False Positives**: Labeling an unsupported or ungrounded statement as entailed due to lexical token overlap.
> - **False Negatives**: Failing to recognize valid semantic paraphrases, numerical conversions, or commonsense inferences.
> In VERITAS, NLI and confidence gates function as **calibrated operational safety barriers and risk-reduction filters**, not as truth-granting oracles.

---

## 2. Architecture & Pipeline Flow

```
                      +-----------------------------+
                      |       Verifier Verdict      |
                      | (Label, Conf, Explanation,  |
                      |    Cited Passage IDs)       |
                      +-----------------------------+
                                     |
                                     v
                       [Stage 1: Sentence Splitting]
                     (Preserves decimals & abbreviations)
                                     |
                                     v
                       [Stage 2: Passage Pairing]
                      (Maps sentences to cited evidence)
                                     |
                                     v
                       [Stage 3: NLI Entailment]
                  (Premise: Passage, Hypothesis: Sentence)
                                     |
                                     v
                   [Stage 4: Faithfulness Assessment]
                (faithfulness = supported / total sentences)
                                     |
                                     v
                     [Stage 5: Dual Threshold Gate]
                   - Check 1: confidence >= min_confidence?
                   - Check 2: faithfulness >= min_faithfulness?
                                     |
                       +-------------+-------------+
                       |                           |
                 (Both Pass)                 (Either Fails)
                       |                           |
                       v                           v
              [Retain Verdict]              [Policy Action]
              abstained = False             - "abstain" (default):
                                               label -> "Unverified"
                                               abstained = True
                                            - "flag_only":
                                               retain label,
                                               flagged_only = True
                                                   |
                                                   v
                                     +---------------------------+
                                     |  Abstention Audit Record  |
                                     | (Original label/conf,     |
                                     |  downgrade reasons, NLI)  |
                                     +---------------------------+
```

---

## 3. Delivered Components

| Module | Location | Description |
| :--- | :--- | :--- |
| **NLI & Faithfulness** | [src/veritas/faithfulness/nli.py](file:///e:/veritas/src/veritas/faithfulness/nli.py) | Sentence splitter, `BaseNLIModel`, `MockNLIModel`, `TransformerNLIModel`, and `FaithfulnessEvaluator`. |
| **Abstention Gate** | [src/veritas/faithfulness/abstain.py](file:///e:/veritas/src/veritas/faithfulness/abstain.py) | `AbstentionGate`, `AbstentionConfig`, `AbstentionAuditRecord`, and `apply_abstention()`. |
| **Configuration** | [configs/faithfulness.yaml](file:///e:/veritas/configs/faithfulness.yaml) | Declarative configuration with thresholds and execution mode. |
| **Test Suite** | [tests/faithfulness/test_faithfulness.py](file:///e:/veritas/tests/faithfulness/test_faithfulness.py) | 15 deterministic unit tests covering supported, contradicted, neutral, missing evidence, and empty explanation cases. |
| **Threshold Sweep CLI** | [scripts/evaluate_abstention_thresholds.py](file:///e:/veritas/scripts/evaluate_abstention_thresholds.py) | Grid sweep script producing tabular metrics, summary CSV/JSON, and a standalone SVG plot. |

---

## 4. Usage Example

```python
from veritas.faithfulness.abstain import AbstentionGate, AbstentionConfig
from veritas.faithfulness.nli import MockNLIModel

gate = AbstentionGate(
    config=AbstentionConfig(min_confidence=0.60, min_faithfulness=0.70),
    nli_model=MockNLIModel(default_label="entailment"),
)

final_label, final_conf, abstained, audit = gate.evaluate(
    claim_id="c_101",
    claim="State carbon emissions declined by 18 percent.",
    label="True",
    confidence=0.85,
    explanation="Official environmental agency report verifies an 18 percent decline.",
    evidence_passages=[{"passage_id": "p1", "text": "State emissions fell 18 percent from 2020 to 2023."}],
    cited_evidence_ids=["p1"],
)

print(f"Final Label: {final_label} (Abstained: {abstained})")
print(f"Faithfulness Score: {audit.faithfulness_score}")
```
