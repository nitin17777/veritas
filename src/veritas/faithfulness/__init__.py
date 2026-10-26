"""VERITAS P7 Faithfulness Checking and Abstention package."""

from veritas.faithfulness.abstain import (
    AbstentionAuditRecord,
    AbstentionConfig,
    AbstentionGate,
    apply_abstention,
)
from veritas.faithfulness.nli import (
    BaseNLIModel,
    DEFAULT_CONTRADICTION_THRESHOLD,
    DEFAULT_ENTAILMENT_THRESHOLD,
    FaithfulnessEvaluator,
    FaithfulnessResult,
    MockNLIModel,
    NLIPrediction,
    SentenceFaithfulness,
    TransformerNLIModel,
    split_sentences,
)

__all__ = [
    "split_sentences",
    "NLIPrediction",
    "SentenceFaithfulness",
    "FaithfulnessResult",
    "BaseNLIModel",
    "MockNLIModel",
    "TransformerNLIModel",
    "FaithfulnessEvaluator",
    "AbstentionConfig",
    "AbstentionAuditRecord",
    "AbstentionGate",
    "apply_abstention",
    "DEFAULT_ENTAILMENT_THRESHOLD",
    "DEFAULT_CONTRADICTION_THRESHOLD",
]
