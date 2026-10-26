"""VERITAS P7 Abstention Gate and Audit Logging.

Evaluates verifier confidence and explanation faithfulness against evidence passages.
If either threshold fails:
- In "abstain" mode (default): Downgrades verdict label to "Unverified" and sets abstained=True.
- In "flag_only" mode: Preserves original label, sets abstained=False, and sets flagged_only=True.

Preserves the complete audit record including original label, confidence, and per-sentence NLI evaluations
so any downgrade or abstention decision can be rigorously inspected and reproduced.

DISCLAIMER:
Abstention based on NLI and confidence thresholds is an operational risk-mitigation policy.
It does NOT guarantee zero error in downstream fact-verification and should be treated as a
calibrated safety barrier rather than an infallible verification oracle.
"""

from __future__ import annotations

import copy
import datetime
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from veritas.config import load_config
from veritas.faithfulness.nli import (
    BaseNLIModel,
    DEFAULT_ENTAILMENT_THRESHOLD,
    FaithfulnessEvaluator,
    FaithfulnessResult,
    MockNLIModel,
    TransformerNLIModel,
)

logger = logging.getLogger("veritas.faithfulness.abstain")

DEFAULT_MIN_CONFIDENCE = 0.60
DEFAULT_MIN_FAITHFULNESS = 0.70
DEFAULT_ABSTAIN_LABEL = "Unverified"


# ---------------------------------------------------------------------------
# Data Models and Configurations
# ---------------------------------------------------------------------------


@dataclass
class AbstentionConfig:
    """Configuration for faithfulness checking and verdict abstention.

    Attributes:
        min_confidence: Minimum verifier confidence required to avoid abstention [0.0, 1.0].
        min_faithfulness: Minimum proportion of supported explanation sentences [0.0, 1.0].
        entailment_threshold: NLI probability threshold to consider a sentence entailed [0.0, 1.0].
        mode: Policy mode: 'abstain' (downgrades label) or 'flag_only' (attaches warning flag).
        abstain_label: Label assigned when abstaining (default: 'Unverified').
        nli_model_name: Identifier for NLI model backend ('mock', 'cross-encoder/nli-deberta-v3-small', etc.).
        device: Compute device for neural NLI ('cpu' or 'cuda').
    """

    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    min_faithfulness: float = DEFAULT_MIN_FAITHFULNESS
    entailment_threshold: float = DEFAULT_ENTAILMENT_THRESHOLD
    mode: str = "abstain"  # 'abstain' or 'flag_only'
    abstain_label: str = DEFAULT_ABSTAIN_LABEL
    nli_model_name: str = "mock"
    device: str = "cpu"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AbstentionConfig:
        """Create configuration from dictionary with parameter filtering."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> AbstentionConfig:
        """Load configuration from a YAML file."""
        cfg_dict = load_config(yaml_path)
        faith_dict = cfg_dict.get("faithfulness", cfg_dict.get("abstention", cfg_dict))
        return cls.from_dict(faith_dict)


@dataclass
class AbstentionAuditRecord:
    """Detailed audit record preserving the original verdict and abstention rationale.

    Attributes:
        claim_id: Identifier of the evaluated claim.
        claim: Claim text.
        original_label: Verifier verdict before faithfulness checking.
        final_label: Final verdict label after applying abstention policy.
        original_confidence: Verifier confidence before gate.
        final_confidence: Final confidence score.
        faithfulness_score: Measured explanation faithfulness score in [0.0, 1.0].
        abstained: True if verdict was downgraded to abstain_label.
        flagged_only: True if thresholds failed but mode was 'flag_only'.
        abstention_reasons: Specific reasons for abstention or flagging.
        sentence_details: Per-sentence NLI support and contradiction scores.
        thresholds_used: Mapping of threshold values applied.
        mode_used: Execution policy ('abstain' or 'flag_only').
        timestamp: ISO-8601 timestamp of evaluation.
    """

    claim_id: str
    claim: str
    original_label: str
    final_label: str
    original_confidence: float
    final_confidence: float
    faithfulness_score: float
    abstained: bool
    flagged_only: bool
    abstention_reasons: list[str]
    sentence_details: list[dict[str, Any]]
    thresholds_used: dict[str, float]
    mode_used: str
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert audit record to serializable dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Abstention Gate Engine
# ---------------------------------------------------------------------------


class AbstentionGate:
    """Enforces confidence and faithfulness safety gates on fact-verification verdicts.

    Orchestrates:
    1. Sentence-level explanation decomposition.
    2. NLI entailment evaluation against cited evidence passages.
    3. Assessment against configurable min_confidence and min_faithfulness thresholds.
    4. Conservative downgrade to 'Unverified' (or flagging) with complete audit trail.
    """

    def __init__(
        self,
        config: AbstentionConfig | dict[str, Any] | None = None,
        evaluator: FaithfulnessEvaluator | None = None,
        nli_model: BaseNLIModel | None = None,
    ) -> None:
        if isinstance(config, dict):
            self.config = AbstentionConfig.from_dict(config)
        elif isinstance(config, AbstentionConfig):
            self.config = config
        else:
            self.config = AbstentionConfig()

        if evaluator is not None:
            self.evaluator = evaluator
        else:
            if nli_model is not None:
                model = nli_model
            elif self.config.nli_model_name == "mock":
                model = MockNLIModel()
            else:
                model = TransformerNLIModel(
                    model_name=self.config.nli_model_name,
                    device=self.config.device,
                )
            self.evaluator = FaithfulnessEvaluator(
                nli_model=model,
                entailment_threshold=self.config.entailment_threshold,
            )

    def evaluate_verdict(self, verdict: Any) -> tuple[Any, AbstentionAuditRecord]:
        """Apply abstention gate to a VERITAS Verdict object.

        Args:
            verdict: A VERITAS Verdict dataclass instance.

        Returns:
            Tuple of (updated_or_copied_verdict, AbstentionAuditRecord).
        """
        claim_id = getattr(verdict, "claim_id", "unknown_claim")
        claim = getattr(verdict, "claim", "")
        orig_label = getattr(verdict, "label", "Unverified")
        orig_conf = float(getattr(verdict, "confidence", 0.0))
        explanation = getattr(verdict, "explanation", "")
        evidence_passages = getattr(verdict, "retrieved_passages", [])
        evidence_ids = getattr(verdict, "evidence_ids", [])

        final_label, final_conf, abstained, audit_record = self.evaluate(
            claim_id=claim_id,
            claim=claim,
            label=orig_label,
            confidence=orig_conf,
            explanation=explanation,
            evidence_passages=evidence_passages,
            cited_evidence_ids=evidence_ids,
        )

        # Create updated verdict copy
        new_verdict = copy.deepcopy(verdict)
        setattr(new_verdict, "label", final_label)
        setattr(new_verdict, "confidence", final_conf)

        # Attach abstention provenance to model_metadata if present
        meta = getattr(new_verdict, "model_metadata", None)
        if isinstance(meta, dict):
            meta["abstention"] = {
                "abstained": abstained,
                "original_label": orig_label,
                "original_confidence": orig_conf,
                "faithfulness_score": audit_record.faithfulness_score,
                "reasons": audit_record.abstention_reasons,
                "mode": self.config.mode,
            }

        return new_verdict, audit_record

    def evaluate(
        self,
        claim_id: str,
        claim: str,
        label: str,
        confidence: float,
        explanation: str | None,
        evidence_passages: list[dict[str, Any]] | list[Any],
        cited_evidence_ids: list[str] | None = None,
    ) -> tuple[str, float, bool, AbstentionAuditRecord]:
        """Evaluate raw claim verification fields and return final verdict & audit record.

        Args:
            claim_id: Unique claim ID.
            claim: Factual claim statement.
            label: Original verdict label ('True', 'False', etc.).
            confidence: Verifier confidence in [0.0, 1.0].
            explanation: Verifier grounded explanation text.
            evidence_passages: Retrieved or available evidence passages.
            cited_evidence_ids: Specific passage IDs cited by the model.

        Returns:
            Tuple of:
            (final_label, final_confidence, abstained_bool, audit_record).
        """
        # 1. Evaluate explanation faithfulness via NLI
        faith_res = self.evaluator.evaluate(
            explanation=explanation,
            evidence_passages=evidence_passages,
            cited_evidence_ids=cited_evidence_ids,
        )

        # 2. Check thresholds
        reasons: list[str] = []
        is_low_confidence = confidence < self.config.min_confidence
        is_low_faithfulness = faith_res.faithfulness_score < self.config.min_faithfulness

        if is_low_confidence:
            reasons.append(
                f"Low confidence ({confidence:.2f} < threshold {self.config.min_confidence:.2f})"
            )

        if is_low_faithfulness:
            reasons.append(
                f"Low explanation faithfulness ({faith_res.faithfulness_score:.2f} < "
                f"threshold {self.config.min_faithfulness:.2f})"
            )

        if not faith_res.has_valid_evidence:
            reasons.append("No valid cited evidence passages available to ground explanation.")

        threshold_failed = is_low_confidence or is_low_faithfulness

        # 3. Apply policy mode
        final_label = label
        final_conf = confidence
        abstained = False
        flagged_only = False

        if threshold_failed:
            if self.config.mode == "flag_only":
                flagged_only = True
                logger.info(
                    "Claim '%s' failed safety gates but mode is 'flag_only'. Retaining label '%s'.",
                    claim_id,
                    label,
                )
            else:
                # Default 'abstain' mode
                abstained = True
                final_label = self.config.abstain_label
                # Conservatively reduce confidence on abstention
                final_conf = min(confidence, 0.50) if final_label == self.config.abstain_label else confidence
                logger.info(
                    "Claim '%s' abstained: downgraded '%s' -> '%s' (reasons: %s)",
                    claim_id,
                    label,
                    final_label,
                    "; ".join(reasons),
                )

        # 4. Construct immutable audit record
        audit_record = AbstentionAuditRecord(
            claim_id=claim_id,
            claim=claim,
            original_label=label,
            final_label=final_label,
            original_confidence=confidence,
            final_confidence=final_conf,
            faithfulness_score=faith_res.faithfulness_score,
            abstained=abstained,
            flagged_only=flagged_only,
            abstention_reasons=reasons,
            sentence_details=[s.to_dict() for s in faith_res.sentence_details],
            thresholds_used={
                "min_confidence": self.config.min_confidence,
                "min_faithfulness": self.config.min_faithfulness,
                "entailment_threshold": self.config.entailment_threshold,
            },
            mode_used=self.config.mode,
        )

        return final_label, final_conf, abstained, audit_record


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def apply_abstention(
    verdict: Any,
    config: AbstentionConfig | dict[str, Any] | None = None,
    evaluator: FaithfulnessEvaluator | None = None,
    min_confidence: float | None = None,
    min_faithfulness: float | None = None,
    mode: str | None = None,
) -> tuple[Any, AbstentionAuditRecord]:
    """Convenience helper to apply the abstention gate to a Verdict object."""
    if isinstance(config, dict):
        cfg = AbstentionConfig.from_dict(config)
    elif isinstance(config, AbstentionConfig):
        cfg = config
    else:
        cfg = AbstentionConfig()

    if min_confidence is not None:
        cfg.min_confidence = min_confidence
    if min_faithfulness is not None:
        cfg.min_faithfulness = min_faithfulness
    if mode is not None:
        cfg.mode = mode

    gate = AbstentionGate(config=cfg, evaluator=evaluator)
    return gate.evaluate_verdict(verdict)
