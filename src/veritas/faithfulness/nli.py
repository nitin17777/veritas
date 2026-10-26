"""VERITAS P7 Natural Language Inference (NLI) and Faithfulness Checking.

DISCLAIMER ON NLI AND HALLUCINATION DETECTION:
NLI models evaluate premise-hypothesis semantic entailment based on empirical neural representations.
NLI scores DO NOT constitute objective proof of real-world truth, nor do they provide infallible
hallucination detection. NLI models can exhibit false positives (labeling unsupported statements as
entailed due to lexical overlap) and false negatives (failing to recognize valid paraphrases or
inferences). In VERITAS, NLI is employed as a heuristic safeguard and conservative filter,
not as a ground-truth oracle.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

logger = logging.getLogger("veritas.faithfulness.nli")

DEFAULT_ENTAILMENT_THRESHOLD = 0.50
DEFAULT_CONTRADICTION_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Sentence Splitting
# ---------------------------------------------------------------------------

_COMMON_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "gov", "sen", "rep", "gen", "col",
    "vs", "approx", "dept", "inc", "ltd", "corp", "co", "jan", "feb",
    "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "u.s", "u.k", "e.g", "i.e", "al", "no", "vol", "p.m", "a.m",
}


def split_sentences(text: str | None) -> list[str]:
    """Split an explanation or paragraph into clean, individual sentences.

    Preserves numbers with decimals (e.g., '3.5 billion'), currency,
    common titles/abbreviations (e.g., 'U.S.', 'Dr.'), and handles quotes.

    Args:
        text: Input explanation string.

    Returns:
        List of non-empty, stripped sentence strings.
    """
    if not text or not text.strip():
        return []

    cleaned = text.strip()

    # Normalize whitespace
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # Protect decimals: replace '3.5' with '3<DEC>5'
    cleaned = re.sub(r"(\d+)\.(\d+)", r"\1<DEC>\2", cleaned)

    # Protect common abbreviations while preserving original casing
    for abbr in _COMMON_ABBREVIATIONS:
        pattern = re.compile(rf"\b{re.escape(abbr)}\.", re.IGNORECASE)
        cleaned = pattern.sub(lambda m: m.group(0)[:-1] + "<DOT>", cleaned)

    # Split on sentence boundaries: period, exclamation mark, or question mark
    # followed by space and uppercase letter, or end of string, or newline
    sentence_delimiters = re.compile(r"([.!?])(?:\s+(?=[A-Z0-9\"'“‘])|\s*$|\n+)")
    raw_pieces = sentence_delimiters.split(cleaned)

    sentences: list[str] = []
    current_acc = ""

    for piece in raw_pieces:
        if not piece:
            continue
        if piece in {".", "!", "?"}:
            current_acc += piece
            # Restore protected tokens
            restored = current_acc.replace("<DEC>", ".").replace("<DOT>", ".")
            restored = restored.strip()
            if restored:
                sentences.append(restored)
            current_acc = ""
        else:
            if current_acc:
                current_acc += " " + piece
            else:
                current_acc = piece

    if current_acc:
        restored = current_acc.replace("<DEC>", ".").replace("<DOT>", ".")
        restored = restored.strip()
        if restored:
            sentences.append(restored)

    return sentences


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class NLIPrediction:
    """Predicted NLI probability distribution over entailment, neutral, contradiction.

    Attributes:
        entailment: Probability that hypothesis is entailed by premise in [0.0, 1.0].
        neutral: Probability that hypothesis is neutral/indeterminate in [0.0, 1.0].
        contradiction: Probability that hypothesis contradicts premise in [0.0, 1.0].
        predicted_label: Highest probability label ('entailment', 'neutral', 'contradiction').
    """

    entailment: float
    neutral: float
    contradiction: float
    predicted_label: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SentenceFaithfulness:
    """Faithfulness evaluation of a single explanation sentence against cited evidence.

    Attributes:
        sentence_index: 0-indexed position in the explanation.
        sentence_text: The individual explanation sentence evaluated as hypothesis.
        cited_passage_ids: IDs of the evidence passages used as premise.
        premise_text: Truncated or combined premise text used for NLI.
        entailment_score: Highest entailment probability across cited passages.
        neutral_score: Associated neutral probability.
        contradiction_score: Highest contradiction probability across cited passages.
        is_supported: True if entailment_score >= entailment_threshold.
        is_contradicted: True if contradiction_score >= contradiction_threshold.
    """

    sentence_index: int
    sentence_text: str
    cited_passage_ids: list[str]
    premise_text: str
    entailment_score: float
    neutral_score: float
    contradiction_score: float
    is_supported: bool
    is_contradicted: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FaithfulnessResult:
    """Overall explanation faithfulness assessment.

    Attributes:
        faithfulness_score: Proportion of explanation sentences judged supported in [0.0, 1.0].
        total_sentences: Number of explanation sentences evaluated.
        supported_sentences: Number of sentences meeting the entailment threshold.
        contradicted_sentences: Number of sentences judged as contradictory.
        sentence_details: Detailed per-sentence evaluation records.
        has_valid_evidence: True if at least one valid evidence passage was available as premise.
        entailment_threshold: The numerical threshold used to judge support.
        notes: Diagnostic details or failure reasons.
    """

    faithfulness_score: float
    total_sentences: int
    supported_sentences: int
    contradicted_sentences: int
    sentence_details: list[SentenceFaithfulness]
    has_valid_evidence: bool
    entailment_threshold: float = DEFAULT_ENTAILMENT_THRESHOLD
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "faithfulness_score": self.faithfulness_score,
            "total_sentences": self.total_sentences,
            "supported_sentences": self.supported_sentences,
            "contradicted_sentences": self.contradicted_sentences,
            "sentence_details": [s.to_dict() for s in self.sentence_details],
            "has_valid_evidence": self.has_valid_evidence,
            "entailment_threshold": self.entailment_threshold,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# NLI Model Interfaces
# ---------------------------------------------------------------------------


class BaseNLIModel(ABC):
    """Abstract base class establishing the NLI interface."""

    @abstractmethod
    def predict(self, premise: str, hypothesis: str) -> NLIPrediction:
        """Predict NLI distribution for a premise-hypothesis pair."""
        pass

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[NLIPrediction]:
        """Predict NLI distributions for multiple premise-hypothesis pairs."""
        return [self.predict(p, h) for p, h in pairs]


class MockNLIModel(BaseNLIModel):
    """Deterministic, zero-download NLI model for unit testing and offline development.

    Supports:
    - Exact mapping of (premise, hypothesis) -> NLIPrediction.
    - Canned sequential predictions.
    - Keyword/overlap rule generators.
    """

    def __init__(
        self,
        canned_predictions: list[NLIPrediction] | None = None,
        exact_mappings: dict[tuple[str, str], NLIPrediction] | None = None,
        default_label: str = "entailment",
        rule_fn: Callable[[str, str], NLIPrediction] | None = None,
    ) -> None:
        self.canned = list(canned_predictions or [])
        self.exact_mappings = dict(exact_mappings or {})
        self.default_label = default_label
        self.rule_fn = rule_fn
        self.call_history: list[tuple[str, str]] = []

    def predict(self, premise: str, hypothesis: str) -> NLIPrediction:
        self.call_history.append((premise, hypothesis))

        # 1. Exact pair mapping
        if (premise, hypothesis) in self.exact_mappings:
            return self.exact_mappings[(premise, hypothesis)]

        # 2. Custom rule function
        if self.rule_fn is not None:
            return self.rule_fn(premise, hypothesis)

        # 3. Canned sequential queue
        if self.canned:
            return self.canned.pop(0)

        # 4. Fallback based on default label
        if self.default_label == "entailment":
            return NLIPrediction(entailment=0.92, neutral=0.06, contradiction=0.02, predicted_label="entailment")
        elif self.default_label == "contradiction":
            return NLIPrediction(entailment=0.02, neutral=0.06, contradiction=0.92, predicted_label="contradiction")
        else:
            return NLIPrediction(entailment=0.15, neutral=0.75, contradiction=0.10, predicted_label="neutral")


class TransformerNLIModel(BaseNLIModel):
    """Production NLI model using Hugging Face Transformers.

    Lazy-loads sequence classification models such as
    'cross-encoder/nli-deberta-v3-small' or 'roberta-large-mnli'.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._tokenizer = None
        self._model = None
        self._label_map: dict[int, str] = {}

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required to run TransformerNLIModel. "
                "Ensure they are installed or use MockNLIModel for tests."
            ) from exc

        logger.info("Loading NLI model '%s' on %s...", self.model_name, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

        # Resolve model label mapping
        id2label = getattr(self._model.config, "id2label", {})
        label_map: dict[int, str] = {}
        for idx, lbl in id2label.items():
            name = str(lbl).lower()
            if "entail" in name:
                label_map[int(idx)] = "entailment"
            elif "contra" in name:
                label_map[int(idx)] = "contradiction"
            else:
                label_map[int(idx)] = "neutral"

        if len(label_map) < 3:
            # Fallback for standard 3-class MNLI mapping: 0=contradiction, 1=neutral, 2=entailment (or similar)
            label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}

        self._label_map = label_map

    def predict(self, premise: str, hypothesis: str) -> NLIPrediction:
        results = self.predict_batch([(premise, hypothesis)])
        return results[0]

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[NLIPrediction]:
        self._load_model()
        import torch

        if not pairs:
            return []

        premises = [p for p, h in pairs]
        hypotheses = [h for p, h in pairs]

        encoded = self._tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self._model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        results: list[NLIPrediction] = []
        for row in probs:
            entail_prob = 0.0
            neutral_prob = 0.0
            contra_prob = 0.0

            for idx, prob_val in enumerate(row):
                lbl_name = self._label_map.get(idx, "neutral")
                if lbl_name == "entailment":
                    entail_prob += float(prob_val)
                elif lbl_name == "contradiction":
                    contra_prob += float(prob_val)
                else:
                    neutral_prob += float(prob_val)

            # Determine argmax
            best_label = "neutral"
            best_val = neutral_prob
            if entail_prob >= best_val and entail_prob >= contra_prob:
                best_label = "entailment"
                best_val = entail_prob
            elif contra_prob >= best_val:
                best_label = "contradiction"
                best_val = contra_prob

            results.append(
                NLIPrediction(
                    entailment=round(entail_prob, 4),
                    neutral=round(neutral_prob, 4),
                    contradiction=round(contra_prob, 4),
                    predicted_label=best_label,
                )
            )

        return results


# ---------------------------------------------------------------------------
# Faithfulness Evaluator
# ---------------------------------------------------------------------------


class FaithfulnessEvaluator:
    """Evaluates explanation faithfulness by running NLI on cited evidence.

    For each explanation sentence (hypothesis) and cited evidence passage (premise),
    computes entailment and evaluates whether the explanation is grounded.
    """

    def __init__(
        self,
        nli_model: BaseNLIModel | None = None,
        entailment_threshold: float = DEFAULT_ENTAILMENT_THRESHOLD,
        contradiction_threshold: float = DEFAULT_CONTRADICTION_THRESHOLD,
    ) -> None:
        self.nli_model = nli_model or MockNLIModel()
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold

    def evaluate(
        self,
        explanation: str | None,
        evidence_passages: list[dict[str, Any]] | list[Any],
        cited_evidence_ids: list[str] | None = None,
    ) -> FaithfulnessResult:
        """Assess faithfulness of explanation sentences against cited evidence.

        Args:
            explanation: Grounded textual reasoning from the verifier.
            evidence_passages: List of passage objects or dictionaries containing 'text' and 'passage_id'.
            cited_evidence_ids: Optional subset of passage IDs specifically cited by the verifier.

        Returns:
            FaithfulnessResult containing faithfulness score and sentence breakdown.
        """
        # 1. Parse explanation into sentences
        sentences = split_sentences(explanation)
        if not sentences:
            logger.info("Empty explanation provided; returning conservative faithfulness 0.0.")
            return FaithfulnessResult(
                faithfulness_score=0.0,
                total_sentences=0,
                supported_sentences=0,
                contradicted_sentences=0,
                sentence_details=[],
                has_valid_evidence=bool(evidence_passages),
                entailment_threshold=self.entailment_threshold,
                notes="Explanation is empty or contains no valid sentences.",
            )

        # 2. Filter available evidence passages to cited ones
        passages_by_id: dict[str, str] = {}
        for p in evidence_passages:
            if isinstance(p, dict):
                p_id = str(p.get("passage_id") or p.get("id") or "")
                p_text = str(p.get("text") or "")
            else:
                p_id = str(getattr(p, "passage_id", getattr(p, "id", "")))
                p_text = str(getattr(p, "text", ""))

            if p_id and p_text:
                passages_by_id[p_id] = p_text

        # If cited_evidence_ids is provided, restrict to those passages
        target_passages: list[tuple[str, str]] = []
        if cited_evidence_ids:
            for cid in cited_evidence_ids:
                if str(cid) in passages_by_id:
                    target_passages.append((str(cid), passages_by_id[str(cid)]))
        else:
            # Use all available passages if no specific citation filter is specified
            target_passages = list(passages_by_id.items())

        # 3. Handle missing or empty evidence conservatively
        if not target_passages:
            logger.warning("No valid cited evidence passages found for explanation sentences.")
            return FaithfulnessResult(
                faithfulness_score=0.0,
                total_sentences=len(sentences),
                supported_sentences=0,
                contradicted_sentences=0,
                sentence_details=[
                    SentenceFaithfulness(
                        sentence_index=idx,
                        sentence_text=sent,
                        cited_passage_ids=[],
                        premise_text="",
                        entailment_score=0.0,
                        neutral_score=1.0,
                        contradiction_score=0.0,
                        is_supported=False,
                        is_contradicted=False,
                    )
                    for idx, sent in enumerate(sentences)
                ],
                has_valid_evidence=False,
                entailment_threshold=self.entailment_threshold,
                notes="No valid cited evidence passages available as premises.",
            )

        # 4. Evaluate each explanation sentence against candidate premises
        sentence_evals: list[SentenceFaithfulness] = []
        supported_count = 0
        contradicted_count = 0

        for s_idx, sentence in enumerate(sentences):
            best_entailment = 0.0
            best_neutral = 0.0
            best_contradiction = 0.0
            best_premise_snippet = ""
            matched_passage_ids: list[str] = [pid for pid, _ in target_passages]

            # Evaluate sentence against each cited passage (take maximum entailment)
            for pid, p_text in target_passages:
                pred = self.nli_model.predict(premise=p_text, hypothesis=sentence)
                if pred.entailment > best_entailment:
                    best_entailment = pred.entailment
                    best_neutral = pred.neutral
                    best_premise_snippet = p_text[:200]
                if pred.contradiction > best_contradiction:
                    best_contradiction = pred.contradiction

            is_supp = best_entailment >= self.entailment_threshold
            is_contra = best_contradiction >= self.contradiction_threshold

            if is_supp:
                supported_count += 1
            if is_contra:
                contradicted_count += 1

            sentence_evals.append(
                SentenceFaithfulness(
                    sentence_index=s_idx,
                    sentence_text=sentence,
                    cited_passage_ids=matched_passage_ids,
                    premise_text=best_premise_snippet,
                    entailment_score=round(best_entailment, 4),
                    neutral_score=round(best_neutral, 4),
                    contradiction_score=round(best_contradiction, 4),
                    is_supported=is_supp,
                    is_contradicted=is_contra,
                )
            )

        faithfulness = round(supported_count / len(sentences), 4)

        return FaithfulnessResult(
            faithfulness_score=faithfulness,
            total_sentences=len(sentences),
            supported_sentences=supported_count,
            contradicted_sentences=contradicted_count,
            sentence_details=sentence_evals,
            has_valid_evidence=True,
            entailment_threshold=self.entailment_threshold,
            notes=None,
        )
