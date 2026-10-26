"""Tests for VERITAS P7 Faithfulness Checking and Abstention Gate.

Covers:
- Sentence splitting with decimal, abbreviation, and quote preservation.
- Supported explanation cases (high entailment -> high faithfulness).
- Contradicted explanation cases (high contradiction -> low faithfulness).
- Neutral/indeterminate explanation cases (unsupported -> low faithfulness).
- Missing, empty, or ungrounded evidence handling (conservative 0.0).
- Empty/whitespace explanation handling.
- Abstention gate in 'abstain' mode (downgrade to 'Unverified').
- Abstention gate in 'flag_only' mode (preserves original label with warning flag).
- Audit trail integrity (preservation of original verdict, confidence, and sentence records).
- All tests use MockNLIModel to guarantee zero downloads and offline determinism.
"""

from __future__ import annotations

import pytest

from veritas.faithfulness.abstain import (
    AbstentionAuditRecord,
    AbstentionConfig,
    AbstentionGate,
    apply_abstention,
)
from veritas.faithfulness.nli import (
    FaithfulnessEvaluator,
    FaithfulnessResult,
    MockNLIModel,
    NLIPrediction,
    SentenceFaithfulness,
    split_sentences,
)
from veritas.pipeline.pipeline import Verdict


# ---------------------------------------------------------------------------
# Fixtures and Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_passages():
    return [
        {
            "passage_id": "doc_1",
            "text": "The World Health Organization confirmed that vaccination reduces severe COVID-19 hospitalizations by 92 percent.",
        },
        {
            "passage_id": "doc_2",
            "text": "The national unemployment rate dropped to 3.7 percent in December 2023 according to the Bureau of Labor Statistics.",
        },
    ]


# ---------------------------------------------------------------------------
# 1. Sentence Splitting Tests
# ---------------------------------------------------------------------------


class TestSentenceSplitting:
    def test_basic_multi_sentence(self):
        text = "This is the first sentence. This is the second sentence! Is this the third?"
        sentences = split_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "This is the first sentence."
        assert sentences[1] == "This is the second sentence!"
        assert sentences[2] == "Is this the third?"

    def test_preserves_decimals_and_currency(self):
        text = "The budget increased by $3.5 billion in 2023. Inflation reached 4.2 percent overall."
        sentences = split_sentences(text)
        assert len(sentences) == 2
        assert "$3.5 billion" in sentences[0]
        assert "4.2 percent" in sentences[1]

    def test_preserves_common_abbreviations(self):
        text = "Dr. Smith stated that U.S. hospital admissions fell. Prof. Davis confirmed the findings."
        sentences = split_sentences(text)
        assert len(sentences) == 2
        assert "Dr. Smith" in sentences[0]
        assert "U.S. hospital" in sentences[0]
        assert "Prof. Davis" in sentences[1]

    def test_empty_and_whitespace(self):
        assert split_sentences("") == []
        assert split_sentences("   \t  \n  ") == []
        assert split_sentences(None) == []


# ---------------------------------------------------------------------------
# 2. Supported Explanation Case
# ---------------------------------------------------------------------------


class TestSupportedExplanation:
    def test_all_sentences_entailed(self, sample_passages):
        """When evidence fully entails all explanation sentences, faithfulness is 1.0."""
        nli = MockNLIModel(
            default_label="entailment",
            canned_predictions=[
                NLIPrediction(entailment=0.95, neutral=0.03, contradiction=0.02, predicted_label="entailment"),
                NLIPrediction(entailment=0.91, neutral=0.07, contradiction=0.02, predicted_label="entailment"),
            ],
        )
        evaluator = FaithfulnessEvaluator(nli_model=nli, entailment_threshold=0.50)

        explanation = (
            "Vaccination significantly mitigates severe hospitalizations by over 90 percent. "
            "The World Health Organization verified these findings directly."
        )

        res = evaluator.evaluate(
            explanation=explanation,
            evidence_passages=sample_passages,
            cited_evidence_ids=["doc_1"],
        )

        assert res.has_valid_evidence is True
        assert res.total_sentences == 2
        assert res.supported_sentences == 2
        assert res.contradicted_sentences == 0
        assert res.faithfulness_score == 1.0
        assert all(s.is_supported for s in res.sentence_details)


# ---------------------------------------------------------------------------
# 3. Contradicted Explanation Case
# ---------------------------------------------------------------------------


class TestContradictedExplanation:
    def test_explanation_contradicts_evidence(self, sample_passages):
        """When explanation asserts facts contradicted by evidence, faithfulness drops to 0.0."""
        nli = MockNLIModel(
            canned_predictions=[
                NLIPrediction(entailment=0.01, neutral=0.04, contradiction=0.95, predicted_label="contradiction"),
            ]
        )
        evaluator = FaithfulnessEvaluator(nli_model=nli, entailment_threshold=0.50)

        explanation = "The WHO reported that vaccines completely fail to reduce COVID-19 hospitalizations."

        res = evaluator.evaluate(
            explanation=explanation,
            evidence_passages=sample_passages,
            cited_evidence_ids=["doc_1"],
        )

        assert res.total_sentences == 1
        assert res.supported_sentences == 0
        assert res.contradicted_sentences == 1
        assert res.faithfulness_score == 0.0
        assert res.sentence_details[0].is_contradicted is True
        assert res.sentence_details[0].is_supported is False


# ---------------------------------------------------------------------------
# 4. Neutral / Unsupported Explanation Case
# ---------------------------------------------------------------------------


class TestNeutralExplanation:
    def test_explanation_unrelated_to_evidence(self, sample_passages):
        """When evidence is neutral (unrelated facts), entailment is low and sentence is unsupported."""
        nli = MockNLIModel(
            canned_predictions=[
                NLIPrediction(entailment=0.10, neutral=0.85, contradiction=0.05, predicted_label="neutral"),
            ]
        )
        evaluator = FaithfulnessEvaluator(nli_model=nli, entailment_threshold=0.50)

        explanation = "The ancient Romans built extensive aqueduct networks across the Mediterranean."

        res = evaluator.evaluate(
            explanation=explanation,
            evidence_passages=sample_passages,
            cited_evidence_ids=["doc_1"],
        )

        assert res.total_sentences == 1
        assert res.supported_sentences == 0
        assert res.faithfulness_score == 0.0
        assert res.sentence_details[0].is_supported is False


# ---------------------------------------------------------------------------
# 5. Missing Evidence and Conservative Handling
# ---------------------------------------------------------------------------


class TestMissingEvidenceHandling:
    def test_no_cited_evidence_passages(self):
        """When no evidence passages are provided, faithfulness score is conservatively 0.0."""
        evaluator = FaithfulnessEvaluator(nli_model=MockNLIModel())
        explanation = "This is an unsupported explanation without any evidence."

        res = evaluator.evaluate(
            explanation=explanation,
            evidence_passages=[],
            cited_evidence_ids=[],
        )

        assert res.has_valid_evidence is False
        assert res.faithfulness_score == 0.0
        assert res.supported_sentences == 0
        assert "No valid cited evidence" in (res.notes or "")

    def test_cited_id_not_in_passages(self, sample_passages):
        """When cited evidence ID does not exist in the retrieved passages, handled conservatively."""
        evaluator = FaithfulnessEvaluator(nli_model=MockNLIModel())
        explanation = "This explanation references a non-existent passage."

        res = evaluator.evaluate(
            explanation=explanation,
            evidence_passages=sample_passages,
            cited_evidence_ids=["non_existent_doc_999"],
        )

        assert res.has_valid_evidence is False
        assert res.faithfulness_score == 0.0


# ---------------------------------------------------------------------------
# 6. Empty Explanation Handling
# ---------------------------------------------------------------------------


class TestEmptyExplanationHandling:
    def test_empty_or_whitespace_explanation(self, sample_passages):
        """Empty explanation string returns 0.0 faithfulness with zero sentences."""
        evaluator = FaithfulnessEvaluator(nli_model=MockNLIModel())

        for empty_text in ["", "   \n\t  ", None]:
            res = evaluator.evaluate(
                explanation=empty_text,
                evidence_passages=sample_passages,
                cited_evidence_ids=["doc_1"],
            )
            assert res.total_sentences == 0
            assert res.faithfulness_score == 0.0
            assert res.supported_sentences == 0


# ---------------------------------------------------------------------------
# 7. Abstention Gate Tests (Abstain vs Flag-Only)
# ---------------------------------------------------------------------------


class TestAbstentionGate:
    def test_pass_all_thresholds_no_abstention(self, sample_passages):
        """High confidence and high faithfulness retains original verdict."""
        nli = MockNLIModel(default_label="entailment")
        gate = AbstentionGate(
            config=AbstentionConfig(min_confidence=0.60, min_faithfulness=0.70),
            evaluator=FaithfulnessEvaluator(nli_model=nli),
        )

        final_label, final_conf, abstained, audit = gate.evaluate(
            claim_id="c1",
            claim="Vaccines reduce hospitalizations.",
            label="True",
            confidence=0.92,
            explanation="WHO confirmed that vaccines reduce hospitalizations by 92 percent.",
            evidence_passages=sample_passages,
            cited_evidence_ids=["doc_1"],
        )

        assert not abstained
        assert not audit.flagged_only
        assert final_label == "True"
        assert final_conf == 0.92
        assert audit.faithfulness_score == 1.0
        assert audit.abstention_reasons == []

    def test_low_confidence_triggers_abstention(self, sample_passages):
        """Low confidence triggers abstention even if explanation is faithful."""
        nli = MockNLIModel(default_label="entailment")
        gate = AbstentionGate(
            config=AbstentionConfig(min_confidence=0.70, min_faithfulness=0.50),
            evaluator=FaithfulnessEvaluator(nli_model=nli),
        )

        final_label, final_conf, abstained, audit = gate.evaluate(
            claim_id="c2",
            claim="Vaccines reduce hospitalizations.",
            label="True",
            confidence=0.45,  # Below 0.70 threshold
            explanation="Vaccines reduce hospitalizations.",
            evidence_passages=sample_passages,
            cited_evidence_ids=["doc_1"],
        )

        assert abstained is True
        assert final_label == "Unverified"
        assert audit.original_label == "True"
        assert any("Low confidence" in r for r in audit.abstention_reasons)

    def test_low_faithfulness_triggers_abstention(self, sample_passages):
        """Low faithfulness triggers abstention even if confidence is high."""
        nli = MockNLIModel(default_label="neutral")  # Produces 0.0 faithfulness
        gate = AbstentionGate(
            config=AbstentionConfig(min_confidence=0.50, min_faithfulness=0.70),
            evaluator=FaithfulnessEvaluator(nli_model=nli),
        )

        final_label, final_conf, abstained, audit = gate.evaluate(
            claim_id="c3",
            claim="Vaccines reduce hospitalizations.",
            label="True",
            confidence=0.95,
            explanation="Unsupported statement unrelated to evidence.",
            evidence_passages=sample_passages,
            cited_evidence_ids=["doc_1"],
        )

        assert abstained is True
        assert final_label == "Unverified"
        assert audit.original_label == "True"
        assert audit.faithfulness_score == 0.0
        assert any("Low explanation faithfulness" in r for r in audit.abstention_reasons)

    def test_flag_only_mode_preserves_label(self, sample_passages):
        """In flag_only mode, original label is preserved even when thresholds fail."""
        nli = MockNLIModel(default_label="neutral")
        gate = AbstentionGate(
            config=AbstentionConfig(mode="flag_only", min_confidence=0.80, min_faithfulness=0.80),
            evaluator=FaithfulnessEvaluator(nli_model=nli),
        )

        final_label, final_conf, abstained, audit = gate.evaluate(
            claim_id="c4",
            claim="Unemployment is down.",
            label="True",
            confidence=0.60,
            explanation="Unemployment is down.",
            evidence_passages=sample_passages,
            cited_evidence_ids=["doc_2"],
        )

        assert abstained is False
        assert audit.flagged_only is True
        assert final_label == "True"  # Preserved!
        assert len(audit.abstention_reasons) > 0


# ---------------------------------------------------------------------------
# 8. Verdict Integration and Audit Trail
# ---------------------------------------------------------------------------


class TestVerdictIntegration:
    def test_apply_abstention_to_verdict_object(self, sample_passages):
        """Verdict object is updated and audit trail is attached to metadata."""
        nli = MockNLIModel(default_label="neutral")
        verdict = Verdict(
            claim_id="v_101",
            claim="Test claim statement.",
            claim_date=None,
            label="True",
            confidence=0.90,
            evidence_ids=["doc_1"],
            explanation="Unfaithful statement.",
            parse_failed=False,
            retrieved_passages=sample_passages,
            queries_used=["test query"],
            stage_timings_s={},
            model_metadata={"model": "gpt-4"},
            raw_llm_output=None,
            run_id=None,
        )

        updated_verdict, audit = apply_abstention(
            verdict=verdict,
            min_confidence=0.50,
            min_faithfulness=0.70,
            evaluator=FaithfulnessEvaluator(nli_model=nli),
        )

        assert updated_verdict.label == "Unverified"
        assert audit.original_label == "True"
        assert audit.final_label == "Unverified"
        assert audit.abstained is True
        assert "abstention" in updated_verdict.model_metadata
        assert updated_verdict.model_metadata["abstention"]["abstained"] is True

        # Ensure original verdict object was not mutated in place
        assert verdict.label == "True"
