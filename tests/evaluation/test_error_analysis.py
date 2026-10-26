"""Unit and integration tests for VERITAS Error Analysis workflow.

Verifies:
- Accurate extraction of incorrect predictions (correct predictions excluded).
- Integrity: Raw predictions are strictly immutable.
- Retention of human annotations across repeated export runs.
- Diagnostic heuristic suggestions vs. human ground-truth distinction.
- Category proportion and counts computation.
- Comprehensive markdown report generation with representative failure examples.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import pytest

from veritas.evaluation.error_analysis import (
    ERROR_CATEGORIES,
    ErrorRecord,
    compute_error_statistics,
    export_errors_to_csv,
    extract_prediction_errors,
    generate_error_summary_report,
    infer_suggested_error_category,
    load_annotations_from_csv,
)


@pytest.fixture
def sample_gold_and_predictions() -> tuple[list[dict], list[dict]]:
    gold = [
        {
            "id": "c1",
            "claim": "Claim 1: The Earth orbits the Sun.",
            "label": "True",
            "claim_date": "2024-01-01",
            "evidence_ids": ["doc_astro_1"],
        },
        {
            "id": "c2",
            "claim": "Claim 2: Drinking bleach cures viral infections.",
            "label": "False",
            "claim_date": "2020-04-15",
            "evidence_ids": ["doc_cdc_bleach"],
        },
        {
            "id": "c3",
            "claim": "Claim 3: Nonfarm payrolls rose in February 2024.",
            "label": "True",
            "claim_date": "2024-03-08",
            "evidence_ids": ["doc_bls_jobs"],
        },
        {
            "id": "c4",
            "claim": "Claim 4: Jupiter has no moons.",
            "label": "False",
            "claim_date": "2023-01-01",
            "evidence_ids": ["doc_jupiter_moons"],
        },
    ]

    preds = [
        # c1: Correct prediction (should be skipped by error extraction)
        {
            "claim_id": "c1",
            "claim": "Claim 1: The Earth orbits the Sun.",
            "label": "True",
            "confidence": 0.98,
            "explanation": "doc_astro_1 confirms heliocentric orbit.",
            "evidence_ids": ["doc_astro_1"],
            "retrieved_passages": [{"passage_id": "doc_astro_1", "text": "Heliocentric model", "score": 0.95}],
            "parse_failed": False,
        },
        # c2: Incorrect prediction - LLM misinterpretation (cited doc_cdc_bleach but said True)
        {
            "claim_id": "c2",
            "claim": "Claim 2: Drinking bleach cures viral infections.",
            "label": "True",
            "confidence": 0.85,
            "explanation": "Evidence mentions bleach.",
            "evidence_ids": ["doc_cdc_bleach"],
            "retrieved_passages": [
                {
                    "passage_id": "doc_cdc_bleach",
                    "text": "Drinking bleach is hazardous and does not cure disease.",
                    "domain": "cdc.gov",
                    "date": "2020-04-10",
                    "score": 0.92,
                }
            ],
            "parse_failed": False,
        },
        # c3: Incorrect prediction - Retrieval miss (gold doc_bls_jobs not retrieved)
        {
            "claim_id": "c3",
            "claim": "Claim 3: Nonfarm payrolls rose in February 2024.",
            "label": "Unverified",
            "confidence": 0.30,
            "explanation": "No relevant jobs data retrieved.",
            "evidence_ids": [],
            "retrieved_passages": [
                {"passage_id": "doc_unrelated", "text": "Unrelated passage about cooking.", "score": 0.10}
            ],
            "parse_failed": False,
        },
        # c4: Incorrect prediction - Parsing failure
        {
            "claim_id": "c4",
            "claim": "Claim 4: Jupiter has no moons.",
            "label": "Unverified",
            "confidence": 0.0,
            "explanation": "JSON parse failed",
            "evidence_ids": [],
            "retrieved_passages": [],
            "parse_failed": True,
        },
    ]

    return gold, preds


class TestErrorExtraction:
    def test_only_incorrect_predictions_extracted(self, sample_gold_and_predictions) -> None:
        gold, preds = sample_gold_and_predictions
        errors = extract_prediction_errors(gold, preds)

        # 4 total claims: 1 correct (c1), 3 incorrect (c2, c3, c4)
        assert len(errors) == 3
        err_ids = {e.claim_id for e in errors}
        assert "c1" not in err_ids
        assert err_ids == {"c2", "c3", "c4"}

    def test_raw_predictions_remain_unmodified(self, sample_gold_and_predictions) -> None:
        gold, preds = sample_gold_and_predictions
        preds_backup = copy.deepcopy(preds)

        extract_prediction_errors(gold, preds)

        assert preds == preds_backup, "extract_prediction_errors modified input predictions!"

    def test_diagnostic_heuristics(self, sample_gold_and_predictions) -> None:
        gold, preds = sample_gold_and_predictions
        errors = extract_prediction_errors(gold, preds)
        by_id = {e.claim_id: e for e in errors}

        # c4 had parse_failed=True
        assert by_id["c4"].suggested_category == "Parsing failure"

        # c3 was retrieval miss (gold doc_bls_jobs not retrieved)
        assert by_id["c3"].suggested_category == "Retrieval miss"

        # c2 cited evidence but predicted opposite
        assert by_id["c2"].suggested_category == "LLM misinterpretation"

        # Automated suggestions are NOT ground truth
        for e in errors:
            assert e.annotated_category == "", "annotated_category should be empty by default!"
            assert e.verification_status == "unreviewed"


class TestAnnotationPreservation:
    def test_merge_retains_human_annotations(self, sample_gold_and_predictions) -> None:
        gold, preds = sample_gold_and_predictions

        # Simulate prior human review of c2
        prior_annotations = {
            "c2": {
                "annotated_category": "LLM misinterpretation",
                "annotator_id": "annotator_alice",
                "annotator_notes": "Model reversed the polarity despite CDC passage warning against bleach.",
                "verification_status": "verified",
            }
        }

        errors = extract_prediction_errors(gold, preds, existing_annotations=prior_annotations)
        by_id = {e.claim_id: e for e in errors}

        c2_err = by_id["c2"]
        assert c2_err.annotated_category == "LLM misinterpretation"
        assert c2_err.annotator_id == "annotator_alice"
        assert "reversed the polarity" in c2_err.annotator_notes
        assert c2_err.verification_status == "verified"

        # Unreviewed claims remain blank
        assert by_id["c3"].annotated_category == ""
        assert by_id["c3"].verification_status == "unreviewed"

    def test_csv_export_and_reimport(self, sample_gold_and_predictions, tmp_path: Path) -> None:
        gold, preds = sample_gold_and_predictions
        errors = extract_prediction_errors(gold, preds)

        # Manually annotate one error
        errors[0].annotated_category = "LLM misinterpretation"
        errors[0].annotator_id = "reviewer_1"
        errors[0].annotator_notes = "Verified failure"
        errors[0].verification_status = "verified"

        csv_out = tmp_path / "errors.csv"
        export_errors_to_csv(errors, csv_out)
        assert csv_out.exists()

        # Re-import annotations
        loaded_ann = load_annotations_from_csv(csv_out)
        assert errors[0].claim_id in loaded_ann
        assert loaded_ann[errors[0].claim_id]["annotated_category"] == "LLM misinterpretation"
        assert loaded_ann[errors[0].claim_id]["annotator_id"] == "reviewer_1"


class TestErrorStatisticsAndReporting:
    def test_compute_error_statistics(self, sample_gold_and_predictions) -> None:
        gold, preds = sample_gold_and_predictions
        errors = extract_prediction_errors(gold, preds)

        stats = compute_error_statistics(errors, total_evaluated_claims=4)

        assert stats["total_claims"] == 4
        assert stats["total_errors"] == 3
        assert stats["error_rate"] == 0.75
        assert stats["suggested_counts"]["Parsing failure"] == 1
        assert stats["suggested_counts"]["Retrieval miss"] == 1
        assert stats["suggested_counts"]["LLM misinterpretation"] == 1

        # Proportions sum up appropriately
        assert sum(stats["suggested_proportions"].values()) == pytest.approx(1.0, rel=1e-3)

    def test_generate_error_summary_report(self, sample_gold_and_predictions) -> None:
        gold, preds = sample_gold_and_predictions
        errors = extract_prediction_errors(gold, preds)

        # Annotate one error
        errors[0].annotated_category = "LLM misinterpretation"
        errors[0].annotator_notes = "Clear failure to process negation."

        report_md = generate_error_summary_report(
            errors,
            total_evaluated_claims=4,
            dataset_name="LIAR",
            run_id="run_test_001",
        )

        assert "# VERITAS Qualitative Error Analysis Report: `run_test_001`" in report_md
        assert "Total Incorrect Predictions**: `3`" in report_md
        assert "Error Category Distribution" in report_md
        for cat in ERROR_CATEGORIES:
            assert cat in report_md
        assert "Representative Failure Patterns with Concrete Examples" in report_md
        assert "Claim 2: Drinking bleach cures viral infections" in report_md
