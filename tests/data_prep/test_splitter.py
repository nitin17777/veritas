"""Tests for the VERITAS dataset splitting and evidence leakage filtering module.

Covers:
- filter_evidence_leakage: disabled passthrough, no-date passthrough, date-based dropping,
  fact-checker domain evidence, records not mutated in place
- assign_splits: official split preservation, stratified splitting, cross-split ID uniqueness,
  exact-text duplicate detection, topic holdout (auto and explicit), strict mode errors
- save_splits: JSONL output files, split report JSON file
- _stratified_split: ratio enforcement, label proportionality
- _select_heldout_topic: minimum record threshold, deterministic selection
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from veritas.data_prep.splitter import (
    DEFAULT_FACT_CHECKER_DOMAINS,
    DEFAULT_SPLIT_RATIOS,
    LeakageFilterStats,
    SplitReport,
    TopicHoldoutReport,
    _claim_fingerprint,
    _find_duplicate_ids,
    _select_heldout_topic,
    _stratified_split,
    assign_splits,
    filter_evidence_leakage,
    save_splits,
)

import random as _stdlib_random

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LABELS = ["True", "Partially True", "Misleading", "False", "Unverified"]


def _make_record(
    rec_id: str,
    claim: str = "Some factual claim.",
    label: str = "True",
    claim_date: str | None = None,
    topic: str | None = None,
    split: str = "train",
    evidence: list[dict] | None = None,
) -> dict[str, Any]:
    return {
        "id": rec_id,
        "claim": claim,
        "label": label,
        "claim_date": claim_date,
        "topic": topic,
        "split": split,
        "evidence": evidence if evidence is not None else [],
    }


def _make_evidence(
    text: str = "Supporting text.",
    url: str | None = "https://example.com",
    date: str | None = None,
    domain: str | None = "example.com",
) -> dict[str, Any]:
    return {"text": text, "url": url, "date": date, "domain": domain}


def _make_corpus(n: int = 50) -> list[dict[str, Any]]:
    """Generate a balanced corpus of ``n`` records across all 5 labels."""
    records = []
    for i in range(n):
        records.append(
            _make_record(
                rec_id=f"rec:{i}",
                claim=f"Claim number {i}.",
                label=_LABELS[i % len(_LABELS)],
                topic=f"topic_{i % 5}",  # 5 topics, each with n/5 records
            )
        )
    return records


# ===========================================================================
# filter_evidence_leakage
# ===========================================================================


class TestFilterEvidenceLeakage:
    def test_disabled_returns_records_unchanged(self):
        rec = _make_record(
            "r1",
            claim_date="2023-01-01",
            evidence=[_make_evidence(date="2024-06-01")],
        )
        result, stats = filter_evidence_leakage([rec], enabled=False)
        assert len(result) == 1
        assert result[0]["evidence"][0]["date"] == "2024-06-01"
        assert stats.evidence_dropped == 0

    def test_disabled_counts_total_evidence(self):
        records = [
            _make_record("r1", evidence=[_make_evidence(), _make_evidence()]),
            _make_record("r2", evidence=[_make_evidence()]),
        ]
        _, stats = filter_evidence_leakage(records, enabled=False)
        assert stats.total_evidence_items == 3
        assert stats.total_records_examined == 2

    def test_no_claim_date_keeps_all_evidence(self):
        rec = _make_record(
            "r1",
            claim_date=None,
            evidence=[_make_evidence(date="2099-01-01")],
        )
        result, stats = filter_evidence_leakage([rec])
        assert len(result[0]["evidence"]) == 1
        assert stats.records_without_date == 1
        assert stats.evidence_dropped == 0

    def test_evidence_before_date_kept(self):
        rec = _make_record(
            "r1",
            claim_date="2023-06-01",
            evidence=[_make_evidence(date="2023-05-31")],
        )
        result, stats = filter_evidence_leakage([rec])
        assert len(result[0]["evidence"]) == 1
        assert stats.evidence_dropped == 0

    def test_evidence_same_day_kept(self):
        rec = _make_record(
            "r1",
            claim_date="2023-06-01",
            evidence=[_make_evidence(date="2023-06-01")],
        )
        result, stats = filter_evidence_leakage([rec])
        assert len(result[0]["evidence"]) == 1
        assert stats.evidence_dropped == 0

    def test_evidence_after_date_dropped(self):
        rec = _make_record(
            "r1",
            claim_date="2023-06-01",
            evidence=[
                _make_evidence(date="2023-05-01"),  # kept
                _make_evidence(date="2023-07-01"),  # dropped
                _make_evidence(date="2024-01-01"),  # dropped
            ],
        )
        result, stats = filter_evidence_leakage([rec])
        assert len(result[0]["evidence"]) == 1
        assert result[0]["evidence"][0]["date"] == "2023-05-01"
        assert stats.evidence_dropped == 2
        assert stats.records_affected == 1

    def test_evidence_without_date_kept(self):
        rec = _make_record(
            "r1",
            claim_date="2023-06-01",
            evidence=[_make_evidence(date=None)],
        )
        result, stats = filter_evidence_leakage([rec])
        assert len(result[0]["evidence"]) == 1
        assert stats.evidence_dropped == 0

    def test_fact_checker_domain_evidence_after_date_dropped(self):
        """Evidence from fact-checker domains is dropped if it post-dates the claim."""
        rec = _make_record(
            "r1",
            claim_date="2023-06-01",
            evidence=[
                _make_evidence(
                    text="PolitiFact analysis",
                    url="https://politifact.com/...",
                    date="2023-07-01",
                    domain="politifact.com",
                )
            ],
        )
        result, stats = filter_evidence_leakage(
            [rec], fact_checker_domains=DEFAULT_FACT_CHECKER_DOMAINS
        )
        assert len(result[0]["evidence"]) == 0
        assert stats.evidence_dropped == 1

    def test_original_records_not_mutated(self):
        """filter_evidence_leakage must not mutate the caller's original records."""
        original_ev = _make_evidence(date="2099-01-01")
        rec = _make_record("r1", claim_date="2023-06-01", evidence=[original_ev])
        original_evidence_list = rec["evidence"]

        filter_evidence_leakage([rec])

        # Original list object must be untouched
        assert rec["evidence"] is original_evidence_list

    def test_multiple_records_mixed_dates(self):
        records = [
            _make_record(
                "r1",
                claim_date="2023-06-01",
                evidence=[
                    _make_evidence(date="2023-05-01"),
                    _make_evidence(date="2024-01-01"),
                ],
            ),
            _make_record(
                "r2",
                claim_date=None,
                evidence=[_make_evidence(date="2024-01-01")],
            ),
            _make_record(
                "r3",
                claim_date="2022-01-01",
                evidence=[_make_evidence(date="2021-12-31")],
            ),
        ]
        result, stats = filter_evidence_leakage(records)
        assert stats.total_records_examined == 3
        assert stats.records_with_date == 2
        assert stats.records_without_date == 1
        assert stats.evidence_dropped == 1
        assert stats.records_affected == 1

    def test_empty_records_list(self):
        result, stats = filter_evidence_leakage([])
        assert result == []
        assert stats.total_records_examined == 0


# ===========================================================================
# _stratified_split
# ===========================================================================


class TestStratifiedSplit:
    def test_ratios_must_sum_to_one(self):
        records = [_make_record(f"r{i}") for i in range(10)]
        rng = _stdlib_random.Random(42)
        with pytest.raises(ValueError, match="sum to 1.0"):
            _stratified_split(records, {"train": 0.7, "test": 0.2}, rng)

    def test_all_records_assigned(self):
        records = _make_corpus(100)
        rng = _stdlib_random.Random(42)
        result = _stratified_split(records, DEFAULT_SPLIT_RATIOS, rng)
        total = sum(len(v) for v in result.values())
        assert total == 100

    def test_no_record_in_multiple_splits(self):
        records = _make_corpus(100)
        rng = _stdlib_random.Random(42)
        result = _stratified_split(records, DEFAULT_SPLIT_RATIOS, rng)
        all_ids = [r["id"] for split in result.values() for r in split]
        assert len(all_ids) == len(set(all_ids))

    def test_label_proportions_approximately_preserved(self):
        """Each label should be distributed proportionally across splits."""
        records = _make_corpus(500)
        rng = _stdlib_random.Random(42)
        result = _stratified_split(records, DEFAULT_SPLIT_RATIOS, rng)

        for label in _LABELS:
            label_total = sum(1 for r in records if r["label"] == label)
            train_count = sum(1 for r in result["train"] if r["label"] == label)
            expected_train = DEFAULT_SPLIT_RATIOS["train"] * label_total
            # Allow ±2 record tolerance due to rounding
            assert abs(train_count - expected_train) <= 2, (
                f"Label {label}: expected ~{expected_train:.0f} in train, got {train_count}"
            )

    def test_reproducible_with_same_seed(self):
        records = _make_corpus(200)
        rng1 = _stdlib_random.Random(99)
        rng2 = _stdlib_random.Random(99)
        r1 = _stratified_split(records[:], DEFAULT_SPLIT_RATIOS, rng1)
        r2 = _stratified_split(records[:], DEFAULT_SPLIT_RATIOS, rng2)
        assert [r["id"] for r in r1["train"]] == [r["id"] for r in r2["train"]]

    def test_different_seeds_produce_different_splits(self):
        records = _make_corpus(200)
        rng1 = _stdlib_random.Random(1)
        rng2 = _stdlib_random.Random(2)
        r1 = _stratified_split(records[:], DEFAULT_SPLIT_RATIOS, rng1)
        r2 = _stratified_split(records[:], DEFAULT_SPLIT_RATIOS, rng2)
        # Very unlikely to produce identical train splits with different seeds on 200 records
        assert [r["id"] for r in r1["train"]] != [r["id"] for r in r2["train"]]


# ===========================================================================
# _select_heldout_topic
# ===========================================================================


class TestSelectHeldoutTopic:
    def test_returns_most_common_topic(self):
        # topic_A appears 10x, topic_B appears 3x, topic_C appears 2x
        records = (
            [_make_record(f"a{i}", topic="topic_A") for i in range(10)]
            + [_make_record(f"b{i}", topic="topic_B") for i in range(3)]
            + [_make_record(f"c{i}", topic="topic_C") for i in range(2)]
        )
        rng = _stdlib_random.Random(42)
        chosen = _select_heldout_topic(records, rng)
        assert chosen == "topic_A"

    def test_respects_minimum_records_threshold(self):
        records = [_make_record(f"a{i}", topic="small_topic") for i in range(3)]
        rng = _stdlib_random.Random(42)
        # min_records default is 5, so this topic should not be returned
        chosen = _select_heldout_topic(records, rng, min_records=5)
        assert chosen is None

    def test_no_topics_returns_none(self):
        records = [_make_record(f"r{i}", topic=None) for i in range(20)]
        rng = _stdlib_random.Random(42)
        assert _select_heldout_topic(records, rng) is None

    def test_deterministic_alphabetical_tiebreak(self):
        """When two topics have the same count, the alphabetically earlier one wins."""
        records = (
            [_make_record(f"a{i}", topic="alpha") for i in range(10)]
            + [_make_record(f"b{i}", topic="beta") for i in range(10)]
        )
        rng = _stdlib_random.Random(42)
        chosen = _select_heldout_topic(records, rng)
        assert chosen == "alpha"


# ===========================================================================
# assign_splits
# ===========================================================================


class TestAssignSplits:
    def test_empty_records_raises(self):
        with pytest.raises(ValueError, match="empty record list"):
            assign_splits([])

    def test_official_splits_preserved(self):
        """When all records have valid splits, they must not be reassigned."""
        records = [
            _make_record("r1", split="train"),
            _make_record("r2", split="validation"),
            _make_record("r3", split="test"),
        ]
        result, report = assign_splits(records, heldout_topic=None)
        assert report.strategy == "official"
        assert {r["id"]: r["split"] for r in result} == {
            "r1": "train",
            "r2": "validation",
            "r3": "test",
        }

    def test_stratified_split_when_no_official_splits(self):
        """Records without a split field trigger stratified assignment."""
        records = []
        for i in range(100):
            rec = _make_record(f"r{i}", label=_LABELS[i % len(_LABELS)])
            del rec["split"]  # remove official split to trigger stratification
            records.append(rec)

        result, report = assign_splits(records, heldout_topic=None)
        assert report.strategy == "stratified"
        assigned_splits = {r["split"] for r in result}
        assert assigned_splits <= {"train", "validation", "test"}

    def test_preserve_official_splits_false_forces_stratification(self):
        records = [
            _make_record(f"r{i}", label=_LABELS[i % len(_LABELS)])
            for i in range(50)
        ]
        result, report = assign_splits(
            records,
            preserve_official_splits=False,
            heldout_topic=None,
        )
        assert report.strategy == "stratified"

    def test_all_records_assigned_to_a_split(self):
        records = _make_corpus(60)
        result, _ = assign_splits(records, heldout_topic=None)
        assert all("split" in r and r["split"] in {"train", "validation", "test"} for r in result)

    def test_no_cross_split_duplicates_after_stratified(self):
        records = _make_corpus(100)
        for r in records:
            del r["split"]
        result, report = assign_splits(records, heldout_topic=None)
        assert report.cross_split_duplicate_ids == []

    def test_cross_split_duplicate_id_detected_strict(self):
        """Injecting the same ID into two different splits raises in strict mode."""
        records = [
            _make_record("dup_id", split="train"),
            _make_record("dup_id", claim="Different claim.", split="test"),
        ]
        with pytest.raises(ValueError, match="multiple splits"):
            assign_splits(records, heldout_topic=None, strict=True)

    def test_cross_split_duplicate_id_logged_non_strict(self):
        records = [
            _make_record("dup_id", split="train"),
            _make_record("dup_id", claim="Different claim.", split="test"),
        ]
        _, report = assign_splits(records, heldout_topic=None, strict=False)
        assert "dup_id" in report.cross_split_duplicate_ids

    def test_exact_text_duplicate_detected(self):
        records = [
            _make_record("r1", claim="Identical claim text.", split="train"),
            _make_record("r2", claim="Identical claim text.", split="validation"),
        ]
        _, report = assign_splits(records, heldout_topic=None)
        assert set(report.duplicate_claim_ids) == {"r1", "r2"}

    def test_auto_heldout_topic_selected(self):
        records = _make_corpus(50)  # topic_0 .. topic_4, each 10 records
        result, report = assign_splits(records, heldout_topic="auto")
        assert report.heldout_topic is not None
        assert report.heldout_topic.record_count > 0
        # All records with the chosen topic should be heldout_topic
        chosen = report.heldout_topic.topic
        for r in result:
            if r.get("topic") == chosen:
                assert r["split"] == "heldout_topic"

    def test_explicit_heldout_topic(self):
        records = _make_corpus(50)
        result, report = assign_splits(records, heldout_topic="topic_0")
        assert report.heldout_topic is not None
        assert report.heldout_topic.topic == "topic_0"
        for r in result:
            if r.get("topic") == "topic_0":
                assert r["split"] == "heldout_topic"

    def test_heldout_topic_none_skips_holdout(self):
        records = _make_corpus(50)
        _, report = assign_splits(records, heldout_topic=None)
        assert report.heldout_topic is None
        # No record should have heldout_topic split
        # (corpus starts with "train" splits, official splits preserved)
        assert not any(r["split"] == "heldout_topic" for r in records)

    def test_split_stats_populated(self):
        records = _make_corpus(50)
        result, report = assign_splits(records, heldout_topic=None)
        split_names = {s.split for s in report.split_stats}
        assert "train" in split_names or "heldout_topic" in split_names

    def test_total_record_count_preserved(self):
        n = 80
        records = _make_corpus(n)
        result, report = assign_splits(records, heldout_topic="auto")
        total = sum(s.record_count for s in report.split_stats)
        assert total == n

    def test_reproducibility_same_seed(self):
        records_a = _make_corpus(100)
        records_b = _make_corpus(100)
        for r in records_a:
            del r["split"]
        for r in records_b:
            del r["split"]

        result_a, _ = assign_splits(records_a, heldout_topic=None, random_seed=7)
        result_b, _ = assign_splits(records_b, heldout_topic=None, random_seed=7)

        ids_a = sorted(r["id"] + r["split"] for r in result_a)
        ids_b = sorted(r["id"] + r["split"] for r in result_b)
        assert ids_a == ids_b

    def test_dataset_id_in_report(self):
        records = [_make_record("r1")]
        _, report = assign_splits(records, dataset_id="my_dataset", heldout_topic=None)
        assert report.dataset_id == "my_dataset"

    def test_random_seed_in_report(self):
        records = [_make_record("r1")]
        _, report = assign_splits(records, random_seed=123, heldout_topic=None)
        assert report.random_seed == 123

    def test_deduplication_policy_in_report(self):
        records = [_make_record("r1")]
        _, report = assign_splits(records, heldout_topic=None)
        assert "exact" in report.deduplication_policy.lower()

    def test_invalid_record_non_strict_logs_warning(self):
        bad_rec = {
            "id": "bad",
            "claim": "A claim.",
            "label": "NOT_VALID_LABEL",
            "claim_date": None,
            "topic": None,
            "split": "train",
            "evidence": [],
        }
        _, report = assign_splits([bad_rec], heldout_topic=None, strict=False)
        assert any("bad" in w for w in report.warnings)

    def test_invalid_record_strict_raises(self):
        bad_rec = {
            "id": "bad",
            "claim": "A claim.",
            "label": "NOT_VALID_LABEL",
            "claim_date": None,
            "topic": None,
            "split": "train",
            "evidence": [],
        }
        with pytest.raises(ValueError):
            assign_splits([bad_rec], heldout_topic=None, strict=True)

    def test_heldout_topic_records_absent_from_other_splits(self):
        """No record should appear in both heldout_topic and any other split."""
        records = _make_corpus(50)
        result, report = assign_splits(records, heldout_topic="auto")
        held_topic = report.heldout_topic.topic if report.heldout_topic else None
        if held_topic:
            for r in result:
                if r.get("topic") == held_topic:
                    assert r["split"] == "heldout_topic"


# ===========================================================================
# save_splits
# ===========================================================================


class TestSaveSplits:
    def test_jsonl_files_created_per_split(self, tmp_path: Path):
        records = [
            _make_record("r1", split="train"),
            _make_record("r2", split="train"),
            _make_record("r3", split="validation"),
            _make_record("r4", split="test"),
        ]
        counts = save_splits(records, tmp_path)

        assert (tmp_path / "train.jsonl").is_file()
        assert (tmp_path / "validation.jsonl").is_file()
        assert (tmp_path / "test.jsonl").is_file()
        assert counts == {"train": 2, "validation": 1, "test": 1}

    def test_jsonl_content_is_valid_json(self, tmp_path: Path):
        records = [_make_record("r1", split="train")]
        save_splits(records, tmp_path)

        with (tmp_path / "train.jsonl").open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["id"] == "r1"

    def test_report_saved_when_provided(self, tmp_path: Path):
        records = [_make_record("r1", split="train")]
        _, report = assign_splits(records, heldout_topic=None)
        save_splits(records, tmp_path, report=report)

        report_file = tmp_path / "split_report.json"
        assert report_file.is_file()
        data = json.loads(report_file.read_text("utf-8"))
        assert data["dataset_id"] == "unknown"

    def test_report_not_saved_when_not_provided(self, tmp_path: Path):
        records = [_make_record("r1", split="train")]
        save_splits(records, tmp_path, report=None)
        assert not (tmp_path / "split_report.json").is_file()

    def test_output_dir_created_if_absent(self, tmp_path: Path):
        out = tmp_path / "new" / "nested" / "dir"
        records = [_make_record("r1", split="train")]
        save_splits(records, out)
        assert out.is_dir()
        assert (out / "train.jsonl").is_file()

    def test_heldout_topic_split_written(self, tmp_path: Path):
        records = [
            _make_record("r1", split="heldout_topic", topic="held"),
        ]
        counts = save_splits(records, tmp_path)
        assert (tmp_path / "heldout_topic.jsonl").is_file()
        assert counts["heldout_topic"] == 1


# ===========================================================================
# _claim_fingerprint and _find_duplicate_ids
# ===========================================================================


class TestFingerprint:
    def test_same_text_same_fingerprint(self):
        assert _claim_fingerprint("Hello World") == _claim_fingerprint("Hello World")

    def test_case_insensitive(self):
        assert _claim_fingerprint("HELLO WORLD") == _claim_fingerprint("hello world")

    def test_whitespace_normalised(self):
        assert _claim_fingerprint("hello   world") == _claim_fingerprint("hello world")

    def test_different_text_different_fingerprint(self):
        assert _claim_fingerprint("foo") != _claim_fingerprint("bar")

    def test_find_duplicates_empty(self):
        assert _find_duplicate_ids([]) == []

    def test_find_duplicates_no_dups(self):
        records = [
            _make_record("r1", claim="Claim one."),
            _make_record("r2", claim="Claim two."),
        ]
        assert _find_duplicate_ids(records) == []

    def test_find_duplicates_detected(self):
        records = [
            _make_record("r1", claim="Same claim."),
            _make_record("r2", claim="Same claim."),
            _make_record("r3", claim="Different claim."),
        ]
        dups = _find_duplicate_ids(records)
        assert set(dups) == {"r1", "r2"}

    def test_find_duplicates_case_insensitive(self):
        records = [
            _make_record("r1", claim="Same CLAIM."),
            _make_record("r2", claim="same claim."),
        ]
        dups = _find_duplicate_ids(records)
        assert set(dups) == {"r1", "r2"}


# ===========================================================================
# Integration: assign_splits + filter_evidence_leakage + save_splits
# ===========================================================================


class TestIntegration:
    def test_full_pipeline_with_leakage_filter(self, tmp_path: Path):
        """End-to-end pipeline: split assignment -> leakage filter -> save."""
        records = []
        for i in range(30):
            ev_date = "2023-07-01" if i % 3 == 0 else "2022-01-01"
            records.append(
                _make_record(
                    f"rec:{i}",
                    claim=f"Claim {i}.",
                    label=_LABELS[i % len(_LABELS)],
                    claim_date="2023-06-01",
                    topic=f"topic_{i % 3}",
                    evidence=[_make_evidence(date=ev_date)],
                )
            )

        # Step 1: assign splits (no official splits provided)
        for r in records:
            del r["split"]
        assigned, split_report = assign_splits(
            records, dataset_id="integration", heldout_topic="auto", random_seed=42
        )

        # Step 2: filter leakage
        filtered, leak_stats = filter_evidence_leakage(assigned, enabled=True)
        assert leak_stats.evidence_dropped > 0  # some future-dated evidence dropped

        # Step 3: save
        counts = save_splits(filtered, tmp_path / "output", report=split_report)
        assert sum(counts.values()) == 30

        # Verify JSONL files are valid
        for split_name, count in counts.items():
            fpath = tmp_path / "output" / f"{split_name}.jsonl"
            recs = [json.loads(line) for line in fpath.read_text("utf-8").splitlines()]
            assert len(recs) == count

    def test_full_pipeline_without_leakage_filter(self, tmp_path: Path):
        records = [
            _make_record(
                f"r{i}",
                claim=f"Claim {i}.",
                label=_LABELS[i % len(_LABELS)],
                claim_date="2023-06-01",
                topic=f"topic_{i % 4}",
                evidence=[_make_evidence(date="2099-01-01")],
            )
            for i in range(40)
        ]
        for r in records:
            del r["split"]

        assigned, _ = assign_splits(records, heldout_topic="auto")
        filtered, leak_stats = filter_evidence_leakage(assigned, enabled=False)

        # No evidence dropped when filter disabled
        assert leak_stats.evidence_dropped == 0
        # All future-dated evidence retained
        for r in filtered:
            assert len(r["evidence"]) == 1
