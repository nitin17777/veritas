"""Regression and unit tests for LIAR dataset ingestion and TSV boundary recovery."""

import csv
import json
from pathlib import Path

import pytest

from veritas.data_prep.liar_loader import (
    COUNT_COLUMNS,
    SOURCE_COLUMNS,
    _parse_row,
    load_liar,
    read_liar_tsv_records,
)
from veritas.data_prep.schema import validate_claim_record

SAMPLE_ROW = [
    "12345.json",
    "mostly-true",
    "The unemployment rate dropped by 2 percent last year.",
    "economy,jobs",
    "john-doe",
    "Governor",
    "Ohio",
    "republican",
    "1",
    "2",
    "3",
    "4",
    "0",
    "a campaign speech in Columbus",
]


def _make_tsv_line(fields: list[str]) -> str:
    return "\t".join(fields) + "\n"


def test_parse_row_valid():
    """Verify standard 14-field row parsing and schema compliance."""
    record = _parse_row(SAMPLE_ROW, "train")

    assert record["id"] == "liar:12345.json"
    assert record["claim"] == "The unemployment rate dropped by 2 percent last year."
    assert record["label"] == "True"  # mostly-true maps to True
    assert record["topic"] == "economy,jobs"
    assert record["split"] == "train"
    assert record["metadata"]["source_id"] == "12345.json"
    assert record["metadata"]["speaker"] == "john-doe"
    assert record["metadata"]["speaker_job"] == "Governor"
    assert record["metadata"]["state_info"] == "Ohio"
    assert record["metadata"]["party_affiliation"] == "republican"
    assert record["metadata"]["speaker_history"] == {
        "barely_true": 1,
        "false": 2,
        "half_true": 3,
        "mostly_true": 4,
        "pants_fire": 0,
    }
    assert record["metadata"]["context"] == "a campaign speech in Columbus"

    # Schema validation must pass
    validate_claim_record(record)


def test_line_2155_malformed_quote_pattern():
    """Reproduce the exact line 2155/2156 pattern that broke standard CSV readers.

    In line 2155, an opening quote was closed with LaTeX '' instead of \".
    Standard csv.reader treats the quote as open, swallowing line 2156 until closed
    by \"The Daily Show\", producing a corrupted 3-column row.
    Our loader treats quotes as literal text and parses both records intact.
    """
    line_2155 = (
        "638.json\tfalse\t\"The fact is that although we have had a president "
        "who is opposed to abortion over the last eight years, abortions have not "
        "gone down.''	abortion	barack-obama	President	Illinois	democrat	"
        "70	71	160	163	9	a TV interview with megachurch pastor Rick Warren in Lake Forest, Calif.\n"
    )
    line_2156 = (
        "2724.json\ttrue\tMost of the jobs that we lost were lost before the economic "
        "policies we put in place had any effect.	economy,job-accomplishments,jobs,stimulus	"
        "barack-obama	President	Illinois	democrat	70	71	160	163	9	"
        "an interview on \"The Daily Show with Jon Stewart\"\n"
    )

    # 1. Demonstrate standard CSV reader failure / corruption
    csv_rows = list(csv.reader([line_2155, line_2156], delimiter="\t"))
    # Standard reader merges both into a single row of 3 columns
    assert len(csv_rows) == 1
    assert len(csv_rows[0]) == 3

    # 2. Verify corrected loader parses both records separately and validly
    records, audit = read_liar_tsv_records([line_2155, line_2156], split="train")

    assert len(records) == 2
    assert audit.physical_lines == 2
    assert audit.valid_records == 2
    assert audit.unresolved_records == 0
    assert audit.quote_recovered >= 1

    # Check record 1
    assert records[0]["metadata"]["source_id"] == "638.json"
    assert records[0]["label"] == "False"
    assert "The fact is that although" in records[0]["claim"]
    assert records[0]["metadata"]["speaker_history"]["pants_fire"] == 9
    validate_claim_record(records[0])

    # Check record 2
    assert records[1]["metadata"]["source_id"] == "2724.json"
    assert records[1]["label"] == "True"
    assert "Most of the jobs that we lost" in records[1]["claim"]
    assert records[1]["metadata"]["context"] == 'an interview on "The Daily Show with Jon Stewart"'
    validate_claim_record(records[1])


def test_silent_swallowing_line_1281_pattern():
    """Reproduce line 1281 pattern where standard csv.reader silently swallowed records.

    Line 1281 opened a quote with LaTeX '', which remained unclosed until line 1293,
    silently swallowing intermediate records without raising an exception because the
    resulting row coincidentally had 14 columns.
    """
    row_1 = [
        "1606.json",
        "mostly-true",
        '"Hospitals, doctors, MRIs, surgeries and so forth are more expensive\'\'',
        "health-care",
        "mitt-romney",
        "Former governor",
        "Massachusetts",
        "republican",
        "34",
        "32",
        "58",
        "33",
        "19",
        "a Fox News Sunday interview",
    ]
    row_2 = [
        "9874.json",
        "barely-true",
        "Obamacare cuts seniors Medicare.",
        "health-care,medicare",
        "ed-gillespie",
        "Republican strategist",
        "Washington, D.C.",
        "republican",
        "2",
        "3",
        "2",
        "2",
        "1",
        "a campaign email.",
    ]
    row_3 = [
        "1638.json",
        "half-true",
        'Marco Rubio controlled funds that "only $4,000 went to candidates."',
        "elections,ethics",
        "charlie-crist",
        "None",
        "Florida",
        "democrat",
        "15",
        "9",
        "20",
        "19",
        "2",
        "a debate on Fox News Sunday",
    ]

    lines = [_make_tsv_line(row_1), _make_tsv_line(row_2), _make_tsv_line(row_3)]

    # Standard reader merges row 1, 2, 3 into a single row
    csv_rows = list(csv.reader(lines, delimiter="\t"))
    assert len(csv_rows) == 1  # 2 records silently swallowed!

    # Corrected loader parses all 3 records individually
    records, audit = read_liar_tsv_records(lines, split="train")
    assert len(records) == 3
    assert audit.valid_records == 3
    assert [r["metadata"]["source_id"] for r in records] == ["1606.json", "9874.json", "1638.json"]
    for r in records:
        validate_claim_record(r)


def test_multiline_record_recovery_stitched():
    """Verify recovery of records split across physical lines by embedded newlines."""
    # Line 1 has 3 fields (id, label, partial statement)
    line_1 = "9999.json\tfalse\t\"First half of statement\n"
    # Line 2 has continuation statement and remaining 11 fields (total 12 fields)
    line_2 = (
        "second half of statement\"\teconomy\tjane-doe\tMayor\tTexas\t"
        "democrat\t0\t1\t2\t3\t0\tin a speech\n"
    )

    records, audit = read_liar_tsv_records([line_1, line_2], split="train")

    assert len(records) == 1
    assert audit.multiline_recovered == 1
    assert audit.valid_records == 1
    assert records[0]["metadata"]["source_id"] == "9999.json"
    assert "First half of statement second half of statement" in records[0]["claim"]
    assert records[0]["label"] == "False"
    assert records[0]["topic"] == "economy"
    validate_claim_record(records[0])


def test_unescaped_tab_recovery():
    """Verify recovery when an accidental tab is embedded in statement."""
    # 15 columns instead of 14, caused by an unescaped tab inside statement
    parts_15 = [
        "5555.json",
        "true",
        "We created 10,000",
        "new jobs this year.",  # Accidental tab split this statement
        "jobs",
        "speaker-x",
        "Senator",
        "California",
        "democrat",
        "1",
        "2",
        "3",
        "4",
        "5",
        "in a debate",
    ]
    assert len(parts_15) == 15

    line = _make_tsv_line(parts_15)
    records, audit = read_liar_tsv_records([line], split="train")

    assert len(records) == 1
    assert audit.tab_repaired == 1
    assert records[0]["metadata"]["source_id"] == "5555.json"
    assert "We created 10,000 new jobs this year." in records[0]["claim"]
    validate_claim_record(records[0])


def test_unrecoverable_malformed_row_raises_strict():
    """In strict mode, unrecoverable fragments must raise a ValueError."""
    # 3 fields, followed by a complete 14-field row (cannot be stitched)
    line_fragment = "111.json\tfalse\tBroken statement without continuation\n"
    line_normal = _make_tsv_line(SAMPLE_ROW)

    with pytest.raises(ValueError, match="Malformed row with 3 fields could not be recovered"):
        read_liar_tsv_records([line_fragment, line_normal], split="train", strict=True)


def test_unrecoverable_malformed_row_logs_non_strict():
    """In non-strict mode, unrecoverable fragments are audited without crashing."""
    line_fragment = "111.json\tfalse\tBroken statement without continuation\n"
    line_normal = _make_tsv_line(SAMPLE_ROW)

    records, audit = read_liar_tsv_records([line_fragment, line_normal], split="train", strict=False)

    assert len(records) == 1
    assert audit.unresolved_records == 1
    assert records[0]["metadata"]["source_id"] == "12345.json"
    assert any(r.action_type == "unresolved_fragment" for r in audit.repairs)


def test_duplicate_source_id_detected_strict():
    """Duplicate IDs within the same split must raise ValueError in strict mode."""
    line_1 = _make_tsv_line(SAMPLE_ROW)
    line_2 = _make_tsv_line(SAMPLE_ROW)

    with pytest.raises(ValueError, match="Duplicate source ID '12345.json'"):
        read_liar_tsv_records([line_1, line_2], split="train", strict=True)


def test_duplicate_source_id_logged_non_strict():
    """Duplicate IDs in non-strict mode are tracked in audit report."""
    line_1 = _make_tsv_line(SAMPLE_ROW)
    line_2 = _make_tsv_line(SAMPLE_ROW)

    records, audit = read_liar_tsv_records([line_1, line_2], split="train", strict=False)

    assert len(records) == 2
    assert "12345.json" in audit.duplicate_ids


def test_missing_claim_id_raises():
    """A row with an empty claim ID must fail validation."""
    row = list(SAMPLE_ROW)
    row[0] = "   "
    with pytest.raises(ValueError, match="Claim ID is empty"):
        _parse_row(row, "train")


def test_empty_statement_raises():
    """A row with an empty statement must fail validation."""
    row = list(SAMPLE_ROW)
    row[2] = "   "
    with pytest.raises(ValueError, match="Statement is empty"):
        _parse_row(row, "train")


def test_unknown_label_raises():
    """A row with an unrecognized label must fail validation."""
    row = list(SAMPLE_ROW)
    row[1] = "completely-bogus-label"
    with pytest.raises(ValueError, match="Unknown label"):
        _parse_row(row, "train")


def test_invalid_history_counts_raises():
    """Non-integer history counts must fail validation."""
    row = list(SAMPLE_ROW)
    row[8] = "three"
    with pytest.raises(ValueError, match="Invalid integer count"):
        _parse_row(row, "train")


def test_wrong_column_count_raises():
    """Rows with invalid column counts must fail validation."""
    row = SAMPLE_ROW[:10]
    with pytest.raises(ValueError, match="Expected 14 columns, got 10"):
        _parse_row(row, "train")


def test_load_liar_end_to_end(tmp_path: Path):
    """End-to-end integration test of load_liar on raw LIAR data."""
    raw_dir = Path("data/raw/liar")
    if not (raw_dir / "train.tsv").is_file():
        pytest.skip("Raw LIAR dataset files not present in data/raw/liar")

    out_dir = tmp_path / "processed_liar"
    summary = load_liar(raw_dir=raw_dir, output_dir=out_dir, generate_report=True, strict=True)

    # Exact expected record counts per split matching William Yang Wang (ACL 2017)
    assert summary["train"] == 10269
    assert summary["validation"] == 1284
    assert summary["test"] == 1283
    assert sum(summary.values()) == 12836

    # Verify JSONL output files
    for split, count in summary.items():
        jsonl_file = out_dir / f"{split}.jsonl"
        assert jsonl_file.is_file()

        records = []
        with jsonl_file.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                validate_claim_record(rec)
                records.append(rec)

        assert len(records) == count

    # Verify report files
    assert (out_dir / "repair_report.json").is_file()
    assert (out_dir / "repair_report.md").is_file()

    with (out_dir / "repair_report.json").open("r", encoding="utf-8") as f:
        rep_data = json.load(f)
        assert rep_data["total_records"] == 12836
        assert rep_data["total_valid"] == 12836
        assert rep_data["total_unresolved"] == 0
        assert len(rep_data["cross_split_duplicates"]) == 0
