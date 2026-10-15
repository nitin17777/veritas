
import csv
import json
import logging
from pathlib import Path
from typing import Any

from veritas.data_prep.label_map import map_label
from veritas.data_prep.text_cleaning import clean_text

logger = logging.getLogger(__name__)

SOURCE_COLUMNS = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "speaker_job",
    "state_info",
    "party_affiliation",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_fire_counts",
    "context",
]

SPLIT_FILES = {
    "train": "train.tsv",
    "validation": "valid.tsv",
    "test": "test.tsv",
}


def _optional_text(value: str | None) -> str | None:
    """Convert blank or missing text into None."""
    if value is None:
        return None

    cleaned = clean_text(value)
    return cleaned if cleaned else None


def _parse_row(row: list[str], split: str) -> dict[str, Any]:
    if len(row) != len(SOURCE_COLUMNS):
        raise ValueError(
            f"Expected {len(SOURCE_COLUMNS)} columns, got {len(row)}"
        )

    source = dict(zip(SOURCE_COLUMNS, row))

    claim_id = clean_text(source["id"])
    claim = clean_text(source["statement"])

    if not claim_id:
        raise ValueError("Claim ID is empty.")
    if not claim:
        raise ValueError("Statement is empty.")

    return {
        "id": f"liar:{claim_id}",
        "claim": claim,
        "label": map_label(source["label"], "liar"),
        "claim_date": None,
        "topic": _optional_text(source["subject"]),
        "split": split,
        "evidence": [],
        "metadata": {
            "dataset": "liar",
            "source_id": claim_id,
            "speaker": _optional_text(source["speaker"]),
            "speaker_job": _optional_text(source["speaker_job"]),
            "state_info": _optional_text(source["state_info"]),
            "party_affiliation": _optional_text(
                source["party_affiliation"]
            ),
            "speaker_history": {
                "barely_true": int(source["barely_true_counts"]),
                "false": int(source["false_counts"]),
                "half_true": int(source["half_true_counts"]),
                "mostly_true": int(source["mostly_true_counts"]),
                "pants_fire": int(source["pants_fire_counts"]),
            },
            "context": _optional_text(source["context"]),
        },
    }


def load_liar(
    raw_dir: str | Path = "data/raw/liar",
    output_dir: str | Path = "data/processed/liar",
) -> dict[str, int]:
    """Normalize LIAR TSV files into one JSONL file per split."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}

    for split, filename in SPLIT_FILES.items():
        input_path = raw_dir / filename
        output_path = output_dir / f"{split}.jsonl"
        count = 0

        if not input_path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {input_path}")

        with input_path.open(
            "r", encoding="utf-8", newline=""
        ) as source_file, output_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as output_file:
            reader = csv.reader(source_file, delimiter="\t",quoting=csv.QUOTE_NONE,)

            for line_number, row in enumerate(reader, start=1):
                try:
                    record = _parse_row(row, split)
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"{input_path}:{line_number}: {exc}"
                    ) from exc

                output_file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                count += 1

        counts[split] = count
        logger.info("Processed %s: %d records", split, count)

    logger.info("Total normalized LIAR records: %d", sum(counts.values()))
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    summary = load_liar()
    print("LIAR processing complete:", summary)
