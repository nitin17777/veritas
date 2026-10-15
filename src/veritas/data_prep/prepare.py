"""Unified Data Preparation Pipeline Command for VERITAS.

Executes dataset ingestion across one or several configured datasets,
verifying presence, tracking label distributions, and producing normalized
VERITAS ClaimRecord JSONL datasets.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from veritas.config import ROOT_DIR, load_config
from veritas.data_prep.averitec_loader import load_averitec
from veritas.data_prep.custom_loader import load_custom_recent
from veritas.data_prep.fever_loader import load_fever
from veritas.data_prep.label_map import (
    PROPOSED_AVERITEC_LABEL_MAP,
    PROPOSED_RAWFC_LABEL_MAP,
)
from veritas.data_prep.liar_loader import load_liar
from veritas.data_prep.manifest import DATASET_MANIFESTS, get_dataset_manifest
from veritas.data_prep.rawfc_loader import load_rawfc

logger = logging.getLogger("veritas.data_prep")


def run_data_prep(
    datasets: list[str] | None = None,
    raw_root: str | Path = "data/raw",
    processed_root: str | Path = "data/processed",
    config_path: str | Path | None = None,
    strict: bool = False,
    approve_proposed_mappings: bool = False,
) -> dict[str, Any]:
    """Run data ingestion across selected or configured datasets.

    Parameters:
        datasets: List of dataset names to process ('liar', 'fever', 'rawfc', 'averitec', 'custom_recent').
                  If None, reads from config_path or defaults to ['liar'].
        raw_root: Root directory where raw dataset subfolders reside.
        processed_root: Root directory where normalized jsonl folders will be saved.
        config_path: Optional YAML config to load settings from.
        strict: If True, raises exceptions on malformed rows or missing files.
        approve_proposed_mappings: If True, activates proposed label mappings for RAWFC/AVeriTeC.
    """
    raw_root = Path(raw_root)
    processed_root = Path(processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)

    # Resolve dataset list
    if datasets is None:
        if config_path and Path(config_path).is_file():
            cfg = load_config(config_path)
            datasets = cfg.get("data_prep", {}).get("datasets", ["liar"])
        else:
            datasets = ["liar"]

    overall_summary: dict[str, Any] = {
        "datasets_requested": datasets,
        "results": {},
        "skipped": [],
    }

    for ds_name in datasets:
        ds_clean = ds_name.lower().replace("-", "_")
        ds_raw_dir = raw_root / ds_clean
        ds_out_dir = processed_root / ds_clean

        logger.info("=" * 60)
        logger.info("Processing dataset: %s", ds_clean)
        logger.info("=" * 60)

        # Check if raw dataset exists before attempting to load
        if not ds_raw_dir.is_dir():
            msg = (
                f"Raw data directory for '{ds_clean}' not found at '{ds_raw_dir}'. "
                f"Skipping ingestion. To download, consult the data source manifest."
            )
            logger.info(msg)
            overall_summary["skipped"].append({"dataset": ds_clean, "reason": msg})
            continue

        try:
            if ds_clean == "liar":
                counts = load_liar(raw_dir=ds_raw_dir, output_dir=ds_out_dir, strict=strict)
            elif ds_clean == "fever":
                counts = load_fever(raw_dir=ds_raw_dir, output_dir=ds_out_dir, strict=strict)
            elif ds_clean == "rawfc":
                mapping = PROPOSED_RAWFC_LABEL_MAP if approve_proposed_mappings else None
                counts = load_rawfc(
                    raw_dir=ds_raw_dir,
                    output_dir=ds_out_dir,
                    label_map=mapping,
                    strict=strict,
                )
            elif ds_clean == "averitec":
                mapping = PROPOSED_AVERITEC_LABEL_MAP if approve_proposed_mappings else None
                counts = load_averitec(
                    raw_dir=ds_raw_dir,
                    output_dir=ds_out_dir,
                    label_map=mapping,
                    strict=strict,
                )
            elif ds_clean == "custom_recent":
                counts = load_custom_recent(
                    raw_dir=ds_raw_dir,
                    output_dir=ds_out_dir,
                    strict=strict,
                )
            else:
                raise ValueError(f"Unknown dataset name: {ds_clean}")

            overall_summary["results"][ds_clean] = {
                "status": "success",
                "counts": counts,
                "total_records": sum(counts.values()),
            }
            logger.info("Completed %s: %s records", ds_clean, counts)

        except Exception as exc:
            logger.error("Error ingesting dataset %s: %s", ds_clean, exc)
            overall_summary["results"][ds_clean] = {
                "status": "error",
                "error": str(exc),
            }
            if strict:
                raise

    summary_file = processed_root / "ingestion_summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2, ensure_ascii=False)

    logger.info("Saved ingestion summary to %s", summary_file)
    return overall_summary


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="VERITAS Data Preparation Pipeline")
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Datasets to process (liar, fever, rawfc, averitec, custom_recent)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Attempt all supported datasets",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Path to raw datasets root directory",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Path to processed datasets root directory",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast on any malformed record",
    )
    parser.add_argument(
        "--approve-proposed-mappings",
        action="store_true",
        help="Activate proposed label mappings for RAWFC and AVeriTeC for evaluation",
    )
    parser.add_argument(
        "--manifest",
        action="store_true",
        help="Print data source manifest and exit",
    )

    args = parser.parse_args()

    if args.manifest:
        print("=" * 80)
        print("VERITAS DATA SOURCE MANIFEST")
        print("=" * 80)
        for name, meta in DATASET_MANIFESTS.items():
            print(f"\n[{name.upper()}] - {meta.full_name}")
            print(f"  Citation: {meta.paper_citation}")
            print(f"  URL: {meta.source_url}")
            print(f"  License: {meta.license}")
            print(f"  Format: {meta.file_format}")
            print(f"  Row Counts: {meta.row_counts}")
            print(f"  Mapping Status: {meta.mapping_status.upper()}")
            print(f"  Proposed Mapping: {meta.proposed_veritas_mapping}")
        return 0

    target_datasets = list(DATASET_MANIFESTS.keys()) if args.all else args.datasets

    summary = run_data_prep(
        datasets=target_datasets,
        raw_root=args.raw_dir,
        processed_root=args.output_dir,
        config_path=args.config,
        strict=args.strict,
        approve_proposed_mappings=args.approve_proposed_mappings,
    )

    print("\nIngestion Execution Summary:")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
