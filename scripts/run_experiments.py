#!/usr/bin/env python3
"""VERITAS Experiment Suite Runner CLI.

Executes experiments for research questions RQ1 through RQ7 and overall comparisons:
  RQ1: LLM-only vs. RAG
  RQ2: BM25 vs. Dense vs. Hybrid retrieval
  RQ3: Credibility scoring vs. No credibility vs. Random credibility
  RQ4: Temporal scoring vs. No temporal scoring
  RQ5: Free prompt vs. Evidence-constrained prompt
  RQ6: Held-out topic or dataset generalization
  RQ7: Explanation faithfulness and abstention behavior
  Overall: Full system vs. implemented baselines

Usage examples:
  # Run a single experiment:
  python scripts/run_experiments.py --config configs/experiments/rq1_rag.yaml --smoke-test

  # Run all experiments for a specific research question:
  python scripts/run_experiments.py --rq rq1 --smoke-test

  # Run the full research suite across all RQs in smoke-test mode:
  python scripts/run_experiments.py --all --smoke-test

  # Regenerate summary table and comparative plots from existing results:
  python scripts/run_experiments.py --summary
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from veritas.experiments.analysis import (
    aggregate_experiment_results,
    export_summary_table_csv,
    generate_comparative_svg_chart,
    generate_markdown_summary_table,
)
from veritas.experiments.config import (
    ExperimentConfig,
    load_config,
    validate_factor_isolation,
)
from veritas.experiments.fixtures import build_fixture_retriever
from veritas.experiments.runner import ExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("veritas.run_experiments")

RQ_CONFIG_MAP: dict[str, list[str]] = {
    "rq1": ["configs/experiments/rq1_llm_only.yaml", "configs/experiments/rq1_rag.yaml"],
    "rq2": [
        "configs/experiments/rq2_bm25.yaml",
        "configs/experiments/rq2_dense.yaml",
        "configs/experiments/rq2_hybrid.yaml",
    ],
    "rq3": [
        "configs/experiments/rq3_credibility.yaml",
        "configs/experiments/rq3_no_credibility.yaml",
        "configs/experiments/rq3_random_credibility.yaml",
    ],
    "rq4": [
        "configs/experiments/rq4_temporal.yaml",
        "configs/experiments/rq4_no_temporal.yaml",
    ],
    "rq5": [
        "configs/experiments/rq5_evidence_constrained.yaml",
        "configs/experiments/rq5_free_prompt.yaml",
    ],
    "rq6": [
        "configs/experiments/rq6_in_domain.yaml",
        "configs/experiments/rq6_held_out.yaml",
    ],
    "rq7": [
        "configs/experiments/rq7_abstention.yaml",
        "configs/experiments/rq7_no_abstention.yaml",
    ],
    "overall": [
        "configs/experiments/overall_baseline_llm_only.yaml",
        "configs/experiments/overall_baseline_bm25_only.yaml",
        "configs/experiments/overall_full_system.yaml",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VERITAS experiments and generate comparative reports."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--config",
        type=str,
        help="Path to a single experiment YAML configuration file.",
    )
    group.add_argument(
        "--rq",
        type=str,
        choices=list(RQ_CONFIG_MAP.keys()),
        help="Research Question to evaluate (rq1, rq2, ..., rq7, overall).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all defined research experiments across all RQs.",
    )
    group.add_argument(
        "--summary",
        action="store_true",
        help="Skip running experiments; regenerate summary table and plots from existing results.",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Enable smoke-test mode with synthetic fixture data and mock retriever.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Overwrite previous runs instead of resuming.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/experiments",
        help="Root directory for saving experiment results and summary artifacts.",
    )
    return parser.parse_args()


def run_single_experiment(
    config_path: str | Path,
    smoke_test: bool = False,
    resume: bool = True,
    shared_retriever: Any = None,
) -> None:
    """Load configuration, validate factor isolation, and execute the experiment."""
    cfg_file = Path(config_path)
    if not cfg_file.is_absolute():
        cfg_file = ROOT / cfg_file

    logger.info("=" * 70)
    logger.info("Executing Experiment from: %s", cfg_file)
    exp_cfg = ExperimentConfig.from_yaml(cfg_file)

    if smoke_test:
        exp_cfg.smoke_test = True

    # Run experiment
    runner = ExperimentRunner(
        config=exp_cfg,
        retriever=shared_retriever,
    )
    report = runner.run(resume=resume)

    acc = report.classification.get("accuracy")
    f1 = report.classification.get("macro_f1")
    logger.info(
        "Finished '%s' | Accuracy: %s | Macro-F1: %s",
        exp_cfg.name,
        f"{acc:.3f}" if acc is not None else "N/A",
        f"{f1:.3f}" if f1 is not None else "N/A",
    )


def update_summary_artifacts(results_dir: str | Path) -> None:
    """Aggregate results and write Markdown table, CSV, and comparative SVG plot."""
    base_dir = Path(results_dir)
    if not base_dir.is_absolute():
        base_dir = ROOT / base_dir

    records = aggregate_experiment_results(base_dir)
    if not records:
        logger.warning("No experiment records found in %s to summarize.", base_dir)
        return

    md_table = generate_markdown_summary_table(records)
    summary_md_path = base_dir / "summary_table.md"
    summary_md_path.write_text(md_table, encoding="utf-8")

    csv_path = base_dir / "summary_table.csv"
    export_summary_table_csv(records, csv_path)

    svg_path = base_dir / "comparison_plot.svg"
    generate_comparative_svg_chart(records, svg_path)

    print("\n" + "=" * 80)
    print("VERITAS EXPERIMENT SUITE SUMMARY TABLE")
    print("=" * 80)
    print(md_table)
    print(f"Summary table saved to: {summary_md_path}")
    print(f"Summary CSV saved to:   {csv_path}")
    print(f"Comparison plot saved:  {svg_path}\n")


def main() -> None:
    args = parse_args()

    if args.summary:
        update_summary_artifacts(args.output_dir)
        return

    # Determine which configs to run
    configs_to_run: list[str] = []
    if args.config:
        configs_to_run.append(args.config)
    elif args.rq:
        configs_to_run.extend(RQ_CONFIG_MAP[args.rq.lower()])
    elif args.all:
        for rq_key in sorted(RQ_CONFIG_MAP.keys()):
            configs_to_run.extend(RQ_CONFIG_MAP[rq_key])
    else:
        # Default behavior: run smoke-test on RQ1
        logger.info("No experiment specified. Defaulting to --rq rq1 with --smoke-test.")
        args.smoke_test = True
        configs_to_run.extend(RQ_CONFIG_MAP["rq1"])

    # Build shared fixture retriever if in smoke-test mode to reuse across runs
    shared_retriever = build_fixture_retriever() if args.smoke_test else None

    # Execute all scheduled experiment configs
    for cfg_path in configs_to_run:
        run_single_experiment(
            config_path=cfg_path,
            smoke_test=args.smoke_test,
            resume=not args.no_resume,
            shared_retriever=shared_retriever,
        )

    # Generate final cross-experiment summary table and plots
    update_summary_artifacts(args.output_dir)


if __name__ == "__main__":
    main()
