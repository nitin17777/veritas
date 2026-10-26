#!/usr/bin/env python
"""Threshold sweep and coverage-versus-accuracy evaluation for VERITAS P7 Abstention.

This script sweeps confidence and faithfulness thresholds over verification results,
evaluating the tradeoff between:
1. Coverage: The proportion of claims for which the system issues a definitive verdict (non-abstained).
2. Accuracy: Accuracy on the subset of retained/covered claims.

Usage:
    # Fast standalone smoke test on synthetic evaluation claims:
    python scripts/evaluate_abstention_thresholds.py --smoke-test

    # Full evaluation over saved batch verdicts:
    python scripts/evaluate_abstention_thresholds.py --input data/verdicts.jsonl --output-dir results
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from veritas.faithfulness.abstain import (
    AbstentionConfig,
    AbstentionGate,
)
from veritas.faithfulness.nli import (
    FaithfulnessEvaluator,
    MockNLIModel,
    NLIPrediction,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("veritas.scripts.evaluate_abstention")


# ---------------------------------------------------------------------------
# Synthetic Benchmark Dataset for Smoke Testing
# ---------------------------------------------------------------------------


def generate_synthetic_sweep_records() -> list[dict[str, Any]]:
    """Generate 20 representative claims with varied confidence and faithfulness."""
    records = []
    # High confidence, faithful, correct (True -> True)
    for i in range(1, 9):
        records.append({
            "claim_id": f"syn_correct_{i}",
            "claim": f"Verified factual claim {i}.",
            "gold_label": "True",
            "predicted_label": "True",
            "confidence": 0.85 + (i % 3) * 0.04,
            "explanation": f"Official verified report {i} clearly proves this statement.",
            "evidence": [{"passage_id": f"p_{i}", "text": f"Official verified report {i} clearly proves this statement."}],
            "entailment_prob": 0.95,
        })

    # High confidence, faithful, correct (False -> False)
    for i in range(1, 5):
        records.append({
            "claim_id": f"syn_refuted_{i}",
            "claim": f"Fabricated claim {i}.",
            "gold_label": "False",
            "predicted_label": "False",
            "confidence": 0.88 + (i % 2) * 0.05,
            "explanation": f"Credible health agency refuted assertion {i} directly.",
            "evidence": [{"passage_id": f"p_ref_{i}", "text": f"Credible health agency refuted assertion {i} directly."}],
            "entailment_prob": 0.92,
        })

    # Low confidence, uncertain (Incorrect guess)
    for i in range(1, 5):
        records.append({
            "claim_id": f"syn_uncertain_{i}",
            "claim": f"Unclear claim {i}.",
            "gold_label": "False",
            "predicted_label": "True",  # Mistake made under low confidence
            "confidence": 0.45 + (i % 3) * 0.05,
            "explanation": f"Ambiguous evidence snippet {i}.",
            "evidence": [{"passage_id": f"p_unc_{i}", "text": f"Ambiguous snippet {i}."}],
            "entailment_prob": 0.55,
        })

    # Hallucinated explanation (Unfaithful: premise does not entail explanation)
    for i in range(1, 5):
        records.append({
            "claim_id": f"syn_hallucinated_{i}",
            "claim": f"Misrepresented claim {i}.",
            "gold_label": "Misleading",
            "predicted_label": "True",  # Mistake made due to ungrounded reasoning
            "confidence": 0.82,
            "explanation": f"Completely ungrounded hallucinated fact {i} never mentioned in source.",
            "evidence": [{"passage_id": f"p_hal_{i}", "text": f"Actual source text discusses different topics entirely {i}."}],
            "entailment_prob": 0.12,  # Low entailment!
        })

    return records


# ---------------------------------------------------------------------------
# SVG Plot Generator (Zero-Dependency)
# ---------------------------------------------------------------------------


def generate_svg_plot(
    data_points: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate a clean, standalone SVG plot of Coverage vs. Accuracy."""
    width, height = 800, 500
    margin = 70

    plot_w = width - 2 * margin
    plot_h = height - 2 * margin

    # Filter points with valid coverage
    valid_points = [p for p in data_points if p["coverage"] > 0]
    if not valid_points:
        return

    # Sort points by coverage ascending
    sorted_points = sorted(valid_points, key=lambda p: (p["coverage"], p["accuracy"]))

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '  <style>',
        '    .title { font-family: -apple-system, sans-serif; font-size: 18px; font-weight: bold; fill: #1e293b; }',
        '    .subtitle { font-family: -apple-system, sans-serif; font-size: 12px; fill: #64748b; }',
        '    .axis { stroke: #94a3b8; stroke-width: 1.5; }',
        '    .grid { stroke: #e2e8f0; stroke-width: 1; stroke-dasharray: 4,4; }',
        '    .label { font-family: -apple-system, sans-serif; font-size: 12px; fill: #475569; }',
        '    .dot { fill: #2563eb; stroke: #1e40af; stroke-width: 1.5; }',
        '    .line { fill: none; stroke: #3b82f6; stroke-width: 2.5; }',
        '    .tooltip { font-family: -apple-system, sans-serif; font-size: 10px; fill: #1e293b; }',
        '  </style>',
        '  <rect width="100%" height="100%" fill="#ffffff" />',
        f'  <text x="{margin}" y="35" class="title">VERITAS P7: Coverage vs. Accuracy Tradeoff</text>',
        f'  <text x="{margin}" y="52" class="subtitle">Threshold Sweep on Verification Confidence &amp; Explanation Faithfulness</text>',
    ]

    # Grid & Axes
    # X axis: Coverage (0% to 100%)
    # Y axis: Accuracy (50% to 100%)
    y_min, y_max = 0.50, 1.00

    for pct in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        x = margin + pct * plot_w
        svg_lines.append(f'  <line x1="{x}" y1="{margin}" x2="{x}" y2="{margin + plot_h}" class="grid" />')
        svg_lines.append(f'  <text x="{x}" y="{margin + plot_h + 20}" class="label" text-anchor="middle">{int(pct * 100)}%</text>')

    for y_val in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        y = margin + plot_h - ((y_val - y_min) / (y_max - y_min)) * plot_h
        svg_lines.append(f'  <line x1="{margin}" y1="{y}" x2="{margin + plot_w}" y2="{y}" class="grid" />')
        svg_lines.append(f'  <text x="{margin - 10}" y="{y + 4}" class="label" text-anchor="end">{int(y_val * 100)}%</text>')

    # Axis lines
    svg_lines.append(f'  <line x1="{margin}" y1="{margin + plot_h}" x2="{margin + plot_w}" y2="{margin + plot_h}" class="axis" />')
    svg_lines.append(f'  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{margin + plot_h}" class="axis" />')

    # Axis titles
    svg_lines.append(f'  <text x="{margin + plot_w / 2}" y="{height - 15}" class="label" text-anchor="middle" font-weight="bold">Coverage (% of Claims Not Abstained)</text>')
    svg_lines.append(f'  <text x="20" y="{margin + plot_h / 2}" class="label" text-anchor="middle" transform="rotate(-90 20 {margin + plot_h / 2})" font-weight="bold">Accuracy on Covered Claims</text>')

    # Plot line & points
    point_coords = []
    for pt in sorted_points:
        cov = pt["coverage"]
        acc = max(min(pt["accuracy"], y_max), y_min)
        cx = margin + cov * plot_w
        cy = margin + plot_h - ((acc - y_min) / (y_max - y_min)) * plot_h
        point_coords.append((cx, cy, pt))

    if len(point_coords) > 1:
        pts_str = " ".join([f"{cx:.1f},{cy:.1f}" for cx, cy, _ in point_coords])
        svg_lines.append(f'  <polyline points="{pts_str}" class="line" />')

    for cx, cy, pt in point_coords:
        svg_lines.append(f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" class="dot">')
        svg_lines.append(f'    <title>Conf>={pt["min_conf"]:.2f}, Faith>={pt["min_faith"]:.2f} -> Cov={int(pt["coverage"]*100)}%, Acc={int(pt["accuracy"]*100)}%</title>')
        svg_lines.append('  </circle>')

    svg_lines.append('</svg>')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(svg_lines))


# ---------------------------------------------------------------------------
# Sweep Engine
# ---------------------------------------------------------------------------


def run_threshold_sweep(
    records: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Execute grid sweep across confidence and faithfulness thresholds."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Grid of candidate thresholds
    conf_thresholds = [0.40, 0.50, 0.60, 0.70, 0.80]
    faith_thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]

    sweep_results: list[dict[str, Any]] = []

    print("\n" + "=" * 80)
    print("VERITAS P7: Threshold Sweep (Confidence & Faithfulness)")
    print("=" * 80)
    print(f"{'Min Conf':<10} | {'Min Faith':<10} | {'Covered':<10} | {'Coverage %':<12} | {'Accuracy %':<12} | {'Abstained %':<12}")
    print("-" * 80)

    for min_c in conf_thresholds:
        for min_f in faith_thresholds:
            covered_count = 0
            correct_count = 0
            abstained_count = 0

            for rec in records:
                conf = float(rec.get("confidence", 0.0))
                entail_prob = float(rec.get("entailment_prob", 0.50))
                gold_label = rec.get("gold_label")
                pred_label = rec.get("predicted_label")

                # Mock NLI predictor for the record's entailment probability
                nli_model = MockNLIModel(
                    rule_fn=lambda p, h, ep=entail_prob: NLIPrediction(
                        entailment=ep,
                        neutral=round(max(0.0, 1.0 - ep - 0.05), 4),
                        contradiction=0.05,
                        predicted_label="entailment" if ep >= 0.5 else "neutral",
                    )
                )

                gate = AbstentionGate(
                    config=AbstentionConfig(min_confidence=min_c, min_faithfulness=min_f),
                    evaluator=FaithfulnessEvaluator(nli_model=nli_model, entailment_threshold=0.50),
                )

                final_label, _, abstained, _ = gate.evaluate(
                    claim_id=rec.get("claim_id", "c"),
                    claim=rec.get("claim", ""),
                    label=pred_label,
                    confidence=conf,
                    explanation=rec.get("explanation", ""),
                    evidence_passages=rec.get("evidence", []),
                )

                if abstained:
                    abstained_count += 1
                else:
                    covered_count += 1
                    if final_label == gold_label:
                        correct_count += 1

            total = len(records)
            cov_pct = round(covered_count / total, 4) if total > 0 else 0.0
            acc_pct = round(correct_count / covered_count, 4) if covered_count > 0 else 1.0
            abs_pct = round(abstained_count / total, 4) if total > 0 else 0.0

            row = {
                "min_conf": min_c,
                "min_faith": min_f,
                "total_claims": total,
                "covered_claims": covered_count,
                "abstained_claims": abstained_count,
                "coverage": cov_pct,
                "accuracy": acc_pct,
                "abstention_rate": abs_pct,
            }
            sweep_results.append(row)

            print(
                f"{min_c:<10.2f} | {min_f:<10.2f} | {covered_count:<10} | "
                f"{cov_pct * 100:<11.1f}% | {acc_pct * 100:<11.1f}% | {abs_pct * 100:<11.1f}%"
            )

    print("=" * 80)

    # 1. Save results to JSON
    json_path = output_dir / "abstention_sweep_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)
    logger.info("Saved JSON sweep results to %s", json_path)

    # 2. Save results to CSV
    csv_path = output_dir / "abstention_sweep_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_results[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_results)
    logger.info("Saved CSV sweep summary to %s", csv_path)

    # 3. Generate SVG plot
    svg_path = output_dir / "coverage_vs_accuracy.svg"
    generate_svg_plot(sweep_results, svg_path)
    logger.info("Generated standalone Coverage vs. Accuracy plot: %s", svg_path)

    # 4. Optional PNG plot via Matplotlib if installed
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
        plt.figure(figsize=(8, 5))
        valid_pts = [p for p in sweep_results if p["coverage"] > 0]
        xs = [p["coverage"] * 100 for p in valid_pts]
        ys = [p["accuracy"] * 100 for p in valid_pts]
        plt.scatter(xs, ys, color="blue", alpha=0.7)
        plt.title("VERITAS P7: Coverage vs. Accuracy Tradeoff")
        plt.xlabel("Coverage (% of Claims Not Abstained)")
        plt.ylabel("Accuracy on Covered Claims (%)")
        plt.grid(True, linestyle="--", alpha=0.6)
        png_path = output_dir / "coverage_vs_accuracy.png"
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Generated PNG plot: %s", png_path)
    except ImportError:
        logger.debug("matplotlib not installed; SVG generated as standalone vector plot.")

    return sweep_results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VERITAS P7 Abstention Thresholds")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast threshold sweep on synthetic claims")
    parser.add_argument("--input", type=str, default=None, help="Input verdicts JSONL file")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save sweep plots and data")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    if args.smoke_test or args.input is None:
        logger.info("Running threshold sweep in smoke-test mode...")
        records = generate_synthetic_sweep_records()
    else:
        in_path = Path(args.input)
        if not in_path.exists():
            logger.error("Input file '%s' does not exist.", in_path)
            sys.exit(1)
        records = []
        with in_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        logger.info("Loaded %d records from %s", len(records), in_path)

    run_threshold_sweep(records, out_dir)


if __name__ == "__main__":
    main()
