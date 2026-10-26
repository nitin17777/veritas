"""Cross-experiment analysis, summary table aggregation, and visualization.

Generates:
- Comprehensive comparative summary table in GitHub-style Markdown.
- Machine-readable comparative CSV across all executed research comparisons.
- Standalone vector graphics (SVG) comparing Accuracy, Macro-F1, Faithfulness, and Latency.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from veritas.config import ROOT_DIR

logger = logging.getLogger("veritas.experiments.analysis")


def aggregate_experiment_results(results_dir: str | Path = "results/experiments") -> list[dict[str, Any]]:
    """Scan experiment results directory and aggregate summary metrics across all runs."""
    base_dir = Path(results_dir)
    if not base_dir.is_absolute():
        base_dir = ROOT_DIR / base_dir

    if not base_dir.exists():
        return []

    records: list[dict[str, Any]] = []

    for exp_dir in sorted(base_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        meta_file = exp_dir / "experiment_metadata.json"
        report_file = exp_dir / "report.json"

        if not report_file.exists():
            continue

        try:
            report_data = json.loads(report_file.read_text(encoding="utf-8"))
            meta_data = {}
            if meta_file.exists():
                meta_data = json.loads(meta_file.read_text(encoding="utf-8"))

            clf = report_data.get("classification", {})
            ret = report_data.get("retrieval", {})
            sys_q = report_data.get("system_quality", {})
            lat = report_data.get("latency", {})
            run_meta = report_data.get("run_metadata", {})

            rec = {
                "name": meta_data.get("name", exp_dir.name),
                "rq": meta_data.get("research_question", run_meta.get("research_question", "RQ")),
                "factor": meta_data.get("comparison_factor", ""),
                "factor_value": str(meta_data.get("factor_value", "")),
                "accuracy": clf.get("accuracy"),
                "macro_f1": clf.get("macro_f1"),
                "f1_true": (clf.get("per_class") or {}).get("True", {}).get("f1"),
                "f1_false": (clf.get("per_class") or {}).get("False", {}).get("f1"),
                "recall_at_5": (ret.get("recall_at_k") or {}).get("5"),
                "ndcg_at_5": (ret.get("ndcg_at_k") or {}).get("5"),
                "faithfulness": sys_q.get("faithfulness_score"),
                "abstention_rate": sys_q.get("abstention_rate"),
                "parse_failure_rate": sys_q.get("parse_failure_rate"),
                "latency_avg_s": lat.get("mean_latency_s"),
                "latency_p95_s": (lat.get("percentiles_s") or {}).get("p95"),
                "total_claims": run_meta.get("total_evaluated_claims", 0),
            }
            records.append(rec)
        except Exception as exc:
            logger.warning("Could not read report in %s: %s", exp_dir, exc)

    return records


def generate_markdown_summary_table(records: list[dict[str, Any]]) -> str:
    """Format aggregated experiment metrics into a Markdown summary table."""
    if not records:
        return "_No experiment records found._\n"

    headers = [
        "Experiment",
        "RQ",
        "Factor Under Test",
        "Accuracy",
        "Macro-F1",
        "Faithfulness",
        "Abstain Rate",
        "Recall@5",
        "Latency p95 (s)",
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for r in records:
        acc_str = f"{r['accuracy']:.3f}" if r.get("accuracy") is not None else "-"
        f1_str = f"{r['macro_f1']:.3f}" if r.get("macro_f1") is not None else "-"
        faith_str = f"{r['faithfulness']:.3f}" if r.get("faithfulness") is not None else "-"
        abst_str = f"{r['abstention_rate']:.3f}" if r.get("abstention_rate") is not None else "-"
        rec5_str = f"{r['recall_at_5']:.3f}" if r.get("recall_at_5") is not None else "-"
        lat_str = f"{r['latency_p95_s']:.3f}" if r.get("latency_p95_s") is not None else "-"

        factor_str = f"{r.get('factor', '')}: {r.get('factor_value', '')}".strip(": ")
        if not factor_str:
            factor_str = "-"

        row = [
            f"`{r['name']}`",
            r.get("rq", "-"),
            factor_str,
            acc_str,
            f1_str,
            faith_str,
            abst_str,
            rec5_str,
            lat_str,
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def export_summary_table_csv(records: list[dict[str, Any]], output_path: str | Path) -> None:
    """Export summary metrics to CSV."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        return

    fieldnames = list(records[0].keys())
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def generate_comparative_svg_chart(
    records: list[dict[str, Any]],
    output_path: str | Path,
    title: str = "VERITAS Empirical Comparison Across Research Questions",
) -> None:
    """Generate an SVG bar chart comparing Accuracy and Macro-F1 across experiment runs."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        return

    # Canvas dimensions
    width = max(800, len(records) * 85 + 150)
    height = 420
    margin_left = 80
    margin_right = 40
    margin_top = 60
    margin_bottom = 100

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    n_bars = len(records)
    group_width = plot_width / max(1, n_bars)
    bar_width = min(28.0, group_width * 0.35)

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        f'<rect width="{width}" height="{height}" fill="#0d1117" rx="8" />',
        f'<text x="{width / 2}" y="32" font-family="Segoe UI, Helvetica, sans-serif" font-size="16" font-weight="600" fill="#f0f6fc" text-anchor="middle">{title}</text>',
    ]

    # Gridlines and Y-axis marks (0.0 to 1.0)
    for y_val in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y_pos = margin_top + plot_height - (y_val * plot_height)
        svg_lines.append(
            f'<line x1="{margin_left}" y1="{y_pos:.1f}" x2="{width - margin_right}" y2="{y_pos:.1f}" stroke="#30363d" stroke-dasharray="3,3" />'
        )
        svg_lines.append(
            f'<text x="{margin_left - 10}" y="{y_pos + 4:.1f}" font-family="sans-serif" font-size="11" fill="#8b949e" text-anchor="end">{y_val:.1f}</text>'
        )

    # Plot bars
    for idx, r in enumerate(records):
        center_x = margin_left + idx * group_width + (group_width / 2.0)
        acc = r.get("accuracy") or 0.0
        f1 = r.get("macro_f1") or 0.0

        h_acc = acc * plot_height
        y_acc = margin_top + plot_height - h_acc
        x_acc = center_x - bar_width - 2

        h_f1 = f1 * plot_height
        y_f1 = margin_top + plot_height - h_f1
        x_f1 = center_x + 2

        # Accuracy bar (Cyan/Teal)
        svg_lines.append(
            f'<rect x="{x_acc:.1f}" y="{y_acc:.1f}" width="{bar_width:.1f}" height="{h_acc:.1f}" fill="#388bfd" rx="2" />'
        )
        # Macro-F1 bar (Purple)
        svg_lines.append(
            f'<rect x="{x_f1:.1f}" y="{y_f1:.1f}" width="{bar_width:.1f}" height="{h_f1:.1f}" fill="#a371f7" rx="2" />'
        )

        # X-label (experiment name)
        name_label = r["name"]
        if len(name_label) > 16:
            name_label = name_label[:14] + ".."
        svg_lines.append(
            f'<text x="{center_x:.1f}" y="{margin_top + plot_height + 20}" font-family="sans-serif" font-size="10" fill="#c9d1d9" text-anchor="middle" transform="rotate(30, {center_x:.1f}, {margin_top + plot_height + 20})">{name_label}</text>'
        )

    # Legend
    legend_y = height - 20
    svg_lines.append(
        f'<rect x="{margin_left}" y="{legend_y - 10}" width="14" height="14" fill="#388bfd" rx="2" />'
    )
    svg_lines.append(
        f'<text x="{margin_left + 22}" y="{legend_y + 2}" font-family="sans-serif" font-size="12" fill="#c9d1d9">Accuracy</text>'
    )
    svg_lines.append(
        f'<rect x="{margin_left + 120}" y="{legend_y - 10}" width="14" height="14" fill="#a371f7" rx="2" />'
    )
    svg_lines.append(
        f'<text x="{margin_left + 142}" y="{legend_y + 2}" font-family="sans-serif" font-size="12" fill="#c9d1d9">Macro-F1</text>'
    )

    svg_lines.append("</svg>")
    p.write_text("\n".join(svg_lines), encoding="utf-8")
    logger.info("Saved comparative SVG plot to %s", p)
