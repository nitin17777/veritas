"""Evaluation Report Generator for VERITAS P9.

Compiles classification, retrieval, faithfulness, latency, and abstention metrics
along with complete run metadata (model version, config, seed, dataset version)
into machine-readable JSON/CSV files and standalone SVG visual plots.
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from veritas.evaluation.metrics import (
    STANDARD_LABEL_ORDER,
    aggregate_multi_run_metrics,
    compute_classification_metrics,
    compute_coverage_accuracy_curve,
    compute_latency_metrics,
    compute_system_quality_metrics,
    match_gold_and_predictions,
)
from veritas.evaluation.retrieval_metrics import evaluate_retrieval_batch

logger = logging.getLogger("veritas.evaluation.report")


@dataclass
class EvaluationReport:
    """Complete evaluation report for a VERITAS verification experiment.

    Attributes:
        run_metadata: Model name, version, seed, dataset version, config.
        classification: Accuracy, Macro-F1, per-class metrics, confusion matrix.
        retrieval: Recall@k, nDCG@k, MRR@k (when gold evidence exists).
        system_quality: Parse failure rate, abstention rate, faithfulness, hallucination proxy.
        latency: Average, p50, p90, p95, p99 stage timings.
        coverage_accuracy_curve: Data points along confidence thresholds.
        multi_run_aggregate: Mean and spread across repeated runs if applicable.
    """

    run_metadata: dict[str, Any]
    classification: dict[str, Any]
    retrieval: dict[str, Any]
    system_quality: dict[str, Any]
    latency: dict[str, Any]
    coverage_accuracy_curve: list[dict[str, Any]]
    multi_run_aggregate: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary representation."""
        return asdict(self)


def build_evaluation_report(
    gold_records: list[dict[str, Any]],
    prediction_records: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
    model_version: str = "default",
    dataset_name: str = "dataset",
    dataset_version: str = "v1.0",
    seed: int | None = 42,
    run_id: str | None = None,
    k_values: list[int] | None = None,
) -> EvaluationReport:
    """Build a comprehensive EvaluationReport from matched gold and prediction records."""
    matched = match_gold_and_predictions(gold_records, prediction_records, strict_match=True)

    y_true = [str(g.get("label") or "") for g, _ in matched]
    y_pred = [str(p.get("label") or "") for _, p in matched]

    # 1. Classification metrics
    clf_metrics = compute_classification_metrics(y_true, y_pred, labels=STANDARD_LABEL_ORDER)

    # 2. Retrieval metrics (strictly over claims with valid gold evidence)
    retrieval_items: list[dict[str, Any]] = []
    for g, p in matched:
        gold_ev = g.get("evidence") or g.get("evidence_ids") or []
        gold_ids = []
        if isinstance(gold_ev, list):
            for item in gold_ev:
                if isinstance(item, dict):
                    gid = item.get("passage_id") or item.get("id")
                    if gid:
                        gold_ids.append(str(gid))
                elif isinstance(item, str):
                    gold_ids.append(item)

        retrieved_ev = p.get("retrieved_passages") or p.get("evidence_ids") or []
        retrieved_ids = []
        if isinstance(retrieved_ev, list):
            for item in retrieved_ev:
                if isinstance(item, dict):
                    rid = item.get("passage_id") or item.get("id")
                    if rid:
                        retrieved_ids.append(str(rid))
                elif isinstance(item, str):
                    retrieved_ids.append(item)

        retrieval_items.append({"gold_ids": gold_ids, "retrieved_ids": retrieved_ids})

    ret_metrics = evaluate_retrieval_batch(retrieval_items, k_values=k_values)

    # 3. System and quality metrics
    sys_metrics = compute_system_quality_metrics(matched)

    # 4. Latency metrics
    latencies: list[float] = []
    for _, p in matched:
        # Check timings dict or direct latency field
        timings = p.get("stage_timings_s")
        if isinstance(timings, dict) and timings:
            latencies.append(sum(float(v) for v in timings.values() if isinstance(v, (int, float))))
        elif p.get("latency_seconds") is not None:
            latencies.append(float(p["latency_seconds"]))

    lat_metrics = compute_latency_metrics(latencies)

    # 5. Coverage vs. accuracy curve
    cov_curve = compute_coverage_accuracy_curve(matched)

    # 6. Run metadata
    run_meta = {
        "run_id": run_id or f"eval_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_version": model_version,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "seed": seed,
        "total_evaluated_claims": len(matched),
        "config": config or {},
    }

    return EvaluationReport(
        run_metadata=run_meta,
        classification=clf_metrics,
        retrieval=ret_metrics,
        system_quality=sys_metrics,
        latency=lat_metrics,
        coverage_accuracy_curve=cov_curve,
    )


# ---------------------------------------------------------------------------
# Formatting and Exporting
# ---------------------------------------------------------------------------


def format_console_report(report: EvaluationReport) -> str:
    """Format report into human-readable console string."""
    meta = report.run_metadata
    clf = report.classification
    sys_q = report.system_quality
    lat = report.latency
    ret = report.retrieval

    lines = [
        "=" * 78,
        f"VERITAS P9 EVALUATION REPORT: {meta.get('dataset_name')} ({meta.get('model_version')})",
        "=" * 78,
        f"Timestamp: {meta.get('timestamp')} | Total Claims: {meta.get('total_evaluated_claims')}",
        "-" * 78,
        "1. CLASSIFICATION PERFORMANCE:",
        f"   Accuracy        : {clf.get('accuracy')}",
        f"   Macro-F1        : {clf.get('macro_f1')}",
        f"   Macro-Precision : {clf.get('macro_precision')}",
        f"   Macro-Recall    : {clf.get('macro_recall')}",
        "",
        "   Per-Class Breakdown:",
        f"   {'Label':<18} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'Support':<8}",
        "   " + "-" * 62,
    ]

    for lbl, stats in clf.get("per_class", {}).items():
        lines.append(
            f"   {lbl:<18} | {stats.get('precision', 0.0):<10.4f} | "
            f"{stats.get('recall', 0.0):<10.4f} | {stats.get('f1', 0.0):<10.4f} | "
            f"{stats.get('support', 0):<8}"
        )

    lines.extend([
        "",
        "2. RETRIEVAL PERFORMANCE (Claims with gold evidence):",
        f"   Status          : {ret.get('status')} (Evaluated on {ret.get('eligible_claims')} claims)",
    ])
    if ret.get("status") == "available":
        for m_name, val in ret.get("metrics", {}).items():
            lines.append(f"   {m_name:<16}: {val}")
    else:
        lines.append("   (Retrieval metrics omitted: dataset does not annotate gold evidence)")

    lines.extend([
        "",
        "3. SYSTEM QUALITY & FAITHFULNESS:",
        f"   JSON Parse Failure Rate : {sys_q.get('json_parse_failure_rate')}",
        f"   Abstention Rate         : {sys_q.get('abstention_rate')}",
        f"   Mean Faithfulness       : {sys_q.get('mean_faithfulness')}",
        f"   Hallucination Proxy     : {sys_q.get('hallucination_rate_proxy')}",
        "",
        "4. LATENCY PERCENTILES (Seconds):",
        f"   Average : {lat.get('avg_latency_s')}",
        f"   P50     : {lat.get('p50_latency_s')}",
        f"   P90     : {lat.get('p90_latency_s')}",
        f"   P95     : {lat.get('p95_latency_s')}",
        f"   P99     : {lat.get('p99_latency_s')}",
        "=" * 78,
    ])

    return "\n".join(lines)


def save_report_artifacts(report: EvaluationReport, output_dir: Path) -> dict[str, Path]:
    """Save machine-readable JSON, CSV summaries, and standalone SVG plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, Path] = {}

    # 1. Full JSON report
    json_path = output_dir / "evaluation_report.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    saved_paths["json"] = json_path

    # 2. Per-class CSV summary
    per_class_csv = output_dir / "per_class_metrics.csv"
    with per_class_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "precision", "recall", "f1", "support"])
        for lbl, stats in report.classification.get("per_class", {}).items():
            writer.writerow([
                lbl,
                stats.get("precision"),
                stats.get("recall"),
                stats.get("f1"),
                stats.get("support"),
            ])
    saved_paths["per_class_csv"] = per_class_csv

    # 3. Summary scalars CSV
    summary_csv = output_dir / "summary_metrics.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["accuracy", report.classification.get("accuracy")])
        writer.writerow(["macro_f1", report.classification.get("macro_f1")])
        writer.writerow(["json_parse_failure_rate", report.system_quality.get("json_parse_failure_rate")])
        writer.writerow(["abstention_rate", report.system_quality.get("abstention_rate")])
        writer.writerow(["mean_faithfulness", report.system_quality.get("mean_faithfulness")])
        writer.writerow(["hallucination_rate_proxy", report.system_quality.get("hallucination_rate_proxy")])
        writer.writerow(["avg_latency_s", report.latency.get("avg_latency_s")])
        writer.writerow(["p95_latency_s", report.latency.get("p95_latency_s")])
    saved_paths["summary_csv"] = summary_csv

    # 4. Standalone SVG Coverage vs. Accuracy plot
    svg_path = output_dir / "coverage_vs_accuracy.svg"
    _generate_svg_curve(report.coverage_accuracy_curve, svg_path)
    saved_paths["svg_curve"] = svg_path

    return saved_paths


def _generate_svg_curve(points: list[dict[str, Any]], out_path: Path) -> None:
    """Generate standalone SVG plot for coverage vs accuracy curve."""
    width, height = 750, 450
    margin = 65
    plot_w, plot_h = width - 2 * margin, height - 2 * margin

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '  <style>',
        '    .title { font-family: sans-serif; font-size: 16px; font-weight: bold; fill: #0f172a; }',
        '    .label { font-family: sans-serif; font-size: 11px; fill: #475569; }',
        '    .grid { stroke: #e2e8f0; stroke-width: 1; stroke-dasharray: 4,4; }',
        '    .axis { stroke: #64748b; stroke-width: 1.5; }',
        '    .line { fill: none; stroke: #2563eb; stroke-width: 2.5; }',
        '    .dot { fill: #1d4ed8; stroke: #ffffff; stroke-width: 1.5; }',
        '  </style>',
        '  <rect width="100%" height="100%" fill="#ffffff" />',
        f'  <text x="{margin}" y="35" class="title">VERITAS P9: Coverage vs. Accuracy Tradeoff</text>',
    ]

    # Grid
    for pct in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        x = margin + pct * plot_w
        svg.append(f'  <line x1="{x}" y1="{margin}" x2="{x}" y2="{margin + plot_h}" class="grid" />')
        svg.append(f'  <text x="{x}" y="{margin + plot_h + 18}" class="label" text-anchor="middle">{int(pct*100)}%</text>')

    for y_pct in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        y = margin + plot_h - ((y_pct - 0.5) / 0.5) * plot_h
        svg.append(f'  <line x1="{margin}" y1="{y}" x2="{margin + plot_w}" y2="{y}" class="grid" />')
        svg.append(f'  <text x="{margin - 10}" y="{y + 4}" class="label" text-anchor="end">{int(y_pct*100)}%</text>')

    svg.append(f'  <line x1="{margin}" y1="{margin + plot_h}" x2="{margin + plot_w}" y2="{margin + plot_h}" class="axis" />')
    svg.append(f'  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{margin + plot_h}" class="axis" />')

    svg.append(f'  <text x="{margin + plot_w/2}" y="{height - 12}" class="label" text-anchor="middle" font-weight="bold">Coverage (%)</text>')
    svg.append(f'  <text x="18" y="{margin + plot_h/2}" class="label" text-anchor="middle" transform="rotate(-90 18 {margin + plot_h/2})" font-weight="bold">Accuracy (%)</text>')

    coords = []
    for pt in points:
        cov = pt.get("coverage", 0.0)
        acc = max(0.5, min(1.0, pt.get("accuracy", 1.0)))
        cx = margin + cov * plot_w
        cy = margin + plot_h - ((acc - 0.5) / 0.5) * plot_h
        coords.append((cx, cy))

    if len(coords) > 1:
        pts_str = " ".join([f"{cx:.1f},{cy:.1f}" for cx, cy in coords])
        svg.append(f'  <polyline points="{pts_str}" class="line" />')

    for cx, cy in coords:
        svg.append(f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" class="dot" />')

    svg.append('</svg>')

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(svg))
