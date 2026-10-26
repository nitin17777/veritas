"""VERITAS Error Analysis Workflow.

Exports, categorizes, tracks, and analyzes prediction errors for qualitative review.

Error Categories:
- Retrieval miss
- Incorrect credibility ranking
- Stale or temporally inappropriate evidence
- LLM misinterpretation
- Label noise or ambiguous annotation
- Parsing failure
- Other / unknown

Guarantees:
- Never modifies raw predictions.
- Automated suggestions are strictly preliminary heuristics; never treated as ground truth.
- Human review annotations are retained and merged across repeated export runs.
- Produces spreadsheet-ready CSVs and Markdown reports with representative failure patterns.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import datetime
import json
import logging
from pathlib import Path
from typing import Any

from veritas.evaluation.metrics import match_gold_and_predictions

logger = logging.getLogger("veritas.evaluation.error_analysis")

# Canonical error taxonomy
ERROR_CATEGORIES: list[str] = [
    "Retrieval miss",
    "Incorrect credibility ranking",
    "Stale or temporally inappropriate evidence",
    "LLM misinterpretation",
    "Label noise or ambiguous annotation",
    "Parsing failure",
    "Other / unknown",
]


@dataclass
class ErrorRecord:
    """Detailed record of an incorrect fact verification prediction.

    Captures claim details, retrieved evidence, source metadata, model explanations,
    abstention status, parsing status, diagnostic suggestions, and human review annotations.
    """

    claim_id: str
    claim: str
    gold_label: str
    predicted_label: str
    confidence: float
    explanation: str
    retrieved_evidence_ids: list[str] = field(default_factory=list)
    retrieved_sources_summary: str = ""
    cited_evidence_ids: list[str] = field(default_factory=list)
    gold_evidence_ids: list[str] = field(default_factory=list)
    claim_date: str | None = None
    faithfulness_score: float | None = None
    abstained: bool = False
    original_label: str | None = None
    abstention_reasons: list[str] = field(default_factory=list)
    parse_failed: bool = False
    attempts: int = 1
    latency_seconds: float | None = None
    stage_timings: dict[str, float] = field(default_factory=dict)
    raw_llm_output: str | None = None

    # Diagnostic Heuristic (provisional suggestion ONLY; never treated as ground truth)
    suggested_category: str = "Other / unknown"
    suggestion_rationale: str = ""

    # Human Review Fields (retained across exports)
    annotated_category: str = ""
    annotator_id: str = ""
    annotator_notes: str = ""
    verification_status: str = "unreviewed"  # unreviewed | in_review | verified | disputed

    def to_dict(self) -> dict[str, Any]:
        """Convert record to complete dictionary."""
        return asdict(self)

    def to_csv_row(self) -> dict[str, Any]:
        """Format record into a clean, spreadsheet-friendly dictionary."""
        return {
            "claim_id": self.claim_id,
            "claim": self.claim,
            "gold_label": self.gold_label,
            "predicted_label": self.predicted_label,
            "confidence": f"{self.confidence:.3f}",
            "explanation": self.explanation.replace("\n", " ").strip(),
            "annotated_category": self.annotated_category,
            "annotator_id": self.annotator_id,
            "annotator_notes": self.annotator_notes,
            "verification_status": self.verification_status,
            "suggested_category": self.suggested_category,
            "suggestion_rationale": self.suggestion_rationale,
            "retrieved_evidence_ids": "; ".join(self.retrieved_evidence_ids),
            "cited_evidence_ids": "; ".join(self.cited_evidence_ids),
            "gold_evidence_ids": "; ".join(self.gold_evidence_ids),
            "retrieved_sources": self.retrieved_sources_summary,
            "claim_date": self.claim_date or "",
            "faithfulness_score": f"{self.faithfulness_score:.3f}" if self.faithfulness_score is not None else "",
            "abstained": str(self.abstained),
            "original_label": self.original_label or "",
            "parse_failed": str(self.parse_failed),
            "attempts": self.attempts,
            "latency_seconds": f"{self.latency_seconds:.3f}" if self.latency_seconds is not None else "",
        }


# ---------------------------------------------------------------------------
# Diagnostic Heuristic Suggester
# ---------------------------------------------------------------------------


def infer_suggested_error_category(
    gold_label: str,
    pred_label: str,
    parse_failed: bool,
    retrieved_passages: list[dict[str, Any]],
    gold_ids: list[str],
    cited_ids: list[str],
    claim_date: str | None,
    faithfulness_score: float | None,
    abstained: bool,
) -> tuple[str, str]:
    """Derive a preliminary diagnostic hypothesis for an error using deterministic heuristics.

    NOTE: This is strictly a preliminary suggestion to accelerate manual triage.
    It is NEVER treated as final ground truth. Human review must verify or revise it.
    """
    # 1. Parsing failure check
    if parse_failed:
        return (
            "Parsing failure",
            "Model output failed structured JSON schema validation or was unparseable after retry.",
        )

    # 2. Retrieval miss check
    if not retrieved_passages:
        return (
            "Retrieval miss",
            "Zero evidence candidates were retrieved from the corpus.",
        )

    retrieved_ids = {
        str(p.get("passage_id") or p.get("id") or "").strip()
        for p in retrieved_passages
        if (p.get("passage_id") or p.get("id"))
    }

    if gold_ids:
        gold_set = {str(gid).strip() for gid in gold_ids}
        if not (gold_set & retrieved_ids):
            return (
                "Retrieval miss",
                f"None of the {len(gold_ids)} gold evidence passages were retrieved in the candidate pool.",
            )

    # 3. Stale or temporally inappropriate evidence check
    if claim_date:
        claim_year = None
        try:
            claim_year = int(claim_date[:4])
        except (ValueError, TypeError):
            pass

        if claim_year is not None:
            dates = [p.get("date") for p in retrieved_passages if p.get("date")]
            years = []
            for d in dates:
                try:
                    years.append(int(str(d)[:4]))
                except (ValueError, TypeError):
                    pass
            if years and all(abs(y - claim_year) >= 3 for y in years):
                return (
                    "Stale or temporally inappropriate evidence",
                    f"Retrieved passage dates ({dates[:3]}) exhibit substantial temporal gap from claim date ({claim_date}).",
                )

    # 4. Incorrect credibility ranking check
    # Check if a low-credibility source was cited or high-credibility source down-ranked
    low_cred_domains = {"twitter.com", "facebook.com", "blogspot.com", "wordpress.com", "unknown"}
    high_cred_domains = {"gov", "edu", "who.int", "cdc.gov", "nature.com", "science.org"}

    top_passage = retrieved_passages[0] if retrieved_passages else {}
    top_domain = str(top_passage.get("domain") or "").lower()

    if any(lcd in top_domain for lcd in low_cred_domains):
        # Look if a higher tier domain was retrieved lower down
        lower_domains = [str(p.get("domain") or "").lower() for p in retrieved_passages[1:5]]
        if any(any(hcd in ld for hcd in high_cred_domains) for ld in lower_domains):
            return (
                "Incorrect credibility ranking",
                f"Top ranked source ({top_domain}) has questionable credibility while authoritative sources were ranked lower.",
            )

    # 5. LLM misinterpretation
    # If gold evidence was retrieved or cited, but label diverged
    if gold_ids and (set(gold_ids) & set(cited_ids)):
        return (
            "LLM misinterpretation",
            "Gold evidence passage was cited by the verifier, but downstream reasoning produced an incorrect label.",
        )

    if cited_ids and pred_label in ("True", "False") and gold_label in ("True", "False"):
        return (
            "LLM misinterpretation",
            f"Verifier cited evidence {cited_ids} but misclassified claim as '{pred_label}' instead of '{gold_label}'.",
        )

    # Default residual fallback
    return (
        "Other / unknown",
        "Discrepancy requires qualitative human inspection to determine root cause.",
    )


# ---------------------------------------------------------------------------
# Error Extraction and Merging
# ---------------------------------------------------------------------------


def format_sources_summary(retrieved_passages: list[dict[str, Any]]) -> str:
    """Format retrieved passages into a concise, readable provenance summary."""
    if not retrieved_passages:
        return "[No passages retrieved]"

    snippets: list[str] = []
    for idx, p in enumerate(retrieved_passages[:5], start=1):
        pid = p.get("passage_id") or p.get("id") or f"p{idx}"
        dom = p.get("domain") or p.get("url") or "unknown_domain"
        dt = p.get("date") or "no_date"
        score = p.get("score")
        score_str = f"score={float(score):.3f}" if score is not None else ""
        meta_str = f"({dom}, {dt}, {score_str})".replace(", )", ")")
        text = str(p.get("text_snippet") or p.get("text") or "").replace("\n", " ")[:100]
        snippets.append(f"[{pid} {meta_str}: \"{text}...\"]")

    return " | ".join(snippets)


def extract_prediction_errors(
    gold_records: list[dict[str, Any]],
    prediction_records: list[dict[str, Any]],
    existing_annotations: dict[str, dict[str, Any]] | None = None,
) -> list[ErrorRecord]:
    """Identify incorrect predictions and compile rich ErrorRecords.

    Args:
        gold_records: List of ground-truth claim dictionaries.
        prediction_records: List of model predictions / verdicts.
        existing_annotations: Optional map of claim_id -> human annotations to retain.

    Returns:
        List of ErrorRecord instances for all cases where gold_label != predicted_label.
    """
    matched = match_gold_and_predictions(gold_records, prediction_records, strict_match=False)
    existing = existing_annotations or {}

    error_records: list[ErrorRecord] = []

    for g, p in matched:
        gold_lbl = str(g.get("label") or "").strip()
        pred_lbl = str(p.get("label") or "").strip()

        # If prediction was correct, skip
        if gold_lbl.lower() == pred_lbl.lower():
            continue

        cid = str(g.get("id") or g.get("claim_id") or p.get("claim_id") or "").strip()
        claim_text = str(g.get("claim") or g.get("text") or p.get("claim") or "").strip()
        claim_date = g.get("claim_date") or g.get("date") or p.get("claim_date")

        # Extract retrieved passages & gold evidence IDs
        passages = p.get("retrieved_passages") or []
        retrieved_ids = [
            str(item.get("passage_id") or item.get("id"))
            for item in passages
            if (item.get("passage_id") or item.get("id"))
        ]
        cited_ids = [str(eid) for eid in p.get("evidence_ids") or []]

        gold_ev = g.get("evidence") or g.get("evidence_ids") or []
        gold_ids: list[str] = []
        if isinstance(gold_ev, list):
            for item in gold_ev:
                if isinstance(item, dict):
                    gid = item.get("passage_id") or item.get("id")
                    if gid:
                        gold_ids.append(str(gid))
                elif isinstance(item, str):
                    gold_ids.append(item)

        # Abstention & Faithfulness info
        meta = p.get("model_metadata") or {}
        abst_info = meta.get("abstention") or {}
        abstained = bool(abst_info.get("abstained") or (pred_lbl == "Unverified" and meta.get("original_label")))
        orig_lbl = abst_info.get("original_label") or meta.get("original_label")
        faith_score = abst_info.get("faithfulness_score") or p.get("faithfulness_score")
        abst_reasons = abst_info.get("abstention_reasons") or []

        # Parsing & Timings
        parse_failed = bool(p.get("parse_failed", False))
        attempts = int(meta.get("attempts", 1))
        timings = p.get("stage_timings_s") or {}
        lat_sec = meta.get("latency_seconds")
        if lat_sec is None and timings:
            lat_sec = sum(float(v) for v in timings.values() if isinstance(v, (int, float)))

        # Diagnostic heuristic suggestion
        sugg_cat, sugg_rat = infer_suggested_error_category(
            gold_label=gold_lbl,
            pred_label=pred_lbl,
            parse_failed=parse_failed,
            retrieved_passages=passages,
            gold_ids=gold_ids,
            cited_ids=cited_ids,
            claim_date=str(claim_date) if claim_date else None,
            faithfulness_score=float(faith_score) if faith_score is not None else None,
            abstained=abstained,
        )

        # Merge previous human review annotations if present
        prev = existing.get(cid, {})
        ann_cat = prev.get("annotated_category", "")
        ann_id = prev.get("annotator_id", "")
        ann_notes = prev.get("annotator_notes", "")
        ver_status = prev.get("verification_status", "unreviewed" if not ann_cat else "in_review")

        sources_summary = format_sources_summary(passages)

        err_rec = ErrorRecord(
            claim_id=cid,
            claim=claim_text,
            gold_label=gold_lbl,
            predicted_label=pred_lbl,
            confidence=float(p.get("confidence", 0.0)),
            explanation=str(p.get("explanation", "")),
            retrieved_evidence_ids=retrieved_ids,
            retrieved_sources_summary=sources_summary,
            cited_evidence_ids=cited_ids,
            gold_evidence_ids=gold_ids,
            claim_date=str(claim_date) if claim_date else None,
            faithfulness_score=float(faith_score) if faith_score is not None else None,
            abstained=abstained,
            original_label=orig_lbl,
            abstention_reasons=abst_reasons,
            parse_failed=parse_failed,
            attempts=attempts,
            latency_seconds=float(lat_sec) if lat_sec is not None else None,
            stage_timings=timings,
            raw_llm_output=p.get("raw_llm_output"),
            suggested_category=sugg_cat,
            suggestion_rationale=sugg_rat,
            annotated_category=ann_cat,
            annotator_id=ann_id,
            annotator_notes=ann_notes,
            verification_status=ver_status,
        )
        error_records.append(err_rec)

    logger.info("Extracted %d incorrect predictions for error analysis.", len(error_records))
    return error_records


# ---------------------------------------------------------------------------
# CSV Persistence and Annotation Merging
# ---------------------------------------------------------------------------


def load_annotations_from_csv(csv_path: str | Path) -> dict[str, dict[str, Any]]:
    """Load previously saved human annotations from CSV keyed by claim_id.

    Preserves manual reviews so re-exporting predictions never overwrites human work.
    """
    p = Path(csv_path)
    if not p.exists():
        return {}

    annotations: dict[str, dict[str, Any]] = {}
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = str(row.get("claim_id", "")).strip()
            if not cid:
                continue
            annotations[cid] = {
                "annotated_category": row.get("annotated_category", "").strip(),
                "annotator_id": row.get("annotator_id", "").strip(),
                "annotator_notes": row.get("annotator_notes", "").strip(),
                "verification_status": row.get("verification_status", "unreviewed").strip(),
            }
    logger.info("Loaded %d existing human annotations from %s", len(annotations), p)
    return annotations


def export_errors_to_csv(errors: list[ErrorRecord], output_path: str | Path) -> None:
    """Export error records into a review-friendly CSV spreadsheet.

    UTF-8 encoded with excel dialect and clean header order.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not errors:
        # Write empty template with headers
        dummy_row = ErrorRecord(
            claim_id="", claim="", gold_label="", predicted_label="", confidence=0.0, explanation=""
        ).to_csv_row()
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(dummy_row.keys()))
            writer.writeheader()
        return

    fieldnames = list(errors[0].to_csv_row().keys())
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for err in errors:
            writer.writerow(err.to_csv_row())

    logger.info("Exported %d error records to %s", len(errors), p)


# ---------------------------------------------------------------------------
# Statistics and Reporting
# ---------------------------------------------------------------------------


def compute_error_statistics(
    errors: list[ErrorRecord],
    total_evaluated_claims: int | None = None,
) -> dict[str, Any]:
    """Compute aggregate counts, proportions, confusion breakdown, and confidence distribution."""
    n_errors = len(errors)
    total_claims = total_evaluated_claims or n_errors

    # Category counts: both annotated (ground truth) and suggested (provisional)
    annotated_counts: dict[str, int] = {cat: 0 for cat in ERROR_CATEGORIES}
    suggested_counts: dict[str, int] = {cat: 0 for cat in ERROR_CATEGORIES}

    verified_count = 0
    confusion_pairs: dict[str, int] = {}
    confidence_bins: dict[str, int] = {"low_[0.0-0.4)": 0, "medium_[0.4-0.7)": 0, "high_[0.7-1.0]": 0}

    for err in errors:
        # Suggested
        s_cat = err.suggested_category if err.suggested_category in suggested_counts else "Other / unknown"
        suggested_counts[s_cat] += 1

        # Annotated
        if err.annotated_category:
            a_cat = err.annotated_category if err.annotated_category in annotated_counts else "Other / unknown"
            annotated_counts[a_cat] += 1
            if err.verification_status in ("verified", "disputed"):
                verified_count += 1

        # Confusion
        pair_key = f"{err.gold_label} -> {err.predicted_label}"
        confusion_pairs[pair_key] = confusion_pairs.get(pair_key, 0) + 1

        # Confidence
        c = err.confidence
        if c < 0.4:
            confidence_bins["low_[0.0-0.4)"] += 1
        elif c < 0.7:
            confidence_bins["medium_[0.4-0.7)"] += 1
        else:
            confidence_bins["high_[0.7-1.0]"] += 1

    # Proportions
    annotated_proportions = {
        cat: round(cnt / n_errors, 4) if n_errors > 0 else 0.0 for cat, cnt in annotated_counts.items()
    }
    suggested_proportions = {
        cat: round(cnt / n_errors, 4) if n_errors > 0 else 0.0 for cat, cnt in suggested_counts.items()
    }

    return {
        "total_claims": total_claims,
        "total_errors": n_errors,
        "error_rate": round(n_errors / total_claims, 4) if total_claims > 0 else 0.0,
        "human_reviewed_errors": sum(annotated_counts.values()),
        "human_verified_errors": verified_count,
        "suggested_counts": suggested_counts,
        "suggested_proportions": suggested_proportions,
        "annotated_counts": annotated_counts,
        "annotated_proportions": annotated_proportions,
        "confusion_breakdown": confusion_pairs,
        "confidence_distribution": confidence_bins,
    }


def generate_error_summary_report(
    errors: list[ErrorRecord],
    total_evaluated_claims: int | None = None,
    dataset_name: str = "dataset",
    run_id: str = "default_run",
) -> str:
    """Produce a comprehensive Markdown report summarizing failure patterns with representative examples."""
    stats = compute_error_statistics(errors, total_evaluated_claims=total_evaluated_claims)
    n_errors = len(errors)

    lines: list[str] = [
        f"# VERITAS Qualitative Error Analysis Report: `{run_id}`",
        "",
        f"- **Dataset Evaluated**: `{dataset_name}`",
        f"- **Total Evaluated Claims**: `{stats['total_claims']}`",
        f"- **Total Incorrect Predictions**: `{stats['total_errors']}` (Error Rate: **{stats['error_rate'] * 100:.2f}%**)",
        f"- **Human-Reviewed Errors**: `{stats['human_reviewed_errors']}` / `{n_errors}` ({stats['human_verified_errors']} fully verified)",
        f"- **Generated At**: `{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`",
        "",
        "---",
        "",
        "## 1. Error Category Distribution",
        "",
        "| Error Category | Suggested Count | Suggested % | Annotated Count | Annotated % | Actionable Remedy |",
        "|:---|:---:|:---:|:---:|:---:|:---|",
    ]

    remedies = {
        "Retrieval miss": "Improve query expansion, add synonyms, or expand corpus coverage",
        "Incorrect credibility ranking": "Calibrate credibility weights (beta) or update domain tiers",
        "Stale or temporally inappropriate evidence": "Tune temporal decay (tau) or enforce date filtering",
        "LLM misinterpretation": "Refine reasoning prompts or provide few-shot exemplars",
        "Label noise or ambiguous annotation": "Audit benchmark dataset label and flag noisy claims",
        "Parsing failure": "Enforce strict JSON retry or schema sanitization",
        "Other / unknown": "Perform manual qualitative review",
    }

    s_counts = stats["suggested_counts"]
    s_props = stats["suggested_proportions"]
    a_counts = stats["annotated_counts"]
    a_props = stats["annotated_proportions"]

    for cat in ERROR_CATEGORIES:
        sc = s_counts.get(cat, 0)
        sp = s_props.get(cat, 0.0) * 100
        ac = a_counts.get(cat, 0)
        ap = a_props.get(cat, 0.0) * 100
        rem = remedies.get(cat, "-")
        lines.append(f"| **{cat}** | {sc} | {sp:.1f}% | {ac} | {ap:.1f}% | {rem} |")

    lines.extend([
        "",
        "> [!NOTE]",
        "> **Methodological Rule**: Suggested categories are heuristic suggestions computed automatically to facilitate triage. They must **never** be treated as ground truth until verified by human reviewers.",
        "",
        "---",
        "",
        "## 2. Label Confusion Breakdown (Errors Only)",
        "",
        "| Gold Label &rarr; Predicted Label | Count | Proportion of Errors |",
        "|:---|:---:|:---:|",
    ])

    for pair, count in sorted(stats["confusion_breakdown"].items(), key=lambda x: x[1], reverse=True):
        prop = (count / n_errors) * 100 if n_errors > 0 else 0.0
        lines.append(f"| `{pair}` | {count} | {prop:.1f}% |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Confidence Distribution of Errors",
        "",
        "| Confidence Bin | Count | % of Errors | Interpretation |",
        "|:---|:---:|:---:|:---|",
    ])

    conf_bins = stats["confidence_distribution"]
    low = conf_bins.get("low_[0.0-0.4)", 0)
    med = conf_bins.get("medium_[0.4-0.7)", 0)
    high = conf_bins.get("high_[0.7-1.0]", 0)

    lines.append(f"| **Low [0.0 - 0.4)** | {low} | {(low/n_errors*100) if n_errors else 0:.1f}% | Uncertain predictions; appropriate candidates for abstention |")
    lines.append(f"| **Medium [0.4 - 0.7)** | {med} | {(med/n_errors*100) if n_errors else 0:.1f}% | Borderline confidence |")
    lines.append(f"| **High [0.7 - 1.0]** | {high} | {(high/n_errors*100) if n_errors else 0:.1f}% | **Overconfident errors** (high risk hallucinations / severe misses) |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Representative Failure Patterns with Concrete Examples",
        "",
    ])

    # Find representative examples per category (prefer human-annotated, fallback to suggested)
    for cat in ERROR_CATEGORIES:
        matching = [e for e in errors if e.annotated_category == cat]
        if not matching:
            matching = [e for e in errors if e.suggested_category == cat]

        if not matching:
            continue

        lines.append(f"### Category: {cat}")
        lines.append("")

        sample = matching[0]
        notes_str = f"**Annotator Notes**: {sample.annotator_notes}\n" if sample.annotator_notes else ""
        lines.extend([
            f"- **Claim ID**: `{sample.claim_id}`",
            f"- **Claim Text**: \"{sample.claim}\"",
            f"- **Gold Label**: `{sample.gold_label}` | **Predicted Label**: `{sample.predicted_label}` (Confidence: {sample.confidence:.2f})",
            f"- **Claim Date**: `{sample.claim_date or 'N/A'}`",
            f"- **Explanation**: {sample.explanation}",
            f"- **Retrieved Sources**: {sample.retrieved_sources_summary or 'None'}",
            f"- **Diagnostic Rationale**: {sample.suggestion_rationale}",
            notes_str,
        ])

    return "\n".join(lines) + "\n"
