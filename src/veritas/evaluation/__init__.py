"""VERITAS P9 Evaluation Package."""

from veritas.evaluation.metrics import (
    EvaluationDataError,
    STANDARD_LABEL_ORDER,
    aggregate_multi_run_metrics,
    compute_classification_metrics,
    compute_coverage_accuracy_curve,
    compute_latency_metrics,
    compute_system_quality_metrics,
    match_gold_and_predictions,
)
from veritas.evaluation.report import (
    EvaluationReport,
    build_evaluation_report,
    format_console_report,
    save_report_artifacts,
)
from veritas.evaluation.retrieval_metrics import (
    compute_dcg_at_k,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_recall_at_k,
    evaluate_retrieval_batch,
)

from veritas.evaluation.error_analysis import (
    ERROR_CATEGORIES,
    ErrorRecord,
    compute_error_statistics,
    export_errors_to_csv,
    extract_prediction_errors,
    generate_error_summary_report,
    load_annotations_from_csv,
)

__all__ = [
    "EvaluationDataError",
    "STANDARD_LABEL_ORDER",
    "match_gold_and_predictions",
    "compute_classification_metrics",
    "compute_latency_metrics",
    "compute_coverage_accuracy_curve",
    "compute_system_quality_metrics",
    "aggregate_multi_run_metrics",
    "compute_recall_at_k",
    "compute_dcg_at_k",
    "compute_ndcg_at_k",
    "compute_mrr_at_k",
    "evaluate_retrieval_batch",
    "EvaluationReport",
    "build_evaluation_report",
    "format_console_report",
    "save_report_artifacts",
    "ERROR_CATEGORIES",
    "ErrorRecord",
    "extract_prediction_errors",
    "export_errors_to_csv",
    "load_annotations_from_csv",
    "compute_error_statistics",
    "generate_error_summary_report",
]
