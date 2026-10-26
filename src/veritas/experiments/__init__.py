"""VERITAS Experiment Suite package."""

from veritas.experiments.config import (
    ExperimentConfig,
    ExperimentConfigError,
    validate_factor_isolation,
)
from veritas.experiments.runner import ExperimentRunner
from veritas.experiments.analysis import (
    aggregate_experiment_results,
    export_summary_table_csv,
    generate_comparative_svg_chart,
    generate_markdown_summary_table,
)

__all__ = [
    "ExperimentConfig",
    "ExperimentConfigError",
    "validate_factor_isolation",
    "ExperimentRunner",
    "aggregate_experiment_results",
    "generate_markdown_summary_table",
    "export_summary_table_csv",
    "generate_comparative_svg_chart",
]
