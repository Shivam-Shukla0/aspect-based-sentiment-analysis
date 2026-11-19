"""Shared evaluation protocol and metric calculations."""

from absa.evaluation.extraction import (
    ExtractionMetrics,
    compute_span_counts,
    evaluate_extraction_spans,
)

__all__ = [
    "ExtractionMetrics",
    "compute_span_counts",
    "evaluate_extraction_spans",
]
