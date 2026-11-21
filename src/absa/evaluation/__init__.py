"""Shared evaluation protocol and metric calculations.

All approaches must be scored on the original SemEval-2014 test split.
"""

from absa.evaluation.e2e import (
    AspectSentimentPair,
    ClassificationMetrics,
    EndToEndMetrics,
    evaluate_end_to_end,
    evaluate_given_aspect_classification,
)
from absa.evaluation.extraction import (
    ExtractionMetrics,
    compute_span_counts,
    evaluate_extraction_spans,
)
from absa.evaluation.extraction_errors import SpanError, span_report
from absa.evaluation.metrics import ClassMetrics, EvaluationResult, evaluate

__all__ = [
    "AspectSentimentPair",
    "ClassMetrics",
    "ClassificationMetrics",
    "EndToEndMetrics",
    "EvaluationResult",
    "ExtractionMetrics",
    "SpanError",
    "compute_span_counts",
    "evaluate",
    "evaluate_end_to_end",
    "evaluate_extraction_spans",
    "evaluate_given_aspect_classification",
    "span_report",
]
