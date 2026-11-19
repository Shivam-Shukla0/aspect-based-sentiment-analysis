"""Shared evaluation protocol and metric calculations."""

from absa.evaluation.e2e import (
    AspectSentimentPair,
    ClassificationMetrics,
    EndToEndMetrics,
    evaluate_end_to_end,
    evaluate_given_aspect_classification,
)

__all__ = [
    "AspectSentimentPair",
    "ClassificationMetrics",
    "EndToEndMetrics",
    "evaluate_end_to_end",
    "evaluate_given_aspect_classification",
]
