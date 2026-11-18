"""Aspect-level record schema and validation."""

from absa.data.schema import (
    RECORD_FIELDS,
    AspectPrediction,
    AspectSpan,
    ExtractionRecord,
    PredictionRecord,
    validate_prediction,
    validate_record,
)

__all__ = [
    "RECORD_FIELDS",
    "AspectPrediction",
    "AspectSpan",
    "ExtractionRecord",
    "PredictionRecord",
    "validate_prediction",
    "validate_record",
]
