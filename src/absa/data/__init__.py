"""Aspect-level record schema and validation."""

from absa.data.bert_dataset import (
    ID_TO_SENTIMENT,
    SENTIMENT_TO_ID,
    BertPairExample,
    build_pair_dataset,
    build_pair_example,
)
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
    "ID_TO_SENTIMENT",
    "RECORD_FIELDS",
    "SENTIMENT_TO_ID",
    "AspectPrediction",
    "AspectSpan",
    "BertPairExample",
    "ExtractionRecord",
    "PredictionRecord",
    "build_pair_dataset",
    "build_pair_example",
    "validate_prediction",
    "validate_record",
]
