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
from absa.data.xml_parser import ParsedSentence, parse_laptop_xml

__all__ = [
    "RECORD_FIELDS",
    "AspectPrediction",
    "AspectSpan",
    "ExtractionRecord",
    "ParsedSentence",
    "PredictionRecord",
    "parse_laptop_xml",
    "validate_prediction",
    "validate_record",
]
