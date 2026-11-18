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
from absa.data.xml_parser import (
    CategoryAnnotation,
    ParsedSentence,
    parse_laptop_xml,
    parse_restaurant_xml,
)

__all__ = [
    "RECORD_FIELDS",
    "AspectPrediction",
    "AspectSpan",
    "CategoryAnnotation",
    "ExtractionRecord",
    "ParsedSentence",
    "PredictionRecord",
    "parse_laptop_xml",
    "parse_restaurant_xml",
    "validate_prediction",
    "validate_record",
]
