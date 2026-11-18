"""Aspect-level record schema and validation."""

from absa.data.normalize import build_processed_files, no_aspect_filename, normalize_sentences
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
    "build_processed_files",
    "no_aspect_filename",
    "normalize_sentences",
    "parse_laptop_xml",
    "parse_restaurant_xml",
    "validate_prediction",
    "validate_record",
]
