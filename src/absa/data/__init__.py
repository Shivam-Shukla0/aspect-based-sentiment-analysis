"""Aspect-level record schema and validation."""

from absa.data.eda import aspect_frequency, class_balance, domain_counts, review_lengths
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
from absa.data.splits import split_train_val
from absa.data.validation import RecordError, ValidationSummary, read_jsonl, validate_records
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
    "RecordError",
    "ValidationSummary",
    "aspect_frequency",
    "build_processed_files",
    "class_balance",
    "domain_counts",
    "no_aspect_filename",
    "normalize_sentences",
    "parse_laptop_xml",
    "parse_restaurant_xml",
    "read_jsonl",
    "review_lengths",
    "split_train_val",
    "validate_prediction",
    "validate_record",
    "validate_records",
]
