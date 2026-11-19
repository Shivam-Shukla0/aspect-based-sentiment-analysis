"""Aspect-level record schema and validation.

``CategoryAnnotation``, ``ParsedSentence``, ``parse_laptop_xml`` and
``parse_restaurant_xml`` need ``lxml``; they are imported lazily (PEP 562)
on first access, so importing this package for the lightweight,
dependency-free names above (``validate_record`` and friends) never
requires lxml to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from absa._lazy import lazy_getattr
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

if TYPE_CHECKING:
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

_LAZY = {
    "CategoryAnnotation": "absa.data.xml_parser",
    "ParsedSentence": "absa.data.xml_parser",
    "parse_laptop_xml": "absa.data.xml_parser",
    "parse_restaurant_xml": "absa.data.xml_parser",
}


def __getattr__(name: str) -> Any:
    """Resolve ``_LAZY`` names by importing their submodule on first access."""
    return lazy_getattr(__name__, _LAZY, name)
