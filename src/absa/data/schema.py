"""Aspect-level record contracts shared by data, models and evaluation.

Three task settings share these contracts:

- Given-aspect classification: an ``AspectRecord`` (``validate_record``) in,
  a sentiment out. Used by the LSTM and BERT classifiers.
- Aspect extraction: an ``ExtractionRecord`` in, annotated ``AspectSpan``
  terms out. Used by extractors that do not assume the aspect is known.
- End-to-end prediction: a review in, a ``PredictionRecord``
  (``validate_prediction``) out. Composes extraction and classification.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict

from absa.config import SENTIMENT_LABELS

RECORD_FIELDS = ("review", "aspect", "sentiment")


class AspectPrediction(TypedDict):
    """One aspect/sentiment pair inside a ``PredictionRecord``."""

    aspect: str
    sentiment: str


class PredictionRecord(TypedDict):
    """End-to-end inference output contract: a review and its predictions."""

    review: str
    predictions: list[AspectPrediction]


@dataclass(frozen=True, slots=True)
class AspectSpan:
    """One annotated aspect term inside an ``ExtractionRecord``.

    ``start`` and ``end`` are character offsets into the enclosing
    ``ExtractionRecord.text``, following the SemEval convention that
    ``text[start:end] == term``.
    """

    term: str
    start: int
    end: int
    polarity: str

    def __post_init__(self) -> None:
        """Reject spans with an empty term or offsets that cannot bound it."""
        if not self.term.strip():
            raise ValueError("span term must be a non-empty string")
        if self.start < 0 or self.end <= self.start:
            raise ValueError(
                f"span offsets must satisfy 0 <= start < end, got {self.start}:{self.end}"
            )


@dataclass(frozen=True, slots=True)
class ExtractionRecord:
    """Aspect-extraction contract: a sentence and its annotated aspect spans.

    Distinct from ``AspectRecord``, which assumes the aspect is already
    known. Used for training and evaluating aspect extraction.
    """

    text: str
    spans: tuple[AspectSpan, ...]

    def __post_init__(self) -> None:
        """Reject an empty text or a span whose offsets do not match its term."""
        if not self.text.strip():
            raise ValueError("extraction record text must be a non-empty string")
        for span in self.spans:
            found = self.text[span.start : span.end]
            if found != span.term:
                raise ValueError(
                    f"span {span.term!r} does not match text[{span.start}:{span.end}] = {found!r}"
                )


def _require_non_empty_str(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field!r} must be a non-empty string")


def _require_label(label: object, allowed_labels: Sequence[str]) -> None:
    if label not in allowed_labels:
        raise ValueError(f"unknown sentiment {label!r}; expected one of {allowed_labels}")


def validate_record(
    record: object, allowed_labels: Sequence[str] = SENTIMENT_LABELS
) -> dict[str, object]:
    """Return ``record`` unchanged if valid, otherwise raise ``ValueError``.

    A valid record is a mapping with non-empty string ``review`` and
    ``aspect`` fields and a ``sentiment`` drawn from ``allowed_labels``.
    """
    if not isinstance(record, dict):
        raise ValueError(f"record must be a dict, got {type(record).__name__}")
    missing = [f for f in RECORD_FIELDS if f not in record]
    if missing:
        raise ValueError(f"record is missing fields: {missing}")
    _require_non_empty_str(record["review"], "review")
    _require_non_empty_str(record["aspect"], "aspect")
    _require_label(record["sentiment"], allowed_labels)
    return record


def validate_prediction(
    record: object, allowed_labels: Sequence[str] = SENTIMENT_LABELS
) -> dict[str, object]:
    """Return ``record`` unchanged if it matches the end-to-end prediction contract.

    A valid prediction record is ``{"review": str, "predictions": [{"aspect":
    str, "sentiment": str}, ...]}``. An empty ``predictions`` list is valid:
    it means no aspect was extracted from the review.
    """
    if not isinstance(record, dict):
        raise ValueError(f"prediction record must be a dict, got {type(record).__name__}")
    missing = [f for f in ("review", "predictions") if f not in record]
    if missing:
        raise ValueError(f"prediction record is missing fields: {missing}")
    _require_non_empty_str(record["review"], "review")
    predictions = record["predictions"]
    if not isinstance(predictions, list):
        raise ValueError(f"'predictions' must be a list, got {type(predictions).__name__}")
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            raise ValueError(
                f"predictions[{index}] must be a dict, got {type(prediction).__name__}"
            )
        prediction_missing = [f for f in ("aspect", "sentiment") if f not in prediction]
        if prediction_missing:
            raise ValueError(f"predictions[{index}] is missing fields: {prediction_missing}")
        _require_non_empty_str(prediction["aspect"], f"predictions[{index}]['aspect']")
        _require_label(prediction["sentiment"], allowed_labels)
    return record
