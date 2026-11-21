"""Validate processed aspect records before they reach training.

Independent of the parser's own checks, which already reject malformed XML
at parse time (see ``absa.data.xml_parser``): this operates on records that
have already been written, such as a processed JSONL file, so a defect is
still caught if bad data reaches training by some other path. Problems are
reported, not repaired; a caller decides what to do with an invalid record.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from absa.config import SENTIMENT_LABELS

ERROR_TYPES = ("empty_review", "empty_aspect", "unknown_sentiment", "offset_mismatch")


@dataclass(frozen=True, slots=True)
class RecordError:
    """One defect found in one record, identified by its position in the input."""

    index: int
    error_type: str
    detail: str


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    """Validation outcome: every error found, plus counts by error type."""

    total: int
    errors: tuple[RecordError, ...]

    @property
    def counts(self) -> dict[str, int]:
        """Return the number of errors of each type, including types with zero."""
        counter = Counter(error.error_type for error in self.errors)
        return {error_type: counter.get(error_type, 0) for error_type in ERROR_TYPES}

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation of this summary."""
        return {
            "total_records": self.total,
            "invalid_records": len({error.index for error in self.errors}),
            "error_counts": self.counts,
            "errors": [
                {"index": error.index, "error_type": error.error_type, "detail": error.detail}
                for error in self.errors
            ],
        }

    def save(self, path: Path) -> None:
        """Write this summary as JSON to ``path``, creating parent directories."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def _validate_one(index: int, record: object, allowed_labels: Sequence[str]) -> list[RecordError]:
    errors = []
    if not isinstance(record, Mapping):
        errors.append(RecordError(index, "empty_review", "record is not a mapping"))
        return errors
    review = record.get("review")
    aspect = record.get("aspect")
    sentiment = record.get("sentiment")
    if not isinstance(review, str) or not review.strip():
        errors.append(RecordError(index, "empty_review", "'review' is missing or empty"))
    if not isinstance(aspect, str) or not aspect.strip():
        errors.append(RecordError(index, "empty_aspect", "'aspect' is missing or empty"))
    if sentiment not in allowed_labels:
        errors.append(
            RecordError(
                index, "unknown_sentiment", f"sentiment {sentiment!r} not in {allowed_labels}"
            )
        )
    start, end = record.get("start"), record.get("end")
    if start is not None or end is not None:
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not isinstance(review, str)
            or not isinstance(aspect, str)
            or start < 0
            or end > len(review)
        ):
            errors.append(
                RecordError(
                    index,
                    "offset_mismatch",
                    f"review[{start}:{end}] = ..., expected aspect {aspect!r}",
                )
            )
        else:
            found = review[start:end]
            if found != aspect:
                errors.append(
                    RecordError(
                        index,
                        "offset_mismatch",
                        f"review[{start}:{end}] = {found!r}, expected aspect {aspect!r}",
                    )
                )
    return errors


def validate_records(
    records: Iterable[object], allowed_labels: Sequence[str] = SENTIMENT_LABELS
) -> ValidationSummary:
    """Validate every record and return one summary covering all of them.

    Checks ``sentiment`` against ``allowed_labels``, that ``review`` and
    ``aspect`` are non-empty strings, and that ``review[start:end] ==
    aspect`` whenever both offsets are present (offsets are not part of the
    minimal record contract, so their absence is not itself an error).
    """
    all_errors: list[RecordError] = []
    total = 0
    for index, record in enumerate(records):
        total += 1
        all_errors.extend(_validate_one(index, record, allowed_labels))
    return ValidationSummary(total=total, errors=tuple(all_errors))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read one JSON object per line from ``path``, skipping blank lines."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
