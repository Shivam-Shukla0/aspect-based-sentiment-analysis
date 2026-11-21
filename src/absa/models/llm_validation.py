"""Validation, error handling and retry wrapper for LLM predictions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from absa.config import SENTIMENT_LABELS


@dataclass(frozen=True)
class ValidatedAspect:
    """A validated aspect prediction aligned with review text."""

    aspect: str
    sentiment: str
    start: int
    end: int


@dataclass
class ValidationReport:
    """Diagnostic report tracking validation statistics and errors."""

    total_candidates: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    retries_used: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def is_successful(self) -> bool:
        """Return True if no candidate was rejected (also True when there were no candidates)."""
        return self.invalid_count == 0


def _find_unused(review: str, aspect: str, used: set[int]) -> int:
    """Return the first occurrence of ``aspect`` in ``review`` not claimed yet, or -1.

    An exact match is preferred; if there is none, a case-insensitive match is used.
    Starting positions already in ``used`` are skipped, so an aspect that is mentioned
    twice maps to two different spans instead of the first one twice.
    """
    for text, needle in ((review, aspect), (review.lower(), aspect.lower())):
        start = text.find(needle)
        while start != -1:
            if start not in used:
                return start
            start = text.find(needle, start + 1)
    return -1


def validate_aspect_predictions(
    review: str,
    raw_predictions: Sequence[dict[str, str]],
    report: ValidationReport | None = None,
) -> list[ValidatedAspect]:
    """Validate extracted aspect predictions against review text and schema labels.

    Args:
        review: Original review string.
        raw_predictions: List of dicts with 'aspect' and 'sentiment'.
        report: Optional ValidationReport to record diagnostic statistics.

    Returns:
        List of validated and offset-aligned ValidatedAspect objects.
    """
    rep = report if report is not None else ValidationReport()
    validated: list[ValidatedAspect] = []
    used: set[int] = set()

    for item in raw_predictions:
        rep.total_candidates += 1
        aspect = item.get("aspect", "").strip()
        sentiment = item.get("sentiment", "").strip().lower()

        if not aspect:
            rep.invalid_count += 1
            rep.errors.append("Empty aspect string")
            continue

        if sentiment not in SENTIMENT_LABELS:
            rep.invalid_count += 1
            rep.errors.append(f"Invalid sentiment label: {sentiment!r}")
            continue

        start = _find_unused(review, aspect, used)
        if start == -1:
            rep.invalid_count += 1
            rep.errors.append(f"Aspect {aspect!r} not found in review text")
            continue

        end = start + len(aspect)
        used.add(start)
        validated.append(
            ValidatedAspect(aspect=review[start:end], sentiment=sentiment, start=start, end=end)
        )
        rep.valid_count += 1

    return validated


TRANSIENT_ERRORS: tuple[type[Exception], ...] = (RuntimeError, OSError)


def _attempt(
    generate_fn: Callable[[], list[dict[str, str]]], retry_on: tuple[type[Exception], ...]
) -> tuple[list[dict[str, str]] | None, Exception | None]:
    """Run one attempt; return ``(result, None)`` or ``(None, error)`` for a retryable error."""
    try:
        return generate_fn(), None
    except retry_on as err:
        return None, err


def retry_llm_inference(
    generate_fn: Callable[[], list[dict[str, str]]],
    max_retries: int = 3,
    report: ValidationReport | None = None,
    retry_on: tuple[type[Exception], ...] = TRANSIENT_ERRORS,
) -> list[dict[str, str]]:
    """Run ``generate_fn``, retrying only errors listed in ``retry_on``.

    Network and service failures (``RuntimeError`` from the client, ``OSError`` and its
    subclasses such as ``ConnectionError`` and ``TimeoutError``) are retried up to
    ``max_retries`` attempts in total. Any other exception, such as a parse error or a
    programming mistake, is raised immediately, because repeating it cannot help.

    Raises:
        ValueError: If ``max_retries`` is less than 1.
        RuntimeError: If every attempt failed with a retryable error.
    """
    if max_retries < 1:
        raise ValueError(f"max_retries must be at least 1, got {max_retries}")
    rep = report if report is not None else ValidationReport()
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        result, err = _attempt(generate_fn, retry_on)
        if result is not None:
            return result
        last_err = err
        rep.retries_used += 1
        rep.errors.append(f"Attempt {attempt} failed: {err}")
    raise RuntimeError(
        f"LLM generation failed after {max_retries} attempts: {last_err}"
    ) from last_err
