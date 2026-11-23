"""Validation, error handling and retry wrapper for LLM predictions."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from absa.config import SENTIMENT_LABELS
from absa.models.llm_errors import LLMClientError


@dataclass(frozen=True)
class ValidatedAspect:
    """A validated aspect prediction aligned with review text.

    This is separate from the shared ``AspectPrediction`` contract (#50) on purpose: it
    carries character offsets (``start``/``end``) recovered by locating the aspect in the
    review, which the model output itself does not contain. When #50 lands it should be
    converted to, or replaced by, the shared type.
    """

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
    retries_used: int = 0  # retries performed after a failed attempt
    errors: list[str] = field(default_factory=list)

    @property
    def is_successful(self) -> bool:
        """Return True if no candidate was rejected (also True when there were no candidates)."""
        return self.invalid_count == 0


def _find_unused(review: str, aspect: str, used: set[int]) -> tuple[int, int] | None:
    """Return the span of the first occurrence of ``aspect`` in ``review`` not claimed yet.

    An exact match is preferred; if there is none, a case-insensitive match is used. The
    search runs on the original review (not on a lower-cased copy, whose length can differ),
    so the returned offsets always index into ``review``. Starting positions already in
    ``used`` are skipped, so an aspect that is mentioned twice maps to two different spans.
    """
    # Two passes: exact case first (most precise), then case-insensitive as a fallback for
    # when the model changed the capitalisation of the aspect.
    for flags in (0, re.IGNORECASE):
        # re.escape: aspect text is literal, so characters like "(" or "." must not act as regex.
        for match in re.finditer(re.escape(aspect), review, flags):
            if match.start() not in used:
                return match.start(), match.end()
    return None


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

    Note:
        Repeated mentions of one aspect are assigned to successive occurrences in the order
        the model listed them, which is only the text order if the model followed it. The
        prompt should therefore ask for aspects in order of appearance. Nested aspects such
        as "battery" and "battery life" both start at the same position, so only the first
        one listed is kept.
    """
    # Use the caller's report if given (to accumulate statistics), else a throwaway one.
    rep = report if report is not None else ValidationReport()
    validated: list[ValidatedAspect] = []
    # Start offsets already assigned, so two mentions of the same word map to two positions.
    used: set[int] = set()

    for item in raw_predictions:
        rep.total_candidates += 1
        aspect = item.get("aspect", "").strip()
        sentiment = item.get("sentiment", "").strip().lower()

        # Each check below rejects the candidate and records why, instead of raising, so one
        # bad item never discards the valid ones.
        if not aspect:
            rep.invalid_count += 1
            rep.errors.append("Empty aspect string")
            continue

        if sentiment not in SENTIMENT_LABELS:
            rep.invalid_count += 1
            rep.errors.append(f"Invalid sentiment label: {sentiment!r}")
            continue

        span = _find_unused(review, aspect, used)
        if span is None:
            rep.invalid_count += 1
            rep.errors.append(f"Aspect {aspect!r} not found in review text")
            continue

        start, end = span
        used.add(start)
        # Use the text as it appears in the review (not the model's spelling) so offsets and
        # text always agree.
        validated.append(
            ValidatedAspect(aspect=review[start:end], sentiment=sentiment, start=start, end=end)
        )
        rep.valid_count += 1

    return validated


# Errors worth retrying by default: service/network problems, not programming mistakes.
TRANSIENT_ERRORS: tuple[type[Exception], ...] = (LLMClientError, OSError)


def _attempt(
    generate_fn: Callable[[], list[dict[str, str]]], retry_on: tuple[type[Exception], ...]
) -> tuple[list[dict[str, str]] | None, Exception | None]:
    """Run one attempt; return ``(result, None)`` or ``(None, error)`` for a retryable error."""
    try:
        return generate_fn(), None
    except retry_on as err:
        # The service said the request itself is wrong (e.g. 400/401): repeating it is pointless.
        if isinstance(err, LLMClientError) and not err.retryable:
            raise
        return None, err


def retry_llm_inference(
    generate_fn: Callable[[], list[dict[str, str]]],
    max_retries: int = 3,
    report: ValidationReport | None = None,
    retry_on: tuple[type[Exception], ...] = TRANSIENT_ERRORS,
) -> list[dict[str, str]]:
    """Run ``generate_fn``, retrying only errors listed in ``retry_on``.

    Network and service failures (``LLMClientError`` marked retryable, ``OSError`` and its
    subclasses such as ``ConnectionError`` and ``TimeoutError``) are retried up to
    ``max_retries`` attempts in total. Any other exception, such as a parse error, a rejected
    request (``LLMClientError`` with ``retryable=False``), ``NotImplementedError`` or another
    programming mistake, is raised immediately, because repeating it cannot help. There is
    no delay between attempts. ``report.retries_used`` counts the retries actually made.

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
        # Note: an empty list is a valid result; only None means "this attempt failed".
        if result is not None:
            return result
        last_err = err
        # The final failed attempt is not a "retry", because nothing follows it.
        if attempt < max_retries:
            rep.retries_used += 1
        rep.errors.append(f"Attempt {attempt} failed: {err}")
    raise RuntimeError(
        f"LLM generation failed after {max_retries} attempts: {last_err}"
    ) from last_err
