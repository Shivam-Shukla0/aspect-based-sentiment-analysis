"""Validation, error handling and retry wrapper for LLM predictions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from absa.config import SENTIMENT_LABELS


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
    retries_used: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def is_successful(self) -> bool:
        """Return True if at least one valid prediction exists or zero candidates were given."""
        return self.invalid_count == 0


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

        start = review.find(aspect)
        if start == -1:
            rep.invalid_count += 1
            rep.errors.append(f"Aspect {aspect!r} not found in review text")
            continue

        end = start + len(aspect)
        validated.append(ValidatedAspect(aspect=aspect, sentiment=sentiment, start=start, end=end))
        rep.valid_count += 1

    return validated


def _attempt_generate(
    generate_fn: Callable[[], list[dict[str, str]]],
) -> tuple[list[dict[str, str]] | None, Exception | None]:
    """Execute a single attempt of generation catching exceptions."""
    try:
        return generate_fn(), None
    except Exception as err:  # noqa: BLE001
        return None, err


def retry_llm_inference(
    generate_fn: Callable[[], list[dict[str, str]]],
    max_retries: int = 3,
    report: ValidationReport | None = None,
) -> list[dict[str, str]]:
    """Execute LLM generation function with bounded retries on failure."""
    rep = report if report is not None else ValidationReport()
    last_err: Exception | None = None

    for attempt in range(max_retries):
        result, err = _attempt_generate(generate_fn)
        if err is None and result is not None:
            return result
        last_err = err
        rep.retries_used += 1
        rep.errors.append(f"Attempt {attempt + 1} failed: {err}")

    if last_err:
        raise RuntimeError(
            f"LLM generation failed after {max_retries} attempts: {last_err}"
        ) from last_err
    return []
