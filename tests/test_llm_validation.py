"""Tests for LLM prediction validation and retry error handling."""

from __future__ import annotations

import pytest

from absa.models.llm_errors import LLMClientError
from absa.models.llm_validation import (
    ValidatedAspect,
    ValidationReport,
    retry_llm_inference,
    validate_aspect_predictions,
)


def test_validate_aspect_predictions_valid() -> None:
    review = "The battery life is stellar, but the screen is dim."
    raw = [
        {"aspect": "battery life", "sentiment": "positive"},
        {"aspect": "screen", "sentiment": "negative"},
    ]
    report = ValidationReport()
    validated = validate_aspect_predictions(review, raw, report=report)

    assert len(validated) == 2
    assert isinstance(validated[0], ValidatedAspect)
    assert validated[0].aspect == "battery life"
    assert validated[0].sentiment == "positive"
    assert review[validated[0].start : validated[0].end] == "battery life"
    assert report.valid_count == 2
    assert report.invalid_count == 0
    assert report.is_successful is True


def test_validate_aspect_predictions_invalid_sentiment() -> None:
    review = "Good food."
    raw = [{"aspect": "food", "sentiment": "superb"}]  # Not in allowed labels
    report = ValidationReport()
    validated = validate_aspect_predictions(review, raw, report=report)

    assert len(validated) == 0
    assert report.invalid_count == 1
    assert any("Invalid sentiment label" in err for err in report.errors)
    assert report.is_successful is False


def test_validate_aspect_predictions_hallucinated_aspect() -> None:
    review = "The pasta was great."
    raw = [{"aspect": "ambience", "sentiment": "positive"}]  # Not in review
    report = ValidationReport()
    validated = validate_aspect_predictions(review, raw, report=report)

    assert len(validated) == 0
    assert report.invalid_count == 1
    assert any("not found in review text" in err for err in report.errors)


def test_retry_llm_inference_eventual_success() -> None:
    attempts = 0

    def flaky_generate() -> list[dict[str, str]]:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ConnectionError("Temporary timeout")
        return [{"aspect": "sound", "sentiment": "positive"}]

    report = ValidationReport()
    results = retry_llm_inference(flaky_generate, max_retries=3, report=report)
    assert len(results) == 1
    assert report.retries_used == 1


def test_retry_llm_inference_exhausted() -> None:
    def broken_generate() -> list[dict[str, str]]:
        raise ConnectionError("Service unavailable")

    report = ValidationReport()
    with pytest.raises(RuntimeError, match="LLM generation failed after 2 attempts"):
        retry_llm_inference(broken_generate, max_retries=2, report=report)
    assert report.retries_used == 1  # two attempts, one retry


def test_retry_does_not_repeat_non_transient_errors() -> None:
    attempts = 0

    def bad_output() -> list[dict[str, str]]:
        nonlocal attempts
        attempts += 1
        raise ValueError("Could not parse response")

    report = ValidationReport()
    with pytest.raises(ValueError, match="Could not parse"):
        retry_llm_inference(bad_output, max_retries=3, report=report)
    assert attempts == 1
    assert report.retries_used == 0


@pytest.mark.parametrize("bad", [0, -1])
def test_retry_rejects_non_positive_max_retries(bad: int) -> None:
    with pytest.raises(ValueError, match="max_retries"):
        retry_llm_inference(list, max_retries=bad)


def test_repeated_aspect_gets_successive_spans() -> None:
    review = "The screen is bright but the screen is small."
    raw = [
        {"aspect": "screen", "sentiment": "positive"},
        {"aspect": "screen", "sentiment": "negative"},
    ]
    validated = validate_aspect_predictions(review, raw)
    assert [(v.start, v.end) for v in validated] == [(4, 10), (29, 35)]
    assert all(review[v.start : v.end] == v.aspect for v in validated)


def test_third_mention_of_a_twice_mentioned_aspect_is_rejected() -> None:
    review = "The screen is bright but the screen is small."
    raw = [{"aspect": "screen", "sentiment": "positive"}] * 3
    report = ValidationReport()
    validated = validate_aspect_predictions(review, raw, report=report)
    assert len(validated) == 2
    assert report.invalid_count == 1


def test_case_insensitive_fallback_keeps_the_review_spelling() -> None:
    review = "Battery Life is great."
    validated = validate_aspect_predictions(
        review, [{"aspect": "battery life", "sentiment": "positive"}]
    )
    assert [(v.aspect, v.start, v.end) for v in validated] == [("Battery Life", 0, 12)]


def test_offsets_stay_correct_when_lowercasing_changes_the_length() -> None:
    review = "\u0130stanbul has a screen and the screen is bright."
    validated = validate_aspect_predictions(review, [{"aspect": "SCREEN", "sentiment": "positive"}])
    assert len(validated) == 1
    assert review[validated[0].start : validated[0].end] == validated[0].aspect == "screen"
    assert validated[0].start == review.index("screen")


def test_runtime_error_subclasses_that_are_bugs_are_not_retried() -> None:
    attempts = 0

    def not_implemented() -> list[dict[str, str]]:
        nonlocal attempts
        attempts += 1
        raise NotImplementedError

    with pytest.raises(NotImplementedError):
        retry_llm_inference(not_implemented, max_retries=3)
    assert attempts == 1


def test_client_errors_marked_retryable_are_retried() -> None:
    calls = 0

    def flaky() -> list[dict[str, str]]:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise LLMClientError("busy")
        return []

    report = ValidationReport()
    assert retry_llm_inference(flaky, max_retries=3, report=report) == []
    assert report.retries_used == 2


def test_client_errors_marked_not_retryable_are_raised_at_once() -> None:
    calls = 0

    def rejected() -> list[dict[str, str]]:
        nonlocal calls
        calls += 1
        raise LLMClientError("bad key", retryable=False)

    with pytest.raises(LLMClientError, match="bad key"):
        retry_llm_inference(rejected, max_retries=3)
    assert calls == 1


def test_nested_aspects_keep_only_the_first_listed() -> None:
    review = "Battery life is great."
    raw = [
        {"aspect": "Battery", "sentiment": "positive"},
        {"aspect": "Battery life", "sentiment": "positive"},
    ]
    report = ValidationReport()
    validated = validate_aspect_predictions(review, raw, report=report)
    assert [v.aspect for v in validated] == ["Battery"]
    assert report.invalid_count == 1
