"""Tests for LLM prediction validation and retry error handling."""

from __future__ import annotations

import pytest

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


@pytest.mark.parametrize("max_retries", [0, -1])
def test_retry_llm_inference_rejects_non_positive_limit(max_retries: int) -> None:
    calls = 0

    def generate() -> list[dict[str, str]]:
        nonlocal calls
        calls += 1
        return []

    with pytest.raises(ValueError, match="max_retries must be at least 1"):
        retry_llm_inference(generate, max_retries=max_retries)
    assert calls == 0


def test_is_successful_with_no_candidates() -> None:
    report = ValidationReport()
    assert validate_aspect_predictions("Any text.", [], report=report) == []
    assert report.total_candidates == 0
    assert report.is_successful is True


def test_is_successful_all_valid() -> None:
    report = ValidationReport()
    raw = [{"aspect": "food", "sentiment": "positive"}]
    validate_aspect_predictions("Good food.", raw, report=report)
    assert report.is_successful is True


def test_is_successful_false_when_some_invalid() -> None:
    report = ValidationReport()
    raw = [
        {"aspect": "food", "sentiment": "positive"},
        {"aspect": "ambience", "sentiment": "positive"},
    ]
    validated = validate_aspect_predictions("Good food.", raw, report=report)
    assert len(validated) == 1
    assert report.valid_count == 1
    assert report.invalid_count == 1
    assert report.is_successful is False


def test_is_successful_false_when_all_invalid() -> None:
    report = ValidationReport()
    raw = [{"aspect": "ambience", "sentiment": "positive"}]
    assert validate_aspect_predictions("Good food.", raw, report=report) == []
    assert report.valid_count == 0
    assert report.is_successful is False


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
        raise ValueError("Unrecoverable error")

    report = ValidationReport()
    with pytest.raises(RuntimeError, match="LLM generation failed after 2 attempts"):
        retry_llm_inference(broken_generate, max_retries=2, report=report)
    assert report.retries_used == 2
