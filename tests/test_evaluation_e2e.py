"""Tests for given-aspect and end-to-end evaluation."""

from __future__ import annotations

import pytest

from absa.evaluation.e2e import (
    ClassificationMetrics,
    EndToEndMetrics,
    evaluate_end_to_end,
    evaluate_given_aspect_classification,
)


def test_evaluate_given_aspect_classification() -> None:
    gold = ["positive", "negative", "neutral", "positive"]
    pred = ["positive", "negative", "positive", "positive"]

    metrics = evaluate_given_aspect_classification(gold, pred)
    assert isinstance(metrics, ClassificationMetrics)
    assert metrics.accuracy == 0.75
    assert "positive" in metrics.per_class
    assert "negative" in metrics.per_class
    assert metrics.per_class["negative"]["precision"] == 1.0
    assert metrics.per_class["negative"]["recall"] == 1.0
    assert metrics.macro_f1 > 0.0
    assert metrics.to_dict()["accuracy"] == 0.75


def test_evaluate_given_aspect_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        evaluate_given_aspect_classification(["positive"], ["positive", "negative"])


def test_evaluate_given_aspect_empty_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_given_aspect_classification([], [])


def test_evaluate_end_to_end_mismatched_sentence_counts() -> None:
    with pytest.raises(ValueError, match="same number of sentences"):
        evaluate_end_to_end([[(0, 5, "positive")], []], [[(0, 5, "positive")]])


def test_evaluate_end_to_end_matching() -> None:
    gold = [[(0, 5, "positive"), (10, 15, "negative")]]
    pred = [[(0, 5, "positive"), (10, 15, "negative")]]

    metrics = evaluate_end_to_end(gold, pred)
    assert isinstance(metrics, EndToEndMetrics)
    assert metrics.tp == 2
    assert metrics.fp == 0
    assert metrics.fn == 0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0


def test_evaluate_end_to_end_span_match_sentiment_mismatch() -> None:
    gold = [[(0, 5, "positive")]]
    pred = [[(0, 5, "negative")]]  # Span matches, but sentiment differs

    metrics = evaluate_end_to_end(gold, pred)
    assert metrics.tp == 0
    assert metrics.fp == 1
    assert metrics.fn == 1
    assert metrics.f1 == 0.0


def test_evaluate_end_to_end_empty() -> None:
    metrics = evaluate_end_to_end([], [])
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_end_to_end_compares_sentiment_exactly() -> None:
    metrics = evaluate_end_to_end([[(0, 5, "positive")]], [[(0, 5, "Positive")]])
    assert (metrics.tp, metrics.fp, metrics.fn) == (0, 1, 1)
