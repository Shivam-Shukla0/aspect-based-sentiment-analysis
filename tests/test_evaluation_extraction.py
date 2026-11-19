"""Tests for aspect extraction span evaluation."""

from __future__ import annotations

from absa.evaluation.extraction import (
    ExtractionMetrics,
    compute_span_counts,
    evaluate_extraction_spans,
)


def test_compute_span_counts_perfect() -> None:
    gold = [(0, 5), (10, 15)]
    pred = [(0, 5), (10, 15)]
    tp, fp, fn = compute_span_counts(gold, pred, exact_match=True)
    assert tp == 2
    assert fp == 0
    assert fn == 0


def test_compute_span_counts_exact_vs_relaxed() -> None:
    gold = [(0, 10)]
    pred = [(0, 8)]  # Sub-span overlap

    tp_exact, fp_exact, fn_exact = compute_span_counts(gold, pred, exact_match=True)
    assert tp_exact == 0
    assert fp_exact == 1
    assert fn_exact == 1

    tp_rel, fp_rel, fn_rel = compute_span_counts(gold, pred, exact_match=False)
    assert tp_rel == 1
    assert fp_rel == 0
    assert fn_rel == 0


def test_evaluate_extraction_spans() -> None:
    gold_list = [[(0, 5)], [(10, 20), (25, 30)]]
    pred_list = [[(0, 5)], [(10, 20)]]

    metrics = evaluate_extraction_spans(gold_list, pred_list, exact_match=True)
    assert isinstance(metrics, ExtractionMetrics)
    assert metrics.tp == 2
    assert metrics.fp == 0
    assert metrics.fn == 1
    assert metrics.precision == 1.0
    assert metrics.recall == round(2 / 3, 4)
    assert metrics.f1 == round((2 * 1.0 * (2 / 3)) / (1.0 + 2 / 3), 4)
    assert metrics.to_dict()["tp"] == 2


def test_evaluate_extraction_spans_empty() -> None:
    metrics = evaluate_extraction_spans([], [], exact_match=True)
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
