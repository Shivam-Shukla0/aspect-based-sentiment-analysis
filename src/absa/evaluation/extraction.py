"""Span-level precision, recall and F1 metrics for aspect term extraction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

Span = tuple[int, int]


@dataclass(frozen=True)
class ExtractionMetrics:
    """Evaluation metrics for aspect span extraction."""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

    def to_dict(self) -> dict[str, float | int]:
        """Return metrics as a dictionary."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


def compute_span_counts(
    gold_spans: Sequence[Span],
    pred_spans: Sequence[Span],
    *,
    exact_match: bool = True,
) -> tuple[int, int, int]:
    """Calculate true positives, false positives, and false negatives between spans.

    Args:
        gold_spans: Gold character offset spans [(start, end), ...].
        pred_spans: Predicted character offset spans [(start, end), ...].
        exact_match: If True, requires exact boundary match; if False, requires overlap.

    Returns:
        Tuple of (true_positives, false_positives, false_negatives).

    Note:
        Matching is greedy and one-to-one: each predicted span, in the order given, claims
        the first unmatched gold span it matches. With ``exact_match=True`` the result does
        not depend on order. With overlap matching it can: a prediction that overlaps two
        gold spans takes the earlier one, which may leave a later prediction with no match
        even though a different pairing would score higher. Pass predictions sorted by
        start offset for reproducible results.
    """
    matched_gold: set[int] = set()
    tp = 0

    for p_start, p_end in pred_spans:
        for g_idx, (g_start, g_end) in enumerate(gold_spans):
            if g_idx in matched_gold:
                continue
            is_match = (
                (p_start == g_start and p_end == g_end)
                if exact_match
                else (max(p_start, g_start) < min(p_end, g_end))
            )
            if is_match:
                matched_gold.add(g_idx)
                tp += 1
                break

    fp = len(pred_spans) - tp
    fn = len(gold_spans) - tp
    return tp, fp, fn


def evaluate_extraction_spans(
    gold_spans_list: Sequence[Sequence[Span]],
    pred_spans_list: Sequence[Sequence[Span]],
    *,
    exact_match: bool = True,
) -> ExtractionMetrics:
    """Evaluate aspect span extraction across multiple sentences.

    Args:
        gold_spans_list: Sequence of gold span sequences per sentence.
        pred_spans_list: Sequence of predicted span sequences per sentence.
        exact_match: If True, requires exact boundary match.

    Returns:
        ExtractionMetrics containing precision, recall, and F1 (unrounded).

    Raises:
        ValueError: If the two lists do not cover the same number of sentences.
    """
    if len(gold_spans_list) != len(pred_spans_list):
        raise ValueError(
            "gold and predicted span lists must have the same number of sentences: "
            f"{len(gold_spans_list)} != {len(pred_spans_list)}"
        )
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for gold, pred in zip(gold_spans_list, pred_spans_list, strict=True):
        tp, fp, fn = compute_span_counts(gold, pred, exact_match=exact_match)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    denom_p = total_tp + total_fp
    denom_r = total_tp + total_fn
    precision = total_tp / denom_p if denom_p > 0 else 0.0
    recall = total_tp / denom_r if denom_r > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return ExtractionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        tp=total_tp,
        fp=total_fp,
        fn=total_fn,
    )
