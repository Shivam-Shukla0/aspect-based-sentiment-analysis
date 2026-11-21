"""Evaluation framework for given-aspect sentiment classification and end-to-end ABSA."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from absa.config import SENTIMENT_LABELS

AspectSentimentPair = tuple[int, int, str]


@dataclass(frozen=True)
class ClassificationMetrics:
    """Metrics for given-aspect sentiment classification."""

    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, object]:
        """Return metrics as a dictionary."""
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "per_class": self.per_class,
        }


@dataclass(frozen=True)
class EndToEndMetrics:
    """Metrics for joint aspect extraction and sentiment prediction."""

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


def evaluate_given_aspect_classification(
    gold_labels: Sequence[str],
    pred_labels: Sequence[str],
    labels: Sequence[str] = SENTIMENT_LABELS,
) -> ClassificationMetrics:
    """Compute accuracy, macro F1, and per-class precision/recall/F1 for sentiment labels."""
    if not gold_labels:
        raise ValueError("gold_labels must not be empty")
    if len(gold_labels) != len(pred_labels):
        raise ValueError(
            f"gold and predicted labels must have the same length: "
            f"{len(gold_labels)} != {len(pred_labels)}"
        )

    correct = sum(1 for g, p in zip(gold_labels, pred_labels, strict=True) if g == p)
    accuracy = correct / len(gold_labels)

    per_class: dict[str, dict[str, float]] = {}
    f1_scores: list[float] = []

    for lbl in labels:
        tp = sum(1 for g, p in zip(gold_labels, pred_labels, strict=True) if g == lbl and p == lbl)
        fp = sum(1 for g, p in zip(gold_labels, pred_labels, strict=True) if g != lbl and p == lbl)
        fn = sum(1 for g, p in zip(gold_labels, pred_labels, strict=True) if g == lbl and p != lbl)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[lbl] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }
        if (tp + fn) > 0:
            f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return ClassificationMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class=per_class,
    )


def evaluate_end_to_end(
    gold_pairs_list: Sequence[Sequence[AspectSentimentPair]],
    pred_pairs_list: Sequence[Sequence[AspectSentimentPair]],
) -> EndToEndMetrics:
    """Evaluate end-to-end ABSA requiring both span offsets and sentiment to match exactly."""
    if len(gold_pairs_list) != len(pred_pairs_list):
        raise ValueError(
            "gold and predicted pair lists must have the same number of sentences: "
            f"{len(gold_pairs_list)} != {len(pred_pairs_list)}"
        )
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for gold_pairs, pred_pairs in zip(gold_pairs_list, pred_pairs_list, strict=True):
        matched_gold: set[int] = set()
        tp = 0

        for p_start, p_end, p_sent in pred_pairs:
            for g_idx, (g_start, g_end, g_sent) in enumerate(gold_pairs):
                if g_idx in matched_gold:
                    continue
                if p_start == g_start and p_end == g_end and p_sent.lower() == g_sent.lower():
                    matched_gold.add(g_idx)
                    tp += 1
                    break

        total_tp += tp
        total_fp += len(pred_pairs) - tp
        total_fn += len(gold_pairs) - tp

    denom_p = total_tp + total_fp
    denom_r = total_tp + total_fn
    precision = total_tp / denom_p if denom_p > 0 else 0.0
    recall = total_tp / denom_r if denom_r > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return EndToEndMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        tp=total_tp,
        fp=total_fp,
        fn=total_fn,
    )
