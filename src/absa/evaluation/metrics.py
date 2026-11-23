"""Shared precision/recall/F1 evaluation, used by every modelling approach.

Every approach (LSTM, BERT, LLM) must be scored with ``evaluate`` so results
are comparable. ``zero_division=0`` is used throughout: a class with no true
or no predicted examples gets a score of zero instead of a warning or a
``ZeroDivisionError``, which matters because this project treats warnings as
errors in tests.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from absa.config import SENTIMENT_LABELS


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    """Precision, recall, F1 and support for one class."""

    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Full evaluation result: per-class metrics plus macro/micro F1 and accuracy."""

    labels: tuple[str, ...]
    accuracy: float
    macro_f1: float
    micro_f1: float
    per_class: tuple[ClassMetrics, ...]
    report: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation of this result."""
        return {
            "labels": list(self.labels),
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "micro_f1": self.micro_f1,
            "per_class": [
                {
                    "label": class_metrics.label,
                    "precision": class_metrics.precision,
                    "recall": class_metrics.recall,
                    "f1": class_metrics.f1,
                    "support": class_metrics.support,
                }
                for class_metrics in self.per_class
            ],
            "report": self.report,
        }

    def save(self, path: Path) -> None:
        """Write this result as JSON to ``path``, creating parent directories."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


# Imbalance: positive is 53.4% of laptop and 65.0% of restaurant test records, so always
# answering positive already gets that accuracy. Macro F1 (the average of per-class F1) treats
# each class equally, and results are compared against this majority baseline. A class-weighted
# loss was also tried for the LSTM; the best validation setting was the unweighted one.
#
# Metrics per task: sentiment uses accuracy and macro F1 with per-class precision, recall, F1
# and a confusion matrix. Extraction uses precision, recall, F1 by exact span match (start and
# end must match), plus overlap match as a looser figure. End-to-end counts a prediction only
# if span and sentiment both match.
def evaluate(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = SENTIMENT_LABELS
) -> EvaluationResult:
    """Score predictions against gold labels using a fixed label order.

    ``labels`` fixes both the per-class output order and which classes are
    scored; a sample whose gold or predicted label is outside ``labels`` is
    not counted toward any class's precision or recall, but still counts
    toward accuracy. Classes with no true or no predicted examples score
    zero rather than raising or warning. scikit-learn is imported here, not at
    module level, so importing this package does not require it.
    """
    # Imported lazily so the package imports without scikit-learn installed.
    from sklearn.metrics import (  # noqa: PLC0415  (optional dependency, see docstring)
        accuracy_score,
        classification_report,
        precision_recall_fscore_support,
    )

    # Guard inputs first: sklearn would otherwise fail with a less helpful message.
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length: {len(y_true)} != {len(y_pred)}"
        )
    if not y_true:
        raise ValueError("y_true and y_pred must not be empty")
    labels = tuple(labels)  # freeze so the order is fixed for every metric below
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    # One per-class call feeds both the per-class breakdown and macro F1;
    # micro F1 is a single scalar call.
    # Macro F1 = unweighted mean of per-class F1: every class counts equally, so a rare
    # class (e.g. "conflict") weighs as much as a frequent one. Micro F1 pools all samples.
    macro_f1 = sum(float(value) for value in f1) / len(f1) if len(f1) else 0.0
    micro_f1 = float(
        precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average="micro", zero_division=0
        )[2]
    )
    per_class = tuple(
        ClassMetrics(
            label=label,
            precision=float(precision[index]),
            recall=float(recall[index]),
            f1=float(f1[index]),
            support=int(support[index]),
        )
        for index, label in enumerate(labels)
    )
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    return EvaluationResult(
        labels=labels,
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(macro_f1),
        micro_f1=float(micro_f1),
        per_class=per_class,
        report=str(report),
    )
