"""Evaluate a trained LSTM checkpoint and analyse its prediction errors.

Uses the shared metrics module (``absa.evaluation``) so results are
comparable with the other approaches, and adds the confusion matrix and
per-example error analysis this model's evaluation issue asks for.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix

from absa.config import SENTIMENT_LABELS
from absa.evaluation import EvaluationResult, evaluate
from absa.models.lstm_attention import AbsaLstmModel
from absa.models.lstm_dataset import AspectDataset, make_dataloader
from absa.models.lstm_tokens import Vocabulary


@dataclass(frozen=True, slots=True)
class PredictionError:
    """One misclassified (review, aspect) example."""

    review: str
    aspect: str
    true_label: str
    predicted_label: str


@dataclass(frozen=True, slots=True)
class ErrorAnalysis:
    """Every misclassified example, with counts by (true, predicted) label pair."""

    errors: tuple[PredictionError, ...]

    @property
    def counts_by_pair(self) -> dict[str, int]:
        """Return the number of errors for each ``"true->predicted"`` label pair."""
        pairs = Counter(f"{error.true_label}->{error.predicted_label}" for error in self.errors)
        return dict(pairs)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation."""
        return {
            "total_errors": len(self.errors),
            "counts_by_pair": self.counts_by_pair,
            "examples": [
                {
                    "review": error.review,
                    "aspect": error.aspect,
                    "true_label": error.true_label,
                    "predicted_label": error.predicted_label,
                }
                for error in self.errors
            ],
        }


@dataclass(frozen=True, slots=True)
class TestEvaluationReport:
    """Full test-set evaluation: shared metrics, a confusion matrix and error analysis."""

    metrics: EvaluationResult
    confusion: list[list[int]]
    labels: tuple[str, ...]
    errors: ErrorAnalysis

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation of the full report."""
        return {
            "metrics": self.metrics.to_dict(),
            "confusion_matrix": self.confusion,
            "labels": list(self.labels),
            "errors": self.errors.to_dict(),
        }

    def save(self, path: Path) -> None:
        """Write this report as JSON to ``path``, creating parent directories."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def predict(
    model: AbsaLstmModel, vocab: Vocabulary, records: Sequence[Mapping[str, object]]
) -> list[str]:
    """Run inference over ``records`` and return one predicted label per record, in order."""
    model.eval()
    loader = make_dataloader(AspectDataset(records, vocab), vocab.pad_id, batch_size=32)
    predictions: list[str] = []
    with torch.no_grad():
        for batch in loader:
            logits, _ = model(
                batch["review_ids"],
                batch["review_lengths"],
                batch["aspect_ids"],
                batch["aspect_lengths"],
            )
            predictions.extend(SENTIMENT_LABELS[index] for index in logits.argmax(dim=-1).tolist())
    return predictions


def analyze_errors(
    records: Sequence[Mapping[str, object]], y_true: Sequence[str], y_pred: Sequence[str]
) -> ErrorAnalysis:
    """Collect every record whose predicted label differs from its true label."""
    errors = [
        PredictionError(str(record["review"]), str(record["aspect"]), true, predicted)
        for record, true, predicted in zip(records, y_true, y_pred, strict=True)
        if true != predicted
    ]
    return ErrorAnalysis(tuple(errors))


def evaluate_test_set(
    model: AbsaLstmModel,
    vocab: Vocabulary,
    records: Sequence[Mapping[str, object]],
    labels: Sequence[str] = SENTIMENT_LABELS,
) -> TestEvaluationReport:
    """Evaluate ``model`` on ``records`` (the frozen test set): metrics, confusion, errors."""
    y_true = [str(record["sentiment"]) for record in records]
    y_pred = predict(model, vocab, records)
    metrics = evaluate(y_true, y_pred, labels=labels)
    confusion = confusion_matrix(y_true, y_pred, labels=list(labels)).tolist()
    errors = analyze_errors(records, y_true, y_pred)
    return TestEvaluationReport(
        metrics=metrics, confusion=confusion, labels=tuple(labels), errors=errors
    )
