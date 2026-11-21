"""Predictors for the given-aspect classifiers (LSTM and BERT).

Both trained models classify the sentiment of an aspect term that is supplied with the
review. They do not find aspects themselves, so ``predict`` (joint extraction and
classification) is not available and says so; ``predict_aspects`` takes the terms.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from absa.inference.interface import (
    AspectPrediction,
    InferenceError,
    PredictionRecord,
    require_checkpoint,
    validate_prediction_record,
)

Classify = Callable[[str, str], str]


class AspectClassifierPredictor:
    """Wrap a ``(review, aspect) -> label`` function in the shared predictor interface."""

    def __init__(self, classify: Classify) -> None:
        """Store the classification function."""
        self._classify = classify

    def predict(self, review: str) -> PredictionRecord:
        """Refuse joint prediction: these models need the aspect terms."""
        raise InferenceError(
            f"this model classifies supplied aspect terms and cannot find them in {review!r}; "
            "use predict_aspects"
        )

    def predict_aspect(self, review: str, aspect: str) -> str:
        """Return the sentiment label of ``aspect`` in ``review``."""
        try:
            return self._classify(review, aspect)
        except InferenceError:
            raise
        except Exception as err:
            raise InferenceError(f"Prediction failed: {err}") from err

    def predict_aspects(self, review: str, aspects: Sequence[str]) -> PredictionRecord:
        """Classify every term in ``aspects`` and return a validated record.

        Raises:
            InferenceError: If the review or an aspect is empty, or the model fails.
            PredictionValidationError: If an aspect does not occur in the review.
        """
        if not review.strip() or not aspects or any(not aspect.strip() for aspect in aspects):
            raise InferenceError("a review and at least one non-empty aspect are required")
        predictions = tuple(
            AspectPrediction(aspect, self.predict_aspect(review, aspect)) for aspect in aspects
        )
        return validate_prediction_record(PredictionRecord(review, predictions))


def load_lstm_predictor(checkpoint: str | Path, device: str = "cpu") -> AspectClassifierPredictor:
    """Load an LSTM checkpoint written by training and wrap it as a predictor.

    Raises:
        CheckpointNotFoundError: If ``checkpoint`` does not exist.
    """
    path = require_checkpoint(checkpoint)
    from absa.models import load_checkpoint, predict  # noqa: PLC0415  (needs torch)

    model, vocab = load_checkpoint(path)

    def classify(review: str, aspect: str) -> str:
        # The dataset class reads a label for every record; it is not used for prediction.
        record = {"review": review, "aspect": aspect, "sentiment": "neutral"}
        return predict(model, vocab, [record], device=device)[0]

    return AspectClassifierPredictor(classify)


def load_bert_predictor(model_dir: str | Path, device: str = "cpu") -> AspectClassifierPredictor:
    """Load a fine-tuned BERT directory written by training and wrap it as a predictor.

    Raises:
        CheckpointNotFoundError: If ``model_dir`` does not exist.
    """
    path = require_checkpoint(model_dir)
    from absa.models import BertConfig, load_bert, predict_bert  # noqa: PLC0415  (needs torch)

    model, tokenizer, labels = load_bert(path)
    config = BertConfig(device=device, bf16=False, drop_conflict="conflict" not in labels)

    def classify(review: str, aspect: str) -> str:
        return predict_bert(model, tokenizer, [{"review": review, "aspect": aspect}], config)[0]

    return AspectClassifierPredictor(classify)
