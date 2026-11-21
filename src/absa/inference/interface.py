"""Common model inference interface, predictor protocol and model registry."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from absa.config import SENTIMENT_LABELS
from absa.data.schema import validate_record


class InferenceError(Exception):
    """Base exception for all inference errors."""


class ModelNotFoundError(InferenceError):
    """Raised when an unknown model name is requested."""


class CheckpointNotFoundError(InferenceError):
    """Raised when a model checkpoint file or directory does not exist."""


class PredictionValidationError(InferenceError):
    """Raised when model output fails validation."""


@dataclass(frozen=True)
class AspectPrediction:
    """A predicted aspect term and sentiment."""

    aspect: str
    sentiment: str

    def to_dict(self) -> dict[str, str]:
        """Return as dictionary."""
        return {"aspect": self.aspect, "sentiment": self.sentiment}


@dataclass(frozen=True)
class PredictionRecord:
    """Shared inference output contract for a single review."""

    review: str
    predictions: tuple[AspectPrediction, ...]

    def to_dict(self) -> dict[str, object]:
        """Return as dictionary conforming to shared inference contract."""
        return {
            "review": self.review,
            "predictions": [p.to_dict() for p in self.predictions],
        }


class Predictor(Protocol):
    """Protocol implemented by all model inference backends."""

    def predict(self, review: str) -> PredictionRecord:
        """Perform joint aspect extraction and sentiment prediction for a review."""
        ...

    def predict_aspect(self, review: str, aspect: str) -> str:
        """Classify sentiment for a known, supplied aspect in the review."""
        ...


_REGISTRY: dict[str, Callable[[], Predictor]] = {}


def register_predictor(name: str, factory: Callable[[], Predictor]) -> None:
    """Register a predictor factory under a model name."""
    _REGISTRY[name.lower().strip()] = factory


def get_predictor(name: str) -> Predictor:
    """Instantiate and return a predictor by registered name.

    Raises:
        ModelNotFoundError: If name is not registered.
    """
    key = name.lower().strip()
    if key not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise ModelNotFoundError(f"Model {name!r} is not registered. Available models: {available}")
    return _REGISTRY[key]()


def list_available_models() -> list[str]:
    """Return sorted list of currently registered model names."""
    return sorted(_REGISTRY.keys())


def require_checkpoint(path: str | Path) -> Path:
    """Return ``path`` as a ``Path`` if it exists.

    Raises:
        CheckpointNotFoundError: If the checkpoint is missing.
    """
    checkpoint = Path(path)
    if not checkpoint.exists():
        raise CheckpointNotFoundError(f"Checkpoint not found: {checkpoint}")
    return checkpoint


def validate_prediction_record(
    record: PredictionRecord, allowed_labels: Sequence[str] = SENTIMENT_LABELS
) -> PredictionRecord:
    """Return ``record`` unchanged if it satisfies the shared record contract.

    The review must be a non-empty string, and every prediction needs a non-empty
    aspect that occurs in the review (case-insensitive) and a sentiment from
    ``allowed_labels``.

    Raises:
        PredictionValidationError: If the record or any prediction is invalid.
    """
    if not isinstance(record, PredictionRecord):
        raise PredictionValidationError(f"expected a PredictionRecord, got {type(record).__name__}")
    for prediction in record.predictions:
        try:
            validate_record(
                {
                    "review": record.review,
                    "aspect": prediction.aspect,
                    "sentiment": prediction.sentiment,
                },
                allowed_labels,
            )
        except ValueError as err:
            raise PredictionValidationError(f"invalid prediction {prediction!r}: {err}") from err
        if prediction.aspect.lower() not in record.review.lower():
            raise PredictionValidationError(
                f"aspect {prediction.aspect!r} does not occur in the review"
            )
    return record


def predict_validated(predictor: Predictor, review: str) -> PredictionRecord:
    """Run ``predictor.predict`` and validate its output.

    Raises:
        PredictionValidationError: If the predictor output is invalid.
    """
    return validate_prediction_record(predictor.predict(review))
