"""Common model inference interface, predictor protocol and model registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from absa.config import SENTIMENT_LABELS


class InferenceError(Exception):
    """Base exception for all inference errors."""


class ModelNotFoundError(InferenceError):
    """Raised when an unknown model name is requested."""


class PredictionValidationError(InferenceError):
    """Raised when model output fails validation."""


class CheckpointNotFoundError(InferenceError):
    """Raised when a model checkpoint file does not exist."""


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


def clear_registry() -> None:
    """Clear all registered predictor factories (primarily used in testing)."""
    _REGISTRY.clear()


def require_checkpoint(path: Path) -> Path:
    """Return ``path`` if it is an existing checkpoint file.

    Raises:
        CheckpointNotFoundError: If no file exists at ``path``.
    """
    if not path.is_file():
        raise CheckpointNotFoundError(f"Checkpoint not found: {path}")
    return path


def validate_prediction(record: PredictionRecord) -> PredictionRecord:
    """Return ``record`` if it follows the shared inference contract.

    The review and every aspect must be non-empty text and every sentiment one of
    ``absa.config.SENTIMENT_LABELS``. An empty prediction list is valid: it means no
    aspect was found.

    Raises:
        PredictionValidationError: If any part of the record is invalid.
    """
    if not isinstance(record.review, str) or not record.review.strip():
        raise PredictionValidationError("review must be a non-empty string")
    for index, prediction in enumerate(record.predictions):
        if not isinstance(prediction.aspect, str) or not prediction.aspect.strip():
            raise PredictionValidationError(f"predictions[{index}] has an empty aspect")
        if prediction.sentiment not in SENTIMENT_LABELS:
            raise PredictionValidationError(
                f"predictions[{index}] has unknown sentiment {prediction.sentiment!r}; "
                f"expected one of {SENTIMENT_LABELS}"
            )
    return record


def predict_validated(predictor: Predictor, review: str) -> PredictionRecord:
    """Run ``predictor`` on ``review`` and return a validated result.

    Raises:
        PredictionValidationError: If the output breaks the contract.
        InferenceError: If the predictor itself fails for any other reason.
    """
    try:
        record = predictor.predict(review)
    except InferenceError:
        raise
    except Exception as err:  # any backend failure becomes a typed error
        raise InferenceError(f"Prediction failed: {err}") from err
    return validate_prediction(record)
