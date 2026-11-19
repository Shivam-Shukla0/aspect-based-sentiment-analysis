"""Common model inference interface, predictor protocol and model registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


class InferenceError(Exception):
    """Base exception for all inference errors."""


class ModelNotFoundError(InferenceError):
    """Raised when an unknown model name is requested."""


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


def clear_registry() -> None:
    """Clear all registered predictor factories (primarily used in testing)."""
    _REGISTRY.clear()
