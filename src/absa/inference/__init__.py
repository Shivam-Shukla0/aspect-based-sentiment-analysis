"""End-to-end inference protocol and common model interface."""

from absa.inference.interface import (
    AspectPrediction,
    InferenceError,
    ModelNotFoundError,
    PredictionRecord,
    PredictionValidationError,
    Predictor,
    clear_registry,
    get_predictor,
    list_available_models,
    register_predictor,
)

__all__ = [
    "AspectPrediction",
    "InferenceError",
    "ModelNotFoundError",
    "PredictionRecord",
    "PredictionValidationError",
    "Predictor",
    "clear_registry",
    "get_predictor",
    "list_available_models",
    "register_predictor",
]
