"""End-to-end inference protocol and common model interface."""

from absa.inference.interface import (
    AspectPrediction,
    CheckpointNotFoundError,
    InferenceError,
    ModelNotFoundError,
    PredictionRecord,
    PredictionValidationError,
    Predictor,
    clear_registry,
    get_predictor,
    list_available_models,
    predict_validated,
    register_predictor,
    require_checkpoint,
    validate_prediction,
)

__all__ = [
    "AspectPrediction",
    "CheckpointNotFoundError",
    "InferenceError",
    "ModelNotFoundError",
    "PredictionRecord",
    "PredictionValidationError",
    "Predictor",
    "clear_registry",
    "get_predictor",
    "list_available_models",
    "predict_validated",
    "register_predictor",
    "require_checkpoint",
    "validate_prediction",
]
