"""End-to-end inference protocol and common model interface."""

from absa.inference.classifiers import (
    AspectClassifierPredictor,
    load_bert_predictor,
    load_lstm_predictor,
)
from absa.inference.interface import (
    AspectPrediction,
    CheckpointNotFoundError,
    DuplicateModelError,
    InferenceError,
    ModelNotFoundError,
    PredictionRecord,
    PredictionValidationError,
    Predictor,
    get_predictor,
    list_available_models,
    predict_validated,
    register_predictor,
    require_checkpoint,
    validate_prediction_record,
)

__all__ = [
    "AspectClassifierPredictor",
    "AspectPrediction",
    "CheckpointNotFoundError",
    "DuplicateModelError",
    "InferenceError",
    "ModelNotFoundError",
    "PredictionRecord",
    "PredictionValidationError",
    "Predictor",
    "get_predictor",
    "list_available_models",
    "load_bert_predictor",
    "load_lstm_predictor",
    "predict_validated",
    "register_predictor",
    "require_checkpoint",
    "validate_prediction_record",
]
