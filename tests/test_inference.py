"""Tests for common inference interface and model registry."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from absa.inference.interface import (
    AspectPrediction,
    CheckpointNotFoundError,
    InferenceError,
    ModelNotFoundError,
    PredictionRecord,
    PredictionValidationError,
    clear_registry,
    get_predictor,
    list_available_models,
    predict_validated,
    register_predictor,
    require_checkpoint,
    validate_prediction,
)


class DummyStubPredictor:
    """Dummy predictor for testing."""

    def predict(self, review: str) -> PredictionRecord:
        return PredictionRecord(
            review=review,
            predictions=(AspectPrediction(aspect="sound", sentiment="positive"),),
        )

    def predict_aspect(self, review: str, aspect: str) -> str:
        del review, aspect
        return "positive"


@pytest.fixture(autouse=True)
def _reset_registry() -> Generator[None, None, None]:
    clear_registry()
    yield
    clear_registry()


def test_register_and_get_predictor() -> None:
    register_predictor("dummy", DummyStubPredictor)
    assert list_available_models() == ["dummy"]

    predictor = get_predictor("dummy")
    assert isinstance(predictor, DummyStubPredictor)

    res = predictor.predict("The sound is crisp.")
    assert isinstance(res, PredictionRecord)
    assert res.review == "The sound is crisp."
    assert len(res.predictions) == 1
    assert res.predictions[0].aspect == "sound"
    assert res.predictions[0].sentiment == "positive"
    assert res.to_dict()["predictions"] == [{"aspect": "sound", "sentiment": "positive"}]

    assert predictor.predict_aspect("The sound is crisp.", "sound") == "positive"


def test_get_unregistered_model() -> None:
    with pytest.raises(ModelNotFoundError, match="Model 'unknown' is not registered"):
        get_predictor("unknown")


def test_case_insensitive_registry() -> None:
    register_predictor("BERT", DummyStubPredictor)
    assert "bert" in list_available_models()
    pred = get_predictor("bert")
    assert isinstance(pred, DummyStubPredictor)


class _BrokenPredictor:
    def predict(self, review: str) -> PredictionRecord:
        raise RuntimeError(f"model crashed on {review!r}")

    def predict_aspect(self, review: str, aspect: str) -> str:
        del review, aspect
        return "positive"


class _BadOutputPredictor:
    def __init__(self, record: PredictionRecord) -> None:
        self._record = record

    def predict(self, review: str) -> PredictionRecord:
        del review
        return self._record

    def predict_aspect(self, review: str, aspect: str) -> str:
        del review, aspect
        return "positive"


def test_require_checkpoint_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CheckpointNotFoundError, match="not found"):
        require_checkpoint(tmp_path / "missing.pt")


def test_require_checkpoint_returns_existing_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"x")
    assert require_checkpoint(checkpoint) == checkpoint


def test_valid_record_passes_validation() -> None:
    record = PredictionRecord("Good battery.", (AspectPrediction("battery", "positive"),))
    assert validate_prediction(record) is record


def test_empty_prediction_list_is_valid() -> None:
    record = PredictionRecord("Nothing to say.", ())
    assert validate_prediction(record) is record


@pytest.mark.parametrize(
    "record",
    [
        PredictionRecord("   ", ()),
        PredictionRecord("ok", (AspectPrediction("", "positive"),)),
        PredictionRecord("ok", (AspectPrediction("battery", "happy"),)),
    ],
)
def test_invalid_record_raises_validation_error(record: PredictionRecord) -> None:
    with pytest.raises(PredictionValidationError):
        validate_prediction(record)


def test_predict_validated_rejects_bad_model_output() -> None:
    bad = PredictionRecord("ok", (AspectPrediction("battery", "happy"),))
    with pytest.raises(PredictionValidationError):
        predict_validated(_BadOutputPredictor(bad), "ok")


def test_predict_validated_wraps_backend_failures() -> None:
    with pytest.raises(InferenceError, match="Prediction failed") as info:
        predict_validated(_BrokenPredictor(), "ok")
    assert isinstance(info.value.__cause__, RuntimeError)


def test_predict_validated_returns_good_output() -> None:
    good = PredictionRecord("ok", (AspectPrediction("battery", "negative"),))
    assert predict_validated(_BadOutputPredictor(good), "ok") == good
