"""Tests for common inference interface and model registry."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from absa.inference import interface
from absa.inference.interface import (
    AspectPrediction,
    CheckpointNotFoundError,
    DuplicateModelError,
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
def _isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test its own empty registry."""
    registry: dict[str, Callable[[], Predictor]] = {}
    monkeypatch.setattr(interface, "_REGISTRY", registry)


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


def _record(
    *predictions: AspectPrediction, review: str = "The sound is crisp."
) -> PredictionRecord:
    return PredictionRecord(review=review, predictions=tuple(predictions))


def test_validate_prediction_record_accepts_valid_output() -> None:
    record = _record(AspectPrediction("Sound", "positive"))
    assert validate_prediction_record(record) is record


def test_validate_prediction_record_accepts_no_predictions() -> None:
    record = _record()
    assert validate_prediction_record(record) is record


@pytest.mark.parametrize(
    "prediction",
    [
        AspectPrediction("sound", "happy"),
        AspectPrediction("", "positive"),
        AspectPrediction("battery", "positive"),
    ],
)
def test_validate_prediction_record_rejects_bad_predictions(prediction: AspectPrediction) -> None:
    with pytest.raises(PredictionValidationError):
        validate_prediction_record(_record(prediction))


def test_validate_prediction_record_rejects_wrong_type() -> None:
    with pytest.raises(PredictionValidationError, match="expected a PredictionRecord"):
        validate_prediction_record({"review": "x"})  # type: ignore[arg-type]


def test_predict_validated_passes_and_fails() -> None:
    assert predict_validated(DummyStubPredictor(), "The sound is crisp.").predictions

    class BadPredictor(DummyStubPredictor):
        def predict(self, review: str) -> PredictionRecord:
            return _record(AspectPrediction("sound", "great"), review=review)

    with pytest.raises(PredictionValidationError, match="invalid prediction"):
        predict_validated(BadPredictor(), "The sound is crisp.")


def test_require_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    with pytest.raises(CheckpointNotFoundError, match="Checkpoint not found"):
        require_checkpoint(checkpoint)
    checkpoint.write_bytes(b"")
    assert require_checkpoint(str(checkpoint)) == checkpoint


def test_duplicate_registration_is_rejected_unless_replaced() -> None:
    class OtherPredictor(DummyStubPredictor):
        pass

    register_predictor("x", DummyStubPredictor)
    with pytest.raises(DuplicateModelError, match="already registered"):
        register_predictor("X", OtherPredictor)
    assert isinstance(get_predictor("x"), DummyStubPredictor)
    assert type(get_predictor("x")) is DummyStubPredictor

    register_predictor("X", OtherPredictor, replace=True)
    assert type(get_predictor("x")) is OtherPredictor


class _RaisingPredictor:
    def predict(self, review: str) -> PredictionRecord:
        raise RuntimeError(f"model crashed on {review!r}")

    def predict_aspect(self, review: str, aspect: str) -> str:
        del review, aspect
        return "positive"


class _FixedPredictor:
    def __init__(self, record: PredictionRecord) -> None:
        self._record = record

    def predict(self, review: str) -> PredictionRecord:
        del review
        return self._record

    def predict_aspect(self, review: str, aspect: str) -> str:
        del review, aspect
        return "positive"


def test_predict_validated_wraps_unexpected_backend_errors() -> None:
    with pytest.raises(interface.InferenceError, match="Prediction failed") as info:
        interface.predict_validated(_RaisingPredictor(), "ok")
    assert isinstance(info.value.__cause__, RuntimeError)


@pytest.mark.parametrize("review", ["", "   "])
def test_predict_validated_reports_an_empty_input_as_bad_input(review: str) -> None:
    with pytest.raises(interface.InferenceError, match="input review"):
        interface.predict_validated(_RaisingPredictor(), review)


def test_predict_validated_rejects_a_record_for_a_different_review() -> None:
    other = PredictionRecord("some other text", ())
    with pytest.raises(interface.PredictionValidationError, match="differs"):
        interface.predict_validated(_FixedPredictor(other), "the real review")


def test_predict_validated_accepts_a_matching_record() -> None:
    record = PredictionRecord("The battery is weak.", (AspectPrediction("battery", "negative"),))
    assert interface.predict_validated(_FixedPredictor(record), "The battery is weak.") == record
