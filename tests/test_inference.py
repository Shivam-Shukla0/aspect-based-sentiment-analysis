"""Tests for common inference interface and model registry."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from absa.inference.interface import (
    AspectPrediction,
    ModelNotFoundError,
    PredictionRecord,
    clear_registry,
    get_predictor,
    list_available_models,
    register_predictor,
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
