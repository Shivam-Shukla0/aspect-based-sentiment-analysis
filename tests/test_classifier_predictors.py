from pathlib import Path

import pytest

from absa import models
from absa.inference import (
    AspectClassifierPredictor,
    CheckpointNotFoundError,
    InferenceError,
    PredictionValidationError,
    load_bert_predictor,
    load_lstm_predictor,
)

REVIEW = "The battery is weak but the screen is excellent."


def _classify(_review: str, aspect: str) -> str:
    return "negative" if aspect == "battery" else "positive"


def test_predict_aspects_returns_a_validated_record() -> None:
    record = AspectClassifierPredictor(_classify).predict_aspects(REVIEW, ["battery", "screen"])
    assert record.to_dict() == {
        "review": REVIEW,
        "predictions": [
            {"aspect": "battery", "sentiment": "negative"},
            {"aspect": "screen", "sentiment": "positive"},
        ],
    }


def test_joint_prediction_is_refused_with_a_clear_message() -> None:
    with pytest.raises(InferenceError, match="predict_aspects"):
        AspectClassifierPredictor(_classify).predict(REVIEW)


@pytest.mark.parametrize(
    ("review", "aspects"), [("", ["battery"]), (REVIEW, []), (REVIEW, ["battery", " "])]
)
def test_empty_input_is_rejected(review: str, aspects: list[str]) -> None:
    with pytest.raises(InferenceError, match="required"):
        AspectClassifierPredictor(_classify).predict_aspects(review, aspects)


def test_an_aspect_missing_from_the_review_is_rejected() -> None:
    with pytest.raises(PredictionValidationError, match="does not occur"):
        AspectClassifierPredictor(_classify).predict_aspects(REVIEW, ["keyboard"])


def test_a_failing_model_is_wrapped_and_an_inference_error_passes_through() -> None:
    def broken(_review: str, _aspect: str) -> str:
        raise RuntimeError("out of memory")

    with pytest.raises(InferenceError, match="out of memory"):
        AspectClassifierPredictor(broken).predict_aspect(REVIEW, "battery")

    def refusing(_review: str, _aspect: str) -> str:
        raise InferenceError("refused")

    with pytest.raises(InferenceError, match="refused"):
        AspectClassifierPredictor(refusing).predict_aspect(REVIEW, "battery")


def test_loaders_report_a_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(CheckpointNotFoundError):
        load_lstm_predictor(tmp_path / "missing.pt")
    with pytest.raises(CheckpointNotFoundError):
        load_bert_predictor(tmp_path / "missing")


def test_lstm_loader_classifies_through_the_trained_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "lstm_checkpoint.pt"
    checkpoint.write_bytes(b"")
    seen: list[object] = []

    def fake_predict(
        _model: object, _vocab: object, records: list[dict[str, str]], device: str
    ) -> list[str]:
        seen.append((records[0]["aspect"], device))
        return ["negative"]

    monkeypatch.setitem(vars(models), "load_checkpoint", lambda _path: ("model", "vocab"))
    monkeypatch.setitem(vars(models), "predict", fake_predict)
    predictor = load_lstm_predictor(checkpoint, device="cpu")
    assert predictor.predict_aspect(REVIEW, "battery") == "negative"
    assert seen == [("battery", "cpu")]


@pytest.mark.parametrize(
    "labels",
    [("positive", "negative", "neutral"), ("positive", "negative", "neutral", "conflict")],
)
def test_bert_loader_uses_the_label_set_of_the_saved_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, labels: tuple[str, ...]
) -> None:
    configs: list[object] = []

    def fake_predict(
        _model: object, _tokenizer: object, records: list[dict[str, str]], config: _FakeConfig
    ) -> list[str]:
        configs.append(config)
        return ["positive"] * len(records)

    monkeypatch.setitem(vars(models), "load_bert", lambda _path: ("model", "tokenizer", labels))
    monkeypatch.setitem(vars(models), "predict_bert", fake_predict)
    monkeypatch.setitem(vars(models), "BertConfig", _FakeConfig)
    predictor = load_bert_predictor(tmp_path, device="cpu")
    assert predictor.predict_aspect(REVIEW, "screen") == "positive"
    assert configs[0].drop_conflict is ("conflict" not in labels)  # type: ignore[attr-defined]


class _FakeConfig:
    def __init__(self, *, device: str, bf16: bool, drop_conflict: bool) -> None:
        self.device, self.bf16, self.drop_conflict = device, bf16, drop_conflict
