import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from absa.config import SENTIMENT_LABELS
from absa.models import AbsaLstmModel, LstmConfig, Vocabulary, evaluate_test_set, predict, train


def _rec(review: str, aspect: str, sentiment: str) -> dict[str, object]:
    return {"review": review, "aspect": aspect, "sentiment": sentiment}


TRAIN_RECORDS = [
    _rec("The battery is weak.", "battery", "negative"),
    _rec("Great screen on this one.", "screen", "positive"),
    _rec("The speed really impressed me.", "speed", "positive"),
    _rec("Terrible keyboard, keys stick.", "keyboard", "negative"),
    _rec("The price is fair for what it offers.", "price", "neutral"),
    _rec("Battery life is excellent here.", "battery life", "positive"),
    _rec("Screen scratches too easily.", "screen", "negative"),
    _rec("Keyboard feels solid and responsive.", "keyboard", "positive"),
]
TEST_RECORDS = [
    _rec("Speed is fine, nothing special.", "speed", "neutral"),
    _rec("Price is too high for this quality.", "price", "negative"),
]
CONFIG = LstmConfig(
    device="cpu", embedding_dim=8, hidden_size=4, batch_size=4, epochs=2, val_fraction=0.25
)


def _trained_model() -> tuple[AbsaLstmModel, Vocabulary]:
    model, vocab, _ = train(TRAIN_RECORDS, TRAIN_RECORDS[:2], CONFIG)
    return model, vocab


def test_predict_returns_one_label_per_record() -> None:
    model, vocab = _trained_model()
    predictions = predict(model, vocab, TEST_RECORDS)
    assert len(predictions) == len(TEST_RECORDS)
    assert all(p in SENTIMENT_LABELS for p in predictions)


def test_evaluate_test_set_report_structure() -> None:
    model, vocab = _trained_model()
    report = evaluate_test_set(model, vocab, TEST_RECORDS)
    assert report.metrics.accuracy >= 0.0
    assert len(report.confusion) == len(report.labels)
    assert all(len(row) == len(report.labels) for row in report.confusion)


def test_confusion_matrix_diagonal_matches_correct_predictions() -> None:
    model, vocab = _trained_model()
    report = evaluate_test_set(model, vocab, TEST_RECORDS)
    correct = sum(report.confusion[i][i] for i in range(len(report.labels)))
    total = sum(sum(row) for row in report.confusion)
    expected_correct = round(report.metrics.accuracy * len(TEST_RECORDS))
    assert correct == expected_correct
    assert total == len(TEST_RECORDS)


def test_errors_are_only_misclassified_records() -> None:
    model, vocab = _trained_model()
    report = evaluate_test_set(model, vocab, TEST_RECORDS)
    for error in report.errors.errors:
        assert error.true_label != error.predicted_label


def test_error_count_matches_incorrect_predictions() -> None:
    model, vocab = _trained_model()
    report = evaluate_test_set(model, vocab, TEST_RECORDS)
    predictions = predict(model, vocab, TEST_RECORDS)
    expected = sum(
        1
        for record, prediction in zip(TEST_RECORDS, predictions, strict=True)
        if record["sentiment"] != prediction
    )
    assert len(report.errors.errors) == expected


def test_report_to_dict_is_json_serialisable() -> None:
    model, vocab = _trained_model()
    report = evaluate_test_set(model, vocab, TEST_RECORDS)
    serialised = json.dumps(report.to_dict())
    assert "confusion_matrix" in serialised


def test_report_save_writes_file(tmp_path: Path) -> None:
    model, vocab = _trained_model()
    report = evaluate_test_set(model, vocab, TEST_RECORDS)
    out = tmp_path / "nested" / "report.json"
    report.save(out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["labels"] == list(report.labels)
