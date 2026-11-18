import json
from pathlib import Path

import pytest

from absa.evaluation import evaluate

Y_TRUE = ["positive", "positive", "negative", "neutral"]
Y_PRED = ["positive", "negative", "negative", "neutral"]
LABELS = ("positive", "negative", "neutral")


def test_accuracy_matches_hand_count() -> None:
    result = evaluate(Y_TRUE, Y_PRED, labels=LABELS)
    assert result.accuracy == pytest.approx(3 / 4)


def test_per_class_metrics_match_hand_computation() -> None:
    result = evaluate(Y_TRUE, Y_PRED, labels=LABELS)
    by_label = {c.label: c for c in result.per_class}
    assert by_label["positive"].precision == pytest.approx(1.0)
    assert by_label["positive"].recall == pytest.approx(0.5)
    assert by_label["positive"].f1 == pytest.approx(2 / 3)
    assert by_label["positive"].support == 2
    assert by_label["negative"].precision == pytest.approx(0.5)
    assert by_label["negative"].recall == pytest.approx(1.0)
    assert by_label["negative"].support == 1
    assert by_label["neutral"].precision == pytest.approx(1.0)
    assert by_label["neutral"].recall == pytest.approx(1.0)
    assert by_label["neutral"].support == 1


def test_macro_and_micro_f1_match_hand_computation() -> None:
    result = evaluate(Y_TRUE, Y_PRED, labels=LABELS)
    assert result.macro_f1 == pytest.approx((2 / 3 + 2 / 3 + 1.0) / 3)
    assert result.micro_f1 == pytest.approx(result.accuracy)


def test_label_order_is_preserved_in_output() -> None:
    result = evaluate(Y_TRUE, Y_PRED, labels=LABELS)
    assert result.labels == LABELS
    assert tuple(c.label for c in result.per_class) == LABELS


def test_class_with_no_examples_scores_zero_without_crashing() -> None:
    result = evaluate(Y_TRUE, Y_PRED, labels=(*LABELS, "conflict"))
    conflict = next(c for c in result.per_class if c.label == "conflict")
    assert conflict.precision == 0.0
    assert conflict.recall == 0.0
    assert conflict.f1 == 0.0
    assert conflict.support == 0


def test_mismatched_lengths_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        evaluate(["positive"], ["positive", "negative"])


def test_empty_input_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        evaluate([], [])


def test_report_mentions_every_label() -> None:
    result = evaluate(Y_TRUE, Y_PRED, labels=LABELS)
    for label in LABELS:
        assert label in result.report


def test_to_dict_is_json_serialisable() -> None:
    result = evaluate(Y_TRUE, Y_PRED, labels=LABELS)
    serialised = json.dumps(result.to_dict())
    assert "macro_f1" in serialised


def test_save_writes_json_file(tmp_path: Path) -> None:
    result = evaluate(Y_TRUE, Y_PRED, labels=LABELS)
    out = tmp_path / "nested" / "result.json"
    result.save(out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["accuracy"] == pytest.approx(result.accuracy)


def test_evaluation_result_is_frozen() -> None:
    result = evaluate(Y_TRUE, Y_PRED, labels=LABELS)
    with pytest.raises(AttributeError):
        result.accuracy = 0.0  # type: ignore[misc]
