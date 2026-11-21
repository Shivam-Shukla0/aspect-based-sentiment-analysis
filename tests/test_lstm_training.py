import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")
import torch

from absa.models import LstmConfig, load_checkpoint, run_training, save_checkpoint, train
from absa.models.lstm_training import _class_weights


def _rec(review: str, aspect: str, sentiment: str) -> dict[str, object]:
    return {"review": review, "aspect": aspect, "sentiment": sentiment}


RECORDS = [
    _rec("The battery is weak.", "battery", "negative"),
    _rec("Great screen on this one.", "screen", "positive"),
    _rec("The speed really impressed me.", "speed", "positive"),
    _rec("Terrible keyboard, keys stick.", "keyboard", "negative"),
    _rec("The price is fair for what it offers.", "price", "neutral"),
    _rec("Battery life is excellent here.", "battery life", "positive"),
    _rec("Screen scratches too easily.", "screen", "negative"),
    _rec("Keyboard feels solid and responsive.", "keyboard", "positive"),
    _rec("Speed is fine, nothing special.", "speed", "neutral"),
    _rec("Price is too high for this quality.", "price", "negative"),
]
TINY_CONFIG = LstmConfig(
    embedding_dim=8, hidden_size=4, batch_size=4, epochs=1, val_fraction=0.3, seed=42
)


def _records_with_ids(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{**r, "sentence_id": str(i)} for i, r in enumerate(records)]


def test_same_seed_gives_same_first_epoch_loss() -> None:
    train_records = _records_with_ids(RECORDS)[:8]
    val_records = _records_with_ids(RECORDS)[8:]
    _, _, history_a = train(train_records, val_records, TINY_CONFIG)
    _, _, history_b = train(train_records, val_records, TINY_CONFIG)
    assert history_a[0].train_loss == history_b[0].train_loss


def test_different_seed_can_give_different_first_epoch_loss() -> None:
    train_records = _records_with_ids(RECORDS)[:8]
    val_records = _records_with_ids(RECORDS)[8:]
    _, _, history_a = train(train_records, val_records, TINY_CONFIG)
    other_config = LstmConfig(
        embedding_dim=8, hidden_size=4, batch_size=4, epochs=1, val_fraction=0.3, seed=7
    )
    _, _, history_b = train(train_records, val_records, other_config)
    assert history_a[0].train_loss != history_b[0].train_loss


def test_training_produces_one_epoch_metrics_entry_per_epoch() -> None:
    train_records = _records_with_ids(RECORDS)[:8]
    val_records = _records_with_ids(RECORDS)[8:]
    config = LstmConfig(embedding_dim=8, hidden_size=4, batch_size=4, epochs=3, val_fraction=0.3)
    _, _, history = train(train_records, val_records, config)
    assert [epoch_metrics.epoch for epoch_metrics in history] == [1, 2, 3]


def test_loss_decreases_with_enough_epochs_on_a_tiny_dataset() -> None:
    train_records = _records_with_ids(RECORDS)[:8]
    val_records = _records_with_ids(RECORDS)[8:]
    config = LstmConfig(
        embedding_dim=8,
        hidden_size=4,
        batch_size=8,
        epochs=30,
        learning_rate=0.01,
        val_fraction=0.3,
    )
    _, _, history = train(train_records, val_records, config)
    assert history[-1].train_loss < history[0].train_loss


def test_checkpoint_round_trip_reproduces_predictions(tmp_path: Path) -> None:
    train_records = _records_with_ids(RECORDS)[:8]
    val_records = _records_with_ids(RECORDS)[8:]
    model, vocab, _ = train(train_records, val_records, TINY_CONFIG)
    model.eval()

    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(checkpoint_path, model, vocab, TINY_CONFIG)
    loaded_model, loaded_vocab = load_checkpoint(checkpoint_path)
    loaded_model.eval()

    assert loaded_vocab.tokens == vocab.tokens
    review_ids = torch.tensor([vocab.encode(["the", "battery", "is", "weak"])])
    aspect_ids = torch.tensor([vocab.encode(["battery"])])
    lengths = torch.tensor([review_ids.shape[1]])
    aspect_lengths = torch.tensor([aspect_ids.shape[1]])

    with torch.no_grad():
        original_logits, _ = model(review_ids, lengths, aspect_ids, aspect_lengths)
        loaded_logits, _ = loaded_model(review_ids, lengths, aspect_ids, aspect_lengths)
    assert torch.equal(original_logits, loaded_logits)


def test_run_training_writes_checkpoint_and_log(tmp_path: Path) -> None:
    records = _records_with_ids(RECORDS)
    result = run_training(records, TINY_CONFIG, output_dir=tmp_path)
    assert result.checkpoint_path.is_file()
    log_path = tmp_path / "lstm_training_log.json"
    assert log_path.is_file()

    log = json.loads(log_path.read_text(encoding="utf-8"))
    assert log["config"]["seed"] == TINY_CONFIG.seed
    assert len(log["history"]) == TINY_CONFIG.epochs
    assert log["best_epoch"] == result.best_epoch


class _Score:
    def __init__(self, macro_f1: float) -> None:
        self.macro_f1 = macro_f1


def _state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in model.state_dict().items()}


def test_train_returns_the_best_epoch_not_the_last(monkeypatch: pytest.MonkeyPatch) -> None:
    records = _records_with_ids(RECORDS)
    train_records, val_records = records[:7], records[7:]

    def scripted(values: list[float]) -> None:
        scores = iter(values)

        def fake_evaluate(*_args: object, **_kwargs: object) -> _Score:
            return _Score(next(scores))

        monkeypatch.setattr("absa.models.lstm_training.evaluate", fake_evaluate)

    config_three = LstmConfig(
        embedding_dim=8, hidden_size=4, batch_size=4, epochs=3, val_fraction=0.3, seed=42
    )
    config_two = LstmConfig(
        embedding_dim=8, hidden_size=4, batch_size=4, epochs=2, val_fraction=0.3, seed=42
    )
    scripted([0.2, 0.9, 0.4])
    model_three, _, history = train(train_records, val_records, config_three)
    scripted([0.2, 0.9])
    model_two, _, _ = train(train_records, val_records, config_two)

    assert [round(h.val_macro_f1, 1) for h in history] == [0.2, 0.9, 0.4]
    expected = _state(model_two)
    for key, value in _state(model_three).items():
        assert torch.equal(value, expected[key])


def test_class_weights_favour_rare_classes_and_zero_absent_ones() -> None:
    records = [_rec("a", "a", "positive")] * 6 + [_rec("b", "b", "negative")] * 2
    weights = _class_weights(records)
    assert weights.tolist() == pytest.approx([8 / (2 * 6), 8 / (2 * 2), 0.0, 0.0])


def test_training_with_class_weighting_runs_and_is_logged(tmp_path: Path) -> None:
    config = LstmConfig(
        embedding_dim=8,
        hidden_size=4,
        batch_size=4,
        epochs=2,
        val_fraction=0.3,
        seed=42,
        class_weighting=True,
    )
    result = run_training(_records_with_ids(RECORDS), config, output_dir=tmp_path)
    log = json.loads((tmp_path / "lstm_training_log.json").read_text(encoding="utf-8"))
    assert log["config"]["class_weighting"] is True
    assert len(result.history) == 2
