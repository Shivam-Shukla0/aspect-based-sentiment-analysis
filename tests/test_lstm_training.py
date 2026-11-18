import json
from pathlib import Path

import torch

from absa.models import LstmConfig, load_checkpoint, run_training, save_checkpoint, train


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
