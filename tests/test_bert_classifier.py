import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("sklearn")
import torch
from transformers import BertConfig as HfBertConfig
from transformers import BertForSequenceClassification, BertTokenizerFast

from absa.models.bert_classifier import (
    BertConfig,
    evaluate_bert_test,
    load_bert,
    predict_bert,
    run_bert_training,
    train_bert,
)

WORDS = [
    "the",
    "battery",
    "is",
    "weak",
    "great",
    "screen",
    "on",
    "this",
    "one",
    "speed",
    "really",
    "impressed",
    "me",
    "terrible",
    "keyboard",
    "keys",
    "stick",
    "price",
    "fair",
    "for",
    "what",
    "it",
    "offers",
    "life",
    "excellent",
    "here",
    "scratches",
    "too",
    "easily",
    "feels",
    "solid",
    "and",
    "responsive",
    "fine",
    "nothing",
    "special",
    "high",
    "quality",
    ",",
    ".",
]


def _rec(review: str, aspect: str, sentiment: str, index: int) -> dict[str, object]:
    return {"review": review, "aspect": aspect, "sentiment": sentiment, "sentence_id": str(index)}


RECORDS = [
    _rec(review, aspect, sentiment, index)
    for index, (review, aspect, sentiment) in enumerate(
        [
            ("The battery is weak.", "battery", "negative"),
            ("Great screen on this one.", "screen", "positive"),
            ("The speed really impressed me.", "speed", "positive"),
            ("Terrible keyboard, keys stick.", "keyboard", "negative"),
            ("The price is fair for what it offers.", "price", "neutral"),
            ("Battery life is excellent here.", "battery life", "positive"),
            ("Screen scratches too easily.", "screen", "negative"),
            ("Keyboard feels solid and responsive.", "keyboard", "positive"),
            ("Speed is fine, nothing special.", "speed", "neutral"),
            ("Price is too high for this quality.", "price", "negative"),
            ("Mixed feelings about the fan.", "fan", "conflict"),
        ]
    )
]


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A random two-layer-free BERT with a word-level vocabulary, small enough for a test."""
    directory = tmp_path_factory.mktemp("tiny_bert")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *sorted(set(WORDS))]
    (directory / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    BertTokenizerFast(vocab_file=str(directory / "vocab.txt")).save_pretrained(directory)
    torch.manual_seed(0)
    config = HfBertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    BertForSequenceClassification(config).save_pretrained(directory)
    return directory


def _config(model_dir: Path, **changes: object) -> BertConfig:
    values: dict[str, object] = {
        "model_name": str(model_dir),
        "batch_size": 4,
        "epochs": 2,
        "max_length": 32,
        "val_fraction": 0.3,
        "device": "cpu",
        "bf16": False,
        "learning_rate": 1e-3,
    }
    values.update(changes)
    return BertConfig(**values)  # type: ignore[arg-type]


def test_labels_follow_the_drop_conflict_option() -> None:
    assert BertConfig().labels == ("positive", "negative", "neutral")
    assert BertConfig(drop_conflict=False).labels == ("positive", "negative", "neutral", "conflict")


def test_training_returns_a_history_and_keeps_the_best_epoch(tiny_model_dir: Path) -> None:
    records = [r for r in RECORDS if r["sentiment"] != "conflict"]
    model, _, history = train_bert(records[:7], records[7:], _config(tiny_model_dir))
    assert [epoch.epoch for epoch in history] == [1, 2]
    assert all(0.0 <= epoch.val_macro_f1 <= 1.0 for epoch in history)
    assert model.config.num_labels == 3  # type: ignore[union-attr]


def test_same_seed_gives_the_same_first_epoch_loss(tiny_model_dir: Path) -> None:
    records = [r for r in RECORDS if r["sentiment"] != "conflict"]
    config = _config(tiny_model_dir, epochs=1)
    first = train_bert(records[:7], records[7:], config)[2][0].train_loss
    second = train_bert(records[:7], records[7:], config)[2][0].train_loss
    assert first == pytest.approx(second)


def test_run_saves_a_model_that_reloads_and_predicts_the_same(
    tiny_model_dir: Path, tmp_path: Path
) -> None:
    config = _config(tiny_model_dir)
    result = run_bert_training(RECORDS, config, output_dir=tmp_path)
    log = json.loads((tmp_path / "bert_training_log.json").read_text(encoding="utf-8"))
    assert log["best_epoch"] == result.best_epoch
    assert log["config"]["drop_conflict"] is True

    model, tokenizer, labels = load_bert(result.model_dir)
    assert labels == ("positive", "negative", "neutral")
    unlabeled = [{"review": r["review"], "aspect": r["aspect"]} for r in RECORDS]
    first = predict_bert(model, tokenizer, unlabeled, config)
    model2, tokenizer2, _ = load_bert(result.model_dir)
    assert predict_bert(model2, tokenizer2, unlabeled, config) == first
    assert set(first) <= set(labels)


def test_evaluation_report_has_the_shared_shape(tiny_model_dir: Path, tmp_path: Path) -> None:
    config = _config(tiny_model_dir)
    result = run_bert_training(RECORDS, config, output_dir=tmp_path)
    model, tokenizer, _ = load_bert(result.model_dir)
    test_records = [r for r in RECORDS if r["sentiment"] != "conflict"]
    report = evaluate_bert_test(model, tokenizer, test_records, config)
    assert report.labels == config.labels
    assert len(report.confusion) == 3
    assert sum(sum(row) for row in report.confusion) == len(test_records)
    assert 0.0 <= report.metrics.accuracy <= 1.0
