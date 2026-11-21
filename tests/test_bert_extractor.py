import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("sklearn")
import torch
import transformers
from transformers import BertConfig as HfBertConfig
from transformers import BertForSequenceClassification, BertForTokenClassification

from absa.extraction import Sentence
from absa.models.bert_extractor import (
    IGNORE_INDEX,
    ExtractorConfig,
    align_labels,
    first_piece_tags,
    load_extractor,
    predict_spans,
    run_extractor_training,
    train_extractor,
)
from scripts import train_extractor as script

TEXTS = [
    "The battery is weak.",
    "Great screen on this one.",
    "The speed really impressed me.",
    "Terrible keyboard, keys stick.",
    "The price is fair.",
    "Battery life is excellent here.",
    "Screen scratches too easily.",
    "Keyboard feels solid and responsive.",
    "Speed is fine, nothing special.",
    "Price is too high for this quality.",
]
SPANS = [(4, 11), (6, 12), (4, 9), (9, 17), (4, 9), (0, 12), (0, 6), (0, 8), (0, 5), (0, 5)]
SENTENCES = [
    Sentence(str(i), text, ((start, end, "positive"),))
    for i, (text, (start, end)) in enumerate(zip(TEXTS, SPANS, strict=True))
]


def _tiny_dir(tmp_path: Path, name: str, *, token_level: bool) -> Path:
    directory = tmp_path / name
    directory.mkdir()
    words = {
        w.lower() for text in TEXTS for w in text.replace(",", " , ").replace(".", " . ").split()
    }
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *sorted(words)]
    (directory / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    transformers.BertTokenizerFast(vocab_file=str(directory / "vocab.txt")).save_pretrained(
        directory
    )  # type: ignore[attr-defined]
    torch.manual_seed(0)
    config = HfBertConfig(  # type: ignore[call-arg]
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        num_labels=3,
    )
    model_class = BertForTokenClassification if token_level else BertForSequenceClassification
    model_class(config).save_pretrained(directory)  # type: ignore[no-untyped-call]
    return directory


@pytest.fixture(scope="module")
def tiny_extractor(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _tiny_dir(tmp_path_factory.mktemp("ext"), "extractor", token_level=True)


@pytest.fixture(scope="module")
def tiny_classifier(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _tiny_dir(tmp_path_factory.mktemp("clf"), "classifier", token_level=False)


def _config(model_dir: Path, **changes: object) -> ExtractorConfig:
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
    return ExtractorConfig(**values)  # type: ignore[arg-type]


def test_labels_are_kept_on_first_pieces_only() -> None:
    word_ids = [None, 0, 1, 1, 2, None]
    assert align_labels(word_ids, [0, 1, 2]) == [IGNORE_INDEX, 0, 1, IGNORE_INDEX, 2, IGNORE_INDEX]


def test_word_tags_are_read_from_first_pieces_and_default_to_outside() -> None:
    word_ids = [None, 0, 1, 1, None]
    assert first_piece_tags(word_ids, [0, 1, 2, 0, 0], n_words=3) == ["B-ASP", "I-ASP", "O"]


def test_training_returns_a_history_and_keeps_the_best_epoch(tiny_extractor: Path) -> None:
    model, _, history, skipped = train_extractor(
        SENTENCES[:7], SENTENCES[7:], _config(tiny_extractor)
    )
    assert [epoch.epoch for epoch in history] == [1, 2]
    assert all(0.0 <= epoch.val_f1 <= 1.0 for epoch in history)
    assert skipped == 0
    assert model.config.num_labels == 3  # type: ignore[union-attr]


def test_sentences_with_untaggable_spans_are_counted_and_skipped(tiny_extractor: Path) -> None:
    overlapping = Sentence("o", "The battery is weak.", ((4, 11, "x"), (4, 9, "x")))
    _, _, _, skipped = train_extractor(
        [*SENTENCES[:7], overlapping], SENTENCES[7:], _config(tiny_extractor, epochs=1)
    )
    assert skipped == 1


def test_run_saves_a_model_that_reloads_and_predicts_valid_spans(
    tiny_extractor: Path, tmp_path: Path
) -> None:
    config = _config(tiny_extractor)
    result = run_extractor_training(SENTENCES, config, output_dir=tmp_path)
    log = json.loads((tmp_path / "extractor_training_log.json").read_text(encoding="utf-8"))
    assert log["best_epoch"] == result.best_epoch
    model, tokenizer = load_extractor(result.model_dir)
    spans = predict_spans(model, tokenizer, TEXTS, config)
    assert len(spans) == len(TEXTS)
    for text, found in zip(TEXTS, spans, strict=True):
        assert all(0 <= start < end <= len(text) for start, end in found)
    assert predict_spans(model, tokenizer, TEXTS, config) == spans


def _write_split(path: Path, name: str, sentences: list[Sentence]) -> None:
    records = [
        {
            "sentence_id": s.sentence_id,
            "review": s.text,
            "aspect": s.text[start:end],
            "start": start,
            "end": end,
            "sentiment": label,
        }
        for s in sentences
        for start, end, label in s.spans
    ]
    (path / name).write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    (path / name.replace(".jsonl", "_no_aspect.jsonl")).write_text(
        json.dumps({"sentence_id": "n", "review": "The keys stick."}) + "\n", encoding="utf-8"
    )


def test_script_tunes_scores_the_test_file_and_runs_end_to_end(
    tiny_extractor: Path, tiny_classifier: Path, tmp_path: Path
) -> None:
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    _write_split(data, "laptop_train.jsonl", SENTENCES)
    _write_split(data, "laptop_test.jsonl", SENTENCES[:5])
    common = ["--domain", "laptop", "--data-dir", str(data), "--output-dir", str(out)]
    argv = [
        *common,
        "--model-name",
        str(tiny_extractor),
        "--lr",
        "1e-3",
        "--epochs",
        "1",
        "--batch-size",
        "4",
        "--device",
        "cpu",
        "--classifier-dir",
        str(tiny_classifier),
    ]
    assert script.main(argv) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["test_sentences"] == 6  # five annotated sentences plus the no-aspect one
    assert 0.0 <= summary["test_exact"]["f1"] <= summary["test_overlap"]["f1"] <= 1.0
    assert set(summary["end_to_end"]) == {"precision", "recall", "f1", "tp", "fp", "fn"}
    report = json.loads((out / "test_report.json").read_text(encoding="utf-8"))
    assert set(report["error_counts"]) == {"boundary", "missed", "spurious"}
    assert Path(summary["checkpoint"]).is_dir()


def test_script_without_a_classifier_reports_extraction_only(
    tiny_extractor: Path, tmp_path: Path
) -> None:
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    _write_split(data, "laptop_train.jsonl", SENTENCES)
    (data / "laptop_test.jsonl").write_text(
        (data / "laptop_train.jsonl").read_text(encoding="utf-8"), encoding="utf-8"
    )
    argv = ["--domain", "laptop", "--data-dir", str(data), "--output-dir", str(out)]
    argv += [
        "--model-name",
        str(tiny_extractor),
        "--lr",
        "1e-3",
        "--epochs",
        "1",
        "--device",
        "cpu",
    ]
    assert script.main(argv) == 0
    assert "end_to_end" not in json.loads((out / "summary.json").read_text(encoding="utf-8"))
