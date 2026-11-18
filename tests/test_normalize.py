from pathlib import Path

from absa.data import (
    AspectSpan,
    ParsedSentence,
    build_processed_files,
    normalize_sentences,
    validate_record,
)
from absa.data.normalize import no_aspect_filename, write_jsonl

WITH_ASPECTS = ParsedSentence(
    sentence_id="1",
    text="The battery is weak but the screen is excellent.",
    spans=(
        AspectSpan(term="battery", start=4, end=11, polarity="negative"),
        AspectSpan(term="screen", start=28, end=34, polarity="POSITIVE"),
    ),
)
WITHOUT_ASPECTS = ParsedSentence(sentence_id="2", text="Great value for the price.", spans=())


def test_one_record_per_aspect() -> None:
    records, no_aspect = normalize_sentences([WITH_ASPECTS], domain="laptop")
    assert len(records) == 2
    assert no_aspect == []


def test_sentence_id_and_domain_are_preserved() -> None:
    records, _ = normalize_sentences([WITH_ASPECTS], domain="laptop")
    assert all(r["sentence_id"] == "1" for r in records)
    assert all(r["domain"] == "laptop" for r in records)


def test_offsets_are_preserved() -> None:
    records, _ = normalize_sentences([WITH_ASPECTS], domain="laptop")
    battery = next(r for r in records if r["aspect"] == "battery")
    assert battery["start"] == 4
    assert battery["end"] == 11


def test_sentiment_label_is_lowercased() -> None:
    records, _ = normalize_sentences([WITH_ASPECTS], domain="laptop")
    screen = next(r for r in records if r["aspect"] == "screen")
    assert screen["sentiment"] == "positive"


def test_no_aspect_sentences_kept_separately_not_dropped() -> None:
    records, no_aspect = normalize_sentences([WITHOUT_ASPECTS], domain="laptop")
    assert records == []
    assert len(no_aspect) == 1
    assert no_aspect[0]["sentence_id"] == "2"
    assert no_aspect[0]["review"] == "Great value for the price."


def test_output_validates_against_schema() -> None:
    records, _ = normalize_sentences([WITH_ASPECTS], domain="laptop")
    for record in records:
        validate_record(record)


def test_no_aspect_filename_convention() -> None:
    assert no_aspect_filename("laptop", "train") == "laptop_train_no_aspect.jsonl"


def test_build_processed_files_writes_both_files(tmp_path: Path) -> None:
    records_path, no_aspect_path = build_processed_files(
        [WITH_ASPECTS, WITHOUT_ASPECTS], domain="laptop", split="train", processed_dir=tmp_path
    )
    assert records_path == tmp_path / "laptop_train.jsonl"
    assert no_aspect_path == tmp_path / "laptop_train_no_aspect.jsonl"
    assert records_path.is_file()
    assert no_aspect_path.is_file()
    assert len(records_path.read_text(encoding="utf-8").splitlines()) == 2
    assert len(no_aspect_path.read_text(encoding="utf-8").splitlines()) == 1


def test_running_twice_gives_identical_output(tmp_path: Path) -> None:
    sentences = [WITH_ASPECTS, WITHOUT_ASPECTS]
    first = build_processed_files(sentences, "laptop", "train", tmp_path)
    first_bytes = [p.read_bytes() for p in first]
    second = build_processed_files(sentences, "laptop", "train", tmp_path)
    second_bytes = [p.read_bytes() for p in second]
    assert first_bytes == second_bytes


def test_write_jsonl_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "out.jsonl"
    write_jsonl([{"a": 1}], path)
    assert path.read_text(encoding="utf-8") == '{"a": 1}\n'
