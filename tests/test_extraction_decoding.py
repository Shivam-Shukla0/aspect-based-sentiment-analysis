import pytest

from absa.evaluation import span_report
from absa.extraction import (
    Sentence,
    decode_spans,
    group_sentences,
    repair_tags,
    split_sentences,
)

TEXT = "The battery life is great but the screen is dim."


def _record(sid: str, review: str, start: int, end: int, sentiment: str) -> dict[str, object]:
    return {
        "sentence_id": sid,
        "review": review,
        "aspect": review[start:end],
        "start": start,
        "end": end,
        "sentiment": sentiment,
    }


def test_group_sentences_collects_spans_and_keeps_no_aspect_sentences() -> None:
    records = [
        _record("1", TEXT, 4, 16, "positive"),
        _record("1", TEXT, 30, 36, "negative"),
        _record("2", "Great.", 0, 5, "positive"),
    ]
    sentences = group_sentences(records, [{"sentence_id": "3", "review": "Nothing here."}])
    assert [s.sentence_id for s in sentences] == ["1", "2", "3"]
    assert sentences[0].gold_spans == [(4, 16), (30, 36)]
    assert sentences[0].spans[1] == (30, 36, "negative")
    assert sentences[2].spans == ()


def test_split_keeps_identical_texts_together_and_is_deterministic() -> None:
    sentences = [Sentence(str(i), f"Review number {i % 6}.") for i in range(12)]
    sentences.append(Sentence("x", "  review NUMBER 1. "))
    train, val = split_sentences(sentences, 0.34, seed=1)
    assert {s.text.strip().lower() for s in train}.isdisjoint({s.text.strip().lower() for s in val})
    assert len(train) + len(val) == len(sentences)
    assert split_sentences(sentences, 0.34, seed=1) == (train, val)


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_split_rejects_a_fraction_outside_the_open_interval(fraction: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        split_sentences([Sentence("1", "a b"), Sentence("2", "c d")], fraction, seed=1)


def test_split_rejects_an_empty_side() -> None:
    with pytest.raises(ValueError, match="empty"):
        split_sentences([Sentence("1", "only one text"), Sentence("2", "ONLY one text")], 0.5, 1)


def test_repair_tags_turns_a_stray_inside_tag_into_a_start() -> None:
    assert repair_tags(["I-ASP", "I-ASP", "O", "I-ASP", "B-ASP", "I-ASP"]) == [
        "B-ASP",
        "I-ASP",
        "O",
        "B-ASP",
        "B-ASP",
        "I-ASP",
    ]


def test_decode_maps_tags_back_to_character_spans() -> None:
    tags = ["O", "B-ASP", "I-ASP", "O", "O", "O", "O", "B-ASP", "O", "O", "O"]
    spans = decode_spans(TEXT, tags)
    assert [TEXT[s:e] for s, e in spans] == ["battery life", "screen"]


def test_decode_repairs_invalid_order_and_pads_missing_tags() -> None:
    assert decode_spans(TEXT, ["O", "I-ASP"]) == [(4, 11)]


def test_decode_rejects_too_many_tags_and_unknown_tags() -> None:
    with pytest.raises(ValueError, match="tags for"):
        decode_spans("Hi.", ["O"] * 5)
    with pytest.raises(ValueError, match="unknown"):
        decode_spans("Hi.", ["X"])


def test_span_report_scores_and_categorises_errors() -> None:
    texts = [TEXT, "Fast boot but loud fan."]
    gold = [[(4, 16), (30, 36)], [(0, 9), (14, 22)]]
    predicted = [[(4, 11), (30, 36)], [(0, 9), (10, 13)]]
    report = span_report(texts, gold, predicted)
    assert report["exact"]["tp"] == 2  # type: ignore[index]
    assert report["overlap"]["tp"] == 3  # type: ignore[index]
    assert report["error_counts"] == {"boundary": 1, "missed": 1, "spurious": 1}
    kinds = {e["kind"]: (e["gold"], e["predicted"]) for e in report["error_examples"]}  # type: ignore[attr-defined]
    assert kinds == {
        "boundary": ("battery life", "battery"),
        "missed": ("loud fan", ""),
        "spurious": ("", "but"),
    }
    assert report["gold_spans"] == 4
    assert report["predicted_spans"] == 4
