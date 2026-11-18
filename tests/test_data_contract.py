import pytest

from absa.data import AspectSpan, ExtractionRecord, validate_prediction, validate_record

VALID = {
    "review": "The battery is weak but the screen is excellent.",
    "aspect": "battery",
    "sentiment": "negative",
}

PREDICTIONS = [
    {"aspect": "battery", "sentiment": "negative"},
    {"aspect": "screen", "sentiment": "positive"},
]

VALID_PREDICTION = {
    "review": "The battery is weak but the screen is excellent.",
    "predictions": PREDICTIONS,
}


def test_valid_record_is_returned_unchanged() -> None:
    assert validate_record(dict(VALID)) == VALID


@pytest.mark.parametrize("label", ["positive", "negative", "neutral", "conflict"])
def test_all_semeval_labels_accepted(label: str) -> None:
    validate_record({**VALID, "sentiment": label})


def test_conflict_can_be_excluded_by_caller() -> None:
    with pytest.raises(ValueError, match="unknown sentiment"):
        validate_record(
            {**VALID, "sentiment": "conflict"}, allowed_labels=("positive", "negative", "neutral")
        )


def test_unknown_label_rejected() -> None:
    with pytest.raises(ValueError, match="unknown sentiment"):
        validate_record({**VALID, "sentiment": "happy"})


@pytest.mark.parametrize("field", ["review", "aspect", "sentiment"])
def test_missing_field_rejected(field: str) -> None:
    record = {k: v for k, v in VALID.items() if k != field}
    with pytest.raises(ValueError, match="missing fields"):
        validate_record(record)


@pytest.mark.parametrize("field", ["review", "aspect"])
@pytest.mark.parametrize("bad", ["", "   ", None, 3])
def test_malformed_text_fields_rejected(field: str, bad: object) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        validate_record({**VALID, field: bad})


@pytest.mark.parametrize("bad", [None, "text", ["a"]])
def test_non_dict_rejected(bad: object) -> None:
    with pytest.raises(ValueError, match="must be a dict"):
        validate_record(bad)


def test_shared_sample_records_are_valid(sample_aspect_records: list[dict[str, str]]) -> None:
    assert [validate_record(dict(record)) for record in sample_aspect_records] == (
        sample_aspect_records
    )


def test_valid_prediction_is_returned_unchanged() -> None:
    assert validate_prediction(dict(VALID_PREDICTION)) == VALID_PREDICTION


def test_empty_predictions_list_is_valid() -> None:
    validate_prediction({"review": "No aspects here.", "predictions": []})


@pytest.mark.parametrize("field", ["review", "predictions"])
def test_prediction_missing_field_rejected(field: str) -> None:
    record = {k: v for k, v in VALID_PREDICTION.items() if k != field}
    with pytest.raises(ValueError, match="missing fields"):
        validate_prediction(record)


def test_prediction_non_list_predictions_rejected() -> None:
    with pytest.raises(ValueError, match="must be a list"):
        validate_prediction({**VALID_PREDICTION, "predictions": "battery"})


def test_prediction_entry_non_dict_rejected() -> None:
    with pytest.raises(ValueError, match="predictions\\[0\\] must be a dict"):
        validate_prediction({"review": "x", "predictions": ["battery"]})


@pytest.mark.parametrize("field", ["aspect", "sentiment"])
def test_prediction_entry_missing_field_rejected(field: str) -> None:
    entry = {k: v for k, v in PREDICTIONS[0].items() if k != field}
    with pytest.raises(ValueError, match="missing fields"):
        validate_prediction({"review": "x", "predictions": [entry]})


def test_prediction_entry_unknown_label_rejected() -> None:
    with pytest.raises(ValueError, match="unknown sentiment"):
        validate_prediction(
            {"review": "x", "predictions": [{"aspect": "battery", "sentiment": "happy"}]}
        )


def test_prediction_non_dict_rejected() -> None:
    with pytest.raises(ValueError, match="must be a dict"):
        validate_prediction(["not", "a", "dict"])


def test_aspect_span_round_trips_against_text() -> None:
    text = "The battery is weak."
    span = AspectSpan(term="battery", start=4, end=11, polarity="negative")
    assert text[span.start : span.end] == span.term
    ExtractionRecord(text=text, spans=(span,))


def test_aspect_span_rejects_empty_term() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        AspectSpan(term="", start=0, end=1, polarity="negative")


@pytest.mark.parametrize(("start", "end"), [(-1, 3), (5, 5), (5, 2)])
def test_aspect_span_rejects_bad_offsets(start: int, end: int) -> None:
    with pytest.raises(ValueError, match="start < end"):
        AspectSpan(term="abc", start=start, end=end, polarity="negative")


def test_extraction_record_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        ExtractionRecord(text="  ", spans=())


def test_extraction_record_rejects_mismatched_span() -> None:
    text = "The battery is weak."
    mismatched = AspectSpan(term="screen", start=4, end=11, polarity="negative")
    with pytest.raises(ValueError, match="does not match"):
        ExtractionRecord(text=text, spans=(mismatched,))
