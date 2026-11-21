import pytest

from absa.data import validate_record

VALID = {
    "review": "The battery is weak but the screen is excellent.",
    "aspect": "battery",
    "sentiment": "negative",
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
