import json
from pathlib import Path

import pytest

from absa.data import validate_records
from absa.data.validation import read_jsonl

VALID = {
    "review": "The battery is weak.",
    "aspect": "battery",
    "sentiment": "negative",
    "start": 4,
    "end": 11,
}


def test_valid_records_produce_no_errors() -> None:
    summary = validate_records([VALID, {**VALID, "sentiment": "positive"}])
    assert summary.total == 2
    assert summary.errors == ()
    assert summary.to_dict()["invalid_records"] == 0


@pytest.mark.parametrize("bad_review", ["", "   ", None])
def test_empty_review_detected(bad_review: object) -> None:
    summary = validate_records([{**VALID, "review": bad_review}])
    assert "empty_review" in [e.error_type for e in summary.errors]


@pytest.mark.parametrize("bad_aspect", ["", "   ", None])
def test_empty_aspect_detected(bad_aspect: object) -> None:
    summary = validate_records([{**VALID, "aspect": bad_aspect}])
    assert "empty_aspect" in [e.error_type for e in summary.errors]


def test_unknown_sentiment_detected() -> None:
    summary = validate_records([{**VALID, "sentiment": "happy"}])
    assert [e.error_type for e in summary.errors] == ["unknown_sentiment"]


def test_offset_mismatch_detected() -> None:
    summary = validate_records([{**VALID, "start": 0, "end": 7}])
    assert [e.error_type for e in summary.errors] == ["offset_mismatch"]


def test_missing_offsets_is_not_an_error() -> None:
    record = {k: v for k, v in VALID.items() if k not in ("start", "end")}
    summary = validate_records([record])
    assert summary.errors == ()


def test_multiple_errors_on_one_record_are_all_reported() -> None:
    summary = validate_records([{"review": "", "aspect": "", "sentiment": "bad"}])
    error_types = {e.error_type for e in summary.errors}
    assert error_types == {"empty_review", "empty_aspect", "unknown_sentiment"}
    assert all(e.index == 0 for e in summary.errors)


def test_error_counts_cover_every_type_even_at_zero() -> None:
    summary = validate_records([VALID])
    assert summary.counts == {
        "empty_review": 0,
        "empty_aspect": 0,
        "unknown_sentiment": 0,
        "offset_mismatch": 0,
    }


def test_invalid_records_counts_distinct_indices_not_error_count() -> None:
    summary = validate_records([{"review": "", "aspect": "", "sentiment": "bad"}, VALID])
    assert summary.to_dict()["invalid_records"] == 1


def test_save_writes_json_report(tmp_path: Path) -> None:
    summary = validate_records([VALID, {**VALID, "sentiment": "bad"}])
    out = tmp_path / "nested" / "report.json"
    summary.save(out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["total_records"] == 2
    assert loaded["error_counts"]["unknown_sentiment"] == 1


def test_read_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(f"{json.dumps(VALID)}\n\n{json.dumps(VALID)}\n", encoding="utf-8")
    assert read_jsonl(path) == [VALID, VALID]
