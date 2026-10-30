import pytest

from absa.data.leakage import assert_no_leakage, find_overlap

TRAIN = [
    {"review": "The battery is weak.", "aspect": "battery", "sentiment": "negative"},
    {"review": "Great screen.", "aspect": "screen", "sentiment": "positive"},
]


def test_disjoint_splits_pass() -> None:
    test = [{"review": "Fast boot time.", "aspect": "boot", "sentiment": "positive"}]
    assert find_overlap(TRAIN, test) == set()
    assert_no_leakage(TRAIN, test)


def test_overlap_ignores_case_and_whitespace() -> None:
    test = [{"review": "  the BATTERY is   weak. ", "aspect": "battery", "sentiment": "negative"}]
    assert find_overlap(TRAIN, test) == {"the battery is weak."}
    with pytest.raises(ValueError, match="1 review"):
        assert_no_leakage(TRAIN, test)
