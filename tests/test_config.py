import pytest

import absa
from absa import config


def test_package_imports_and_has_version() -> None:
    assert isinstance(absa.__version__, str)


def test_sentiment_labels_cover_semeval_polarities() -> None:
    assert set(config.SENTIMENT_LABELS) == {"positive", "negative", "neutral", "conflict"}


def test_domains() -> None:
    assert config.DOMAINS == ("laptop", "restaurant")


def test_paths_are_anchored_at_repo_root() -> None:
    assert (config.PROJECT_ROOT / "pyproject.toml").is_file()
    assert config.DATA_RAW_DIR == config.PROJECT_ROOT / "data" / "raw"


@pytest.mark.parametrize(
    ("domain", "split", "expected"),
    [
        ("laptop", "train", "laptop_train.jsonl"),
        ("restaurant", "test", "restaurant_test.jsonl"),
        ("laptop", "val", "laptop_val.jsonl"),
    ],
)
def test_processed_filename(domain: str, split: str, expected: str) -> None:
    assert config.processed_filename(domain, split) == expected


def test_processed_filename_rejects_unknown_domain() -> None:
    with pytest.raises(ValueError, match="unknown domain"):
        config.processed_filename("phone", "train")


def test_processed_filename_rejects_unknown_split() -> None:
    with pytest.raises(ValueError, match="unknown split"):
        config.processed_filename("laptop", "dev")
