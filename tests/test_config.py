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
