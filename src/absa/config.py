"""Project-wide constants shared by all team members."""

from pathlib import Path

# This file lives at <root>/src/absa/config.py, so parents[2] walks up
# config.py -> absa/ -> src/ -> <root>. Every data/output path below hangs off it,
# which keeps paths independent of the directory a script is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# The two SemEval-2014 Task 4 review domains and the three dataset splits.
# "val" is carved out of the official train set (see data/splits.py); "test" is
# the official SemEval test file and is never used for tuning.
DOMAINS = ("laptop", "restaurant")
SPLITS = ("train", "val", "test")

# SemEval-2014 Task 4 polarity labels. "conflict" is kept in the schema;
# experiments that drop it must say so explicitly (see data/README.md).
SENTIMENT_LABELS = ("positive", "negative", "neutral", "conflict")

# Single default seed used everywhere (splits, set_seed) so runs are comparable.
RANDOM_SEED = 42


def processed_filename(domain: str, split: str) -> str:
    """Return the canonical processed-file name for a domain/split pair.

    For example ``processed_filename("laptop", "train")`` is
    ``"laptop_train.jsonl"``. One aspect record is stored per line.
    """
    # Fail loudly on typos here so a bad name can never silently create a stray file.
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; expected one of {DOMAINS}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    return f"{domain}_{split}.jsonl"
