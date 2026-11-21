import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from scripts.train_lstm import main

REVIEWS = [
    ("The battery is weak.", "battery", "negative"),
    ("Great screen on this one.", "screen", "positive"),
    ("The speed really impressed me.", "speed", "positive"),
    ("Terrible keyboard, keys stick.", "keyboard", "negative"),
    ("The price is fair for what it offers.", "price", "neutral"),
    ("Battery life is excellent here.", "battery life", "positive"),
    ("Screen scratches too easily.", "screen", "negative"),
    ("Keyboard feels solid and responsive.", "keyboard", "positive"),
    ("Speed is fine, nothing special.", "speed", "neutral"),
    ("Price is too high for this quality.", "price", "negative"),
    ("Mixed feelings about the fan.", "fan", "conflict"),
]


def _write(path: Path, rows: list[tuple[str, str, str]]) -> None:
    lines = [
        json.dumps({"review": r, "aspect": a, "sentiment": s, "sentence_id": str(i)})
        for i, (r, a, s) in enumerate(rows)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_script_tunes_then_scores_the_test_file_without_conflict(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _write(data / "laptop_train.jsonl", REVIEWS)
    _write(data / "laptop_test.jsonl", REVIEWS)
    out = tmp_path / "out"
    code = main(
        [
            "--domain",
            "laptop",
            "--data-dir",
            str(data),
            "--output-dir",
            str(out),
            "--dropout",
            "0.0",
            "--hidden",
            "4",
            "--lr",
            "0.01",
            "--epochs",
            "1",
            "--device",
            "cpu",
        ]
    )
    assert code == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["runs"]) == 2  # plain and class-weighted
    assert summary["test_records"] == len(REVIEWS) - 1  # conflict removed
    assert summary["chosen"]["drop_conflict"] is True
    assert (out / "test_report.json").is_file()
