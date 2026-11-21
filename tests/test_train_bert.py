import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("sklearn")

from scripts.train_bert import main
from tests.test_bert_classifier import RECORDS, tiny_model_dir

__all__ = ["tiny_model_dir"]


def _write(path: Path) -> None:
    lines = [json.dumps(record) for record in RECORDS]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_script_tunes_then_scores_the_test_file_without_conflict(
    tiny_model_dir: Path, tmp_path: Path
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _write(data / "laptop_train.jsonl")
    _write(data / "laptop_test.jsonl")
    out = tmp_path / "out"
    code = main(
        [
            "--domain", "laptop", "--data-dir", str(data), "--output-dir", str(out),
            "--model-name", str(tiny_model_dir), "--lr", "0.001", "0.002", "--epochs", "1",
            "--batch-size", "4", "--max-length", "32", "--device", "cpu",
        ]
    )  # fmt: skip
    assert code == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert [run["learning_rate"] for run in summary["runs"]] == [0.001, 0.002]
    assert summary["test_records"] == len(RECORDS) - 1
    assert (out / "test_report.json").is_file()
