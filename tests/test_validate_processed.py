import json
from pathlib import Path

import pytest

from scripts.validate_processed import main

VALID = {"review": "The battery is weak.", "aspect": "battery", "sentiment": "negative"}


def test_main_returns_zero_on_valid_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(VALID) + "\n", encoding="utf-8")
    assert main([str(path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["invalid_records"] == 0


def test_main_returns_one_on_invalid_file(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({**VALID, "sentiment": "bad"}) + "\n", encoding="utf-8")
    assert main([str(path)]) == 1


def test_main_writes_report_when_output_given(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(VALID) + "\n", encoding="utf-8")
    report = tmp_path / "report.json"
    main([str(path), "--output", str(report)])
    assert json.loads(report.read_text(encoding="utf-8"))["total_records"] == 1
