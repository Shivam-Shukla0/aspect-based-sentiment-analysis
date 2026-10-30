import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from absa.__main__ import main, validate_lines

GOOD = {"review": "The battery is weak.", "aspect": "battery", "sentiment": "negative"}


def test_validate_lines_accepts_good_records_and_skips_blanks() -> None:
    lines = [json.dumps(GOOD), "", "  \n"]
    assert validate_lines(lines, ["negative"]) == []


def test_validate_lines_reports_line_numbers() -> None:
    lines = [json.dumps(GOOD), "not json", json.dumps({**GOOD, "sentiment": "meh"})]
    errors = validate_lines(lines, ["negative"])
    assert len(errors) == 2
    assert errors[0].startswith("line 2:")
    assert errors[1].startswith("line 3:")


def test_main_ok(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps(GOOD) + "\n")
    assert main(["validate", str(path)]) == 0
    assert capsys.readouterr().out.strip() == "ok"


def test_main_reports_errors(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    path.write_text("{}\n")
    stderr = io.StringIO()
    assert main(["validate", str(path)], stderr=stderr) == 1
    assert "line 1: record is missing fields" in stderr.getvalue()


def test_conflict_can_be_dropped(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps({**GOOD, "sentiment": "conflict"}) + "\n")
    assert main(["validate", str(path)]) == 0
    assert main(["validate", str(path), "--drop-conflict"], stderr=io.StringIO()) == 1


def test_module_runs_as_script() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "absa", "--version"], capture_output=True, text=True, check=True
    )
    assert result.stdout.startswith("absa ")
