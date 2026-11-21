import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from absa.models import LstmConfig, save_checkpoint, train
from scripts.evaluate_lstm import main


def _rec(review: str, aspect: str, sentiment: str) -> dict[str, object]:
    return {"review": review, "aspect": aspect, "sentiment": sentiment}


RECORDS = [
    _rec("The battery is weak.", "battery", "negative"),
    _rec("Great screen on this one.", "screen", "positive"),
    _rec("The speed really impressed me.", "speed", "positive"),
    _rec("Terrible keyboard, keys stick.", "keyboard", "negative"),
]
CONFIG = LstmConfig(embedding_dim=8, hidden_size=4, batch_size=4, epochs=1, val_fraction=0.25)


@pytest.fixture
def checkpoint_and_test_file(tmp_path: Path) -> tuple[Path, Path]:
    model, vocab, _, _ = train(RECORDS, RECORDS[:1], CONFIG)
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(checkpoint_path, model, vocab, CONFIG)
    test_file = tmp_path / "test.jsonl"
    test_file.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n", encoding="utf-8")
    return checkpoint_path, test_file


def test_main_prints_report(
    checkpoint_and_test_file: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint_path, test_file = checkpoint_and_test_file
    exit_code = main([str(checkpoint_path), str(test_file)])
    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert "confusion_matrix" in report


def test_main_writes_output_file(checkpoint_and_test_file: tuple[Path, Path]) -> None:
    checkpoint_path, test_file = checkpoint_and_test_file
    output_path = test_file.parent / "report.json"
    main([str(checkpoint_path), str(test_file), "--output", str(output_path)])
    assert output_path.is_file()
