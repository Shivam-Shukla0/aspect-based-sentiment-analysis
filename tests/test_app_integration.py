"""End to end: a real (tiny) trained LSTM checkpoint served through the Flask app."""

import json
from pathlib import Path

import pytest

pytest.importorskip("flask")
pytest.importorskip("torch")
pytest.importorskip("sklearn")
from absa.models import LstmConfig, run_training
from app import server

RECORDS = [
    {"review": review, "aspect": aspect, "sentiment": sentiment, "sentence_id": str(index)}
    for index, (review, aspect, sentiment) in enumerate(
        [
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
        ]
    )
]
CONFIG = LstmConfig(
    device="cpu", embedding_dim=8, hidden_size=4, batch_size=4, epochs=1, val_fraction=0.3
)


@pytest.fixture
def trained_dir(tmp_path: Path) -> Path:
    result = run_training(RECORDS, CONFIG, output_dir=tmp_path / "lstm_laptop")
    summary = {"checkpoint": str(result.checkpoint_path)}
    (tmp_path / "lstm_laptop" / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return tmp_path


def test_the_page_form_to_prediction_flow_with_a_trained_model(
    trained_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "OUTPUTS_DIR", trained_dir)
    monkeypatch.delenv("ABSA_LSTM_CHECKPOINT", raising=False)
    monkeypatch.delenv("ABSA_BERT_MODEL_DIR", raising=False)
    monkeypatch.setenv("ABSA_DOMAIN", "laptop")
    client = server.create_app().test_client()

    review = "The battery is weak but the screen is great."
    response = client.post(
        "/predict", json={"review": review, "model": "lstm", "aspects": ["battery", "screen"]}
    )
    body = response.get_json()
    assert response.status_code == 200
    assert body["review"] == review
    assert [p["aspect"] for p in body["predictions"]] == ["battery", "screen"]
    assert {p["sentiment"] for p in body["predictions"]} <= {"positive", "negative", "neutral"}
    assert client.get("/models").get_json()["loaded"] == ["lstm"]

    # BERT was not trained here: a clear 503, and the LSTM keeps working.
    missing = client.post(
        "/predict", json={"review": review, "model": "bert", "aspects": ["battery"]}
    )
    assert missing.status_code == 503
    assert "not available" in missing.get_json()["error"]
    assert (
        client.post(
            "/predict", json={"review": review, "model": "llm", "aspects": ["battery"]}
        ).status_code
        == 503
    )
