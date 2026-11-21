from pathlib import Path

import pytest

pytest.importorskip("flask")
from flask.testing import FlaskClient

from absa.inference import AspectClassifierPredictor, CheckpointNotFoundError, InferenceError
from app import server

REVIEW = "The battery is weak but the screen is excellent."


def _good() -> AspectClassifierPredictor:
    return AspectClassifierPredictor(lambda _r, a: "negative" if a == "battery" else "positive")


def _missing() -> AspectClassifierPredictor:
    raise CheckpointNotFoundError("Checkpoint not found: outputs/bert_laptop/model")


def _crashing_loader() -> AspectClassifierPredictor:
    raise RuntimeError("corrupt file")


def _failing() -> AspectClassifierPredictor:
    def classify(_review: str, _aspect: str) -> str:
        raise InferenceError("model blew up")

    return AspectClassifierPredictor(classify)


@pytest.fixture
def client() -> FlaskClient:
    factories = {"lstm": _good, "bert": _missing, "broken": _crashing_loader, "bad": _failing}
    return server.create_app(factories).test_client()


def _post(client: FlaskClient, **fields: object) -> tuple[int, dict[str, object]]:
    body = {"review": REVIEW, "model": "lstm", "aspects": ["battery", "screen"], **fields}
    response = client.post("/predict", json=body)
    return response.status_code, response.get_json()


def test_a_valid_request_returns_the_shared_prediction_shape(client: FlaskClient) -> None:
    status, body = _post(client)
    assert status == 200
    assert body == {
        "review": REVIEW,
        "predictions": [
            {"aspect": "battery", "sentiment": "negative"},
            {"aspect": "screen", "sentiment": "positive"},
        ],
    }


def test_the_model_name_is_case_insensitive(client: FlaskClient) -> None:
    assert _post(client, model=" LSTM ")[0] == 200


@pytest.mark.parametrize(
    "fields",
    [
        {"review": ""},
        {"review": 5},
        {"review": "x" * 2001},
        {"model": ""},
        {"model": 3},
        {"aspects": []},
        {"aspects": "battery"},
        {"aspects": ["battery", ""]},
        {"aspects": ["a"] * 21},
        {"aspects": ["b" * 101]},
    ],
)
def test_invalid_fields_get_a_400_with_a_message(
    client: FlaskClient, fields: dict[str, object]
) -> None:
    status, body = _post(client, **fields)
    assert status == 400
    assert isinstance(body["error"], str)


def test_a_body_that_is_not_a_json_object_gets_a_400(client: FlaskClient) -> None:
    assert client.post("/predict", data="not json").status_code == 400
    assert client.post("/predict", json=[1, 2]).status_code == 400


def test_an_unknown_model_gets_a_404(client: FlaskClient) -> None:
    status, body = _post(client, model="nope")
    assert status == 404
    assert "lstm" in str(body["error"])


def test_an_aspect_that_is_not_in_the_review_gets_a_422(client: FlaskClient) -> None:
    status, body = _post(client, aspects=["keyboard"])
    assert status == 422
    assert "keyboard" in str(body["error"])


def test_a_missing_checkpoint_gets_a_503_and_other_models_still_work(client: FlaskClient) -> None:
    status, body = _post(client, model="bert")
    assert status == 503
    assert "not available" in str(body["error"])
    assert _post(client)[0] == 200


def test_a_loader_that_crashes_gets_a_503_without_a_traceback(client: FlaskClient) -> None:
    status, body = _post(client, model="broken")
    assert status == 503
    assert "corrupt" not in str(body["error"])


def test_a_failing_model_gets_a_500(client: FlaskClient) -> None:
    status, body = _post(client, model="bad")
    assert status == 500
    assert "blew up" in str(body["error"])


def test_models_are_loaded_once() -> None:
    calls: list[int] = []

    def counting() -> AspectClassifierPredictor:
        calls.append(1)
        return _good()

    app_client = server.create_app({"lstm": counting}).test_client()
    for _ in range(3):
        assert _post(app_client)[0] == 200
    assert len(calls) == 1
    assert app_client.get("/models").get_json() == {"models": ["lstm"], "loaded": ["lstm"]}


def test_health_models_and_static_page(client: FlaskClient) -> None:
    assert client.get("/health").get_json() == {"status": "ok"}
    assert client.get("/models").get_json()["models"] == ["bad", "bert", "broken", "lstm"]
    page = client.get("/")
    assert page.status_code == 200
    assert b"Aspect-based sentiment analysis" in page.data
    page.close()


def test_other_http_errors_are_json(client: FlaskClient) -> None:
    missing = client.get("/nothing")
    assert (missing.status_code, "error" in missing.get_json()) == (404, True)
    wrong_method = client.put("/predict")
    assert wrong_method.status_code == 405
    assert "error" in wrong_method.get_json()
    too_big = client.post("/predict", data="x" * (server.MAX_BODY_BYTES + 1))
    assert too_big.status_code == 413


def test_default_factories_offer_the_demo_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ABSA_LSTM_CHECKPOINT", raising=False)
    monkeypatch.delenv("ABSA_BERT_MODEL_DIR", raising=False)
    factories = server.default_factories()
    assert sorted(factories) == ["bert", "llm", "lstm"]
    with pytest.raises(CheckpointNotFoundError):
        factories["llm"]()


def test_checkpoint_comes_from_env_then_summary_then_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("X_PATH", raising=False)
    assert server._checkpoint("X_PATH", tmp_path, "ckpt.pt") == tmp_path / "ckpt.pt"
    (tmp_path / "summary.json").write_text('{"checkpoint": "/somewhere/best.pt"}', encoding="utf-8")
    assert server._checkpoint("X_PATH", tmp_path, "ckpt.pt") == Path("/somewhere/best.pt")
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    assert server._checkpoint("X_PATH", tmp_path, None) == tmp_path / "model"
    monkeypatch.setenv("X_PATH", "/env/path")
    assert server._checkpoint("X_PATH", tmp_path, "ckpt.pt") == Path("/env/path")
