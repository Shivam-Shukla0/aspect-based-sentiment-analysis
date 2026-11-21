"""Flask backend: serves the demo page and a ``/predict`` endpoint over trained classifiers.

The LSTM and BERT models classify aspect terms that the caller supplies, so a request carries
the review, the model name and the aspect terms. Models are loaded on first use and cached.
A model whose checkpoint is missing answers 503 with a message and does not stop the others.

Run from the repository root with ``python -m app.server``. Debug mode is off.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request

from absa.config import OUTPUTS_DIR
from absa.inference import (
    AspectClassifierPredictor,
    CheckpointNotFoundError,
    InferenceError,
    PredictionValidationError,
    load_bert_predictor,
    load_lstm_predictor,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_REVIEW_LENGTH = 2000
MAX_ASPECT_LENGTH = 100
MAX_ASPECTS = 20
MAX_BODY_BYTES = 64 * 1024

Factory = Callable[[], AspectClassifierPredictor]


def _checkpoint(env_name: str, output_dir: Path, filename: str | None) -> Path:
    """Return the path from ``env_name``, else the checkpoint recorded by the tuning script."""
    configured = os.environ.get(env_name)
    if configured:
        return Path(configured)
    summary = output_dir / "summary.json"
    if summary.is_file():
        recorded = json.loads(summary.read_text(encoding="utf-8")).get("checkpoint")
        if recorded:
            return Path(recorded)
    return output_dir / (filename or "model")


def _unavailable_llm() -> AspectClassifierPredictor:
    raise CheckpointNotFoundError(
        "The language-model backend needs an endpoint and a key that are not configured here"
    )


def default_factories() -> dict[str, Factory]:
    """Return the factories for the models the demo page offers.

    The domain comes from ``ABSA_DOMAIN`` (default ``laptop``) and the device from
    ``ABSA_DEVICE`` (default ``cpu``). ``ABSA_LSTM_CHECKPOINT`` and ``ABSA_BERT_MODEL_DIR``
    override the paths read from ``outputs/<model>_<domain>/summary.json``.
    """
    domain = os.environ.get("ABSA_DOMAIN", "laptop")
    device = os.environ.get("ABSA_DEVICE", "cpu")
    lstm_dir = OUTPUTS_DIR / f"lstm_{domain}"
    bert_dir = OUTPUTS_DIR / f"bert_{domain}"
    return {
        "lstm": lambda: load_lstm_predictor(
            _checkpoint("ABSA_LSTM_CHECKPOINT", lstm_dir, "lstm_checkpoint.pt"), device
        ),
        "bert": lambda: load_bert_predictor(
            _checkpoint("ABSA_BERT_MODEL_DIR", bert_dir, "model"), device
        ),
        "llm": _unavailable_llm,
    }


def _error(message: str, status: int) -> tuple[Response, int]:
    return jsonify({"error": message}), status


def _parse(payload: object) -> tuple[str, str, list[str]] | str:
    """Return ``(review, model, aspects)`` from a request body, or an error message."""
    if not isinstance(payload, dict):
        return "the request body must be a JSON object"
    review, model, aspects = payload.get("review"), payload.get("model"), payload.get("aspects")
    if not isinstance(review, str) or not review.strip():
        return "review must be a non-empty string"
    if len(review) > MAX_REVIEW_LENGTH:
        return f"review is longer than {MAX_REVIEW_LENGTH} characters"
    if not isinstance(model, str) or not model.strip():
        return "model must be a non-empty string"
    if not isinstance(aspects, list) or not aspects:
        return "aspects must be a non-empty list of aspect terms that occur in the review"
    if len(aspects) > MAX_ASPECTS:
        return f"at most {MAX_ASPECTS} aspects are allowed"
    if not all(isinstance(a, str) and a.strip() and len(a) <= MAX_ASPECT_LENGTH for a in aspects):
        return f"each aspect must be a non-empty string of at most {MAX_ASPECT_LENGTH} characters"
    return review, model.strip().lower(), [a.strip() for a in aspects]


class _Service:
    """Holds the model registry and the models loaded so far."""

    def __init__(self, registry: Mapping[str, Factory]) -> None:
        self.registry = dict(registry)
        self.loaded: dict[str, AspectClassifierPredictor] = {}
        self._lock = threading.Lock()

    def predictor(self, name: str) -> AspectClassifierPredictor:
        """Return the model ``name``, loading it on first use."""
        with self._lock:
            if name not in self.loaded:
                self.loaded[name] = self.registry[name]()
            return self.loaded[name]

    def predict(self, payload: object) -> tuple[Response, int]:
        """Validate a request body, run the model and build the HTTP answer."""
        parsed = _parse(payload)
        if isinstance(parsed, str):
            return _error(parsed, 400)
        review, model, aspects = parsed
        if model not in self.registry:
            return _error(f"unknown model {model!r}; available: {sorted(self.registry)}", 404)
        missing = [a for a in aspects if a.lower() not in review.lower()]
        if missing:
            return _error(f"aspect terms not found in the review: {missing}", 422)
        try:
            record = self.predictor(model).predict_aspects(review, aspects)
        except CheckpointNotFoundError as err:
            return _error(f"model {model!r} is not available: {err}", 503)
        except PredictionValidationError as err:
            return _error(str(err), 422)
        except InferenceError as err:
            return _error(str(err), 500)
        except Exception:  # a loader failing for any other reason must not leak a traceback
            logging.getLogger(__name__).exception("loading model %s failed", model)
            return _error(f"model {model!r} could not be loaded", 503)
        return jsonify(record.to_dict()), 200


def create_app(factories: Mapping[str, Factory] | None = None) -> Flask:
    """Build the Flask application. ``factories`` maps model names to loader functions."""
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
    app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_BYTES
    service = _Service(default_factories() if factories is None else factories)

    @app.get("/")
    def index() -> Response:
        return app.send_static_file("index.html")

    @app.get("/health")
    def health() -> Response:
        return jsonify({"status": "ok"})

    @app.get("/models")
    def models() -> Response:
        return jsonify({"models": sorted(service.registry), "loaded": sorted(service.loaded)})

    @app.post("/predict")
    def predict() -> tuple[Response, int]:
        if (request.content_length or 0) > MAX_BODY_BYTES:
            return _error(f"the request body is larger than {MAX_BODY_BYTES} bytes", 413)
        return service.predict(request.get_json(silent=True))

    @app.errorhandler(404)
    @app.errorhandler(405)
    @app.errorhandler(413)
    def http_error(error: Any) -> tuple[Response, int]:
        return _error(str(error.description), int(error.code))

    return app


def main() -> None:  # pragma: no cover
    """Serve on ``ABSA_HOST``:``ABSA_PORT`` (default 127.0.0.1:8000) with debug off."""
    host = os.environ.get("ABSA_HOST", "127.0.0.1")
    port = int(os.environ.get("ABSA_PORT", "8000"))
    create_app().run(host=host, port=port, debug=False)


if __name__ == "__main__":  # pragma: no cover
    main()
