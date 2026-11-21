"""Tests for LLM client adapter and structured JSON parsing."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from types import TracebackType
from typing import Any

import pytest

from absa.models.llm import (
    HttpLLMClient,
    MockLLMClient,
    parse_llm_json,
)


def test_mock_client_generate() -> None:
    client = MockLLMClient(
        responses={"prompt_1": '[{"aspect": "battery", "sentiment": "positive"}]'},
        default_response="[]",
    )
    assert client.generate("prompt_1") == '[{"aspect": "battery", "sentiment": "positive"}]'
    assert client.generate("unknown") == "[]"
    assert len(client.prompts_received) == 2


def test_http_client_missing_key() -> None:
    client = HttpLLMClient(api_key="", api_base="http://localhost")
    with pytest.raises(ValueError, match="LLM_API_KEY environment variable is not configured"):
        client.generate("test prompt")


def test_parse_llm_json_clean_array() -> None:
    raw = (
        '[{"aspect": "screen", "sentiment": "positive"}, '
        '{"aspect": "battery", "sentiment": "negative"}]'
    )
    parsed = parse_llm_json(raw)
    assert len(parsed) == 2
    assert parsed[0] == {"aspect": "screen", "sentiment": "positive"}
    assert parsed[1] == {"aspect": "battery", "sentiment": "negative"}


def test_parse_llm_json_markdown_code_fence() -> None:
    raw = """Here are the extracted aspects:
```json
[
  {"aspect": "service", "sentiment": "neutral"}
]
```
Let me know if you need anything else!"""
    parsed = parse_llm_json(raw)
    assert len(parsed) == 1
    assert parsed[0] == {"aspect": "service", "sentiment": "neutral"}


def test_parse_llm_json_empty_array() -> None:
    assert parse_llm_json("[]") == []
    assert parse_llm_json("") == []


def test_parse_llm_json_malformed() -> None:
    with pytest.raises(ValueError, match="Could not parse valid JSON array"):
        parse_llm_json("I cannot identify any aspects in this review.")


class _FakeResponse:
    """Minimal context-manager stand-in for an HTTP response."""

    def __init__(self, body: bytes) -> None:
        self._body = io.BytesIO(body)

    def read(self) -> bytes:
        return self._body.read()

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


def test_http_client_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "secret")
    monkeypatch.setenv("LLM_API_BASE", "http://example.test/v1/")
    client = HttpLLMClient()
    assert client.model == "test-model"
    assert client.api_key == "secret"
    assert client.api_base == "http://example.test/v1"


def test_http_client_generate_success(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(req: urllib.request.Request, timeout: float) -> _FakeResponse:
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["auth"] = req.get_header("Authorization")
        captured["payload"] = json.loads(req.data.decode("utf-8"))  # type: ignore[union-attr]
        body = {"choices": [{"message": {"content": "[]"}}]}
        return _FakeResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = HttpLLMClient(model="m", api_key="k", api_base="http://example.test/v1", timeout=5.0)
    assert client.generate("hello") == "[]"
    assert captured["url"] == "http://example.test/v1/chat/completions"
    assert captured["timeout"] == 5.0
    assert captured["auth"] == "Bearer k"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "hello"}]


def test_http_client_generate_url_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(*_args: object, **_kwargs: object) -> _FakeResponse:
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = HttpLLMClient(model="m", api_key="k", api_base="http://example.test")
    with pytest.raises(RuntimeError, match="HTTP LLM generation failed"):
        client.generate("hello")


def test_parse_llm_json_embedded_array_without_fence() -> None:
    raw = 'Sure! [{"aspect": "price", "sentiment": "Negative "}] Hope this helps.'
    assert parse_llm_json(raw) == [{"aspect": "price", "sentiment": "negative"}]


def test_parse_llm_json_skips_invalid_items() -> None:
    raw = '[{"aspect": "a"}, "text", {"aspect": "b", "sentiment": "positive"}]'
    assert parse_llm_json(raw) == [{"aspect": "b", "sentiment": "positive"}]


def test_parse_llm_json_invalid_json_inside_brackets() -> None:
    with pytest.raises(ValueError, match="Could not parse valid JSON array"):
        parse_llm_json("[{aspect: broken}]")


def test_parse_llm_json_invalid_json_in_fence() -> None:
    with pytest.raises(ValueError, match="Could not parse valid JSON array"):
        parse_llm_json('```json\n[{"aspect": "x", "sentiment": "positive"} 5]\n```')


def test_http_client_requires_model_and_base(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LLM_MODEL", "LLM_API_KEY", "LLM_API_BASE"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="LLM_MODEL"):
        HttpLLMClient(api_key="k", api_base="http://example.test").generate("hi")
    with pytest.raises(ValueError, match="LLM_API_BASE"):
        HttpLLMClient(api_key="k", model="m").generate("hi")
