"""Tests for LLM client adapter and structured JSON parsing."""

from __future__ import annotations

import urllib.error

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
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _client_with_response(monkeypatch: pytest.MonkeyPatch, outcome: object) -> HttpLLMClient:
    def fake_urlopen(request: object, timeout: float) -> _FakeResponse:
        del request, timeout
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeResponse(outcome if isinstance(outcome, bytes) else b"")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return HttpLLMClient(api_key="test-key", api_base="http://localhost")


def test_http_client_returns_message_content(monkeypatch: pytest.MonkeyPatch) -> None:
    body = b'{"choices": [{"message": {"content": "[]"}}]}'
    client = _client_with_response(monkeypatch, body)
    assert client.generate("prompt") == "[]"


@pytest.mark.parametrize(
    "outcome",
    [
        urllib.error.URLError("connection refused"),
        TimeoutError("timed out"),
        b"not json",
        b'{"choices": []}',
        b'{"other": 1}',
    ],
)
def test_http_client_wraps_failures_in_runtime_error(
    monkeypatch: pytest.MonkeyPatch, outcome: object
) -> None:
    client = _client_with_response(monkeypatch, outcome)
    with pytest.raises(RuntimeError, match="HTTP LLM"):
        client.generate("prompt")
