"""Tests for LLM client adapter and structured JSON parsing."""

from __future__ import annotations

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
