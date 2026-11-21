"""Language model client protocol, HTTP adapter and structured JSON parsing."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Protocol

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)
JSON_ARRAY_RE = re.compile(r"\[\s*\{.*?\}\s*\]", re.DOTALL)


class LLMClient(Protocol):
    """Protocol for language model text generation."""

    def generate(self, prompt: str) -> str:
        """Generate text completion from a prompt string."""
        ...


class MockLLMClient:
    """Mock client returning canned responses for deterministic testing."""

    def __init__(
        self, responses: dict[str, str] | None = None, default_response: str = "[]"
    ) -> None:
        """Initialize mock responses map and default output."""
        self.responses: dict[str, str] = responses or {}
        self.default_response: str = default_response
        self.prompts_received: list[str] = []

    def generate(self, prompt: str) -> str:
        """Return registered or default response."""
        self.prompts_received.append(prompt)
        return self.responses.get(prompt, self.default_response)


class HttpLLMClient:
    """HTTP client for API endpoints reading credentials from environment variables."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize HTTP client configuration."""
        self.model: str = model or os.getenv("LLM_MODEL") or "gpt-3.5-turbo"
        self.api_key: str = api_key or os.getenv("LLM_API_KEY") or ""
        base = api_base or os.getenv("LLM_API_BASE") or "https://api.openai.com/v1"
        self.api_base: str = base.rstrip("/")
        self.timeout: float = timeout

    def generate(self, prompt: str) -> str:
        """Invoke API endpoint and return text content.

        Raises:
            ValueError: If no API key is configured.
            RuntimeError: On a network error, a timeout or a response that is not valid
                chat-completion JSON.
        """
        if not self.api_key:
            raise ValueError("LLM_API_KEY environment variable is not configured")

        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            }
        ).encode("utf-8")

        req = urllib.request.Request(url, data=payload, headers=headers)  # noqa: S310
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
                return str(data["choices"][0]["message"]["content"])
        except (urllib.error.URLError, TimeoutError) as err:
            raise RuntimeError(f"HTTP LLM generation failed: {err}") from err
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as err:
            raise RuntimeError(f"HTTP LLM returned an unexpected response: {err!r}") from err


def _validate_raw_list(items: list[object]) -> list[dict[str, str]]:
    """Filter and type-cast list of objects into dicts with aspect and sentiment."""
    return [
        {
            "aspect": str(item["aspect"]),
            "sentiment": str(item["sentiment"]).lower().strip(),
        }
        for item in items
        if isinstance(item, dict) and "aspect" in item and "sentiment" in item
    ]


def _extract_json_candidate(text: str) -> str | None:
    """Attempt to locate a JSON array string within the text."""
    if text.startswith("[") and text.endswith("]"):
        return text

    fence_match = JSON_FENCE_RE.search(text)
    if fence_match:
        return fence_match.group(1).strip()

    array_match = JSON_ARRAY_RE.search(text)
    if array_match:
        return array_match.group(0).strip()

    return None


def parse_llm_json(response: str) -> list[dict[str, str]]:
    """Extract and parse a JSON array of aspect-sentiment objects from raw LLM output.

    Args:
        response: Raw string completion from the language model.

    Returns:
        List of parsed aspect-sentiment dictionaries.

    Raises:
        ValueError: If no valid JSON array can be parsed.
    """
    cleaned = response.strip()
    if not cleaned:
        return []

    candidate = _extract_json_candidate(cleaned)
    if candidate:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return _validate_raw_list(parsed)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse valid JSON array from response: {cleaned[:100]!r}")
