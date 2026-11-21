"""Language model client protocol, HTTP adapter and structured JSON parsing."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Protocol


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
        """Initialize HTTP client configuration.

        Model, API key and base URL have no vendor defaults: pass them explicitly or set
        ``LLM_MODEL``, ``LLM_API_KEY`` and ``LLM_API_BASE``.
        """
        self.model: str = model or os.getenv("LLM_MODEL") or ""
        self.api_key: str = api_key or os.getenv("LLM_API_KEY") or ""
        base = api_base or os.getenv("LLM_API_BASE") or ""
        self.api_base: str = base.rstrip("/")
        self.timeout: float = timeout

    def generate(self, prompt: str) -> str:
        """Invoke API endpoint and return text content."""
        if not self.api_key:
            raise ValueError("LLM_API_KEY environment variable is not configured")
        if not self.model:
            raise ValueError("LLM_MODEL environment variable is not configured")
        if not self.api_base:
            raise ValueError("LLM_API_BASE environment variable is not configured")

        parsed_base = urllib.parse.urlparse(self.api_base)
        is_local = parsed_base.hostname in {"localhost", "127.0.0.1", "::1"}
        if parsed_base.scheme != "https" and not (parsed_base.scheme == "http" and is_local):
            raise ValueError("LLM_API_BASE must use https (http is only allowed for localhost)")

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
        except urllib.error.URLError as err:
            raise RuntimeError(f"HTTP LLM generation failed: {err}") from err


def _validate_raw_list(items: list[object]) -> list[dict[str, str]]:
    """Type-cast list of objects into dicts with aspect and sentiment.

    Raises:
        ValueError: If any item is not an object with both an ``aspect`` and a
            ``sentiment`` key. Dropping such items silently would look like "no aspects".
    """
    parsed: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict) or "aspect" not in item or "sentiment" not in item:
            raise ValueError(f"item must be an object with 'aspect' and 'sentiment': {item!r}")
        parsed.append(
            {
                "aspect": str(item["aspect"]),
                "sentiment": str(item["sentiment"]).lower().strip(),
            }
        )
    return parsed


def _find_json_array(text: str) -> list[object] | None:
    """Return the first JSON array in ``text`` that is empty or made of objects.

    Scans every ``[`` so that prose, code fences and empty arrays (``"Sure! []"``) work.
    A single object that is not wrapped in an array is not accepted.
    """
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "[":
            continue
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return value
    return None


def parse_llm_json(response: str) -> list[dict[str, str]]:
    """Extract and parse a JSON array of aspect-sentiment objects from raw LLM output.

    The array may be surrounded by prose or a Markdown code fence, and ``[]`` is a valid
    answer meaning that no aspects were found. A single JSON object that is not inside an
    array is rejected.

    Args:
        response: Raw string completion from the language model.

    Returns:
        List of parsed aspect-sentiment dictionaries.

    Raises:
        ValueError: If no JSON array can be found, or an item lacks ``aspect`` or
            ``sentiment``.
    """
    cleaned = response.strip()
    if not cleaned:
        return []

    items = _find_json_array(cleaned)
    if items is None:
        raise ValueError(f"Could not parse valid JSON array from response: {cleaned[:100]!r}")
    return _validate_raw_list(items)
