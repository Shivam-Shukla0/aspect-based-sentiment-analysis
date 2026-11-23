"""Language model client protocol, HTTP adapter and structured JSON parsing."""

from __future__ import annotations

import http.client
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Protocol

from absa.models.llm_errors import LLMClientError


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
        # Record every prompt so tests can assert exactly what would have been sent.
        return self.responses.get(prompt, self.default_response)


# Status codes that mean "try again later" (rate limit / server-side problem).
_HTTP_TOO_MANY_REQUESTS = 429
_HTTP_SERVER_ERROR = 500


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
        """Invoke API endpoint and return text content.

        Raises:
            ValueError: If no API key is configured.
            LLMClientError: On a network error, a timeout, an HTTP error, a response that is
                not valid chat-completion JSON, or a reply without text. Its ``retryable``
                flag tells a retry wrapper whether another attempt makes sense.
        """
        # Credentials come from the environment, never from source code; fail clearly if unset.
        if not self.api_key:
            raise ValueError("LLM_API_KEY environment variable is not configured")
        if not self.model:
            raise ValueError("LLM_MODEL environment variable is not configured")
        if not self.api_base:
            raise ValueError("LLM_API_BASE environment variable is not configured")

        # Security: only https is allowed (the API key travels in a header); plain http is
        # permitted for a local server only.
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
                "temperature": 0.0,  # 0 = as deterministic as the service allows
            }
        ).encode("utf-8")

        req = urllib.request.Request(url, data=payload, headers=headers)  # noqa: S310
        return self._send(req)

    def _send(self, req: urllib.request.Request) -> str:
        """Send ``req`` and return the reply text, raising ``LLMClientError`` on any failure."""
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
        # HTTPError must be caught before OSError (it is a subclass of it).
        except urllib.error.HTTPError as err:
            # 429 and 5xx may succeed later; other 4xx (bad request, bad key) never will.
            retryable = err.code == _HTTP_TOO_MANY_REQUESTS or err.code >= _HTTP_SERVER_ERROR
            raise LLMClientError(
                f"HTTP LLM generation failed: status {err.code}", retryable=retryable
            ) from err
        except (OSError, http.client.HTTPException) as err:
            raise LLMClientError(f"HTTP LLM generation failed: {err}") from err
        except (ValueError, KeyError, IndexError, TypeError) as err:
            raise LLMClientError(f"HTTP LLM returned an unexpected response: {err!r}") from err
        if not isinstance(content, str):
            raise LLMClientError("HTTP LLM returned no text content", retryable=False)
        return content


def _validate_raw_list(items: list[object]) -> list[dict[str, str]]:
    """Type-cast list of objects into dicts with aspect and sentiment.

    Raises:
        ValueError: If any item is not an object with both an ``aspect`` and a
            ``sentiment`` key, or if either value is null. Dropping such items silently
            would look like "no aspects", and ``str(None)`` would turn null into ``"None"``.
    """
    parsed: list[dict[str, str]] = []
    # Strict on purpose: silently dropping a malformed item would look like "no aspects".
    for item in items:
        if not isinstance(item, dict) or "aspect" not in item or "sentiment" not in item:
            raise ValueError(f"item must be an object with 'aspect' and 'sentiment': {item!r}")
        for field in ("aspect", "sentiment"):
            if item[field] is None:
                raise ValueError(f"'{field}' must not be null: {item!r}")
        parsed.append(
            {
                "aspect": str(item["aspect"]),
                # Normalise case/whitespace so "Positive " is accepted as "positive".
                "sentiment": str(item["sentiment"]).lower().strip(),
            }
        )
    return parsed


def _find_json_array(text: str) -> list[object] | None:
    """Return the first JSON array in ``text`` that is empty or made of objects.

    Scans every ``[`` so that prose, code fences and empty arrays (``"Sure! []"``) work.
    A single object that is not wrapped in an array is not accepted.
    """
    # raw_decode parses ONE JSON value starting at an index and ignores trailing text, which
    # is exactly what is needed to pull an array out of chatty LLM output.
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "[":
            continue
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue  # this "[" was not the start of valid JSON (e.g. "[note]"); try the next one
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return value
    return None


def parse_llm_json(response: str) -> list[dict[str, str]]:
    """Extract and parse a JSON array of aspect-sentiment objects from raw LLM output.

    The array may be surrounded by prose or a Markdown code fence, and ``[]`` is a valid
    answer meaning that no aspects were found. A single JSON object that is not inside an
    array is rejected. An empty or whitespace-only response is an error, not "no aspects",
    so the caller can retry.

    When several JSON arrays appear in the response, the first one wins: ``"[] then
    [{...}]"`` returns ``[]``.

    Args:
        response: Raw string completion from the language model.

    Returns:
        List of parsed aspect-sentiment dictionaries.

    Raises:
        ValueError: If the response is empty or whitespace-only, no JSON array can be
            found, an item lacks ``aspect`` or ``sentiment``, or either of them is null.
    """
    # An empty reply is an error (not "no aspects") so the retry wrapper can ask again.
    cleaned = response.strip()
    if not cleaned:
        raise ValueError("Empty response from the language model")

    items = _find_json_array(cleaned)
    if items is None:
        # Only the first 100 characters are echoed to keep error messages short.
        raise ValueError(f"Could not parse valid JSON array from response: {cleaned[:100]!r}")
    return _validate_raw_list(items)
