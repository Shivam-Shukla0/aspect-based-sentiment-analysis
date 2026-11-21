"""Errors raised by language-model clients."""

from __future__ import annotations


class LLMClientError(RuntimeError):
    """A language-model call failed.

    ``retryable`` is True for failures that can succeed on another attempt (network errors,
    timeouts, HTTP 429 and 5xx, a garbled response) and False for requests that the service
    rejected (for example HTTP 400 or 401) or that returned no text.
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        """Store the message and whether another attempt is worthwhile."""
        super().__init__(message)
        self.retryable = retryable
