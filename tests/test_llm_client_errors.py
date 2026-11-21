import email.message
import http.client
import io
import urllib.error

import pytest

from absa.models.llm import HttpLLMClient
from absa.models.llm_errors import LLMClientError


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _client(monkeypatch: pytest.MonkeyPatch, outcome: object) -> HttpLLMClient:
    def fake_urlopen(request: object, timeout: float) -> _Response:
        del request, timeout
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, bytes)
        return _Response(outcome)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return HttpLLMClient(model="m", api_key="k", api_base="https://example.test/v1")


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://example.test", code, "error", email.message.Message(), io.BytesIO()
    )


@pytest.mark.parametrize(
    "outcome",
    [
        ConnectionResetError("reset"),
        TimeoutError("timed out"),
        http.client.IncompleteRead(b"partial"),
        http.client.RemoteDisconnected("closed"),
        b"\xff\xfe not utf-8",
        b"not json",
        b'{"choices": []}',
        b'{"other": 1}',
        b"[1, 2]",
    ],
)
def test_transport_and_format_failures_become_retryable_client_errors(
    monkeypatch: pytest.MonkeyPatch, outcome: object
) -> None:
    with pytest.raises(LLMClientError) as info:
        _client(monkeypatch, outcome).generate("prompt")
    assert info.value.retryable is True


def test_null_content_is_an_error_not_the_string_none(monkeypatch: pytest.MonkeyPatch) -> None:
    body = b'{"choices": [{"message": {"content": null}}]}'
    with pytest.raises(LLMClientError, match="no text content") as info:
        _client(monkeypatch, body).generate("prompt")
    assert info.value.retryable is False


@pytest.mark.parametrize("code", [400, 401, 404])
def test_client_side_http_errors_are_not_retryable(
    monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    with pytest.raises(LLMClientError) as info:
        _client(monkeypatch, _http_error(code)).generate("prompt")
    assert info.value.retryable is False


@pytest.mark.parametrize("code", [429, 500, 503])
def test_rate_limit_and_server_errors_are_retryable(
    monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    with pytest.raises(LLMClientError) as info:
        _client(monkeypatch, _http_error(code)).generate("prompt")
    assert info.value.retryable is True
