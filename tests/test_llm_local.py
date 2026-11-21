from typing import Any, ClassVar

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
import torch
import transformers

from absa.models.llm_local import LocalLLMClient


class _Encoding(dict[str, Any]):
    def to(self, _device: object) -> "_Encoding":
        return self


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    @classmethod
    def from_pretrained(cls, _name: str, **_kwargs: object) -> "_Tokenizer":
        return cls()

    def apply_chat_template(self, messages: list[dict[str, str]], **_kwargs: object) -> str:
        return "chat:" + messages[0]["content"]

    def __call__(self, texts: list[str], **_kwargs: object) -> _Encoding:
        return _Encoding(input_ids=torch.ones((len(texts), 3), dtype=torch.long))

    def decode(self, row: torch.Tensor, skip_special_tokens: bool = True) -> str:  # noqa: FBT001, FBT002
        assert skip_special_tokens
        return "answer" + "".join(str(i) for i in row.tolist())


class _Model:
    calls: ClassVar[list[dict[str, object]]] = []

    @classmethod
    def from_pretrained(cls, _name: str, **_kwargs: object) -> "_Model":
        return cls()

    def to(self, _device: object) -> "_Model":
        return self

    def eval(self) -> None:
        return None

    def generate(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        _Model.calls.append(kwargs)
        new = torch.tensor([[7, 8]] * input_ids.shape[0])
        return torch.cat([input_ids, new], dim=1)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> LocalLLMClient:
    monkeypatch.setattr(transformers, "AutoTokenizer", _Tokenizer)
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", _Model)
    return LocalLLMClient("fake", device="cpu", max_new_tokens=5)


def test_batch_returns_only_the_new_tokens_in_order(client: LocalLLMClient) -> None:
    assert client.generate_batch(["a", "b"]) == ["answer78", "answer78"]


def test_decoding_is_greedy_and_bounded(client: LocalLLMClient) -> None:
    client.generate("a")
    kwargs = _Model.calls[-1]
    assert kwargs["do_sample"] is False
    assert kwargs["max_new_tokens"] == 5
    assert kwargs["pad_token_id"] == 0


def test_single_prompt_uses_the_batch_path(client: LocalLLMClient) -> None:
    assert client.generate("a") == "answer78"
