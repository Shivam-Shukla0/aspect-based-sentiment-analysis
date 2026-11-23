"""A language-model client that runs an open-weights chat model on the local GPU.

It implements the ``LLMClient`` protocol, so the prompts, JSON parsing and validation of the
other approaches apply unchanged. Decoding is greedy, so the same prompt gives the same
answer, and ``generate_batch`` runs several prompts at once for speed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from absa.models.lstm_attention import resolve_device


# How the LLM approach runs without an API key: the open-weights model Qwen2.5-7B-Instruct runs
# on the Colab GPU with the transformers library, using a versioned few-shot prompt (v1.0) and
# greedy decoding (max 256 new tokens). Greedy decoding makes results reproducible. Every raw
# answer is saved, so scores can be recomputed without the model; cost is not applicable.
class LocalLLMClient:
    """Chat model loaded with ``transformers``; no API key and no network call at run time."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_new_tokens: int = 256,
    ) -> None:
        """Load the tokenizer and the model (bfloat16 on a GPU) and move it to ``device``."""
        # Imports are local: torch/transformers are large and only needed when a model is loaded.
        import torch  # noqa: PLC0415  (heavy, only needed when a model is loaded)
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._device = resolve_device(device)
        # bfloat16 halves memory on GPU; CPU stays in float32, where bf16 is slow/unsupported.
        dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
        # Decoder-only models must be left-padded so generation continues from real text.
        self._tokenizer: Any = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self._model: Any = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        self._model.to(self._device)
        self._model.eval()  # inference mode: disables dropout

    def generate(self, prompt: str) -> str:
        """Return the model's answer to one prompt."""
        return self.generate_batch([prompt])[0]

    def generate_batch(self, prompts: Sequence[str]) -> list[str]:
        """Return one answer per prompt, in order, using greedy decoding."""
        import torch  # noqa: PLC0415

        # Wrap each prompt in the model's own chat format (roles, special tokens).
        texts = [
            self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            for prompt in prompts
        ]
        # Many chat models define no pad token; reuse end-of-sequence so batching works.
        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self._tokenizer.eos_token_id
        encoded = self._tokenizer(texts, return_tensors="pt", padding=True).to(self._device)
        # no_grad: no gradients are needed to generate text, which saves memory.
        with torch.no_grad():
            output = self._model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
        # generate() returns prompt + answer; cut off the prompt part so only the answer remains.
        new_tokens = output[:, encoded["input_ids"].shape[1] :]
        return [self._tokenizer.decode(row, skip_special_tokens=True) for row in new_tokens]
