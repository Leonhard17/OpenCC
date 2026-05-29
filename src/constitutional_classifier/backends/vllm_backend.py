"""In-process vLLM generation backend.

Loads model weights once into GPU memory and serves generation locally. Uses the
tokenizer's chat template to format OpenAI-style messages, and overrides
:meth:`batch_generate` for a single native engine pass. Mirrors REDACT's
``llms/vllm_backend.py``.
"""

from __future__ import annotations

import os

from .base import GenerationBackend


class VLLMBackend(GenerationBackend):
    def __init__(self, model: str, **vllm_kwargs):
        import multiprocessing

        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        from transformers import AutoTokenizer
        from vllm import LLM

        if "download_dir" not in vllm_kwargs and os.environ.get("HF_HOME"):
            vllm_kwargs["download_dir"] = os.environ["HF_HOME"]

        self._model_name = model
        self._llm = LLM(model=model, **vllm_kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(model)

    def _format(self, messages: list[dict]) -> str:
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _sampling_params(self, model: str, max_tokens, temperature, **kwargs):
        from vllm import SamplingParams

        from ..model_config import get_model_config

        config = get_model_config(model)
        return SamplingParams(
            max_tokens=max_tokens if max_tokens is not None else (config.max_tokens or 1024),
            temperature=temperature if temperature is not None else (config.temperature or 0.0),
            top_p=kwargs.get("top_p", 0.95),
        )

    def generate(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs,
    ) -> str:
        params = self._sampling_params(model, max_tokens, temperature, **kwargs)
        outputs = self._llm.generate([self._format(messages)], params)
        return outputs[0].outputs[0].text.strip()

    def batch_generate(
        self,
        messages_list: list[list[dict]],
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs,
    ) -> list[str]:
        if not messages_list:
            return []
        params = self._sampling_params(model, max_tokens, temperature, **kwargs)
        formatted = [self._format(m) for m in messages_list]
        outputs = self._llm.generate(formatted, params)
        return [o.outputs[0].text.strip() for o in outputs]

    @property
    def backend_name(self) -> str:
        return "vllm"

    @property
    def supports_native_batching(self) -> bool:
        return True

    @property
    def supports_parallel_calls(self) -> bool:
        return False  # GPU contention; batch through batch_generate()
