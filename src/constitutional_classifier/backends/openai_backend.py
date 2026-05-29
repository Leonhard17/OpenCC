"""OpenAI-compatible generation backend.

Works against any ``/v1/chat/completions`` endpoint: OpenAI, Venice, or a local vLLM
server started with ``vllm serve``. Backend-level defaults (max_tokens, temperature) come
from the model's registry entry; explicit per-call kwargs override them.
"""

from __future__ import annotations

import os

from .base import GenerationBackend


class OpenAIBackend(GenerationBackend):
    def __init__(self, api_key: str, base_url: str | None = None):
        import openai

        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

    @classmethod
    def from_env(
        cls,
        api_key_var: str = "OPENAI_API_KEY",
        base_url_var: str = "OPENAI_BASE_URL",
    ) -> "OpenAIBackend":
        # A local vLLM/Ollama server ignores the key but the SDK requires a non-empty one.
        api_key = os.environ.get(api_key_var) or "EMPTY"
        base_url = os.environ.get(base_url_var) or None
        return cls(api_key=api_key, base_url=base_url)

    def generate(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs,
    ) -> str:
        from ..model_config import get_model_config

        config = get_model_config(model)
        resolved_max_tokens = max_tokens if max_tokens is not None else config.max_tokens
        resolved_temp = temperature if temperature is not None else config.temperature

        call_kwargs: dict = {
            "model": config.hf_repo or model,
            "messages": messages,
            **kwargs,
        }
        if resolved_max_tokens is not None:
            call_kwargs["max_tokens"] = resolved_max_tokens
        if resolved_temp is not None:
            call_kwargs["temperature"] = resolved_temp

        response = self._client.chat.completions.create(**call_kwargs)
        content = response.choices[0].message.content
        return content if content is not None else ""

    @property
    def backend_name(self) -> str:
        return "openai"
