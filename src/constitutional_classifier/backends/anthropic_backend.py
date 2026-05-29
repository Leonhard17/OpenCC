"""Anthropic generation backend (native Messages API).

Accepts OpenAI-format messages and converts them: system messages are pulled out into the
``system`` parameter, the rest stay in ``messages``. Mirrors REDACT's
``llms/anthropic_backend.py``.
"""

from __future__ import annotations

import os

from .base import GenerationBackend


class AnthropicBackend(GenerationBackend):
    def __init__(self, api_key: str):
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)

    @classmethod
    def from_env(cls, api_key_var: str = "ANTHROPIC_API_KEY") -> "AnthropicBackend":
        return cls(api_key=os.environ[api_key_var])

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

        system_parts: list[str] = []
        conv: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                conv.append({"role": msg["role"], "content": msg["content"]})

        call_kwargs: dict = {
            "model": config.hf_repo or model,
            "messages": conv,
            # Anthropic requires max_tokens; fall back to a sane default.
            "max_tokens": resolved_max_tokens or 1024,
            **kwargs,
        }
        if system_parts:
            call_kwargs["system"] = "\n\n".join(system_parts)
        if resolved_temp is not None:
            call_kwargs["temperature"] = resolved_temp

        response = self._client.messages.create(**call_kwargs)
        if not response.content:
            return ""
        return response.content[0].text or ""

    @property
    def backend_name(self) -> str:
        return "anthropic"

    @property
    def supports_parallel_calls(self) -> bool:
        return False  # tight RPM/TPM limits
