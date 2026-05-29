"""Backend routing + caching.

Looks up a model's ``backend_type`` in the registry and returns the matching backend
instance, importing optional dependencies lazily. Caching follows REDACT's rule:

* API backends (openai, anthropic) are cached by *type* — one shared instance serves every
  model of that type.
* Heavyweight backends (vllm, hf_classifier) are cached by *model name* since each loads
  unique weights.

A per-model override map takes precedence over all of this — used by tests and custom
deployments to inject a hand-built backend for a specific model.
"""

from __future__ import annotations

from ..model_config import get_model_config
from .base import ClassifierBackend, GenerationBackend

_gen_cache: dict[str, GenerationBackend] = {}
_clf_cache: dict[str, ClassifierBackend] = {}
_gen_overrides: dict[str, GenerationBackend] = {}
_clf_overrides: dict[str, ClassifierBackend] = {}


def get_generation_backend(model: str) -> GenerationBackend:
    if model in _gen_overrides:
        return _gen_overrides[model]

    config = get_model_config(model)
    if config.kind != "generation":
        raise ValueError(f"Model {model!r} is kind={config.kind!r}, not a generation model.")

    bt = config.backend_type
    if bt == "vllm":
        key = f"vllm:{model}"
        if key not in _gen_cache:
            from .vllm_backend import VLLMBackend

            _gen_cache[key] = VLLMBackend(model=config.hf_repo or model)
        return _gen_cache[key]

    if bt in _gen_cache:
        return _gen_cache[bt]

    if bt == "anthropic":
        from .anthropic_backend import AnthropicBackend

        backend: GenerationBackend = AnthropicBackend.from_env()
    elif bt == "openai":
        from .openai_backend import OpenAIBackend

        backend = OpenAIBackend.from_env()
    else:
        raise ValueError(f"Unknown generation backend_type {bt!r} for model {model!r}.")

    _gen_cache[bt] = backend
    return backend


def get_classifier_backend(model: str) -> ClassifierBackend:
    if model in _clf_overrides:
        return _clf_overrides[model]

    config = get_model_config(model)
    if config.kind != "classifier":
        raise ValueError(f"Model {model!r} is kind={config.kind!r}, not a classifier.")
    if config.backend_type != "hf_classifier":
        raise ValueError(
            f"Unknown classifier backend_type {config.backend_type!r} for model {model!r}."
        )

    # A single HFClassifierBackend instance caches each loaded model internally, so the
    # backend object itself can be shared across classifier models.
    key = "hf_classifier"
    if key not in _clf_cache:
        from .hf_classifier_backend import HFClassifierBackend

        _clf_cache[key] = HFClassifierBackend()
    return _clf_cache[key]


def set_model_override(
    model: str,
    *,
    generation: GenerationBackend | None = None,
    classifier: ClassifierBackend | None = None,
) -> None:
    """Force ``model`` to resolve to a specific backend instance (tests / custom setups)."""
    if generation is not None:
        _gen_overrides[model] = generation
    if classifier is not None:
        _clf_overrides[model] = classifier


def clear_backend_caches() -> None:
    _gen_cache.clear()
    _clf_cache.clear()
    _gen_overrides.clear()
    _clf_overrides.clear()
