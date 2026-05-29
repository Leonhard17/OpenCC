"""Model execution backends.

Concrete backend classes are imported lazily by :mod:`router` so that installing only the
extras you need (``[hf]``, ``[vllm]``, ``[anthropic]``, ``[openai]``) is enough.
"""

from __future__ import annotations

from .base import ClassifierBackend, ClassScores, GenerationBackend
from .router import (
    clear_backend_caches,
    get_classifier_backend,
    get_generation_backend,
    set_model_override,
)

__all__ = [
    "GenerationBackend",
    "ClassifierBackend",
    "ClassScores",
    "get_generation_backend",
    "get_classifier_backend",
    "set_model_override",
    "clear_backend_caches",
]
