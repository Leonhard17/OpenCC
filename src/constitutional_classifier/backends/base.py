"""Backend abstractions.

Two families of model backends:

* :class:`GenerationBackend` — text-in / text-out chat models (rephraser, frontier judge).
  Modeled on REDACT's ``llms/base.py``: all backends accept OpenAI-format chat messages and
  declare capability flags so callers can batch without knowing the concrete backend.
* :class:`ClassifierBackend` — text-in / per-label-scores-out models (the TACTIC jailbreak
  detector and content-moderation classifiers).

Keeping the two interfaces separate lets a generation model and a classifier be swapped
independently behind the same pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


# --------------------------------------------------------------------------- generation


class GenerationBackend(ABC):
    """Abstract base class for text-generation backends."""

    @abstractmethod
    def generate(self, messages: list[dict], model: str, **kwargs) -> str:
        """Generate a single response.

        Args:
            messages: Chat messages in OpenAI format
                ``[{"role": "system"|"user"|"assistant", "content": "..."}]``.
            model: Model identifier (a key in the registry).
            **kwargs: Backend-specific parameters (``max_tokens``, ``temperature``, ...).

        Returns:
            The generated text content.
        """

    def batch_generate(
        self, messages_list: list[list[dict]], model: str, **kwargs
    ) -> list[str]:
        """Generate responses for multiple message lists.

        Default implementation loops over :meth:`generate`. Backends like vLLM override
        this for a single native engine pass.
        """
        return [self.generate(msgs, model, **kwargs) for msgs in messages_list]

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Short identifier for the backend type (e.g. ``"openai"``, ``"vllm"``)."""

    @property
    def supports_native_batching(self) -> bool:
        """True if :meth:`batch_generate` is a single engine pass (vLLM)."""
        return False

    @property
    def supports_parallel_calls(self) -> bool:
        """True if :meth:`generate` is safe to call concurrently from threads."""
        return True


# --------------------------------------------------------------------------- classifier


@dataclass
class ClassScores:
    """Per-label probabilities for one classified input.

    ``labels`` and ``scores`` are aligned. For a binary jailbreak detector the labels are
    typically ``["jailbreak"]`` with a single score; for the harm classifier they are the
    harm categories.
    """

    labels: list[str]
    scores: list[float]

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.labels, self.scores))

    def top(self) -> tuple[str, float]:
        """The highest-scoring (label, score) pair."""
        i = max(range(len(self.scores)), key=lambda j: self.scores[j])
        return self.labels[i], self.scores[i]


class ClassifierBackend(ABC):
    """Abstract base class for classification backends."""

    @abstractmethod
    def classify(self, texts: list[str], model: str, **kwargs) -> list[ClassScores]:
        """Return one :class:`ClassScores` per input text."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Short identifier for the backend type (e.g. ``"hf_classifier"``)."""
