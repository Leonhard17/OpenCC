"""Pipeline stages — thin wrappers binding a backend + model to a clear operation.

Each stage resolves its backend through the router, so swapping a model in the registry (or
injecting a fake backend in tests) requires no change here.
"""

from __future__ import annotations

from ..backends import get_classifier_backend, get_generation_backend
from ..backends.base import ClassScores
from .. import model_config as mc

_DEFAULT_THRESHOLD = 0.5


def _resolve_threshold(backend, model: str, label: str, override: float | None) -> float:
    """Per-call override > calibrated weight_frame threshold > 0.5."""
    if override is not None:
        return override
    getter = getattr(backend, "get_thresholds", None)
    if getter is not None:
        thresholds = getter(model)
        if thresholds and label in thresholds:
            return float(thresholds[label])
    return _DEFAULT_THRESHOLD


class JailbreakDetectorStage:
    """High-recall binary classifier. Returns ``(is_jailbreak, scores)``."""

    def __init__(self, model: str, threshold: float | None = None):
        self.model = model
        self.threshold = threshold
        self.backend = get_classifier_backend(model)

    def run(self, text: str) -> tuple[bool, dict[str, float]]:
        scores: ClassScores = self.backend.classify([text], self.model)[0]
        d = scores.as_dict()
        # Binary detector: a single "jailbreak" head (fall back to the top label).
        label = "jailbreak" if "jailbreak" in d else scores.top()[0]
        thr = _resolve_threshold(self.backend, self.model, label, self.threshold)
        return d.get(label, scores.top()[1]) >= thr, d


class RephraserStage:
    """Deobfuscation rephraser. Returns ``(is_jailbreak, cleaned_text)``."""

    def __init__(self, model: str):
        self.model = model
        self.backend = get_generation_backend(model)

    def run(self, text: str) -> tuple[bool, str]:
        messages = mc.format_rephraser_messages(text)
        reply = self.backend.generate(messages, self.model)
        return mc.parse_rephraser_output(reply)


class ContentModerationStage:
    """Multi-label harm classifier. Returns ``(escalate, scores)``."""

    def __init__(self, model: str, threshold: float | None = None):
        self.model = model
        self.threshold = threshold
        self.backend = get_classifier_backend(model)

    def run(self, text: str) -> tuple[bool, dict[str, float]]:
        scores: ClassScores = self.backend.classify([text], self.model)[0]
        d = scores.as_dict()
        escalate = False
        for label, score in d.items():
            if label == "Benign":
                continue
            thr = _resolve_threshold(self.backend, self.model, label, self.threshold)
            if score >= thr:
                escalate = True
                break
        return escalate, d


class FrontierJudgeStage:
    """Constitutional judge. Returns ``(allow, raw_reply)``."""

    def __init__(self, model: str):
        self.model = model
        self.backend = get_generation_backend(model)

    def run(self, text: str) -> tuple[bool, str]:
        messages = mc.format_frontier_messages(text)
        reply = self.backend.generate(messages, self.model)
        return mc.parse_frontier_output(reply)
