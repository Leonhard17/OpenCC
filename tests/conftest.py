"""Fake backends + registry fixtures so the pipeline can be tested with no GPU or network."""

from __future__ import annotations

import pytest

from constitutional_classifier.backends import clear_backend_caches, set_model_override
from constitutional_classifier.backends.base import (
    ClassifierBackend,
    ClassScores,
    GenerationBackend,
)
from constitutional_classifier import model_config as mc


class FakeClassifier(ClassifierBackend):
    """Returns preset scores per exact input text (default: all-benign)."""

    def __init__(self, labels: list[str], default: dict[str, float], table: dict[str, dict]):
        self.labels = labels
        self.default = default
        self.table = table

    def classify(self, texts, model, **kwargs):
        out = []
        for t in texts:
            scores = self.table.get(t, self.default)
            out.append(ClassScores(self.labels, [float(scores.get(l, 0.0)) for l in self.labels]))
        return out

    @property
    def backend_name(self) -> str:
        return "fake_classifier"


class FakeGeneration(GenerationBackend):
    """Maps exact input text -> canned reply (default: echo the input)."""

    def __init__(self, table: dict[str, str]):
        self.table = table

    def generate(self, messages, model, **kwargs):
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return self.table.get(user, user)

    @property
    def backend_name(self) -> str:
        return "fake_generation"


@pytest.fixture(autouse=True)
def _clean_caches():
    clear_backend_caches()
    yield
    clear_backend_caches()


@pytest.fixture
def wire():
    """Return a helper that wires fake backends for each stage and applies overrides."""

    def _wire(*, jb=None, rephrase=None, harm=None, frontier=None):
        if jb is not None:
            set_model_override(
                "jailbreak-detector",
                classifier=FakeClassifier(["jailbreak"], {"jailbreak": 0.0}, jb),
            )
        if rephrase is not None:
            set_model_override("rephraser", generation=FakeGeneration(rephrase))
        if harm is not None:
            from constitutional_classifier.taxonomy import HARM_CATEGORIES

            set_model_override(
                "content-moderation",
                classifier=FakeClassifier(HARM_CATEGORIES, dict.fromkeys(HARM_CATEGORIES, 0.0), harm),
            )
        if frontier is not None:
            set_model_override("frontier-judge", generation=FakeGeneration(frontier))

    return _wire
