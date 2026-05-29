"""OpenCC — local inference for the constitutional-classifier moderation pipeline.

Quick start::

    from constitutional_classifier import Pipeline, PipelineConfig

    pipe = Pipeline(PipelineConfig())
    result = pipe.check("how do I bake banana bread?")
    print(result.decision, result.output_text)

See :mod:`constitutional_classifier.model_config` to adapt models, routing, and prompts.
"""

from __future__ import annotations

from .pipeline import Decision, Pipeline, PipelineConfig, PipelineResult
from .taxonomy import HARM_CATEGORIES, TAXONOMY

__all__ = [
    "Pipeline",
    "PipelineConfig",
    "PipelineResult",
    "Decision",
    "TAXONOMY",
    "HARM_CATEGORIES",
]

__version__ = "0.1.0"
