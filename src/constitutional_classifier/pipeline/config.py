"""Pipeline configuration: which stages are populated and how flagged inputs are handled."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .result import Decision


@dataclass
class PipelineConfig:
    """Names a model (from the registry) for each stage. ``None`` disables the stage.

    Disabled-stage behaviour (by design):

    * ``rephraser is None`` — a jailbreak-detector flag counts directly as a jailbreak
      (terminal flag, fail-closed).
    * ``frontier_judge is None`` — a content-moderation escalation is NOT blocked; the
      input is forwarded ANNOTATED so the downstream model is aware.
    * ``jailbreak_detector is None`` — inputs skip straight to content moderation.
    * ``content_moderation is None`` — no harm check after the (clean) input passes.
    """

    jailbreak_detector: str | None = "jailbreak-detector"
    rephraser: str | None = "rephraser"
    content_moderation: str | None = "content-moderation"
    frontier_judge: str | None = None

    # Action for a terminal jailbreak flag: BLOCK (fail-closed default) or ANNOTATE.
    default_action: Decision = Decision.BLOCK

    # Optional threshold overrides (otherwise each model's calibrated weight_frame
    # thresholds are used).
    jailbreak_threshold: float | None = None
    harm_threshold: float | None = None

    # Server bind.
    host: str = "127.0.0.1"
    port: int = 8000

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        data = dict(data)
        action = data.pop("default_action", None)
        cfg = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        if action is not None:
            cfg.default_action = Decision(action) if isinstance(action, str) else action
        return cfg

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> "PipelineConfig":
        import yaml

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)
