"""Pipeline output types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class Decision(str, Enum):
    """Terminal decision for an input."""

    ALLOW = "allow"  # forward the text unchanged
    BLOCK = "block"  # reject; do not forward
    ANNOTATE = "annotate"  # forward the text plus a safety note


@dataclass
class StageTrace:
    """What one stage did, for observability."""

    stage: str  # "jailbreak_detector" | "rephraser" | "content_moderation" | "frontier_judge"
    outcome: str  # short human-readable outcome, e.g. "benign", "flagged", "escalate"
    scores: dict[str, float] | None = None  # classifier scores, when applicable
    detail: str | None = None  # raw text / reason, when applicable


@dataclass
class PipelineResult:
    decision: Decision
    reason: str
    # The text to forward downstream: the (possibly rephrased) input, with a safety note
    # appended when ``decision == ANNOTATE``. ``None`` when blocked.
    output_text: str | None
    rephrased: bool = False
    trace: list[StageTrace] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["decision"] = self.decision.value
        return d
