"""The moderation pipeline — wires the stages into the reference diagram.

Flow (stages are skipped when their config slot is ``None``):

    input
      │
      ▼  jailbreak_detector ── benign ──────────────┐
      │ flagged                                      │
      ▼  rephraser ── "jailbreak"/can't decode ─► TERMINAL FLAG
      │ cleaned text                                 │
      ├───────────────(cleaned text)─────────────────┤
      ▼  content_moderation ── benign ─► ALLOW        │  (benign input also lands here)
      │ escalate                                      │
      ▼  frontier_judge ── ALLOW ─► ALLOW             │
      │ BLOCK ─► BLOCK            (no judge ─► ANNOTATE w/ frontier note)

TERMINAL FLAG → ``config.default_action``: BLOCK (fail-closed) or ANNOTATE.
"""

from __future__ import annotations

from .. import model_config as mc
from .config import PipelineConfig
from .result import Decision, PipelineResult, StageTrace
from .stages import (
    ContentModerationStage,
    FrontierJudgeStage,
    JailbreakDetectorStage,
    RephraserStage,
)


class Pipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        c = self.config
        self.jailbreak = (
            JailbreakDetectorStage(c.jailbreak_detector, c.jailbreak_threshold)
            if c.jailbreak_detector
            else None
        )
        self.rephraser = RephraserStage(c.rephraser) if c.rephraser else None
        self.content_moderation = (
            ContentModerationStage(c.content_moderation, c.harm_threshold)
            if c.content_moderation
            else None
        )
        self.frontier = FrontierJudgeStage(c.frontier_judge) if c.frontier_judge else None

    # ------------------------------------------------------------------ public API
    def check(self, text: str, mode: Decision | None = None) -> PipelineResult:
        """Run the pipeline. ``mode`` overrides the terminal-flag action for this call."""
        trace: list[StageTrace] = []
        action = mode or self.config.default_action
        moderated_text = text
        rephrased = False

        # --- Stage 1: jailbreak detection -------------------------------------------
        if self.jailbreak is not None:
            is_jb, scores = self.jailbreak.run(text)
            if is_jb:
                trace.append(StageTrace("jailbreak_detector", "flagged", scores=scores))
                # --- Stage 1b: rephrase, or treat flag as a jailbreak --------------
                if self.rephraser is None:
                    return self._terminal_flag(
                        action, text, trace,
                        reason="Jailbreak detected and no rephraser configured.",
                    )
                rp_is_jb, cleaned = self.rephraser.run(text)
                if rp_is_jb:
                    trace.append(
                        StageTrace("rephraser", "jailbreak", detail=cleaned or "(empty)")
                    )
                    return self._terminal_flag(
                        action, text, trace,
                        reason="Rephraser could not decode the input / flagged it as an attack.",
                    )
                trace.append(StageTrace("rephraser", "deobfuscated", detail=cleaned))
                moderated_text = cleaned
                rephrased = True
            else:
                trace.append(StageTrace("jailbreak_detector", "benign", scores=scores))

        # --- Stage 2: content moderation --------------------------------------------
        if self.content_moderation is not None:
            escalate, scores = self.content_moderation.run(moderated_text)
            if not escalate:
                trace.append(StageTrace("content_moderation", "benign", scores=scores))
                return PipelineResult(
                    Decision.ALLOW, "Passed moderation.", moderated_text,
                    rephrased=rephrased, trace=trace,
                )
            trace.append(StageTrace("content_moderation", "escalate", scores=scores))

            # --- Stage 3: frontier judge, or substitute with a note ----------------
            if self.frontier is None:
                annotated = f"{moderated_text}\n\n{mc.FRONTIER_NOTE}"
                trace.append(StageTrace("frontier_judge", "skipped_annotated"))
                return PipelineResult(
                    Decision.ANNOTATE,
                    "Harm escalation with no frontier judge; forwarded with a safety note.",
                    annotated, rephrased=rephrased, trace=trace,
                )
            allow, raw = self.frontier.run(moderated_text)
            trace.append(
                StageTrace("frontier_judge", "allow" if allow else "block", detail=raw)
            )
            if allow:
                return PipelineResult(
                    Decision.ALLOW, "Frontier judge allowed the request.", moderated_text,
                    rephrased=rephrased, trace=trace,
                )
            return PipelineResult(
                Decision.BLOCK, "Frontier judge blocked the request.", None,
                rephrased=rephrased, trace=trace,
            )

        # No content moderation configured: a clean (or rephrased) input passes through.
        return PipelineResult(
            Decision.ALLOW, "No content moderation configured.", moderated_text,
            rephrased=rephrased, trace=trace,
        )

    # ------------------------------------------------------------------ helpers
    def _terminal_flag(
        self, action: Decision, text: str, trace: list[StageTrace], *, reason: str
    ) -> PipelineResult:
        """Map a confirmed jailbreak to the configured action."""
        if action == Decision.ANNOTATE:
            return PipelineResult(
                Decision.ANNOTATE, reason, f"{text}\n\n{mc.ANNOTATION_NOTE}", trace=trace
            )
        return PipelineResult(Decision.BLOCK, reason, None, trace=trace)
