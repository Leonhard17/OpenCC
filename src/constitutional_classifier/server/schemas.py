"""Request/response models for the local HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CheckRequest(BaseModel):
    text: str = Field(..., description="The input to moderate.")
    mode: str | None = Field(
        None,
        description="Override the terminal-flag action: 'block' or 'annotate'. "
        "Defaults to the server's configured action.",
    )


class StageTraceModel(BaseModel):
    stage: str
    outcome: str
    scores: dict[str, float] | None = None
    detail: str | None = None


class CheckResponse(BaseModel):
    decision: str  # "allow" | "block" | "annotate"
    reason: str
    output_text: str | None
    rephrased: bool
    trace: list[StageTraceModel]
