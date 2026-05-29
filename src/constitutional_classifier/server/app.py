"""FastAPI app exposing the pipeline as a local REST-like service.

A single :class:`Pipeline` is built at startup (loading model weights once) and reused for
every request. Bind to 127.0.0.1 so it is reachable only by programs on the same machine.

``fastapi`` is imported at module load, so this module is only importable with the
``[server]`` extra installed — which is exactly when it is used.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from ..pipeline import Decision, Pipeline, PipelineConfig
from .schemas import CheckRequest, CheckResponse


def create_app(config: PipelineConfig | None = None, pipeline: Pipeline | None = None) -> FastAPI:
    """Build the FastAPI app. Pass a prebuilt ``pipeline`` to skip model loading (tests)."""
    app = FastAPI(title="OpenCC Constitutional Classifier", version="0.1.0")
    pipe = pipeline or Pipeline(config)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/config")
    def get_config() -> dict:
        c = pipe.config
        return {
            "jailbreak_detector": c.jailbreak_detector,
            "rephraser": c.rephraser,
            "content_moderation": c.content_moderation,
            "frontier_judge": c.frontier_judge,
            "default_action": c.default_action.value,
        }

    @app.post("/check", response_model=CheckResponse)
    def check(req: CheckRequest) -> CheckResponse:
        mode = None
        if req.mode is not None:
            try:
                mode = Decision(req.mode)
            except ValueError:
                raise HTTPException(400, f"Invalid mode {req.mode!r}; use 'block' or 'annotate'.")
        result = pipe.check(req.text, mode=mode)
        return CheckResponse(**result.to_dict())

    return app
