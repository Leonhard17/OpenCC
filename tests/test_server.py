"""Smoke test for the HTTP layer (skipped if fastapi isn't installed)."""

from __future__ import annotations

import pytest

from constitutional_classifier import Pipeline, PipelineConfig

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from constitutional_classifier.server import create_app  # noqa: E402


def test_check_endpoint(wire):
    wire(jb={}, rephrase={}, harm={})
    pipe = Pipeline(PipelineConfig(frontier_judge=None))
    client = TestClient(create_app(pipeline=pipe))

    assert client.get("/health").json() == {"status": "ok"}

    resp = client.post("/check", json={"text": "how do I bake banana bread?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == "allow"
    assert body["output_text"] == "how do I bake banana bread?"
