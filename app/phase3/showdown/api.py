"""Standalone HTTP service for the SHOWDOWN Phase 3 coordinator."""

from typing import Any

from fastapi import FastAPI

from .engine import decide_move, runtime_identity


app = FastAPI(title="SHOWDOWN Phase 3 Bot", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str | int]:
    """Warm the process without mutating match state."""

    return {"status": "ok", "phase": 3}


@app.get("/showdown/runtime")
def runtime() -> dict[str, Any]:
    """Identify the decision engine used by standalone deployments."""

    return {"router": "standalone-v3", **runtime_identity()}


@app.post("/move")
def move(payload: dict[str, Any]) -> dict[str, str | int]:
    """Return a protocol-v2 move; engine failures degrade to a legal fallback."""

    return decide_move(payload)
