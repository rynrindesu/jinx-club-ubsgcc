from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from .phase1.adaptive_api_1 import solve
from phase01.showdown import decide_move

app = FastAPI(title="Jinx Club Challenge Gateway")


class SolveRequest(BaseModel):
    payload: str


@app.post("/solve")
def solve_endpoint(request: SolveRequest):
    result = solve(request.payload)

    return {
        "adaptOutput": result
    }


@app.get("/health")
def health_endpoint():
    """Warm-up endpoint used by SHOWDOWN before a match."""

    return {"status": "ok"}


@app.post("/move")
def move_endpoint(payload: dict[str, Any]):
    """Return one legal move for a SHOWDOWN protocol request."""

    return decide_move(payload)
