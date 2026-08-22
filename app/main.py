from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from .phase1.adaptive_api_gateway_1.solution import solve
from .phase1.showdown import decide_move
from .phase1.kanchiong_delivery_driver.solution import solve_case

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

@app.post("/kan-cheong-delivery-driver")
def kan_cheong_delivery_driver(
    cases: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        case_id: solve_case(case)
        for case_id, case in cases.items()
    }
