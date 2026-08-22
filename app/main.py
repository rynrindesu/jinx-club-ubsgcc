from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .phase2.adaptive.solution import PayloadValidationError, solve
from .phase1.kanchiong_delivery_driver.solution import solve_case
from .showdown import decide_move
from .phase1.toolbox.server import mcp

mcp_app = mcp.http_app(path="/")


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp_app.lifespan(app):
        yield


app = FastAPI(title="Jinx Club Challenge Gateway", lifespan=lifespan)
app.mount("/mcp", mcp_app)


class SolveRequest(BaseModel):
    payload: str


@app.post("/solve")
def solve_endpoint(request: SolveRequest):
    try:
        return solve(request.payload)
    except PayloadValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


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
