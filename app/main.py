from contextlib import asynccontextmanager
import os
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .phase2.adaptive.solution import PayloadValidationError, solve
from .phase1.kanchiong_delivery_driver.solution import solve_case
from .phase1.ghost_chains import (
    Transaction,
    TransactionConflictError,
    TransactionValidationError,
    reset as reset_phase_one_ghost_chains,
    score_batch as score_phase_one_ghost_chains_batch,
)
from .phase2.ghost_chains import (
    reset as reset_phase_two_ghost_chains,
    score_batch as score_phase_two_ghost_chains_batch,
)
from .phase3.ghost_chains import (
    reset as reset_phase_three_ghost_chains,
    score_batch as score_phase_three_ghost_chains_batch,
)
from .showdown import decide_move
from .phase1.toolbox.server import mcp
from .phase2.toolbox.server import register_tools as register_phase2_toolbox_tools

register_phase2_toolbox_tools(mcp)

mcp_app = mcp.http_app(path="/")


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp_app.lifespan(app):
        yield


app = FastAPI(title="Jinx Club Challenge Gateway", lifespan=lifespan)
app.mount("/mcp", mcp_app)


class SolveRequest(BaseModel):
    payload: str


class GhostChainsResetRequest(BaseModel):
    clearTransactions: Literal[True]


class GhostChainsTransactionsRequest(BaseModel):
    transactions: list[dict[str, Any]]


def _configured_ghost_chains_phase() -> str:
    phase = os.getenv("GHOST_CHAINS_PHASE", "3").strip()
    if phase not in {"1", "2", "3"}:
        raise RuntimeError("GHOST_CHAINS_PHASE must be '1', '2', or '3'")
    return phase


_GHOST_CHAINS_PHASE = _configured_ghost_chains_phase()


def _reset_ghost_chains() -> None:
    if _GHOST_CHAINS_PHASE == "3":
        reset_phase_three_ghost_chains()
    elif _GHOST_CHAINS_PHASE == "2":
        reset_phase_two_ghost_chains()
    else:
        reset_phase_one_ghost_chains()


def _score_ghost_chains_batch(transactions: list[Transaction]) -> list[float]:
    if _GHOST_CHAINS_PHASE == "3":
        return score_phase_three_ghost_chains_batch(transactions)
    if _GHOST_CHAINS_PHASE == "2":
        return score_phase_two_ghost_chains_batch(transactions)
    return score_phase_one_ghost_chains_batch(transactions)


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


@app.get("/ghost-chains/health")
def ghost_chains_health_endpoint():
    """Report availability using the Ghost Chains coordinator contract."""

    return {"status": "ok"}


@app.get("/ghost-chains/runtime")
def ghost_chains_runtime_endpoint() -> dict[str, str]:
    """Expose the deployed phase/model without changing the health contract."""

    runtime = {
        "phase": _GHOST_CHAINS_PHASE,
        "model": (
            "segmented-value-flow-v1"
            if _GHOST_CHAINS_PHASE == "3"
            else "temporal-routes-v1"
        ),
    }
    if revision := os.getenv("RENDER_GIT_COMMIT"):
        runtime["revision"] = revision
    if instance := os.getenv("RENDER_INSTANCE_ID"):
        runtime["instance"] = instance
    return runtime


@app.post("/ghost-chains/reset")
def ghost_chains_reset_endpoint(
    request: GhostChainsResetRequest,
) -> dict[str, bool]:
    """Restore Ghost Chains state to its startup condition."""

    _reset_ghost_chains()
    return {"clearTransactions": request.clearTransactions}


@app.post("/ghost-chains/transactions")
def ghost_chains_transactions_endpoint(
    request: GhostChainsTransactionsRequest,
) -> dict[str, list[dict[str, str | float]]]:
    """Score one ordered transaction batch against the streaming graph."""

    try:
        transactions = [
            Transaction.from_mapping(transaction)
            for transaction in request.transactions
        ]
        scores = _score_ghost_chains_batch(transactions)
    except TransactionConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except TransactionValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {
        "transactions": [
            {"txId": transaction.tx_id, "riskScore": score}
            for transaction, score in zip(transactions, scores, strict=True)
        ]
    }


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
