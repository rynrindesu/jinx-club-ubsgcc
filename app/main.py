from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .phase2.adaptive.solution import PayloadValidationError, solve
from .phase1.kanchiong_delivery_driver.solution import solve_case
from .phase1.ghost_chains import (
    Transaction,
    TransactionConflictError,
    TransactionValidationError,
    reset as reset_ghost_chains,
    score_batch as score_ghost_chains_batch,
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


@app.post("/ghost-chains/reset")
def ghost_chains_reset_endpoint(
    request: GhostChainsResetRequest,
) -> dict[str, bool]:
    """Restore Ghost Chains state to its startup condition."""

    reset_ghost_chains()
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
        scores = score_ghost_chains_batch(transactions)
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
