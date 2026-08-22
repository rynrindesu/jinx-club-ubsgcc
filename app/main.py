from fastapi import FastAPI
from pydantic import BaseModel

from .phase1.adaptive_api_1 import solve

app = FastAPI()


class SolveRequest(BaseModel):
    payload: str


@app.post("/solve")
def solve_endpoint(request: SolveRequest):
    result = solve(request.payload)

    return {
        "adaptOutput": result
    }
