# Ghost Chains — Phase 1

This package implements the structural-risk engine behind the required
`/ghost-chains/health`, `/ghost-chains/reset`, and
`/ghost-chains/transactions` endpoints in `app/main.py`.

## Model

The active 24-hour history is a directed transaction-event graph. For a new
event, the scorer measures its marginal increase in bounded discounted routes:

`K = A + alpha*A^2 + ... + alpha^(L-1)*A^L`

Routes must follow strictly increasing `(createdAt, arrival sequence)` keys.
They use simple entity signatures, with one return to the starting entity
allowed, so parallel payments and repeated laps cannot manufacture route
multiplicity. Pair capacity rewards distinct causal paths, shortest-path
efficiency rewards genuine shortcuts, and closed money routes carry extra
weight while decaying coherently by reciprocal hop count. The ordinary value
of a single direct edge is subtracted, so a
disconnected transfer scores zero. All non-closed evidence shares one cap
before scores saturate into `[0, 1]`, preventing a large acyclic bridge from
outweighing a return loop. Phase 1 deliberately ignores amount, IP, and device
values.

The engine also provides:

- a watermark-defined active window `(M - 24h, M]`;
- out-of-order processing without watermark rollback;
- reference-counted parallel graph edges;
- exact retry scores and conflicting-ID detection;
- a compact SQLite retry ledger, keeping process memory bounded by the window;
- atomic batch conflict validation;
- serialized mutation and atomic reset.

Self-transfers receive a small bounded score but are deliberately excluded from
route construction: a one-entity movement is not a path through counterparties,
and allowing its repeated walks into `K` would amplify unrelated later scores.

## Core interface

```python
from app.phase1.ghost_chains import GhostChainsEngine

engine = GhostChainsEngine()
score = engine.score_transaction(
    {
        "txId": "tx-1",
        "fromUserId": "A",
        "toUserId": "B",
        "amount": 100.0,
        "createdAt": "2026-08-22T00:00:00Z",
    }
)
```

`score_batch(...)`, `reset()`, and `snapshot()` are available on the engine.
The public endpoints use the documented field names. The core parser remains
tolerant of common aliases without changing the state or scoring logic.
