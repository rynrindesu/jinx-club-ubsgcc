# Ghost Chains — Phase 1

This package implements the structural-risk engine behind the required
`/ghost-chains/health`, `/ghost-chains/reset`, and
`/ghost-chains/transactions` endpoints in `app/main.py`.

## Model

The active 24-hour transaction history is a binary directed graph. For a new
edge, the scorer measures its marginal increase in bounded discounted walks:

`K = A + alpha*A^2 + ... + alpha^(L-1)*A^L`

Pair capacity rewards path multiplicity, and diagonal entries (closed money
routes) carry extra weight. The ordinary value of a single direct edge is
subtracted, so a disconnected transfer scores zero. Scores then saturate into
`[0, 1]`. The aggregate open-path contribution also saturates before closed
routes are added, preventing large merchant or payroll stars from outweighing
actual return loops. Phase 1 deliberately ignores amount, IP, and device
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
