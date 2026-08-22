# SHOWDOWN Phase 3

This is a standalone, clean-room Phase 3 service. It does not import the
earlier SHOWDOWN implementations and does not replace the shared gateway.

## Run

```bash
uvicorn app.phase3.showdown.api:app --host 0.0.0.0 --port 8000
```

Register the public HTTPS base URL with the coordinator. The service exposes:

- `GET /health`
- `POST /move`

Runtime learning is process-local. Set `SHOWDOWN_PHASE3_SEED` to load a seed
from a different path; otherwise `knowledge.seed.json` beside this file is
loaded.

## Learn from a scout attempt

Save every raw `/matches/<runId>?download=1` response, then fit a new aggregate
seed:

```bash
python -m app.phase3.showdown.replay match-a.json match-b.json \
  --seed app/phase3/showdown/knowledge.seed.json \
  --output app/phase3/showdown/knowledge.seed.json
```

The command hashes inputs, skips duplicates, and writes only rule evidence and
opponent statistics. Raw replay files are not copied into the repository.

## Simulate

`app.phase3.showdown.simulator` provides deterministic six-seat legs,
scripted opponents, side-pot settlement, benchmarking, and parameter tuning.
It intentionally treats its emitted legal-action bounds as authoritative where
the public coordinator documentation is ambiguous.
