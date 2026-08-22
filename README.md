# Jinx Club Challenge Gateway

One FastAPI deployment hosts handlers for multiple coding challenges.

## Routes

- `POST /solve` — retained mock adaptive-API challenge.
- `POST /move` — SHOWDOWN, dispatched by phase from `app/showdown/`.
- `GET /health` — warm-up and deployment health check.

## Run locally

```bash
python -m pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Render

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`

Register the Render service base URL with SHOWDOWN. The coordinator appends
`/move` when requesting a decision.

Ghost Chains defaults to the Phase 1 engine. Set `GHOST_CHAINS_PHASE=2` in the
deployment environment before a Phase 2 evaluation. The read-only
`GET /ghost-chains/runtime` endpoint reports the active phase and model so a
deployment can be verified before triggering an evaluation. On Render it also
reports the deployed Git revision and instance identifier.

## SHOWDOWN Phase 2

Phase 1 keeps its existing standard-rule policy. Phase 2 is isolated in
`app/phase2/showdown/` and adds:

- showdown-rule learning keyed by the opaque `table_rule` codename;
- repeat-safe ingestion of the rolling `recent_hands` window;
- a candidate-rule ensemble with a transitive pairwise fallback;
- validated scouting comparisons compiled from completed Phase 2 attempts;
- opponent tendencies carried across the four legs of one attempt; and
- uncertainty, re-raise, and `+25`-cushion risk controls.

Validated scouting evidence is replayed into the normal learner by fixed leg
order, so a deployment does not discard discoveries from completed scouting
attempts. New rule knowledge then lives for the server process and carries
across retries without mixing opponent profiles between attempts. The supplied
Uvicorn command runs one worker, which keeps that in-memory state coherent.

The rolling request history captures every completed hand except a leg's final
hand, because no later `/move` callback exists for it. `Phase2State` exposes
`ingest_completed_hands(...)` for an external replay runner; the callback itself
cannot fetch `/matches/<runId>` because the coordinator does not send that run
ID to the bot.
