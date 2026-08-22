# Jinx Club Challenge Gateway

One FastAPI deployment hosts handlers for multiple coding challenges.

## Routes

- `POST /solve` — retained mock adaptive-API challenge.
- `POST /move` — SHOWDOWN, implemented in `app/phase1/showdown/`.
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
