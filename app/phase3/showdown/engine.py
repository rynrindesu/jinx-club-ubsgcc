"""Safe orchestration boundary for one SHOWDOWN Phase 3 decision."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import threading
from typing import Any, Mapping

from .learning import EventKnowledge, RuntimeStore
from .policy import HighVariancePolicy
from .protocol import (
    MoveRequest,
    ProtocolError,
    parse_payload,
    safe_fallback,
    validate_response,
)


_PACKAGE_SEED = Path(__file__).with_name("knowledge.seed.json")
_RUNTIME_LOCK = threading.RLock()


def _configured_seed_path() -> Path:
    configured = os.getenv("SHOWDOWN_PHASE3_SEED")
    return Path(configured).expanduser() if configured else _PACKAGE_SEED


_SEED_PATH = _configured_seed_path()
_store = RuntimeStore(_SEED_PATH)
_policy = HighVariancePolicy()


def _seed_digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "unavailable"


def _rule_summary(knowledge: EventKnowledge) -> dict[str, str]:
    result: dict[str, str] = {}
    for codename, model in sorted(knowledge.rules.items()):
        try:
            posterior = model.posterior()
            if posterior:
                result[codename] = max(posterior, key=posterior.__getitem__)
        except (AttributeError, TypeError, ValueError):
            continue
    return result


# This describes the seed actually loaded at process start. Runtime learning is
# deliberately in-memory and must not make deployment verification drift.
_STARTUP_IDENTITY: dict[str, Any] = {
    "phase3_engine": "app.phase3.showdown.engine",
    "seed_file": _SEED_PATH.name,
    "seed_sha256": _seed_digest(_SEED_PATH),
    "seed_sources": len(_store.knowledge.source_hashes),
    "seed_rules": _rule_summary(_store.knowledge),
}


def decide_move(payload: Mapping[str, Any]) -> dict[str, str | int]:
    """Return one canonical legal move, degrading safely on every failure.

    The coordinator does not retry calls.  Learning and strategy errors are
    therefore contained at this boundary and never become HTTP errors.
    """

    request: MoveRequest | None = None
    try:
        request = parse_payload(payload)
        # The event runs one attempt at a time.  A lock also makes duplicate
        # warm/test calls deterministic under FastAPI's worker thread pool.
        with _RUNTIME_LOCK:
            session = _store.ingest(request)
            proposed = _policy.decide(request, session, _store.knowledge)
        return validate_response(request, proposed)
    except Exception:
        # Safe fallback itself is intentionally tolerant of an unparsed raw
        # payload.  A completely malformed object has no provably legal answer;
        # check is the least destructive final response and matches coordinator
        # substitution semantics when checking is possible.
        try:
            return safe_fallback(payload, request)
        except (ProtocolError, TypeError, ValueError):
            return {"action": "check"}


def reset_runtime_for_tests(knowledge: EventKnowledge | None = None) -> None:
    """Replace process-local state; never called by the public HTTP service."""

    global _store
    with _RUNTIME_LOCK:
        _store = RuntimeStore(knowledge=knowledge or EventKnowledge())


def runtime_snapshot() -> dict[str, Any]:
    """Return a JSON-safe copy for tests and diagnostics, without persisting it."""

    with _RUNTIME_LOCK:
        return _store.knowledge.to_dict()


def runtime_identity() -> dict[str, Any]:
    """Return immutable startup metadata suitable for deployment checks."""

    return {
        **_STARTUP_IDENTITY,
        "seed_rules": dict(_STARTUP_IDENTITY["seed_rules"]),
    }


__all__ = [
    "decide_move",
    "reset_runtime_for_tests",
    "runtime_identity",
    "runtime_snapshot",
]
