"""Streaming Ghost Chains engine with cumulative Phase 2 identity scoring."""

from __future__ import annotations

from datetime import timedelta
import math
from os import PathLike
from typing import Any, Iterable, Mapping

from app.phase1.ghost_chains.solution import (
    EngineSnapshot,
    GhostChainsEngine as StructuralGhostChainsEngine,
)

from .identity import IdentityScorer
from .models import Transaction
from .scoring import DiscountedWalkScorer, ScoreConfig, StructuralScore


class GhostChainsEngine(StructuralGhostChainsEngine):
    """Preserve Phase 1 state semantics while adding identity evidence."""

    def __init__(
        self,
        *,
        scorer: DiscountedWalkScorer | None = None,
        identity_scorer: IdentityScorer | None = None,
        window: timedelta = timedelta(hours=24),
        ledger_path: str | PathLike[str] | None = None,
    ):
        self.identity_scorer = identity_scorer or IdentityScorer()
        super().__init__(scorer=scorer, window=window, ledger_path=ledger_path)

    def _process_unique(self, transaction: Transaction) -> float:
        if self._watermark is None or transaction.created_at > self._watermark:
            self._watermark = transaction.created_at
            self._expire_old_transactions()

        if self._is_outside_window(transaction.created_at):
            score = 0.0
            self._remember(transaction, score)
            return score

        pair = (transaction.sender, transaction.recipient)
        if self._pair_counts[pair] > 0:
            structural = StructuralScore.zero()
        else:
            structural = self.scorer.score_new_edge(
                transaction.sender,
                transaction.recipient,
                self._forward,
                self._reverse,
            )

        identity = self.identity_scorer.score(
            transaction,
            (entry.transaction for entry in self._active.values()),
            self._forward,
            self._reverse,
            structural,
        )
        raw = structural.raw + identity.raw
        if not math.isfinite(raw):
            score = 1.0
        elif raw <= 0:
            score = 0.0
        else:
            score = raw / (raw + self.scorer.config.risk_half_saturation)

        self._remember(transaction, score)
        self._activate(transaction)
        return min(1.0, max(0.0, score))


_default_engine = GhostChainsEngine()


def score_transaction(transaction: Transaction | Mapping[str, Any]) -> float:
    """Score one transaction with the process-lifetime Phase 2 engine."""

    return _default_engine.score_transaction(transaction)


def score_batch(
    transactions: Iterable[Transaction | Mapping[str, Any]],
) -> list[float]:
    """Score a batch sequentially with the process-lifetime Phase 2 engine."""

    return _default_engine.score_batch(transactions)


def reset() -> None:
    """Reset structural, temporal, identity, and retry state."""

    _default_engine.reset()


def solve(
    transactions: Transaction
    | Mapping[str, Any]
    | Iterable[Transaction | Mapping[str, Any]],
) -> float | list[float]:
    """Transport-neutral adapter matching the Phase 1 package interface."""

    if isinstance(transactions, (Transaction, Mapping)):
        return score_transaction(transactions)
    return score_batch(transactions)


__all__ = [
    "EngineSnapshot",
    "GhostChainsEngine",
    "ScoreConfig",
    "Transaction",
    "reset",
    "score_batch",
    "score_transaction",
    "solve",
]
