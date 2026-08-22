"""Streaming Ghost Chains engine with cumulative Phase 3 value scoring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import math
from os import PathLike
from typing import Any, Iterable, Mapping

from app.phase1.ghost_chains.solution import EngineSnapshot
from app.phase2.ghost_chains.solution import (
    GhostChainsEngine as IdentityGhostChainsEngine,
)

from .identity import IdentityEvent, IdentityScore, IdentityScorer
from .models import Transaction
from .scoring import DiscountedWalkScorer, ScoreConfig, StructuralScore
from .value import ValueConfig, ValueEvent, ValueFlowScorer, ValueScore


@dataclass(frozen=True)
class CrossSignalScore:
    """Bounded interactions whose evidence belongs to the same value path."""

    risk: float
    return_reversal: float
    structural_reversal: float
    dropout_continuation: float
    divergence_reversal: float
    aligned_continuation: float
    branch_continuation: float

    @classmethod
    def zero(cls) -> CrossSignalScore:
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class GhostChainsEngine(IdentityGhostChainsEngine):
    """Preserve Phase 1/2 semantics while adding branch-local value evidence."""

    def __init__(
        self,
        *,
        scorer: DiscountedWalkScorer | None = None,
        identity_scorer: IdentityScorer | None = None,
        value_scorer: ValueFlowScorer | None = None,
        window: timedelta = timedelta(hours=24),
        ledger_path: str | PathLike[str] | None = None,
    ):
        super().__init__(
            scorer=scorer,
            identity_scorer=identity_scorer,
            window=window,
            ledger_path=ledger_path,
        )
        self.value_scorer = value_scorer or ValueFlowScorer(
            ValueConfig(max_path_length=self.scorer.config.max_walk_length)
        )

    def _process_unique(self, transaction: Transaction) -> float:
        if self._watermark is None or transaction.created_at > self._watermark:
            self._watermark = transaction.created_at
            self._expire_old_transactions()

        if self._is_outside_window(transaction.created_at):
            score = 0.0
            self._remember(transaction, score)
            return score

        active = tuple(self._active.values())
        sequence = self._next_sequence + 1
        structural = self._score_structural(transaction)
        identity = self.identity_scorer.score(
            IdentityEvent(transaction, sequence),
            (
                IdentityEvent(entry.transaction, entry.sequence)
                for entry in active
            ),
            structural,
        )
        phase_two_risk = self._phase_two_risk(structural, identity)

        value = self.value_scorer.score(
            ValueEvent(transaction, sequence),
            (
                ValueEvent(entry.transaction, entry.sequence)
                for entry in active
            ),
        )
        cross_signal = self._score_cross_signal(structural, identity, value)
        if value.risk == 0.0 and cross_signal.risk == 0.0:
            # Preserve byte-for-byte Phase 2 behavior when Phase 3 has no
            # evidence; even an algebraically equivalent noisy-or can differ
            # by a final floating-point bit.
            score = phase_two_risk
        else:
            score = _noisy_or(phase_two_risk, value.risk, cross_signal.risk)

        self._remember(transaction, score)
        self._activate(transaction)
        return score

    def _phase_two_risk(
        self,
        structural: StructuralScore,
        identity: IdentityScore,
    ) -> float:
        """Use exactly Phase 2's raw-score transform before Phase 3 evidence."""

        raw = structural.raw + identity.raw
        if not math.isfinite(raw):
            return 1.0
        if raw <= 0:
            return 0.0
        risk = raw / (raw + self.scorer.config.risk_half_saturation)
        return min(1.0, max(0.0, risk))

    @staticmethod
    def _score_cross_signal(
        structural: StructuralScore,
        identity: IdentityScore,
        value: ValueScore,
    ) -> CrossSignalScore:
        if not value.hypotheses:
            return CrossSignalScore.zero()

        structural_activation = math.tanh(max(0.0, structural.raw) / 2.0)
        return_activation = math.tanh(max(0.0, structural.closed_route_delta))
        reversal = min(1.0, value.reversal / 0.65)
        continuation = min(1.0, value.continuation / 0.05)
        branch_continuation_strength = min(
            1.0, value.branch_continuation / 0.05
        )

        alignment_raw = math.fsum(
            dimension.alignment for dimension in identity.dimensions
        )
        divergence_raw = math.fsum(
            dimension.divergence for dimension in identity.dimensions
        )
        dropout_raw = math.fsum(
            dimension.dropout for dimension in identity.dimensions
        )
        alignment = 1.0 - math.exp(-alignment_raw)
        divergence = 1.0 - math.exp(-divergence_raw)
        dropout = 1.0 - math.exp(-dropout_raw)

        return_reversal = 0.30 * return_activation * reversal
        structural_reversal = 0.12 * structural_activation * reversal
        dropout_continuation = 0.22 * dropout * continuation
        divergence_reversal = 0.20 * divergence * reversal
        aligned_continuation = (
            0.08 * alignment * continuation * structural_activation
        )
        branch_continuation = 0.08 * branch_continuation_strength
        risk = _noisy_or(
            return_reversal,
            structural_reversal,
            dropout_continuation,
            divergence_reversal,
            aligned_continuation,
            branch_continuation,
        )
        return CrossSignalScore(
            risk=risk,
            return_reversal=return_reversal,
            structural_reversal=structural_reversal,
            dropout_continuation=dropout_continuation,
            divergence_reversal=divergence_reversal,
            aligned_continuation=aligned_continuation,
            branch_continuation=branch_continuation,
        )


def _noisy_or(*signals: float) -> float:
    bounded = (min(1.0, max(0.0, signal)) for signal in signals)
    return min(
        1.0,
        max(0.0, 1.0 - math.prod(1.0 - signal for signal in bounded)),
    )


_default_engine = GhostChainsEngine()


def score_transaction(transaction: Transaction | Mapping[str, Any]) -> float:
    """Score one transaction with the process-lifetime Phase 3 engine."""

    return _default_engine.score_transaction(transaction)


def score_batch(
    transactions: Iterable[Transaction | Mapping[str, Any]],
) -> list[float]:
    """Score a batch sequentially with the process-lifetime Phase 3 engine."""

    return _default_engine.score_batch(transactions)


def reset() -> None:
    """Reset structural, temporal, identity, value, and retry state."""

    _default_engine.reset()


def solve(
    transactions: Transaction
    | Mapping[str, Any]
    | Iterable[Transaction | Mapping[str, Any]],
) -> float | list[float]:
    """Transport-neutral adapter matching the earlier-phase interface."""

    if isinstance(transactions, (Transaction, Mapping)):
        return score_transaction(transactions)
    return score_batch(transactions)


__all__ = [
    "CrossSignalScore",
    "EngineSnapshot",
    "GhostChainsEngine",
    "ScoreConfig",
    "Transaction",
    "reset",
    "score_batch",
    "score_transaction",
    "solve",
]
