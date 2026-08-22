"""Serialized streaming engine for the Ghost Chains Phase 1 challenge."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import heapq
from os import PathLike
import sqlite3
from threading import RLock
from typing import Any, Iterable, Mapping, Sequence

from .models import (
    Transaction,
    TransactionConflictError,
    coerce_transaction,
)
from .scoring import DiscountedWalkScorer, ScoreConfig


@dataclass(frozen=True)
class _LedgerEntry:
    fingerprint: str
    score: float


@dataclass(frozen=True)
class _ActiveEntry:
    transaction: Transaction
    sequence: int


@dataclass(frozen=True)
class EngineSnapshot:
    """Immutable, compact state view useful for diagnostics and tests."""

    watermark: datetime | None
    active_transactions: int
    remembered_transactions: int
    active_edges: tuple[tuple[str, str, int], ...]
    revision: int


class _IdempotencyLedger:
    """Keep all-time retry metadata off the bounded in-memory graph."""

    def __init__(self, path: str | PathLike[str] | None = None):
        # SQLite interprets an empty filename as a temporary on-disk database
        # that is deleted when the connection closes.
        database = "" if path is None else str(path)
        self._connection = sqlite3.connect(database, check_same_thread=False)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS transaction_ledger (
                tx_id TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL,
                score REAL NOT NULL
            )
            """
        )
        self._connection.commit()

    def get(self, tx_id: str) -> _LedgerEntry | None:
        row = self._connection.execute(
            "SELECT fingerprint, score FROM transaction_ledger WHERE tx_id = ?",
            (tx_id,),
        ).fetchone()
        if row is None:
            return None
        return _LedgerEntry(fingerprint=row[0], score=row[1])

    def add(self, tx_id: str, fingerprint: str, score: float) -> None:
        self._connection.execute(
            """
            INSERT INTO transaction_ledger (tx_id, fingerprint, score)
            VALUES (?, ?, ?)
            """,
            (tx_id, fingerprint, score),
        )
        self._connection.commit()

    def clear(self) -> None:
        self._connection.execute("DELETE FROM transaction_ledger")
        self._connection.commit()

    def count(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM transaction_ledger"
        ).fetchone()
        return int(row[0])

    def close(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is None:
            return
        self._connection = None
        connection.close()

    def __del__(self) -> None:
        self.close()


class GhostChainsEngine:
    """Maintain a 24-hour event-time graph and score incoming transactions."""

    def __init__(
        self,
        *,
        scorer: DiscountedWalkScorer | None = None,
        window: timedelta = timedelta(hours=24),
        ledger_path: str | PathLike[str] | None = None,
    ):
        if window <= timedelta(0):
            raise ValueError("window must be positive")
        self.scorer = scorer or DiscountedWalkScorer()
        self.window = window
        self._lock = RLock()
        self._ledger = _IdempotencyLedger(ledger_path)
        self.reset()

    def reset(self) -> None:
        """Atomically clear graph, watermark, expiry queue, and ID ledger."""

        with self._lock:
            self._watermark: datetime | None = None
            self._ledger.clear()
            self._active: dict[str, _ActiveEntry] = {}
            self._expiry_heap: list[tuple[datetime, int, str]] = []
            self._pair_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
            self._forward: defaultdict[str, set[str]] = defaultdict(set)
            self._reverse: defaultdict[str, set[str]] = defaultdict(set)
            self._next_sequence = 0
            self._revision = 0

    def close(self) -> None:
        """Release the temporary idempotency database owned by this engine."""

        with self._lock:
            self._ledger.close()

    def score_transaction(
        self, transaction: Transaction | Mapping[str, Any]
    ) -> float:
        """Score one transaction, returning the original result for a retry."""

        return self.score_batch([transaction])[0]

    def score_batch(
        self,
        transactions: Iterable[Transaction | Mapping[str, Any]],
    ) -> list[float]:
        """Validate a complete batch, then process it in supplied order."""

        canonical = [coerce_transaction(transaction) for transaction in transactions]
        if not canonical:
            return []

        with self._lock:
            self._prevalidate_conflicts(canonical)
            results: list[float] = []
            for transaction in canonical:
                existing = self._ledger.get(transaction.tx_id)
                if existing is not None:
                    # Duplicate handling intentionally precedes watermark and expiry.
                    results.append(existing.score)
                    continue
                results.append(self._process_unique(transaction))
            return results

    def snapshot(self) -> EngineSnapshot:
        """Read state without exposing mutable graph collections."""

        with self._lock:
            active_edges = tuple(
                sorted(
                    (
                        sender,
                        recipient,
                        count,
                    )
                    for (sender, recipient), count in self._pair_counts.items()
                    if count > 0
                )
            )
            return EngineSnapshot(
                watermark=self._watermark,
                active_transactions=len(self._active),
                remembered_transactions=self._ledger.count(),
                active_edges=active_edges,
                revision=self._revision,
            )

    def _prevalidate_conflicts(self, transactions: Sequence[Transaction]) -> None:
        batch_fingerprints: dict[str, str] = {}
        for transaction in transactions:
            batch_fingerprint = batch_fingerprints.get(transaction.tx_id)
            if (
                batch_fingerprint is not None
                and batch_fingerprint != transaction.fingerprint
            ):
                raise TransactionConflictError(transaction.tx_id)
            batch_fingerprints[transaction.tx_id] = transaction.fingerprint

            existing = self._ledger.get(transaction.tx_id)
            if (
                existing is not None
                and existing.fingerprint != transaction.fingerprint
            ):
                raise TransactionConflictError(transaction.tx_id)

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
            score = 0.0
        else:
            score = self.scorer.score_new_edge(
                transaction.sender,
                transaction.recipient,
                self._forward,
                self._reverse,
            ).risk

        self._remember(transaction, score)
        self._activate(transaction)
        return score

    def _remember(self, transaction: Transaction, score: float) -> None:
        self._ledger.add(transaction.tx_id, transaction.fingerprint, score)
        self._revision += 1

    def _activate(self, transaction: Transaction) -> None:
        self._next_sequence += 1
        sequence = self._next_sequence
        self._active[transaction.tx_id] = _ActiveEntry(transaction, sequence)
        heapq.heappush(
            self._expiry_heap,
            (transaction.created_at, sequence, transaction.tx_id),
        )

        pair = (transaction.sender, transaction.recipient)
        self._pair_counts[pair] += 1
        if self._pair_counts[pair] == 1 and transaction.sender != transaction.recipient:
            self._forward[transaction.sender].add(transaction.recipient)
            self._reverse[transaction.recipient].add(transaction.sender)

    def _expire_old_transactions(self) -> None:
        if self._watermark is None:
            return
        cutoff = self._watermark - self.window
        while self._expiry_heap and self._expiry_heap[0][0] <= cutoff:
            _, sequence, tx_id = heapq.heappop(self._expiry_heap)
            active = self._active.get(tx_id)
            if active is None or active.sequence != sequence:
                continue
            del self._active[tx_id]
            transaction = active.transaction
            pair = (transaction.sender, transaction.recipient)
            self._pair_counts[pair] -= 1
            if self._pair_counts[pair] > 0:
                continue

            del self._pair_counts[pair]
            if transaction.sender == transaction.recipient:
                continue
            self._forward[transaction.sender].discard(transaction.recipient)
            self._reverse[transaction.recipient].discard(transaction.sender)
            if not self._forward[transaction.sender]:
                del self._forward[transaction.sender]
            if not self._reverse[transaction.recipient]:
                del self._reverse[transaction.recipient]

    def _is_outside_window(self, created_at: datetime) -> bool:
        if self._watermark is None:
            return False
        return created_at <= self._watermark - self.window


_default_engine = GhostChainsEngine()


def score_transaction(transaction: Transaction | Mapping[str, Any]) -> float:
    """Score one transaction with the process-lifetime default engine."""

    return _default_engine.score_transaction(transaction)


def score_batch(
    transactions: Iterable[Transaction | Mapping[str, Any]],
) -> list[float]:
    """Score a batch with the process-lifetime default engine."""

    return _default_engine.score_batch(transactions)


def reset() -> None:
    """Reset the process-lifetime default engine."""

    _default_engine.reset()


def solve(
    transactions: Transaction
    | Mapping[str, Any]
    | Iterable[Transaction | Mapping[str, Any]],
) -> float | list[float]:
    """Small transport-neutral adapter following other Phase 1 solutions."""

    if isinstance(transactions, (Transaction, Mapping)):
        return score_transaction(transactions)
    return score_batch(transactions)


__all__ = [
    "EngineSnapshot",
    "GhostChainsEngine",
    "ScoreConfig",
    "Transaction",
    "TransactionConflictError",
    "reset",
    "score_batch",
    "score_transaction",
    "solve",
]
