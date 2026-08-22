"""Identity evidence for the cumulative Ghost Chains Phase 2 model."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from typing import Iterable

from .models import PresentValue, Transaction
from .scoring import Adjacency, StructuralScore


@dataclass(frozen=True)
class IdentityConfig:
    """Deterministic parameters for two independent identity dimensions."""

    max_path_length: int = 7
    path_discount: float = 0.65
    structural_scale: float = 2.0
    alignment_weight: float = 1.35
    divergence_weight: float = 0.85
    dropout_weight: float = 1.15
    disconnected_reuse_weight: float = 0.45
    disconnected_capacity: float = 3.0
    ip_reliability: float = 0.75
    device_reliability: float = 1.0

    def __post_init__(self) -> None:
        if self.max_path_length < 1:
            raise ValueError("max_path_length must be positive")
        if not 0 < self.path_discount < 1:
            raise ValueError("path_discount must be between zero and one")
        for name in (
            "structural_scale",
            "alignment_weight",
            "divergence_weight",
            "dropout_weight",
            "disconnected_reuse_weight",
            "disconnected_capacity",
            "ip_reliability",
            "device_reliability",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class IdentityDimensionScore:
    """Explainable evidence contributed by one optional identity field."""

    dimension: str
    raw: float
    alignment: float
    divergence: float
    dropout: float
    disconnected_reuse: float


@dataclass(frozen=True)
class IdentityScore:
    """Combined raw evidence from IP and device, kept as separate dimensions."""

    raw: float
    dimensions: tuple[IdentityDimensionScore, ...]

    @classmethod
    def zero(cls) -> IdentityScore:
        return cls(raw=0.0, dimensions=())


class IdentityScorer:
    """Score identity agreement, trail changes, and cross-component reuse.

    The scorer only reads transactions that are currently active in the
    structural engine.  Consequently expiry, reset, retries, and out-of-order
    event handling remain governed by the same single 24-hour state model.
    """

    def __init__(self, config: IdentityConfig | None = None):
        self.config = config or IdentityConfig()

    def score(
        self,
        transaction: Transaction,
        active_transactions: Iterable[Transaction],
        forward: Adjacency,
        reverse: Adjacency,
        structural: StructuralScore,
    ) -> IdentityScore:
        """Return identity evidence visible immediately before activation."""

        active = tuple(active_transactions)
        if not active:
            return IdentityScore.zero()

        transactions_by_pair: defaultdict[
            tuple[str, str], list[Transaction]
        ] = defaultdict(list)
        for historical in active:
            transactions_by_pair[(historical.sender, historical.recipient)].append(
                historical
            )

        structural_activation = self._structural_activation(structural.raw)
        dimensions = (
            self._score_dimension(
                "ip",
                transaction,
                active,
                transactions_by_pair,
                forward,
                reverse,
                structural_activation,
                self.config.ip_reliability,
            ),
            self._score_dimension(
                "device",
                transaction,
                active,
                transactions_by_pair,
                forward,
                reverse,
                structural_activation,
                self.config.device_reliability,
            ),
        )
        nonzero = tuple(dimension for dimension in dimensions if dimension.raw > 0)
        return IdentityScore(
            raw=math.fsum(dimension.raw for dimension in nonzero),
            dimensions=nonzero,
        )

    def _score_dimension(
        self,
        dimension: str,
        transaction: Transaction,
        active: tuple[Transaction, ...],
        transactions_by_pair: dict[tuple[str, str], list[Transaction]],
        forward: Adjacency,
        reverse: Adjacency,
        structural_activation: float,
        reliability: float,
    ) -> IdentityDimensionScore:
        current_key = _identity_key(getattr(transaction, dimension))
        upstream = self._upstream_weights(
            dimension,
            transaction.sender,
            transactions_by_pair,
            reverse,
        )
        path_mass = math.fsum(upstream.values())
        path_strength = 1.0 - math.exp(-path_mass)

        alignment = 0.0
        divergence = 0.0
        dropout = 0.0
        disconnected_reuse = 0.0

        if path_mass > 0:
            concentration = max(upstream.values()) / path_mass
            structural_context = 0.35 + 0.65 * structural_activation
            if current_key is None:
                dropout = (
                    self.config.dropout_weight
                    * path_strength
                    * (0.5 + 0.5 * concentration)
                    * (0.6 + 0.4 * structural_activation)
                )
            else:
                matching_mass = upstream.get(current_key, 0.0)
                mismatching_mass = max(0.0, path_mass - matching_mass)
                match_ratio = matching_mass / path_mass
                mismatch_ratio = mismatching_mass / path_mass
                if matching_mass > 0:
                    alignment = (
                        self.config.alignment_weight
                        * path_strength
                        * match_ratio**2
                        * (0.25 + 0.75 * structural_activation)
                    )
                if mismatching_mass > 0:
                    divergence = (
                        self.config.divergence_weight
                        * path_strength
                        * mismatch_ratio
                        * structural_context
                    )

        if current_key is not None:
            foreign_components = self._foreign_component_count(
                dimension,
                current_key,
                transaction,
                active,
                forward,
                reverse,
            )
            bounded_components = (
                self.config.disconnected_capacity
                * math.tanh(
                    foreign_components / self.config.disconnected_capacity
                )
            )
            disconnected_reuse = (
                self.config.disconnected_reuse_weight
                * bounded_components
                * (0.75 + 0.5 * structural_activation)
            )

        alignment *= reliability
        divergence *= reliability
        dropout *= reliability
        disconnected_reuse *= reliability
        raw = math.fsum((alignment, divergence, dropout, disconnected_reuse))
        return IdentityDimensionScore(
            dimension=dimension,
            raw=raw,
            alignment=alignment,
            divergence=divergence,
            dropout=dropout,
            disconnected_reuse=disconnected_reuse,
        )

    def _upstream_weights(
        self,
        dimension: str,
        sender: str,
        transactions_by_pair: dict[tuple[str, str], list[Transaction]],
        reverse: Adjacency,
    ) -> dict[str, float]:
        """Aggregate identity values on directed paths ending at the sender."""

        weights: defaultdict[str, float] = defaultdict(float)
        visited = {sender}
        frontier = {sender}
        discount = 1.0

        for _ in range(self.config.max_path_length):
            next_frontier: set[str] = set()
            for recipient in sorted(frontier):
                for predecessor in sorted(reverse.get(recipient, ())):
                    pair_transactions = transactions_by_pair.get(
                        (predecessor, recipient), ()
                    )
                    # Parallel payments share one structural edge.  Averaging
                    # their evidence prevents payment frequency alone from
                    # manufacturing an identity anomaly.
                    pair_weight = discount / max(1, len(pair_transactions))
                    for historical in pair_transactions:
                        key = _identity_key(getattr(historical, dimension))
                        if key is not None:
                            weights[key] += pair_weight
                    if predecessor not in visited:
                        next_frontier.add(predecessor)
            if not next_frontier:
                break
            visited.update(next_frontier)
            frontier = next_frontier
            discount *= self.config.path_discount

        return dict(weights)

    def _foreign_component_count(
        self,
        dimension: str,
        current_key: str,
        transaction: Transaction,
        active: tuple[Transaction, ...],
        forward: Adjacency,
        reverse: Adjacency,
    ) -> int:
        """Count weak components reusing the identity away from this edge."""

        local_nodes = self._weak_component(
            transaction.sender, forward, reverse
        ) | self._weak_component(transaction.recipient, forward, reverse)
        foreign_components: set[frozenset[str]] = set()

        for historical in active:
            if _identity_key(getattr(historical, dimension)) != current_key:
                continue
            if historical.sender in local_nodes or historical.recipient in local_nodes:
                continue
            component = self._weak_component(historical.sender, forward, reverse)
            component.update(
                self._weak_component(historical.recipient, forward, reverse)
            )
            foreign_components.add(frozenset(component))

        return len(foreign_components)

    @staticmethod
    def _weak_component(
        start: str,
        forward: Adjacency,
        reverse: Adjacency,
    ) -> set[str]:
        visited = {start}
        frontier = {start}
        while frontier:
            next_frontier = {
                neighbor
                for node in frontier
                for neighbor in (
                    set(forward.get(node, ())) | set(reverse.get(node, ()))
                )
                if neighbor not in visited
            }
            visited.update(next_frontier)
            frontier = next_frontier
        return visited

    def _structural_activation(self, raw: float) -> float:
        if not math.isfinite(raw):
            return 1.0
        return math.tanh(max(0.0, raw) / self.config.structural_scale)


def _identity_key(value: PresentValue) -> str | None:
    """Turn a useful optional value into a stable, type-sensitive key."""

    if not value.present or value.value is None:
        return None
    if isinstance(value.value, str) and not value.value.strip():
        return None
    try:
        encoded = json.dumps(
            value.value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        return None
    return f"{type(value.value).__name__}:{encoded}"


__all__ = [
    "IdentityConfig",
    "IdentityDimensionScore",
    "IdentityScore",
    "IdentityScorer",
]
