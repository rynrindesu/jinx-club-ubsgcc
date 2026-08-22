"""Time-respecting identity evidence for Ghost Chains Phase 2."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import json
import math
from typing import Iterable

from .models import PresentValue, Transaction
from .scoring import StructuralScore


@dataclass(frozen=True)
class IdentityConfig:
    """Deterministic parameters for two independent identity dimensions."""

    max_path_length: int = 8
    max_route_states: int = 250_000
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
        if self.max_route_states < 1:
            raise ValueError("max_route_states must be positive")
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
class IdentityEvent:
    """One active identity-bearing event with its arrival tie-breaker."""

    transaction: Transaction
    sequence: int

    @property
    def key(self) -> tuple[datetime, int]:
        return self.transaction.created_at, self.sequence


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
    """Score causal identity paths and cross-component reuse.

    Path evidence follows the same strict ``(createdAt, arrival sequence)``
    ordering and simple-route rules as Phase 1 structural scoring. An
    out-of-order candidate can therefore connect to a later active event, but
    an event that is later in event time cannot masquerade as an earlier leg.
    """

    def __init__(self, config: IdentityConfig | None = None):
        self.config = config or IdentityConfig()

    def score(
        self,
        candidate: IdentityEvent,
        active_events: Iterable[IdentityEvent],
        structural: StructuralScore,
    ) -> IdentityScore:
        """Return identity evidence visible immediately before activation."""

        active = tuple(active_events)
        if not active:
            return IdentityScore.zero()

        upstream, downstream = self._causal_context(candidate, active)
        neighbors = self._component_neighbors(active)
        structural_activation = self._structural_activation(structural.raw)
        dimensions = (
            self._score_dimension(
                "ip",
                candidate,
                active,
                upstream,
                downstream,
                neighbors,
                structural_activation,
                self.config.ip_reliability,
            ),
            self._score_dimension(
                "device",
                candidate,
                active,
                upstream,
                downstream,
                neighbors,
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
        candidate: IdentityEvent,
        active: tuple[IdentityEvent, ...],
        upstream: tuple[tuple[IdentityEvent, int], ...],
        downstream: tuple[tuple[IdentityEvent, int], ...],
        neighbors: dict[str, set[str]],
        structural_activation: float,
        reliability: float,
    ) -> IdentityDimensionScore:
        transaction = candidate.transaction
        current_key = _identity_key(getattr(transaction, dimension))
        upstream_weights = self._context_weights(dimension, upstream)
        downstream_weights = self._context_weights(dimension, downstream)
        context_weights = dict(upstream_weights)
        for key, weight in downstream_weights.items():
            context_weights[key] = context_weights.get(key, 0.0) + weight

        context_mass = math.fsum(context_weights.values())
        context_strength = 1.0 - math.exp(-context_mass)
        upstream_mass = math.fsum(upstream_weights.values())
        upstream_strength = 1.0 - math.exp(-upstream_mass)

        alignment = 0.0
        divergence = 0.0
        dropout = 0.0
        disconnected_reuse = 0.0

        if current_key is None and upstream_mass > 0:
            # Only an earlier causal leg can establish a trail that the
            # candidate subsequently drops. A field first appearing on a
            # downstream leg is not retroactively treated as a dropout.
            concentration = max(upstream_weights.values()) / upstream_mass
            dropout = (
                self.config.dropout_weight
                * upstream_strength
                * (0.5 + 0.5 * concentration)
                * (0.6 + 0.4 * structural_activation)
            )
        elif current_key is not None and context_mass > 0:
            matching_mass = context_weights.get(current_key, 0.0)
            mismatching_mass = max(0.0, context_mass - matching_mass)
            match_ratio = matching_mass / context_mass
            mismatch_ratio = mismatching_mass / context_mass
            structural_context = 0.35 + 0.65 * structural_activation
            if matching_mass > 0:
                alignment = (
                    self.config.alignment_weight
                    * context_strength
                    * match_ratio**2
                    * (0.25 + 0.75 * structural_activation)
                )
            if mismatching_mass > 0:
                divergence = (
                    self.config.divergence_weight
                    * context_strength
                    * mismatch_ratio
                    * structural_context
                )

        if current_key is not None:
            foreign_components = self._foreign_component_count(
                dimension,
                current_key,
                transaction,
                active,
                neighbors,
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

    def _causal_context(
        self,
        candidate: IdentityEvent,
        active: tuple[IdentityEvent, ...],
    ) -> tuple[
        tuple[tuple[IdentityEvent, int], ...],
        tuple[tuple[IdentityEvent, int], ...],
    ]:
        """Find active events on valid causal routes containing the candidate."""

        incoming: defaultdict[str, list[IdentityEvent]] = defaultdict(list)
        outgoing: defaultdict[str, list[IdentityEvent]] = defaultdict(list)
        for event in active:
            transaction = event.transaction
            if transaction.sender == transaction.recipient:
                continue
            incoming[transaction.recipient].append(event)
            outgoing[transaction.sender].append(event)
        for events in (*incoming.values(), *outgoing.values()):
            events.sort(key=self._event_sort_key)

        upstream = self._walk_upstream(candidate, incoming)
        downstream = self._walk_downstream(candidate, outgoing)
        return upstream, downstream

    def _walk_upstream(
        self,
        candidate: IdentityEvent,
        incoming: dict[str, list[IdentityEvent]],
    ) -> tuple[tuple[IdentityEvent, int], ...]:
        transaction = candidate.transaction
        found: dict[int, tuple[IdentityEvent, int]] = {}
        stack = [
            (
                transaction.sender,
                candidate.key,
                (transaction.sender, transaction.recipient),
                0,
            )
        ]
        explored_states = 0

        while stack and explored_states < self.config.max_route_states:
            explored_states += 1
            node, boundary, route, distance = stack.pop()
            if route[0] == route[-1] or distance >= self.config.max_path_length - 1:
                continue
            for event in reversed(incoming.get(node, ())):
                if event.key >= boundary:
                    continue
                predecessor = event.transaction.sender
                if predecessor in route and predecessor != route[-1]:
                    continue
                next_distance = distance + 1
                previous = found.get(event.sequence)
                if previous is None or next_distance < previous[1]:
                    found[event.sequence] = (event, next_distance)
                stack.append(
                    (
                        predecessor,
                        event.key,
                        (predecessor, *route),
                        next_distance,
                    )
                )

        return tuple(sorted(found.values(), key=self._context_sort_key))

    def _walk_downstream(
        self,
        candidate: IdentityEvent,
        outgoing: dict[str, list[IdentityEvent]],
    ) -> tuple[tuple[IdentityEvent, int], ...]:
        transaction = candidate.transaction
        found: dict[int, tuple[IdentityEvent, int]] = {}
        stack = [
            (
                transaction.recipient,
                candidate.key,
                (transaction.sender, transaction.recipient),
                0,
            )
        ]
        explored_states = 0

        while stack and explored_states < self.config.max_route_states:
            explored_states += 1
            node, boundary, route, distance = stack.pop()
            if route[0] == route[-1] or distance >= self.config.max_path_length - 1:
                continue
            for event in outgoing.get(node, ()):
                if event.key <= boundary:
                    continue
                recipient = event.transaction.recipient
                if recipient in route and recipient != route[0]:
                    continue
                next_distance = distance + 1
                previous = found.get(event.sequence)
                if previous is None or next_distance < previous[1]:
                    found[event.sequence] = (event, next_distance)
                stack.append(
                    (
                        recipient,
                        event.key,
                        (*route, recipient),
                        next_distance,
                    )
                )

        return tuple(sorted(found.values(), key=self._context_sort_key))

    def _context_weights(
        self,
        dimension: str,
        context: tuple[tuple[IdentityEvent, int], ...],
    ) -> dict[str, float]:
        """Average parallel-event evidence, then discount it by path distance."""

        groups: defaultdict[
            tuple[str, str, int], list[IdentityEvent]
        ] = defaultdict(list)
        for event, distance in context:
            transaction = event.transaction
            groups[(transaction.sender, transaction.recipient, distance)].append(
                event
            )

        weights: defaultdict[str, float] = defaultdict(float)
        for (_, _, distance), events in sorted(groups.items()):
            group_weight = (
                self.config.path_discount ** (distance - 1)
            ) / len(events)
            for event in events:
                key = _identity_key(getattr(event.transaction, dimension))
                if key is not None:
                    weights[key] += group_weight
        return dict(weights)

    @staticmethod
    def _component_neighbors(
        active: tuple[IdentityEvent, ...],
    ) -> dict[str, set[str]]:
        neighbors: defaultdict[str, set[str]] = defaultdict(set)
        for event in active:
            transaction = event.transaction
            if transaction.sender == transaction.recipient:
                continue
            neighbors[transaction.sender].add(transaction.recipient)
            neighbors[transaction.recipient].add(transaction.sender)
        return dict(neighbors)

    def _foreign_component_count(
        self,
        dimension: str,
        current_key: str,
        transaction: Transaction,
        active: tuple[IdentityEvent, ...],
        neighbors: dict[str, set[str]],
    ) -> int:
        """Count weak components reusing the identity away from this edge."""

        local_nodes = self._weak_component(
            transaction.sender, neighbors
        ) | self._weak_component(transaction.recipient, neighbors)
        foreign_components: set[frozenset[str]] = set()

        for event in active:
            historical = event.transaction
            if _identity_key(getattr(historical, dimension)) != current_key:
                continue
            if historical.sender in local_nodes or historical.recipient in local_nodes:
                continue
            component = self._weak_component(historical.sender, neighbors)
            component.update(self._weak_component(historical.recipient, neighbors))
            foreign_components.add(frozenset(component))

        return len(foreign_components)

    @staticmethod
    def _weak_component(start: str, neighbors: dict[str, set[str]]) -> set[str]:
        visited = {start}
        frontier = {start}
        while frontier:
            next_frontier = {
                neighbor
                for node in frontier
                for neighbor in neighbors.get(node, ())
                if neighbor not in visited
            }
            visited.update(next_frontier)
            frontier = next_frontier
        return visited

    def _structural_activation(self, raw: float) -> float:
        if not math.isfinite(raw):
            return 1.0
        return math.tanh(max(0.0, raw) / self.config.structural_scale)

    @staticmethod
    def _event_sort_key(event: IdentityEvent) -> tuple[datetime, int, str, str]:
        transaction = event.transaction
        return (
            transaction.created_at,
            event.sequence,
            transaction.sender,
            transaction.recipient,
        )

    @classmethod
    def _context_sort_key(
        cls,
        item: tuple[IdentityEvent, int],
    ) -> tuple[int, datetime, int, str, str]:
        event, distance = item
        return distance, *cls._event_sort_key(event)


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
    "IdentityEvent",
    "IdentityScore",
    "IdentityScorer",
]
