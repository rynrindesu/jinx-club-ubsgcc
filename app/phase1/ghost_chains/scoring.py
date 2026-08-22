"""Time-respecting structural scoring for Ghost Chains Phase 1."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import math
from typing import AbstractSet, Iterable, Mapping


Adjacency = Mapping[str, AbstractSet[str]]


@dataclass(frozen=True)
class ScoreConfig:
    """Centralized, deterministic structural-scoring parameters."""

    max_walk_length: int = 8
    walk_discount: float = 0.45
    convergence_weight: float = 2.0
    convergence_scale: float = 4.0
    open_route_capacity: float = 2.0
    shortest_path_weight: float = 4.0
    closed_route_weight: float = 14.0
    risk_half_saturation: float = 8.0
    self_transfer_risk: float = 0.12
    max_route_signatures: int = 250_000

    def __post_init__(self) -> None:
        if self.max_walk_length < 1:
            raise ValueError("max_walk_length must be positive")
        if self.max_route_signatures < 1:
            raise ValueError("max_route_signatures must be positive")
        if not 0 < self.walk_discount < 1:
            raise ValueError("walk_discount must be between zero and one")
        for name in (
            "convergence_weight",
            "convergence_scale",
            "open_route_capacity",
            "shortest_path_weight",
            "closed_route_weight",
            "risk_half_saturation",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.self_transfer_risk < 1:
            raise ValueError("self_transfer_risk must be in [0, 1)")


@dataclass(frozen=True)
class TemporalEdge:
    """One active transaction edge with a deterministic event-time key."""

    sender: str
    recipient: str
    created_at: datetime
    sequence: int


@dataclass(frozen=True)
class StructuralScore:
    """Explainable components of one marginal graph score."""

    risk: float
    raw: float
    open_route_delta: float
    closed_route_delta: float

    @classmethod
    def zero(cls) -> StructuralScore:
        return cls(risk=0.0, raw=0.0, open_route_delta=0.0, closed_route_delta=0.0)


@dataclass(frozen=True)
class _RouteState:
    connectivity: Mapping[tuple[str, str], float]
    shortest_distances: Mapping[tuple[str, str], int]
    direct_pairs: frozenset[tuple[str, str]]


class DiscountedWalkScorer:
    """Score marginal capacity added by causal, bounded entity routes."""

    def __init__(self, config: ScoreConfig | None = None):
        self.config = config or ScoreConfig()

    def score_new_event(
        self,
        candidate: TemporalEdge,
        active_events: Iterable[TemporalEdge],
    ) -> StructuralScore:
        """Score a candidate against active events ordered by event time."""

        active = tuple(active_events)
        if candidate.sender == candidate.recipient:
            repeated = any(
                event.sender == candidate.sender
                and event.recipient == candidate.recipient
                for event in active
            )
            return StructuralScore.zero() if repeated else self._self_transfer_score()

        # Only the weak components touching either candidate endpoint can change.
        # Keeping unrelated components out also makes the safety cap local rather
        # than allowing a dense, disconnected graph to affect this score.
        relevant = self._relevant_events(candidate, active)
        before = self._route_state(relevant)
        after = self._route_state((*relevant, candidate))
        candidate_pair = (candidate.sender, candidate.recipient)
        direct_was_active = candidate_pair in before.direct_pairs

        open_deltas: list[float] = []
        closed_deltas: list[float] = []
        for pair in sorted(
            before.connectivity.keys() | after.connectivity.keys()
        ):
            delta = self._pair_value(after.connectivity.get(pair, 0.0)) - (
                self._pair_value(before.connectivity.get(pair, 0.0))
            )
            if pair == candidate_pair and not direct_was_active:
                delta -= self._pair_value(1.0)
            if delta <= 0:
                continue
            if pair[0] == pair[1]:
                closed_deltas.append(delta)
            else:
                open_deltas.append(delta)

        shortest_path_deltas: list[float] = []
        for pair, after_distance in after.shortest_distances.items():
            if pair[0] == pair[1]:
                continue
            before_distance = before.shortest_distances.get(pair)
            if before_distance is None or after_distance >= before_distance:
                continue
            shortest_path_deltas.append(
                (1.0 / after_distance) - (1.0 / before_distance)
            )

        open_delta = math.fsum(open_deltas)
        shortest_path_delta = math.fsum(shortest_path_deltas)
        closed_delta = math.fsum(closed_deltas)
        non_closed_delta = (
            open_delta + self.config.shortest_path_weight * shortest_path_delta
        )
        bounded_non_closed_delta = (
            self.config.open_route_capacity
            * math.tanh(non_closed_delta / self.config.open_route_capacity)
        )
        raw = max(
            0.0,
            bounded_non_closed_delta + self.config.closed_route_weight * closed_delta,
        )
        return StructuralScore(
            risk=self._risk(raw),
            raw=raw,
            open_route_delta=non_closed_delta,
            closed_route_delta=closed_delta,
        )

    def _self_transfer_score(self) -> StructuralScore:
        risk = self.config.self_transfer_risk
        raw = (
            self.config.risk_half_saturation * risk / (1 - risk)
            if risk
            else 0.0
        )
        return StructuralScore(
            risk=risk,
            raw=raw,
            open_route_delta=raw,
            closed_route_delta=0.0,
        )

    @staticmethod
    def _relevant_events(
        candidate: TemporalEdge,
        events: Iterable[TemporalEdge],
    ) -> tuple[TemporalEdge, ...]:
        """Return active events in components the candidate can affect."""

        materialized = tuple(events)
        neighbors: defaultdict[str, set[str]] = defaultdict(set)
        for event in materialized:
            if event.sender == event.recipient:
                continue
            neighbors[event.sender].add(event.recipient)
            neighbors[event.recipient].add(event.sender)

        reachable = {candidate.sender, candidate.recipient}
        pending = list(reachable)
        while pending:
            entity = pending.pop()
            for neighbor in neighbors.get(entity, ()):
                if neighbor in reachable:
                    continue
                reachable.add(neighbor)
                pending.append(neighbor)

        return tuple(
            event
            for event in materialized
            if event.sender in reachable or event.recipient in reachable
        )

    def _route_state(self, events: Iterable[TemporalEdge]) -> _RouteState:
        """Enumerate distinct simple entity routes in strict temporal order."""

        signatures: set[tuple[str, ...]] = set()
        routes_ending_at: defaultdict[str, set[tuple[str, ...]]] = defaultdict(set)

        for event in sorted(
            events,
            key=lambda item: (
                item.created_at,
                item.sequence,
                item.sender,
                item.recipient,
            ),
        ):
            if event.sender == event.recipient:
                continue

            additions = {(event.sender, event.recipient)}
            for route in sorted(routes_ending_at.get(event.sender, ())):
                route_length = len(route) - 1
                if route_length >= self.config.max_walk_length:
                    continue
                if route[0] == route[-1]:
                    continue
                if event.recipient in route and event.recipient != route[0]:
                    continue
                additions.add((*route, event.recipient))

            for route in sorted(additions):
                if route in signatures:
                    continue
                if len(signatures) >= self.config.max_route_signatures:
                    continue
                signatures.add(route)
                routes_ending_at[route[-1]].add(route)

        route_weights: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
        shortest_distances: dict[tuple[str, str], int] = {}
        direct_pairs: set[tuple[str, str]] = set()
        for route in sorted(signatures):
            length = len(route) - 1
            pair = route[0], route[-1]
            weight = (
                1.0 / length
                if pair[0] == pair[1]
                else self.config.walk_discount ** (length - 1)
            )
            route_weights[pair].append(weight)
            previous_distance = shortest_distances.get(pair)
            if previous_distance is None or length < previous_distance:
                shortest_distances[pair] = length
            if length == 1:
                direct_pairs.add(pair)

        connectivity = {
            pair: math.fsum(weights) for pair, weights in route_weights.items()
        }
        return _RouteState(
            connectivity=connectivity,
            shortest_distances=shortest_distances,
            direct_pairs=frozenset(direct_pairs),
        )

    def _pair_value(self, connectivity: float) -> float:
        """Reward a second distinct route more, then approach linear growth."""

        if connectivity <= 0:
            return 0.0
        config = self.config
        convergence = (
            config.convergence_weight
            * config.convergence_scale
            * connectivity
            * (connectivity / (config.convergence_scale + connectivity))
        )
        return connectivity + convergence

    def _risk(self, raw: float) -> float:
        if not math.isfinite(raw):
            return 1.0
        if raw <= 0:
            return 0.0
        risk = raw / (raw + self.config.risk_half_saturation)
        return min(1.0, max(0.0, risk))
