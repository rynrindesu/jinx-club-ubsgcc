"""Active-graph structural scoring for Ghost Chains Phase 1."""

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
    open_route_capacity: float = 4.0
    shortest_path_weight: float = 4.0
    closed_route_weight: float = 14.0
    risk_half_saturation: float = 8.0
    self_transfer_risk: float = 0.12
    max_route_signatures: int = 250_000
    topology_weight: float = 0.8
    temporal_weight: float = 0.2
    temporal_half_life_seconds: float = 12 * 60 * 60

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
        if self.topology_weight < 0 or self.temporal_weight < 0:
            raise ValueError("structural mixture weights must be non-negative")
        if self.temporal_half_life_seconds <= 0:
            raise ValueError("temporal_half_life_seconds must be positive")
        if not math.isclose(
            self.topology_weight + self.temporal_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("structural mixture weights must sum to one")


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
    topology_raw: float = 0.0
    temporal_raw: float = 0.0

    @classmethod
    def zero(cls) -> StructuralScore:
        return cls(risk=0.0, raw=0.0, open_route_delta=0.0, closed_route_delta=0.0)


@dataclass(frozen=True)
class _RouteState:
    connectivity: Mapping[tuple[str, str], float]
    shortest_distances: Mapping[tuple[str, str], int]
    shortest_confidences: Mapping[tuple[str, str], float]
    shortest_completion_keys: Mapping[
        tuple[str, str], tuple[datetime, int]
    ]
    direct_pairs: frozenset[tuple[str, str]]


class DiscountedWalkScorer:
    """Score marginal recurring-flow capacity in the active directed graph.

    Topology is primary: active edges form a graph regardless of the order in
    which their transactions happened to arrive. A smaller time-respecting
    component rewards evidence that is also realizable as an observed flow,
    without allowing timestamp order to erase an otherwise real return path.
    """

    def __init__(self, config: ScoreConfig | None = None):
        self.config = config or ScoreConfig()

    def score_new_event(
        self,
        candidate: TemporalEdge,
        active_events: Iterable[TemporalEdge],
    ) -> StructuralScore:
        """Score a candidate against the active graph and observed flow order."""

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
        topology_before = self._topology_route_state(relevant)
        topology_after = self._topology_route_state((*relevant, candidate))
        temporal_before = self._temporal_route_state(relevant)
        temporal_after = self._temporal_route_state((*relevant, candidate))

        topology = self._state_delta(candidate, topology_before, topology_after)
        topology = self._apply_long_return_floor(
            candidate,
            relevant,
            topology_before,
            topology,
        )
        temporal = self._state_delta(candidate, temporal_before, temporal_after)
        config = self.config
        raw = math.fsum(
            (
                config.topology_weight * topology.raw,
                config.temporal_weight * temporal.raw,
            )
        )
        open_route_delta = math.fsum(
            (
                config.topology_weight * topology.open_route_delta,
                config.temporal_weight * temporal.open_route_delta,
            )
        )
        closed_route_delta = math.fsum(
            (
                config.topology_weight * topology.closed_route_delta,
                config.temporal_weight * temporal.closed_route_delta,
            )
        )
        return StructuralScore(
            risk=self._risk(raw),
            raw=raw,
            open_route_delta=open_route_delta,
            closed_route_delta=closed_route_delta,
            topology_raw=topology.raw,
            temporal_raw=temporal.raw,
        )

    def _state_delta(
        self,
        candidate: TemporalEdge,
        before: _RouteState,
        after: _RouteState,
    ) -> StructuralScore:
        """Return the bounded marginal between two route-state snapshots."""

        candidate_pair = candidate.sender, candidate.recipient
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
            confidence = min(
                before.shortest_confidences.get(pair, 1.0),
                after.shortest_confidences.get(pair, 1.0),
            )
            completion_key = before.shortest_completion_keys.get(pair)
            if completion_key is not None:
                candidate_key = candidate.created_at, candidate.sequence
                if completion_key >= candidate_key:
                    # A late-arriving edge cannot retroactively shorten a route
                    # that had not completed at the candidate's event time.
                    confidence = 0.0
                else:
                    gap = max(
                        0.0,
                        (
                            candidate.created_at - completion_key[0]
                        ).total_seconds(),
                    )
                    confidence *= math.exp(
                        -math.log(2.0)
                        * gap
                        / self.config.temporal_half_life_seconds
                    )
            shortest_path_deltas.append(
                ((1.0 / after_distance) - (1.0 / before_distance))
                * confidence
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

    def _apply_long_return_floor(
        self,
        candidate: TemporalEdge,
        relevant: tuple[TemporalEdge, ...],
        before: _RouteState,
        score: StructuralScore,
    ) -> StructuralScore:
        """Keep a real return visible when bounded route enumeration misses it.

        Simple-route enumeration is deliberately bounded for predictable
        streaming cost. Reachability itself is cheap, however, and a return
        path must not become an ordinary extension merely because it is one
        hop longer than that safety bound.
        """

        pair = candidate.sender, candidate.recipient
        if pair in before.direct_pairs:
            return score

        adjacency = self._adjacency(relevant)
        return_distance = self._shortest_distance(
            candidate.recipient,
            candidate.sender,
            adjacency,
        )
        if return_distance is None:
            return score

        cycle_length = return_distance + 1
        closed_floor = 1.0 / cycle_length
        if score.closed_route_delta >= closed_floor:
            return score

        raw = max(
            0.0,
            score.raw
            + self.config.closed_route_weight
            * (closed_floor - score.closed_route_delta),
        )
        return StructuralScore(
            risk=self._risk(raw),
            raw=raw,
            open_route_delta=score.open_route_delta,
            closed_route_delta=closed_floor,
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

    def _topology_route_state(
        self,
        events: Iterable[TemporalEdge],
    ) -> _RouteState:
        """Calculate bounded discounted-walk capacity in the active graph.

        This is the sparse equivalent of ``A + a*A^2 + ...``. Unlike capped
        path-signature enumeration, its result cannot depend on entity sort
        order, and cycles contribute the repeated-flow capacity that makes
        them structurally distinct from an acyclic route.
        """

        materialized = tuple(events)
        adjacency = self._adjacency(materialized)
        nodes: set[str] = set()
        for event in materialized:
            if event.sender == event.recipient:
                continue
            nodes.add(event.sender)
            nodes.add(event.recipient)

        connectivity: defaultdict[tuple[str, str], float] = defaultdict(float)
        for source in sorted(nodes):
            frontier: dict[str, int] = {source: 1}
            discount = 1.0
            for _ in range(self.config.max_walk_length):
                next_frontier: defaultdict[str, int] = defaultdict(int)
                for node, walk_count in frontier.items():
                    for recipient in adjacency.get(node, ()):
                        next_frontier[recipient] += walk_count
                if not next_frontier:
                    break
                for recipient, walk_count in next_frontier.items():
                    connectivity[(source, recipient)] += discount * walk_count
                frontier = dict(next_frontier)
                discount *= self.config.walk_discount

        shortest_distances = self._all_pair_shortest_distances(adjacency, nodes)
        return _RouteState(
            connectivity=dict(connectivity),
            shortest_distances=shortest_distances,
            shortest_confidences={
                pair: 1.0 for pair in shortest_distances
            },
            shortest_completion_keys={},
            direct_pairs=frozenset(
                (sender, recipient)
                for sender, recipients in adjacency.items()
                for recipient in recipients
            ),
        )

    def _temporal_route_state(self, events: Iterable[TemporalEdge]) -> _RouteState:
        """Enumerate distinct simple entity routes in strict temporal order."""

        signatures: set[tuple[str, ...]] = set()
        inferred_signature_count = 0
        route_factors: dict[tuple[str, ...], float] = {}
        route_completion_keys: dict[
            tuple[str, ...], tuple[datetime, int]
        ] = {}
        routes_ending_at: defaultdict[
            str, dict[tuple[str, ...], datetime]
        ] = defaultdict(dict)

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

            additions = {(event.sender, event.recipient): event.created_at}
            # Dict insertion order is induced by transaction sequence. Keeping
            # that order makes the safety cap invariant to arbitrary entity
            # spelling; lexical route order would retain different hypotheses
            # after an identifier-only rename.
            for route, started_at in routes_ending_at.get(
                event.sender, {}
            ).items():
                route_length = len(route) - 1
                if route_length >= self.config.max_walk_length:
                    continue
                if route[0] == route[-1]:
                    continue
                if event.recipient in route and event.recipient != route[0]:
                    continue
                additions[(*route, event.recipient)] = started_at

            for route, started_at in additions.items():
                is_observed_edge = len(route) == 2
                if (
                    route not in signatures
                    and not is_observed_edge
                    and inferred_signature_count
                    >= self.config.max_route_signatures
                ):
                    continue
                if route not in signatures and not is_observed_edge:
                    inferred_signature_count += 1
                signatures.add(route)
                span = max(0.0, (event.created_at - started_at).total_seconds())
                factor = math.exp(
                    -math.log(2.0)
                    * span
                    / self.config.temporal_half_life_seconds
                )
                completion_key = event.created_at, event.sequence
                previous_factor = route_factors.get(route)
                if (
                    previous_factor is None
                    or factor > previous_factor
                    or (
                        factor == previous_factor
                        and completion_key
                        > route_completion_keys.get(route, completion_key)
                    )
                ):
                    route_factors[route] = factor
                    route_completion_keys[route] = completion_key
                if route[0] == route[-1]:
                    continue
                previous_start = routes_ending_at[route[-1]].get(route)
                if previous_start is None or started_at > previous_start:
                    routes_ending_at[route[-1]][route] = started_at

        return self._state_from_signatures(
            signatures,
            route_factors,
            route_completion_keys,
        )

    def _state_from_signatures(
        self,
        signatures: Iterable[tuple[str, ...]],
        route_factors: Mapping[tuple[str, ...], float] | None = None,
        route_completion_keys: Mapping[
            tuple[str, ...], tuple[datetime, int]
        ]
        | None = None,
    ) -> _RouteState:
        route_weights: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
        shortest_distances: dict[tuple[str, str], int] = {}
        shortest_confidences: dict[tuple[str, str], float] = {}
        shortest_completion_keys: dict[
            tuple[str, str], tuple[datetime, int]
        ] = {}
        direct_pairs: set[tuple[str, str]] = set()
        for route in sorted(signatures):
            length = len(route) - 1
            pair = route[0], route[-1]
            factor = 1.0
            if route_factors is not None:
                factor = route_factors.get(route, 0.0)
            weight = (
                1.0 / length
                if pair[0] == pair[1]
                else self.config.walk_discount ** (length - 1)
            )
            weight *= factor

            if pair[0] == pair[1]:
                # A time-respecting cycle has one real chronological start.
                # Rotating it would imply a different (usually impossible)
                # ordering of the same events.
                route_weights[pair].append(weight)
                continue

            route_weights[pair].append(weight)
            previous_distance = shortest_distances.get(pair)
            completion_key = (
                route_completion_keys.get(route)
                if route_completion_keys is not None
                else None
            )
            if previous_distance is None or length < previous_distance:
                shortest_distances[pair] = length
                shortest_confidences[pair] = factor
                if completion_key is not None:
                    shortest_completion_keys[pair] = completion_key
            elif length == previous_distance:
                previous_confidence = shortest_confidences.get(pair, 0.0)
                previous_completion = shortest_completion_keys.get(pair)
                if (
                    factor > previous_confidence
                    or (
                        factor == previous_confidence
                        and completion_key is not None
                        and (
                            previous_completion is None
                            or completion_key > previous_completion
                        )
                    )
                ):
                    shortest_confidences[pair] = factor
                    if completion_key is not None:
                        shortest_completion_keys[pair] = completion_key
            if length == 1:
                direct_pairs.add(pair)

        connectivity = {
            pair: math.fsum(weights) for pair, weights in route_weights.items()
        }
        return _RouteState(
            connectivity=connectivity,
            shortest_distances=shortest_distances,
            shortest_confidences=shortest_confidences,
            shortest_completion_keys=shortest_completion_keys,
            direct_pairs=frozenset(direct_pairs),
        )

    @staticmethod
    def _adjacency(
        events: Iterable[TemporalEdge],
    ) -> dict[str, set[str]]:
        adjacency: defaultdict[str, set[str]] = defaultdict(set)
        for event in events:
            if event.sender != event.recipient:
                adjacency[event.sender].add(event.recipient)
        return dict(adjacency)

    @classmethod
    def _all_pair_shortest_distances(
        cls,
        adjacency: Mapping[str, AbstractSet[str]],
        nodes: AbstractSet[str],
    ) -> dict[tuple[str, str], int]:
        distances: dict[tuple[str, str], int] = {}
        for source in sorted(nodes):
            frontier = {source}
            visited = {source}
            distance = 0
            while frontier:
                distance += 1
                frontier = {
                    recipient
                    for node in frontier
                    for recipient in adjacency.get(node, ())
                    if recipient not in visited
                }
                for recipient in frontier:
                    distances[(source, recipient)] = distance
                visited.update(frontier)
        return distances

    @staticmethod
    def _shortest_distance(
        source: str,
        target: str,
        adjacency: Mapping[str, AbstractSet[str]],
    ) -> int | None:
        if source == target:
            return 0
        frontier = {source}
        visited = {source}
        distance = 0
        while frontier:
            distance += 1
            frontier = {
                recipient
                for node in frontier
                for recipient in adjacency.get(node, ())
                if recipient not in visited
            }
            if target in frontier:
                return distance
            visited.update(frontier)
        return None

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
