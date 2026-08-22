"""Bounded discounted-walk scoring for Ghost Chains Phase 1."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import AbstractSet, Mapping


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
    fan_in_capacity: float = 0.5
    fan_in_scale: float = 1.0
    closed_route_weight: float = 6.0
    risk_half_saturation: float = 8.0
    self_transfer_risk: float = 0.12

    def __post_init__(self) -> None:
        if self.max_walk_length < 1:
            raise ValueError("max_walk_length must be positive")
        if not 0 < self.walk_discount < 1:
            raise ValueError("walk_discount must be between zero and one")
        for name in (
            "convergence_weight",
            "convergence_scale",
            "open_route_capacity",
            "shortest_path_weight",
            "fan_in_capacity",
            "fan_in_scale",
            "closed_route_weight",
            "risk_half_saturation",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.self_transfer_risk < 1:
            raise ValueError("self_transfer_risk must be in [0, 1)")


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


class DiscountedWalkScorer:
    """Score how much one binary directed edge expands recurring flow paths."""

    def __init__(self, config: ScoreConfig | None = None):
        self.config = config or ScoreConfig()

    def score_new_edge(
        self,
        sender: str,
        recipient: str,
        forward: Adjacency,
        reverse: Adjacency,
    ) -> StructuralScore:
        """Return the exact bounded-walk marginal for an absent graph edge."""

        if recipient in forward.get(sender, ()):
            return StructuralScore.zero()
        if sender == recipient:
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

        affected_sources = self._reverse_neighborhood(sender, reverse)
        novel_route_deltas: list[float] = []
        reinforcement_deltas: list[float] = []
        closed_deltas: list[float] = []
        shortest_path_deltas: list[float] = []
        extra_edge = (sender, recipient)
        direct_edge_baseline = self._pair_value(1.0)

        for source in sorted(affected_sources):
            before = self._connectivity(source, forward)
            after = self._connectivity(source, forward, extra_edge=extra_edge)
            before_distances = self._shortest_distances(source, forward)
            after_distances = self._shortest_distances(
                source, forward, extra_edge=extra_edge
            )
            for target in sorted(after.keys() | before.keys()):
                delta = self._pair_value(after.get(target, 0.0)) - self._pair_value(
                    before.get(target, 0.0)
                )
                if source == sender and target == recipient:
                    delta -= direct_edge_baseline
                if delta <= 0:
                    continue
                if source == target:
                    closed_deltas.append(delta)
                elif before.get(target, 0.0) > 0:
                    reinforcement_deltas.append(delta)
                else:
                    novel_route_deltas.append(delta)

            for target, after_distance in after_distances.items():
                before_distance = before_distances.get(target)
                if (
                    target == source
                    or before_distance is None
                    or after_distance >= before_distance
                ):
                    continue
                shortest_path_deltas.append(
                    (1.0 / after_distance) - (1.0 / before_distance)
                )

        novel_route_delta = math.fsum(novel_route_deltas)
        reinforcement_delta = math.fsum(reinforcement_deltas)
        shortest_path_delta = math.fsum(shortest_path_deltas)
        closed_delta = math.fsum(closed_deltas)
        fan_in_delta = self._fan_in_delta(recipient, reverse)
        non_closed_delta = (
            novel_route_delta
            + reinforcement_delta
            + self.config.shortest_path_weight * shortest_path_delta
            + fan_in_delta
        )
        bounded_non_closed_delta = (
            self.config.open_route_capacity
            * math.tanh(non_closed_delta / self.config.open_route_capacity)
        )
        raw = max(
            0.0,
            bounded_non_closed_delta + self.config.closed_route_weight * closed_delta,
        )
        if not math.isfinite(raw):
            risk = 1.0
        elif raw == 0:
            risk = 0.0
        else:
            risk = raw / (raw + self.config.risk_half_saturation)

        return StructuralScore(
            risk=min(1.0, max(0.0, risk)),
            raw=raw,
            open_route_delta=non_closed_delta,
            closed_route_delta=closed_delta,
        )

    def _fan_in_delta(self, recipient: str, reverse: Adjacency) -> float:
        """Return a small saturated marginal for a reused destination."""

        before_degree = len(reverse.get(recipient, ()))
        if before_degree == 0:
            return 0.0
        config = self.config
        before = config.fan_in_capacity * math.tanh(
            (before_degree - 1) / config.fan_in_scale
        )
        after = config.fan_in_capacity * math.tanh(
            before_degree / config.fan_in_scale
        )
        return after - before

    def _pair_value(self, connectivity: float) -> float:
        """Reward path multiplicity, becoming linear for very dense pairs."""

        if connectivity <= 0:
            return 0.0
        config = self.config
        # This stable form is x + w*x^2/(1+x/s), avoiding an intermediate x^2.
        convergence = (
            config.convergence_weight
            * config.convergence_scale
            * connectivity
            * (connectivity / (config.convergence_scale + connectivity))
        )
        return connectivity + convergence

    def _reverse_neighborhood(
        self, node: str, reverse: Adjacency
    ) -> set[str]:
        """Find sources whose walks can reach the new edge within the bound."""

        visited = {node}
        frontier = {node}
        for _ in range(self.config.max_walk_length - 1):
            next_frontier = {
                predecessor
                for current in frontier
                for predecessor in reverse.get(current, ())
                if predecessor not in visited
            }
            if not next_frontier:
                break
            visited.update(next_frontier)
            frontier = next_frontier
        return visited

    def _shortest_distances(
        self,
        source: str,
        forward: Adjacency,
        *,
        extra_edge: tuple[str, str] | None = None,
    ) -> dict[str, int]:
        """Return bounded directed shortest-path lengths from one source."""

        distances = {source: 0}
        frontier = {source}
        for distance in range(1, self.config.max_walk_length + 1):
            next_frontier: set[str] = set()
            for node in frontier:
                neighbors = set(forward.get(node, ()))
                if extra_edge is not None and node == extra_edge[0]:
                    neighbors.add(extra_edge[1])
                next_frontier.update(
                    neighbor for neighbor in neighbors if neighbor not in distances
                )
            if not next_frontier:
                break
            for node in next_frontier:
                distances[node] = distance
            frontier = next_frontier
        return distances

    def _connectivity(
        self,
        source: str,
        forward: Adjacency,
        *,
        extra_edge: tuple[str, str] | None = None,
    ) -> dict[str, float]:
        """Calculate K[source, *] with sparse bounded-walk dynamic programming."""

        frontier: dict[str, int] = {source: 1}
        connectivity: defaultdict[str, float] = defaultdict(float)
        discount = 1.0

        for _ in range(self.config.max_walk_length):
            next_frontier: defaultdict[str, int] = defaultdict(int)
            for node, walk_count in frontier.items():
                neighbors = list(forward.get(node, ()))
                if extra_edge is not None and node == extra_edge[0]:
                    neighbors.append(extra_edge[1])
                for neighbor in sorted(neighbors):
                    next_frontier[neighbor] += walk_count

            if not next_frontier:
                break
            for target, walk_count in next_frontier.items():
                connectivity[target] += discount * walk_count
            frontier = dict(next_frontier)
            discount *= self.config.walk_discount

        return dict(connectivity)
