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

    max_walk_length: int = 7
    walk_discount: float = 0.45
    convergence_weight: float = 2.0
    convergence_scale: float = 4.0
    open_route_capacity: float = 4.0
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
        open_deltas: list[float] = []
        closed_deltas: list[float] = []
        extra_edge = (sender, recipient)

        for source in sorted(affected_sources):
            before = self._connectivity(source, forward)
            after = self._connectivity(source, forward, extra_edge=extra_edge)
            for target in sorted(after.keys() | before.keys()):
                delta = self._pair_value(after.get(target, 0.0)) - self._pair_value(
                    before.get(target, 0.0)
                )
                if delta <= 0:
                    continue
                if source == target:
                    closed_deltas.append(delta)
                else:
                    open_deltas.append(delta)

        open_delta = math.fsum(open_deltas)
        closed_delta = math.fsum(closed_deltas)
        direct_edge_baseline = self._pair_value(1.0)
        open_route_delta = max(0.0, open_delta - direct_edge_baseline)
        bounded_open_delta = self.config.open_route_capacity * math.tanh(
            open_route_delta / self.config.open_route_capacity
        )
        raw = max(
            0.0,
            bounded_open_delta + self.config.closed_route_weight * closed_delta,
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
            open_route_delta=open_route_delta,
            closed_route_delta=closed_delta,
        )

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
