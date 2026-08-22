"""Cost-aware routing for the Tool-box city maps."""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence
from math import inf
from typing import Any


Graph = Mapping[str, Any]


def next_hop(
    graph: Graph,
    current_node: str,
    destination: str,
    hops_remaining: int | None = None,
    visited_nodes: Sequence[str] = (),
) -> str:
    """Choose the lowest-cost legal next node for a journey.

    Entering a node pays both its incoming edge weight and that node's toll.
    When a hop allowance applies, the selected route is restricted to that
    many remaining edges.  Earlier nodes supplied in ``visited_nodes`` are
    excluded so the caller cannot create a loop.
    """

    adjacency, tolls = _normalise_graph(graph)
    if current_node not in adjacency:
        raise ValueError(f"unknown current node: {current_node}")
    if destination not in adjacency:
        raise ValueError(f"unknown destination: {destination}")
    if current_node == destination:
        raise ValueError("already at the destination")
    if hops_remaining is not None and hops_remaining < 1:
        raise ValueError("at least one hop is required to reach the destination")

    # The node currently occupied is already visited too.  Excluding it stops
    # a self-loop from being returned as the next hop.
    forbidden = set(visited_nodes)
    forbidden.add(current_node)
    forbidden.discard(destination)
    if hops_remaining is None:
        return _unconstrained_next_hop(
            adjacency, tolls, current_node, destination, forbidden
        )
    # A legal route cannot revisit a node, so it never needs more than V - 1
    # edges.  Clamping avoids needless dynamic-programming passes when the
    # supplied allowance is larger than any possible legal path.
    effective_hops = min(hops_remaining, len(adjacency) - 1)
    if effective_hops < 1:
        raise ValueError("destination cannot be reached within the hop allowance")
    return _hop_limited_next_hop(
        adjacency,
        tolls,
        current_node,
        destination,
        effective_hops,
        forbidden,
    )


def _normalise_graph(
    graph: Graph,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    try:
        raw_adjacency = graph["adjacency"]
        raw_tolls = graph["tolls"]
    except KeyError as error:
        raise ValueError("graph must include adjacency and tolls") from error

    if not isinstance(raw_adjacency, Mapping) or not isinstance(raw_tolls, Mapping):
        raise ValueError("graph adjacency and tolls must be objects")

    adjacency: dict[str, dict[str, float]] = {}
    for source, raw_edges in raw_adjacency.items():
        if not isinstance(source, str) or not isinstance(raw_edges, Mapping):
            raise ValueError("graph adjacency must map node names to edge maps")
        adjacency[source] = {}
        for target, weight in raw_edges.items():
            if not isinstance(target, str):
                raise ValueError("graph node names must be strings")
            numeric_weight = float(weight)
            if numeric_weight < 0:
                raise ValueError("edge weights must be non-negative")
            adjacency[source][target] = numeric_weight

    tolls = {node: float(value) for node, value in raw_tolls.items()}
    if set(adjacency) != set(tolls):
        raise ValueError("tolls must list every graph node")
    if any(toll < 0 for toll in tolls.values()):
        raise ValueError("tolls must be non-negative")
    if any(target not in adjacency for edges in adjacency.values() for target in edges):
        raise ValueError("every edge target must be a graph node")
    return adjacency, tolls


def _unconstrained_next_hop(
    adjacency: Mapping[str, Mapping[str, float]],
    tolls: Mapping[str, float],
    current_node: str,
    destination: str,
    forbidden: set[str],
) -> str:
    """Run Dijkstra forward and retain the first hop for the winning path."""

    queue: list[tuple[float, str, str]] = []
    best_cost: dict[str, float] = {current_node: 0.0}
    for neighbour, weight in adjacency[current_node].items():
        if neighbour in forbidden:
            continue
        cost = weight + tolls[neighbour]
        best_cost[neighbour] = cost
        heapq.heappush(queue, (cost, neighbour, neighbour))

    while queue:
        cost, node, first_hop = heapq.heappop(queue)
        if cost != best_cost.get(node):
            continue
        if node == destination:
            return first_hop
        for neighbour, weight in adjacency[node].items():
            if neighbour in forbidden:
                continue
            candidate_cost = cost + weight + tolls[neighbour]
            if candidate_cost < best_cost.get(neighbour, inf):
                best_cost[neighbour] = candidate_cost
                heapq.heappush(queue, (candidate_cost, neighbour, first_hop))

    raise ValueError("destination is unreachable without revisiting a node")


def _hop_limited_next_hop(
    adjacency: Mapping[str, Mapping[str, float]],
    tolls: Mapping[str, float],
    current_node: str,
    destination: str,
    hops_remaining: int,
    forbidden: set[str],
) -> str:
    """Find the best route using at most ``hops_remaining`` edges.

    ``costs`` holds the best (cost, number of hops) pair for each node for the
    previous layer.  Retaining only this layer makes the dynamic program use
    O(V) memory instead of O(hops_remaining * V).  Fewer hops break equal-cost
    ties, which removes gratuitous zero-cost cycles from the chosen route.
    """

    costs: dict[str, tuple[float, int]] = {destination: (0.0, 0)}
    # The selected first hop consumes one edge, so calculate only the
    # remaining H - 1 layers.
    for _ in range(hops_remaining - 1):
        next_costs = dict(costs)  # Arriving early is always allowed.
        for source, edges in adjacency.items():
            if source in forbidden:
                continue
            for target, weight in edges.items():
                if target in forbidden or target not in costs:
                    continue
                remaining_cost, remaining_hops = costs[target]
                candidate = (weight + tolls[target] + remaining_cost, remaining_hops + 1)
                best = next_costs.get(source)
                if best is None or candidate < best:
                    next_costs[source] = candidate
        costs = next_costs

    candidates = [
        (
            weight + tolls[target] + costs[target][0],
            costs[target][1] + 1,
            target,
        )
        for target, weight in adjacency[current_node].items()
        if target not in forbidden and target in costs
    ]
    if not candidates:
        raise ValueError("destination cannot be reached within the hop allowance")
    return min(candidates)[2]
