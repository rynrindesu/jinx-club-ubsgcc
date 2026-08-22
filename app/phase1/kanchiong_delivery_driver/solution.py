import heapq
import json

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from fractions import Fraction
from itertools import count
from typing import Any, Mapping


def parse_iso(value):
    """Convert an ISO-8601 string into a datetime."""
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"

    return datetime.fromisoformat(value)


def exact_seconds_between(later, earlier):
    """Return an exact number of seconds between two datetimes."""
    difference = later - earlier

    whole_seconds = difference.days * 86400 + difference.seconds
    fractional_seconds = Fraction(
        difference.microseconds,
        1_000_000
    )

    return Fraction(whole_seconds) + fractional_seconds


@dataclass(frozen=True)
class TrafficSchedule:
    """Piecewise-constant speeds for one directed edge.

    ``speeds[i]`` applies from ``boundaries[i - 1]`` (or negative infinity
    for ``i == 0``) up to ``boundaries[i]`` (or positive infinity for the
    final entry).  This lets a traversal jump directly to its next traffic
    change rather than repeatedly scanning every obstruction.
    """

    boundaries: tuple[Fraction, ...]
    speeds: tuple[Fraction, ...]

    def segment_at(self, time_value: Fraction) -> tuple[Fraction, Fraction | None]:
        index = bisect_right(self.boundaries, time_value)
        next_change = (
            self.boundaries[index]
            if index < len(self.boundaries)
            else None
        )
        return self.speeds[index], next_change

    @property
    def maximum_speed(self) -> Fraction:
        return max(self.speeds)


EMPTY_SCHEDULE = TrafficSchedule((), (Fraction(1),))


def build_traffic_schedule(
    intervals: list[tuple[Fraction, Fraction, Fraction]],
) -> TrafficSchedule:
    """Combine potentially overlapping obstructions into speed segments.

    The challenge does not specify overlap semantics.  Like the original
    implementation, overlapping obstructions use the most restrictive
    (lowest) speed factor.
    """

    if not intervals:
        return EMPTY_SCHEDULE

    events: dict[Fraction, list[tuple[int, Fraction]]] = defaultdict(list)
    for start, end, factor in intervals:
        events[start].append((1, factor))
        events[end].append((-1, factor))

    active_counts: dict[Fraction, int] = defaultdict(int)
    active_speeds: list[Fraction] = []
    boundaries = sorted(events)
    speeds = [Fraction(1)]

    for boundary in boundaries:
        for change, factor in events[boundary]:
            if change > 0:
                active_counts[factor] += 1
                heapq.heappush(active_speeds, factor)
            else:
                active_counts[factor] -= 1

        while active_speeds and active_counts[active_speeds[0]] <= 0:
            heapq.heappop(active_speeds)

        speeds.append(active_speeds[0] if active_speeds else Fraction(1))

    return TrafficSchedule(tuple(boundaries), tuple(speeds))


def traverse_edge(
    departure: Fraction,
    base_duration: Fraction,
    schedule: TrafficSchedule,
) -> Fraction | None:
    """
    Calculate the arrival time after traversing one directed edge.

    base_duration represents the amount of travel work remaining.
    At speed factor f, f units of work are completed per second.
    """
    # A road blocked when we are still at the node cannot be entered.
    departure_speed, _ = schedule.segment_at(departure)
    if departure_speed <= 0:
        return None

    remaining = base_duration
    current_time = departure

    while remaining > 0:
        current_speed, next_change = schedule.segment_at(current_time)

        if next_change is None:
            if current_speed <= 0:
                return None

            return current_time + remaining / current_speed

        # If a closure began after entering the edge, no progress is
        # made until that closure changes or ends.
        if current_speed <= 0:
            current_time = next_change
            continue

        available_time = next_change - current_time
        possible_progress = available_time * current_speed

        if remaining <= possible_progress:
            return current_time + remaining / current_speed

        remaining -= possible_progress
        current_time = next_change

    return current_time


def optimistic_distances(
    graph: Mapping[
        tuple[int, int],
        list[tuple[tuple[int, int], str, Fraction, TrafficSchedule]],
    ],
    destination: tuple[int, int],
) -> dict[tuple[int, int], Fraction]:
    """Return admissible remaining-time lower bounds for A*.

    Each road is assigned its fastest speed seen in its complete schedule.
    That can only underestimate real travel time, so it is safe for pruning
    even though a real driver cannot wait for a favourable traffic window.
    """

    reverse_graph: dict[tuple[int, int], list[tuple[tuple[int, int], Fraction]]]
    reverse_graph = defaultdict(list)
    for node, edges in graph.items():
        for neighbour, _, base_duration, schedule in edges:
            lower_bound = base_duration / schedule.maximum_speed
            reverse_graph[neighbour].append((node, lower_bound))

    distances = {destination: Fraction(0)}
    sequence = count()
    frontier = [(Fraction(0), next(sequence), destination)]

    while frontier:
        distance, _, node = heapq.heappop(frontier)
        if distances.get(node) != distance:
            continue

        for predecessor, edge_cost in reverse_graph.get(node, []):
            candidate = distance + edge_cost
            previous_distance = distances.get(predecessor)
            if previous_distance is not None and candidate >= previous_distance:
                continue
            distances[predecessor] = candidate
            heapq.heappush(frontier, (candidate, next(sequence), predecessor))

    return distances


def unreachable_result():
    return {
        "total_duration_sec": None,
        "arrival_time": None,
        "path": []
    }


def format_iso(value):
    result = value.isoformat()

    if result.endswith("+00:00"):
        result = result[:-6] + "Z"

    return result


def solve_case(input_data: Mapping[str, Any]) -> dict[str, Any]:
    """Return the required result for one delivery-driver case."""

    start = tuple(input_data["start_coordinate"])
    destination = tuple(input_data["end_coordinate"])

    if start == destination:
        return {
            "total_duration_sec": 0,
            "arrival_time": input_data["start_time"],
            "path": []
        }

    starting_datetime = parse_iso(input_data["start_time"])

    # Key:
    #     (edge_id, from_coordinate, to_coordinate)
    #
    # Value:
    #     list of (start_elapsed, end_elapsed, speed_factor)
    raw_schedules = defaultdict(list)

    # After this time every obstruction has ended.
    obstruction_horizon = Fraction(0)

    for obstruction in input_data["obstructions"]:
        direction = obstruction["edge"]

        key = (
            obstruction["edge_id"],
            tuple(direction["from"]),
            tuple(direction["to"])
        )

        interval_start = exact_seconds_between(
            parse_iso(obstruction["start_time"]),
            starting_datetime
        )

        interval_end = exact_seconds_between(
            parse_iso(obstruction["end_time"]),
            starting_datetime
        )

        # Intervals that have already ended cannot affect a route starting at
        # elapsed time zero.  Clip an already-active interval to departure.
        if interval_end <= interval_start or interval_end <= 0:
            continue
        interval_start = max(interval_start, Fraction(0))

        factor = Fraction(str(obstruction["speed_factor"]))

        raw_schedules[key].append((
            interval_start,
            interval_end,
            factor
        ))

        obstruction_horizon = max(
            obstruction_horizon,
            interval_end
        )

    schedules = {
        key: build_traffic_schedule(intervals)
        for key, intervals in raw_schedules.items()
    }

    # Each adjacency entry contains:
    # (neighbor, edge_id, base_duration, directional_traffic_schedule)
    graph = {
        tuple(node): []
        for node in input_data["nodes"]
    }

    for edge in input_data["edges"]:
        node1 = tuple(edge["node1"])
        node2 = tuple(edge["node2"])
        edge_id = edge["edge_id"]
        base_duration = Fraction(edge["base_duration_sec"])

        forward_schedule = schedules.get(
            (edge_id, node1, node2),
            EMPTY_SCHEDULE,
        )

        reverse_schedule = schedules.get(
            (edge_id, node2, node1),
            EMPTY_SCHEDULE,
        )

        graph.setdefault(node1, []).append((
            node2,
            edge_id,
            base_duration,
            forward_schedule
        ))

        graph.setdefault(node2, []).append((
            node1,
            edge_id,
            base_duration,
            reverse_schedule
        ))

    # These lower bounds both prioritize promising states and identify nodes
    # that cannot reach the destination even before traffic is considered.
    remaining_lower_bound = optimistic_distances(graph, destination)
    if start not in remaining_lower_bound:
        return unreachable_result()

    zero = Fraction(0)
    starting_state = (start, zero)

    # Before the final obstruction ends, multiple arrival times at the
    # same node may all be useful.
    discovered_before_horizon = set()

    # After all obstructions end, only the earliest arrival at each
    # node is useful because the graph is static again.
    best_after_horizon = {}

    if obstruction_horizon > zero:
        discovered_before_horizon.add(starting_state)
    else:
        best_after_horizon[start] = zero

    parent = {}

    sequence = count()

    # Heap entries are ordered by A* priority, then elapsed time:
    # (elapsed_time + optimistic_remaining, elapsed_time, insertion_order, node)
    frontier = [
        (
            remaining_lower_bound[start],
            zero,
            next(sequence),
            start,
        )
    ]

    final_state = None

    while frontier:
        _, elapsed, _, node = heapq.heappop(frontier)
        state = (node, elapsed)

        # Ignore an outdated post-obstruction state.
        if elapsed >= obstruction_horizon:
            if best_after_horizon.get(node) != elapsed:
                continue

        if node == destination:
            final_state = state
            break

        for (
            neighbor,
            edge_id,
            base_duration,
            schedule
        ) in graph.get(node, []):

            # Reaching this node cannot lead to the destination.
            if neighbor not in remaining_lower_bound:
                continue

            arrival = traverse_edge(
                elapsed,
                base_duration,
                schedule
            )

            # This directed road cannot currently be entered.
            if arrival is None:
                continue

            next_state = (neighbor, arrival)

            if arrival < obstruction_horizon:
                if next_state in discovered_before_horizon:
                    continue

                discovered_before_horizon.add(next_state)

            else:
                previous_best = best_after_horizon.get(neighbor)

                if (
                    previous_best is not None
                    and previous_best <= arrival
                ):
                    continue

                best_after_horizon[neighbor] = arrival

            parent[next_state] = (state, edge_id)

            heapq.heappush(
                frontier,
                (
                    arrival + remaining_lower_bound[neighbor],
                    arrival,
                    next(sequence),
                    neighbor,
                )
            )

    if final_state is None:
        return unreachable_result()

    # Reconstruct the path backwards.
    path = []
    current_state = final_state

    while current_state != starting_state:
        previous_state, edge_id = parent[current_state]
        path.append(edge_id)
        current_state = previous_state

    path.reverse()

    elapsed = final_state[1]

    if elapsed.denominator == 1:
        total_duration = int(elapsed)
    else:
        total_duration = float(elapsed)

    elapsed_microseconds = int(
        round(elapsed * 1_000_000)
    )

    arrival_datetime = starting_datetime + timedelta(
        microseconds=elapsed_microseconds
    )

    return {
        "total_duration_sec": total_duration,
        "arrival_time": format_iso(arrival_datetime),
        "path": path
    }


def solve(data: str) -> str:
    """Preserve the original single-case JSON-string interface."""

    return json.dumps(solve_case(json.loads(data)))
       
