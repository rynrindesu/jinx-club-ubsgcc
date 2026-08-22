import heapq
import json

from collections import defaultdict, deque
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


def speed_at(time_value, intervals):
    """
    Return the applicable speed factor.

    Intervals are treated as:
        start_time <= current_time < end_time
    """
    active_factors = [
        factor
        for start, end, factor in intervals
        if start <= time_value < end
    ]

    if not active_factors:
        return Fraction(1)

    # The question does not define overlapping obstructions.
    # This assumes the most restrictive factor applies.
    return min(active_factors)


def traverse_edge(departure, base_duration, intervals):
    """
    Calculate the arrival time after traversing one directed edge.

    base_duration represents the amount of travel work remaining.
    At speed factor f, f units of work are completed per second.
    """
    # A road blocked when we are still at the node cannot be entered.
    if speed_at(departure, intervals) <= 0:
        return None

    remaining = Fraction(base_duration)
    current_time = departure

    while remaining > 0:
        current_speed = speed_at(current_time, intervals)

        # Find the next time at which an obstruction starts or ends.
        future_changes = [
            boundary
            for start, end, _ in intervals
            for boundary in (start, end)
            if boundary > current_time
        ]

        if not future_changes:
            if current_speed <= 0:
                return None

            return current_time + remaining / current_speed

        next_change = min(future_changes)

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
    schedules = defaultdict(list)

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

        # Ignore invalid or empty intervals.
        if interval_end <= interval_start:
            continue

        factor = Fraction(str(obstruction["speed_factor"]))

        schedules[key].append((
            interval_start,
            interval_end,
            factor
        ))

        obstruction_horizon = max(
            obstruction_horizon,
            interval_end
        )

    for intervals in schedules.values():
        intervals.sort(key=lambda interval: (interval[0], interval[1]))

    # Each adjacency entry contains:
    # (neighbor, edge_id, base_duration, directional_intervals)
    graph = {
        tuple(node): []
        for node in input_data["nodes"]
    }

    for edge in input_data["edges"]:
        node1 = tuple(edge["node1"])
        node2 = tuple(edge["node2"])
        edge_id = edge["edge_id"]
        base_duration = Fraction(edge["base_duration_sec"])

        forward_intervals = schedules[
            (edge_id, node1, node2)
        ]

        reverse_intervals = schedules[
            (edge_id, node2, node1)
        ]

        graph.setdefault(node1, []).append((
            node2,
            edge_id,
            base_duration,
            forward_intervals
        ))

        graph.setdefault(node2, []).append((
            node1,
            edge_id,
            base_duration,
            reverse_intervals
        ))

    # Quick check using the road network without considering traffic.
    reachable_nodes = {start}
    search_queue = deque([start])

    while search_queue:
        current = search_queue.popleft()

        for neighbor, _, _, _ in graph.get(current, []):
            if neighbor not in reachable_nodes:
                reachable_nodes.add(neighbor)
                search_queue.append(neighbor)

    if destination not in reachable_nodes:
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

    # Heap entries:
    # (elapsed_time, insertion_order, node)
    frontier = [
        (zero, next(sequence), start)
    ]

    final_state = None

    while frontier:
        elapsed, _, node = heapq.heappop(frontier)
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
            intervals
        ) in graph.get(node, []):

            arrival = traverse_edge(
                elapsed,
                base_duration,
                intervals
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
                (arrival, next(sequence), neighbor)
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
       
