"""Joint meeting-point and venue optimization for Tool-box Phase 3."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .locations import best_meeting_point


def best_outing_plan(
    meeting_window: str,
    own_location: Sequence[int],
    friend_locations: Sequence[Mapping[str, Any]],
    open_venues: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Choose a venue and point that minimize the whole outing's travel.

    For a fixed venue, minimizing everyone else's travel to the point plus the
    point-to-venue leg is equivalent to taking the coordinate-wise median of
    the android, friends, and that venue treated as one extra point.
    """

    start_time, end_time = _parse_window(meeting_window)
    if not open_venues:
        raise ValueError("no venue is open when the meeting ends")

    candidates: list[tuple[int, int, int, str, list[int]]] = []
    for venue in open_venues:
        name = venue.get("name")
        if not isinstance(name, str):
            raise ValueError("each venue must have a name")
        point = best_meeting_point(own_location, [*friend_locations, venue])
        cost = _outing_cost(own_location, friend_locations, point, venue)
        # Coordinates and name make equally good answers reproducible.
        candidates.append((cost, point[0], point[1], name, point))

    _, _, _, venue_name, point = min(candidates)
    return {
        "meeting_window": [start_time, end_time],
        "meeting_point": point,
        "place_to_eat": venue_name,
    }


def _outing_cost(
    own_location: Sequence[int],
    friend_locations: Sequence[Mapping[str, Any]],
    meeting_point: Sequence[int],
    venue: Mapping[str, Any],
) -> int:
    travellers: list[Sequence[object]] = [own_location]
    travellers.extend((location.get("x"), location.get("y")) for location in friend_locations)
    return sum(_distance(traveller, meeting_point) for traveller in travellers) + _distance(
        meeting_point, (venue.get("x"), venue.get("y"))
    )


def _distance(left: Sequence[object], right: Sequence[object]) -> int:
    if len(left) != 2 or len(right) != 2:
        raise ValueError("points must be [x, y] pairs")
    left_x, left_y = left
    right_x, right_y = right
    if not all(type(value) is int for value in (left_x, left_y, right_x, right_y)):
        raise ValueError("point coordinates must be integers")
    return abs(left_x - right_x) + abs(left_y - right_y)


def _parse_window(meeting_window: str) -> tuple[str, str]:
    parts = [part.strip() for part in meeting_window.split(",")]
    if len(parts) != 2 or any(len(part) != 5 for part in parts):
        raise ValueError("meeting window must be 'HH:MM, HH:MM'")
    return parts[0], parts[1]
