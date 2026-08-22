"""Minimum-total-travel meeting points on the Tool-box grid."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


GRID_MIN = 0
GRID_MAX = 9


def best_meeting_point(
    own_location: Sequence[int], friend_locations: Sequence[Mapping[str, Any]]
) -> list[int]:
    """Return a coordinate-wise median, minimizing total Manhattan travel.

    With an even number of travellers there is an interval of equally optimal
    coordinates on either axis. Selecting its lower end is deterministic and
    remains a valid grid cell.
    """

    own_x, own_y = _validate_point(own_location, "own_location")
    points = [(own_x, own_y)]
    for location in friend_locations:
        if not isinstance(location, Mapping):
            raise ValueError("each friend location must be an object")
        points.append(_validate_point((location.get("x"), location.get("y")), "location"))

    x_values = sorted(point[0] for point in points)
    y_values = sorted(point[1] for point in points)
    lower_median = (len(points) - 1) // 2
    return [x_values[lower_median], y_values[lower_median]]


def _validate_point(point: Sequence[object], label: str) -> tuple[int, int]:
    if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) != 2:
        raise ValueError(f"{label} must be an [x, y] pair")
    x, y = point
    if (
        type(x) is not int
        or type(y) is not int
        or not all(GRID_MIN <= value <= GRID_MAX for value in (x, y))
    ):
        raise ValueError(f"{label} coordinates must be integers from 0 through 9")
    return x, y
