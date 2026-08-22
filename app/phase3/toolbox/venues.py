"""Venue availability lookup for Tool-box Phase 3."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


WEEKDAYS = frozenset(
    {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
)


def normalise_day(day: str) -> str:
    """Validate and canonicalise a weekday supplied by the MCP client."""

    normalised = day.strip().capitalize()
    if normalised not in WEEKDAYS:
        raise ValueError("day must be a weekday from Monday through Sunday")
    return normalised


def validate_hour(time: str) -> str:
    """Validate a zero-padded, on-the-hour challenge time."""

    if len(time) != 5 or time[2] != ":" or not time.replace(":", "").isdigit():
        raise ValueError("time must be a zero-padded HH:MM string")
    hour, minute = (int(part) for part in time.split(":"))
    if not 8 <= hour <= 23 or minute != 0:
        raise ValueError("time must be on the hour between 08:00 and 23:00")
    return time


def open_venue_names(payload: Mapping[str, Any], time: str) -> str:
    """Return all venue names whose half-open availability contains ``time``.

    The challenge defines availability as [start, end): a venue opening at the
    requested hour is usable, while one closing at that hour is not.
    """

    raw_venues = payload.get("venues")
    if not isinstance(raw_venues, Sequence) or isinstance(raw_venues, (str, bytes)):
        raise ValueError("venues response must contain a venues array")

    matches: list[str] = []
    for venue in raw_venues:
        if not isinstance(venue, Mapping):
            raise ValueError("each venue must be an object")
        name = venue.get("name")
        intervals = venue.get("available")
        if not isinstance(name, str) or not isinstance(intervals, Sequence) or isinstance(
            intervals, (str, bytes)
        ):
            raise ValueError("each venue must contain a name and availability array")
        if any(_contains_hour(interval, time) for interval in intervals):
            matches.append(name)
    return ", ".join(matches)


def _contains_hour(interval: object, time: str) -> bool:
    if (
        not isinstance(interval, Sequence)
        or isinstance(interval, (str, bytes))
        or len(interval) != 2
        or not all(isinstance(value, str) for value in interval)
    ):
        raise ValueError("each availability interval must be a [start, end] pair")
    start, end = interval
    return start <= time < end
