"""Common-meeting search for Tool-box Phase 3."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .venues import normalise_day, validate_hour


DAY_START = 8 * 60
DAY_END = 23 * 60
Response = Literal["ACCEPTED", "DECLINED", "TENTATIVE"]
WHEN_LINE = re.compile(
    r"^When:\s*(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+"
    r"(\d{2}:\d{2})\s*[-–]\s*(\d{2}:\d{2})\s*$",
    re.MULTILINE,
)
RESPONSE_LINE = re.compile(r"^Response:\s*(ACCEPTED|DECLINED|TENTATIVE)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Interval:
    start: int
    end: int

    def overlaps(self, other: "Interval") -> bool:
        """Return whether two half-open intervals overlap."""

        return self.start < other.end and other.start < self.end


def find_meeting_window(
    day: str,
    people: Sequence[str],
    earliest_time: str,
    latest_time: str,
    duration_minutes: int,
    inbox: str,
    schedules: Mapping[str, Mapping[str, Any]],
) -> str:
    """Find the earliest clean window, then the earliest tentative fallback."""

    canonical_day = normalise_day(day)
    range_start = _to_minutes(validate_hour(earliest_time))
    range_end = _to_minutes(validate_hour(latest_time))
    _validate_request(people, range_start, range_end, duration_minutes)

    hard_blocks, soft_blocks = own_calendar_blocks(inbox, canonical_day)
    for person in people:
        if not isinstance(person, str) or not person.strip():
            raise ValueError("each person must be a non-empty string")
        schedule = schedules.get(person)
        if schedule is None:
            raise ValueError(f"missing schedule for {person}")
        hard_blocks.extend(friend_busy_blocks(schedule))

    tentative_candidate: Interval | None = None
    for start in range(range_start, range_end - duration_minutes + 1, 60):
        candidate = Interval(start, start + duration_minutes)
        if any(candidate.overlaps(block) for block in hard_blocks):
            continue
        if not any(candidate.overlaps(block) for block in soft_blocks):
            return _format_window(candidate)
        if tentative_candidate is None:
            tentative_candidate = candidate

    if tentative_candidate is not None:
        return _format_window(tentative_candidate)
    raise ValueError("no meeting window is available in the requested range")


def own_calendar_blocks(inbox: str, day: str) -> tuple[list[Interval], list[Interval]]:
    """Read only structured Response/When lines, ignoring message prose."""

    if not isinstance(inbox, str):
        raise ValueError("inbox must be text")
    hard_blocks: list[Interval] = []
    soft_blocks: list[Interval] = []
    responses = list(RESPONSE_LINE.finditer(inbox))
    for index, response_match in enumerate(responses):
        # A message's structured When line occurs after its Response line and
        # before the next message response.  Prose outside these lines is ignored.
        next_response = responses[index + 1].start() if index + 1 < len(responses) else len(inbox)
        when_match = WHEN_LINE.search(inbox, response_match.end(), next_response)
        if when_match is None or when_match.group(1) != day:
            continue
        interval = _interval_from_times(when_match.group(2), when_match.group(3))
        response: Response = response_match.group(1)  # type: ignore[assignment]
        if response == "ACCEPTED":
            hard_blocks.append(interval)
        elif response == "TENTATIVE":
            soft_blocks.append(interval)
    return hard_blocks, soft_blocks


def friend_busy_blocks(schedule: Mapping[str, Any]) -> list[Interval]:
    """Validate one friend schedule and return its busy ranges."""

    busy = schedule.get("busy")
    if not isinstance(busy, Sequence) or isinstance(busy, (str, bytes)):
        raise ValueError("schedule response must contain a busy array")
    blocks: list[Interval] = []
    for entry in busy:
        if (
            not isinstance(entry, Sequence)
            or isinstance(entry, (str, bytes))
            or len(entry) != 2
            or not all(isinstance(value, str) for value in entry)
        ):
            raise ValueError("each busy entry must be a [start, end] pair")
        blocks.append(_interval_from_times(entry[0], entry[1]))
    return blocks


def _validate_request(
    people: Sequence[str], range_start: int, range_end: int, duration_minutes: int
) -> None:
    if isinstance(people, (str, bytes)) or not people:
        raise ValueError("at least one friend is required")
    if range_start >= range_end:
        raise ValueError("earliest_time must be before latest_time")
    if duration_minutes <= 0 or duration_minutes % 60:
        raise ValueError("duration_minutes must be a positive whole number of hours")
    if range_start + duration_minutes > range_end:
        raise ValueError("duration does not fit in the requested range")


def _interval_from_times(start: str, end: str) -> Interval:
    start_minutes = _to_minutes(validate_hour(start))
    end_minutes = _to_minutes(validate_hour(end))
    if start_minutes >= end_minutes:
        raise ValueError("an event must end after it starts")
    return Interval(start_minutes, end_minutes)


def _to_minutes(time: str) -> int:
    hour, minute = (int(part) for part in time.split(":"))
    return hour * 60 + minute


def _format_window(window: Interval) -> str:
    return f"{_format_time(window.start)}, {_format_time(window.end)}"


def _format_time(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"
