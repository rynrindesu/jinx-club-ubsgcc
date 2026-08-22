"""Phase 3 venue tools registered on the shared Tool-box MCP server."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any
from urllib.parse import quote

import httpx
from fastmcp import FastMCP

from .venues import normalise_day, open_venue_names, open_venues, validate_hour
from .meetings import find_meeting_window
from .locations import best_meeting_point
from .outings import best_outing_plan


DEFAULT_CHALLENGE_URL = "https://tool-box-2591eaa24fa3.herokuapp.com"
mcp = FastMCP("Tool-box Working Life")


def register_tools(mcp: FastMCP) -> None:
    """Register the Stage 3 Working Life venue availability tool."""

    @mcp.tool(
        name="find_open_venues",
        description=(
            "Return every venue open on a weekday at an exact hour, as the required "
            "comma-separated list of names. Availability includes its start time and "
            "excludes its end time; give the weekday and zero-padded HH:MM from the "
            "question."
        ),
    )
    def find_open_venues(day: str, time: str) -> str:
        canonical_day = normalise_day(day)
        requested_time = validate_hour(time)
        return open_venue_names(_fetch_venues(canonical_day), requested_time)

    @mcp.tool(
        name="find_meeting_time",
        description=(
            "Find the best common meeting window and return 'HH:MM, HH:MM'. Supply "
            "the weekday, every friend's lowercase name, requested start and end, and "
            "duration in minutes. The result is the earliest window with no conflicts "
            "at all; a tentative android calendar event is used only if no clean "
            "window exists anywhere in the requested range."
        ),
    )
    def find_meeting_time(
        day: str,
        people: list[str],
        earliest_time: str,
        latest_time: str,
        duration_minutes: int,
    ) -> str:
        canonical_day = normalise_day(day)
        inbox = _fetch_inbox()
        schedules = {
            person: _fetch_schedule(person, canonical_day) for person in people
        }
        return find_meeting_window(
            canonical_day,
            people,
            earliest_time,
            latest_time,
            duration_minutes,
            inbox,
            schedules,
        )

    @mcp.tool(
        name="find_meeting_point",
        description=(
            "Return the [x, y] grid cell that minimizes total Manhattan travel for "
            "the android and every named friend on a weekday. Supply the android's "
            "starting [x, y] position and every friend's name. All travellers count."
        ),
    )
    def find_meeting_point(
        day: str, people: list[str], own_location: list[int]
    ) -> list[int]:
        canonical_day = normalise_day(day)
        if not people:
            raise ValueError("at least one friend is required")
        if len(set(people)) != len(people):
            raise ValueError("each friend must be listed only once")
        locations = [_fetch_location(person, canonical_day) for person in people]
        return best_meeting_point(own_location, locations)

    @mcp.tool(
        name="plan_outing",
        description=(
            "Plan the complete outing: return the required common meeting window, "
            "a meeting point, and a place to eat after the meeting. It minimizes all "
            "travel to the meeting point plus the one trip from that point to a venue "
            "open exactly when the meeting ends. Supply weekday, every friend's name, "
            "the android's [x, y], requested bounds, and duration in minutes."
        ),
    )
    def plan_outing(
        day: str,
        people: list[str],
        own_location: list[int],
        earliest_time: str,
        latest_time: str,
        duration_minutes: int,
    ) -> dict[str, object]:
        canonical_day = normalise_day(day)
        if not people:
            raise ValueError("at least one friend is required")
        if len(set(people)) != len(people):
            raise ValueError("each friend must be listed only once")
        inbox = _fetch_inbox()
        schedules = {person: _fetch_schedule(person, canonical_day) for person in people}
        meeting_window = find_meeting_window(
            canonical_day,
            people,
            earliest_time,
            latest_time,
            duration_minutes,
            inbox,
            schedules,
        )
        meeting_end = meeting_window.split(",", maxsplit=1)[1].strip()
        venues = open_venues(_fetch_venues(canonical_day), meeting_end)
        locations = [_fetch_location(person, canonical_day) for person in people]
        return best_outing_plan(meeting_window, own_location, locations, venues)


def _venues_url() -> str:
    base_url = os.getenv("TOOLBOX_CHALLENGE_URL", DEFAULT_CHALLENGE_URL).rstrip("/")
    return os.getenv("TOOLBOX_VENUES_URL", f"{base_url}/venues")


def _fetch_venues(day: str) -> dict[str, Any]:
    return _cached_venues(_venues_url(), day)


@lru_cache(maxsize=7)
def _cached_venues(venues_url: str, day: str) -> dict[str, Any]:
    """Fetch each day's fixed venue schedule at most once per process."""

    response = httpx.get(
        f"{venues_url.rstrip('/')}/{day}", timeout=15.0, follow_redirects=True
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("venues response must be an object")
    return payload


def _inbox_url() -> str:
    base_url = os.getenv("TOOLBOX_CHALLENGE_URL", DEFAULT_CHALLENGE_URL).rstrip("/")
    return os.getenv("TOOLBOX_INBOX_URL", f"{base_url}/emails")


def _schedule_url() -> str:
    base_url = os.getenv("TOOLBOX_CHALLENGE_URL", DEFAULT_CHALLENGE_URL).rstrip("/")
    return os.getenv("TOOLBOX_SCHEDULE_URL", f"{base_url}/schedule")


def _location_url() -> str:
    base_url = os.getenv("TOOLBOX_CHALLENGE_URL", DEFAULT_CHALLENGE_URL).rstrip("/")
    return os.getenv("TOOLBOX_LOCATION_URL", f"{base_url}/location")


def _fetch_inbox() -> str:
    response = httpx.get(_inbox_url(), timeout=15.0, follow_redirects=True)
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError:
        return response.text
    return _inbox_text(payload)


def _inbox_text(payload: Any) -> str:
    """Extract mail bodies from the challenge's ``{\"emails\": [...]}`` payload."""

    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("inbox"), str):
        return payload["inbox"]
    if isinstance(payload, dict) and isinstance(payload.get("emails"), list):
        bodies = []
        for email in payload["emails"]:
            if not isinstance(email, dict) or not isinstance(email.get("body"), str):
                raise ValueError("each email must be an object with a text body")
            bodies.append(email["body"])
        return "\n\n".join(bodies)
    raise ValueError(
        "inbox response must be text, an object with an inbox field, or an emails array"
    )


def _fetch_schedule(person: str, day: str) -> dict[str, Any]:
    if not isinstance(person, str) or not person.strip():
        raise ValueError("each person must be a non-empty string")
    response = httpx.get(
        f"{_schedule_url().rstrip('/')}/{quote(person.strip(), safe='')}/{day}",
        timeout=15.0,
        follow_redirects=True,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("schedule response must be an object")
    return payload


def _fetch_location(person: str, day: str) -> dict[str, Any]:
    if not isinstance(person, str) or not person.strip():
        raise ValueError("each person must be a non-empty string")
    response = httpx.get(
        f"{_location_url().rstrip('/')}/{quote(person.strip(), safe='')}/{day}",
        timeout=15.0,
        follow_redirects=True,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("location response must be an object")
    return payload


# This is intentionally a dedicated server.  Earlier phases have their own
# MCP implementations, but Phase 3 must not expose their unrelated tools.
register_tools(mcp)
