"""Phase 3 venue tools registered on the shared Tool-box MCP server."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any
from urllib.parse import quote

import httpx
from fastmcp import FastMCP

from .venues import normalise_day, open_venue_names, validate_hour
from .meetings import find_meeting_window


DEFAULT_CHALLENGE_URL = "https://tool-box-2591eaa24fa3.herokuapp.com"


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
    return os.getenv("TOOLBOX_INBOX_URL", f"{base_url}/inbox")


def _schedule_url() -> str:
    base_url = os.getenv("TOOLBOX_CHALLENGE_URL", DEFAULT_CHALLENGE_URL).rstrip("/")
    return os.getenv("TOOLBOX_SCHEDULE_URL", f"{base_url}/schedule")


def _fetch_inbox() -> str:
    response = httpx.get(_inbox_url(), timeout=15.0, follow_redirects=True)
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError:
        return response.text
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("inbox"), str):
        return payload["inbox"]
    raise ValueError("inbox response must be text or an object with an inbox field")


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
