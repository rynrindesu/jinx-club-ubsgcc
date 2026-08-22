"""Phase 3 venue tools registered on the shared Tool-box MCP server."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import httpx
from fastmcp import FastMCP

from .venues import normalise_day, open_venue_names, validate_hour


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
