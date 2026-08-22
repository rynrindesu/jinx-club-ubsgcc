"""Phase 2 tools registered on the shared Tool-box MCP server."""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import lru_cache

import httpx
from fastmcp import FastMCP

from .routing import next_hop
from .study import study_passages


DEFAULT_CHALLENGE_URL = "https://tool-box-2591eaa24fa3.herokuapp.com"


def register_tools(mcp: FastMCP) -> None:
    """Register the Phase 2 tools alongside the original Tool-box tools."""

    @mcp.tool(
        name="retrieve",
        description=(
            "Return the most relevant passages from the assigned study materials "
            "for a question. Passages, not an answer, are returned and their total "
            "o200k_base token count is at most 900. For school-trip questions, use "
            "these passages first to identify the destination before navigating."
        ),
    )
    def retrieve(query: str) -> list[str]:
        return study_passages(query, _study_materials_url())

    @mcp.tool(
        name="next_route_node",
        description=(
            "Return the adjacent next node on the cheapest route to a destination. "
            "Map cost is edge weight plus the toll of every node entered. Supply "
            "hops_remaining whenever the question gives an allowance; it counts this "
            "move. Also supply previously visited nodes to prevent a forbidden revisit."
        ),
    )
    def next_route_node(
        map_id: str,
        current_node: str,
        destination: str,
        hops_remaining: int | None = None,
        visited_nodes: Sequence[str] = (),
    ) -> str:
        return next_hop(
            _fetch_graph(map_id),
            current_node,
            destination,
            hops_remaining,
            visited_nodes,
        )


def _study_materials_url() -> str:
    base_url = os.getenv("TOOLBOX_CHALLENGE_URL", DEFAULT_CHALLENGE_URL).rstrip("/")
    return os.getenv("TOOLBOX_STUDY_MATERIALS_URL", f"{base_url}/study-materials")


def _fetch_graph(map_id: str) -> dict[str, object]:
    if not map_id.strip():
        raise ValueError("map_id must not be empty")
    base_url = os.getenv("TOOLBOX_CHALLENGE_URL", DEFAULT_CHALLENGE_URL).rstrip("/")
    graph_url = os.getenv("TOOLBOX_GRAPH_URL", f"{base_url}/graph")
    return _cached_graph(graph_url, map_id)


@lru_cache(maxsize=128)
def _cached_graph(graph_url: str, map_id: str) -> dict[str, object]:
    """Fetch each immutable opaque map ID once per process."""

    response = httpx.get(
        graph_url,
        params={"map_id": map_id},
        timeout=15.0,
        follow_redirects=True,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("graph response must be an object")
    return payload
