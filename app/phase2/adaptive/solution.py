import base64
import json
import math
from typing import Any


PRIORITY_MAP = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
}


def adapt_input(adapt_input: dict[str, Any]) -> dict[str, Any]:
    """Convert the Version 1 request model to the Version 2 model."""

    return {
        "id": adapt_input["user"]["id"],
        "name": adapt_input["user"]["fullName"],
        "action": adapt_input["action"].lower(),
        "priority": PRIORITY_MAP[adapt_input["metadata"]["priority"]],
    }


def calculate_slo(
    heartbeats: list[dict[str, Any]], slo_query: dict[str, Any]
) -> dict[str, float | int]:
    """Report SLOs for any queried service in the requested time window."""

    requested_service = slo_query["service"]
    since_timestamp = slo_query["since"]

    matching_heartbeats = [
        heartbeat
        for heartbeat in heartbeats
        if heartbeat["service"] == requested_service
        and heartbeat["timestamp"] >= since_timestamp
    ]

    if not matching_heartbeats:
        return {"availability": 0.0, "p95LatencyMs": 0}

    successful_heartbeats = sum(
        heartbeat["status"] == "OK" for heartbeat in matching_heartbeats
    )
    availability = successful_heartbeats / len(matching_heartbeats)

    latencies = sorted(heartbeat["latencyMs"] for heartbeat in matching_heartbeats)
    p95_index = math.ceil(0.95 * len(latencies)) - 1

    return {
        "availability": availability,
        "p95LatencyMs": latencies[p95_index],
    }


def solve(payload: str) -> dict[str, dict[str, Any]]:
    """Decode a challenge payload and produce adaptation and SLO outputs."""

    decoded = base64.b64decode(payload).decode("utf-8")
    data = json.loads(decoded)

    return {
        "adaptOutput": adapt_input(data["adaptInput"]),
        "sloOutput": calculate_slo(data["heartbeats"], data["sloQuery"]),
    }
