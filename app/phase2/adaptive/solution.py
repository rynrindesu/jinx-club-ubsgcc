import base64
import binascii
import json
import math
from typing import Any


PRIORITY_MAP = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
}


class PayloadValidationError(ValueError):
    """Raised when a decoded challenge payload does not meet the API contract."""


def _required(mapping: dict[str, Any], field: str, location: str) -> Any:
    try:
        return mapping[field]
    except KeyError as error:
        raise PayloadValidationError(f"{location}.{field} is required") from error


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PayloadValidationError(f"{location} must be an object")
    return value


def _text(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise PayloadValidationError(f"{location} must be a string")
    return value


def _number(value: Any, location: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PayloadValidationError(f"{location} must be a number")
    if not math.isfinite(value):
        raise PayloadValidationError(f"{location} must be finite")
    return value


def adapt_input(adapt_input: dict[str, Any]) -> dict[str, Any]:
    """Convert the Version 1 request model to the Version 2 model."""

    adapt_input = _object(adapt_input, "adaptInput")
    user = _object(_required(adapt_input, "user", "adaptInput"), "adaptInput.user")
    metadata = _object(
        _required(adapt_input, "metadata", "adaptInput"), "adaptInput.metadata"
    )
    priority = _text(
        _required(metadata, "priority", "adaptInput.metadata"),
        "adaptInput.metadata.priority",
    )

    if priority not in PRIORITY_MAP:
        raise PayloadValidationError("adaptInput.metadata.priority is unsupported")

    return {
        "id": _required(user, "id", "adaptInput.user"),
        "name": _text(
            _required(user, "fullName", "adaptInput.user"), "adaptInput.user.fullName"
        ),
        "action": _text(
            _required(adapt_input, "action", "adaptInput"), "adaptInput.action"
        ).lower(),
        "priority": PRIORITY_MAP[priority],
    }


def calculate_slo(
    heartbeats: list[dict[str, Any]], slo_query: dict[str, Any]
) -> dict[str, float | int]:
    """Report SLOs for any queried service in the requested time window."""

    if not isinstance(heartbeats, list):
        raise PayloadValidationError("heartbeats must be an array")

    slo_query = _object(slo_query, "sloQuery")
    requested_service = _text(
        _required(slo_query, "service", "sloQuery"), "sloQuery.service"
    )
    since_timestamp = _number(
        _required(slo_query, "since", "sloQuery"), "sloQuery.since"
    )

    heartbeats_by_service: dict[str, list[dict[str, Any]]] = {}
    for index, heartbeat_value in enumerate(heartbeats):
        location = f"heartbeats[{index}]"
        heartbeat = _object(heartbeat_value, location)
        service = _text(_required(heartbeat, "service", location), f"{location}.service")
        timestamp = _number(
            _required(heartbeat, "timestamp", location), f"{location}.timestamp"
        )
        latency_ms = _number(
            _required(heartbeat, "latencyMs", location), f"{location}.latencyMs"
        )
        if latency_ms < 0:
            raise PayloadValidationError(f"{location}.latencyMs must not be negative")
        status = _text(_required(heartbeat, "status", location), f"{location}.status")

        heartbeats_by_service.setdefault(service, []).append(
            {
                "timestamp": timestamp,
                "latencyMs": latency_ms,
                "status": status,
            }
        )

    matching_heartbeats = [
        heartbeat
        for heartbeat in heartbeats_by_service.get(requested_service, [])
        if heartbeat["timestamp"] >= since_timestamp
    ]

    if not matching_heartbeats:
        return {"availability": 0.0, "p95LatencyMs": 0}

    successful_heartbeats = sum(
        heartbeat["status"] == "OK" for heartbeat in matching_heartbeats
    )
    availability = successful_heartbeats / len(matching_heartbeats)

    return {
        "availability": availability,
        "p95LatencyMs": p95_latency_ms(matching_heartbeats),
    }


def p95_latency_ms(heartbeats: list[dict[str, Any]]) -> int | float:
    """Select the sorted latency whose right-inclusive range contains 95%."""

    latencies = sorted(heartbeat["latencyMs"] for heartbeat in heartbeats)
    heartbeat_count = len(latencies)

    for index, latency in enumerate(latencies):
        lower_percent = index / heartbeat_count * 100
        upper_percent = (index + 1) / heartbeat_count * 100
        if lower_percent < 95 <= upper_percent:
            return latency

    # The loop always finds a range for 95, but this keeps the helper total.
    return latencies[-1]


def solve(payload: str) -> dict[str, dict[str, Any]]:
    """Decode a challenge payload and produce adaptation and SLO outputs."""

    if not isinstance(payload, str):
        raise PayloadValidationError("payload must be a Base64 string")

    try:
        decoded = base64.b64decode(payload, validate=True).decode("utf-8")
        data = json.loads(decoded)
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PayloadValidationError(
            "payload must be valid Base64-encoded UTF-8 JSON"
        ) from error

    data = _object(data, "payload")
    return {
        "adaptOutput": adapt_input(_required(data, "adaptInput", "payload")),
        "sloOutput": calculate_slo(
            _required(data, "heartbeats", "payload"),
            _required(data, "sloQuery", "payload"),
        ),
    }
