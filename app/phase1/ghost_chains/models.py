"""Input models and validation for the Ghost Chains Phase 1 engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import math
from typing import Any, Mapping


class TransactionValidationError(ValueError):
    """Raised when a transaction cannot be interpreted safely."""


class TransactionConflictError(ValueError):
    """Raised when one transaction ID is reused for a different payload."""

    status_code = 409

    def __init__(self, tx_id: str):
        self.tx_id = tx_id
        super().__init__(f"transaction {tx_id!r} conflicts with its original payload")


@dataclass(frozen=True)
class PresentValue:
    """Preserve the distinction between an absent field and a present value."""

    present: bool
    value: Any = None

    @classmethod
    def absent(cls) -> PresentValue:
        return cls(present=False)

    @classmethod
    def supplied(cls, value: Any) -> PresentValue:
        return cls(present=True, value=value)


_ID_FIELDS = ("txId", "transactionId", "id")
_SENDER_FIELDS = ("fromUserId", "sender", "senderId", "from", "source")
_RECIPIENT_FIELDS = (
    "toUserId",
    "recipient",
    "recipientId",
    "receiver",
    "receiverId",
    "to",
    "destination",
)
_CREATED_AT_FIELDS = ("createdAt", "timestamp", "occurredAt")
_AMOUNT_FIELDS = ("amount",)
_IP_FIELDS = ("ip", "ipAddress", "sourceIp")
_DEVICE_FIELDS = ("device", "deviceId")
_KNOWN_FIELDS = frozenset(
    _ID_FIELDS
    + _SENDER_FIELDS
    + _RECIPIENT_FIELDS
    + _CREATED_AT_FIELDS
    + _AMOUNT_FIELDS
    + _IP_FIELDS
    + _DEVICE_FIELDS
)


@dataclass(frozen=True)
class Transaction:
    """Canonical Phase 1 transaction.

    Amount and identity evidence are retained for later phases but deliberately
    excluded from structural scoring.
    """

    tx_id: str
    sender: str
    recipient: str
    created_at: datetime
    amount: PresentValue = field(default_factory=PresentValue.absent)
    ip: PresentValue = field(default_factory=PresentValue.absent)
    device: PresentValue = field(default_factory=PresentValue.absent)
    fingerprint: str = field(default="", repr=False, compare=False)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> Transaction:
        if not isinstance(payload, Mapping):
            raise TransactionValidationError("transaction must be an object")

        tx_id = _required_text(payload, _ID_FIELDS, "txId")
        sender = _required_text(payload, _SENDER_FIELDS, "sender")
        recipient = _required_text(payload, _RECIPIENT_FIELDS, "recipient")
        created_at = _parse_timestamp(
            _select_required(payload, _CREATED_AT_FIELDS, "createdAt")
        )
        amount = PresentValue.supplied(
            _required_number(payload, _AMOUNT_FIELDS, "amount")
        )
        ip = _select_optional(payload, _IP_FIELDS, "ip")
        device = _select_optional(payload, _DEVICE_FIELDS, "device")

        extras = {
            str(key): value
            for key, value in payload.items()
            if key not in _KNOWN_FIELDS
        }
        semantic_payload = {
            "txId": tx_id,
            "sender": sender,
            "recipient": recipient,
            "createdAt": _canonical_timestamp(created_at),
            "amount": _present_value_for_hash(amount),
            "ip": _present_value_for_hash(ip),
            "device": _present_value_for_hash(device),
            "extra": extras,
        }
        fingerprint = _fingerprint(semantic_payload)

        return cls(
            tx_id=tx_id,
            sender=sender,
            recipient=recipient,
            created_at=created_at,
            amount=amount,
            ip=ip,
            device=device,
            fingerprint=fingerprint,
        )


def coerce_transaction(value: Transaction | Mapping[str, Any]) -> Transaction:
    """Return a validated transaction without rebuilding canonical instances."""

    if isinstance(value, Transaction):
        if not value.fingerprint:
            raise TransactionValidationError(
                "Transaction instances must be created with Transaction.from_mapping"
            )
        return value
    return Transaction.from_mapping(value)


def _required_text(
    payload: Mapping[str, Any], aliases: tuple[str, ...], label: str
) -> str:
    value = _select_required(payload, aliases, label)
    if not isinstance(value, str) or not value.strip():
        raise TransactionValidationError(f"{label} must be a non-empty string")
    return value


def _required_number(
    payload: Mapping[str, Any], aliases: tuple[str, ...], label: str
) -> int | float | Decimal:
    value = _select_required(payload, aliases, label)
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise TransactionValidationError(f"{label} must be a number")
    try:
        finite = math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        finite = False
    if not finite:
        raise TransactionValidationError(f"{label} must be finite")
    return value


def _select_required(
    payload: Mapping[str, Any], aliases: tuple[str, ...], label: str
) -> Any:
    selected = _matching_values(payload, aliases)
    if not selected:
        raise TransactionValidationError(f"{label} is required")
    _ensure_aliases_agree(selected, label)
    return selected[0][1]


def _select_optional(
    payload: Mapping[str, Any], aliases: tuple[str, ...], label: str
) -> PresentValue:
    selected = _matching_values(payload, aliases)
    if not selected:
        return PresentValue.absent()
    _ensure_aliases_agree(selected, label)
    return PresentValue.supplied(selected[0][1])


def _matching_values(
    payload: Mapping[str, Any], aliases: tuple[str, ...]
) -> list[tuple[str, Any]]:
    return [(alias, payload[alias]) for alias in aliases if alias in payload]


def _ensure_aliases_agree(selected: list[tuple[str, Any]], label: str) -> None:
    if len(selected) < 2:
        return
    first = _canonicalize(selected[0][1])
    if any(_canonicalize(value) != first for _, value in selected[1:]):
        names = ", ".join(name for name, _ in selected)
        raise TransactionValidationError(f"conflicting {label} aliases: {names}")


def _parse_timestamp(value: Any) -> datetime:
    parsed: datetime
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise TransactionValidationError("createdAt must not be empty")
        if text.endswith(("Z", "z")):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as error:
            raise TransactionValidationError(
                "createdAt must be an ISO-8601 timestamp"
            ) from error
    elif isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        try:
            numeric = float(value)
        except (OverflowError, ValueError) as error:
            raise TransactionValidationError("createdAt must be finite") from error
        if not math.isfinite(numeric):
            raise TransactionValidationError("createdAt must be finite")
        # Accommodate common millisecond epochs without making the core depend
        # on one transport representation.
        if abs(numeric) >= 100_000_000_000:
            numeric /= 1000
        try:
            parsed = datetime.fromtimestamp(numeric, tz=timezone.utc)
        except (OverflowError, OSError, ValueError) as error:
            raise TransactionValidationError("createdAt is out of range") from error
    else:
        raise TransactionValidationError(
            "createdAt must be an ISO-8601 timestamp or Unix epoch"
        )

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _canonical_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _present_value_for_hash(value: PresentValue) -> dict[str, Any]:
    if not value.present:
        return {"present": False}
    return {"present": True, "value": value.value}


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _canonicalize(value: Any) -> Any:
    """Convert arbitrary JSON-like values to stable, type-aware data."""

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, datetime):
        normalized = value
        if normalized.tzinfo is None:
            normalized = normalized.replace(tzinfo=timezone.utc)
        return {"$datetime": _canonical_timestamp(normalized)}
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            return {"$number": repr(value)}
        try:
            decimal_value = Decimal(str(value))
            normalized = decimal_value.normalize()
            number = format(normalized, "f")
            if number == "-0":
                number = "0"
            return {"$number": number}
        except (InvalidOperation, ValueError):
            return {"$number": repr(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized_items = [_canonicalize(item) for item in value]
        return sorted(
            normalized_items,
            key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True),
        )
    return {"$type": type(value).__qualname__, "$repr": repr(value)}
