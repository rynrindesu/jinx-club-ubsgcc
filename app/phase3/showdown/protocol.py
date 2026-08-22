"""Protocol v2 parsing and response safety for the phase 3 SHOWDOWN bot.

This module deliberately has no dependencies on the earlier SHOWDOWN phases.  The
wire format is parsed into small immutable dataclasses so the strategy can remain
independent of FastAPI (and can be exercised by the simulator and replay tools).
Unknown object fields are ignored; malformed values that the strategy needs in
order to make a legal decision raise :class:`ProtocolError`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


ACTIONS = frozenset({"fold", "check", "call", "bet", "raise"})
SIZED_ACTIONS = frozenset({"bet", "raise"})
ROUNDS = frozenset({"pre_reveal", "post_reveal"})


class ProtocolError(ValueError):
    """Raised when a payload or proposed response cannot be used safely."""


@dataclass(frozen=True, slots=True)
class Player:
    seat: int
    name: str
    folded: bool
    chip_delta: int
    bet_this_round: int
    stack: int
    all_in: bool
    busted: bool


@dataclass(frozen=True, slots=True)
class ActionRecord:
    round: str
    seat: int
    action: str
    amount: int | None = None
    # The following fields are not present in ordinary /move history, but some
    # replay exports include them.  Keeping them when present improves offline
    # learning without making them required on the live protocol.
    pot_before: int | None = None
    to_call: int | None = None
    live_players: int | None = None
    position: str | None = None


@dataclass(frozen=True, slots=True)
class RecentHand:
    hand_number: int
    community_number: int | None
    winners: tuple[int, ...]
    pot: int
    shown_numbers: dict[int, int]
    actions: tuple[ActionRecord, ...]
    button_seat: int | None = None


@dataclass(frozen=True, slots=True)
class MoveRequest:
    protocol_version: int
    match_id: str
    phase: int
    table_rule: str
    small_blind: int
    big_blind: int
    starting_stack: int
    your_stack: int
    hand_number: int
    total_hands: int
    round: str
    your_number: int
    community_number: int | None
    your_seat: int
    button_seat: int
    pot: int
    to_call: int
    min_raise_to: int | None
    max_raise_to: int | None
    legal_actions: tuple[str, ...]
    players: tuple[Player, ...]
    current_hand_actions: tuple[ActionRecord, ...]
    recent_hands: tuple[RecentHand, ...]
    leg_number: int | None = None
    total_legs: int | None = None

    @property
    def own_player(self) -> Player:
        """Return our seat's player record.

        ``parse_payload`` guarantees that this player exists, so failure here
        means a manually constructed ``MoveRequest`` violated the contract.
        """

        for player in self.players:
            if player.seat == self.your_seat:
                return player
        raise ProtocolError(f"players does not contain your_seat {self.your_seat}")

    @property
    def live_opponents(self) -> tuple[Player, ...]:
        """Opponents still eligible to win the current hand.

        All-in players remain live; folded and busted seats do not.
        """

        return tuple(
            player
            for player in self.players
            if player.seat != self.your_seat
            and not player.folded
            and not player.busted
        )

    @property
    def active_players(self) -> tuple[Player, ...]:
        """Seats that have not busted from the match (including folded seats)."""

        return tuple(player for player in self.players if not player.busted)

    @property
    def players_by_seat(self) -> dict[int, Player]:
        return {player.seat: player for player in self.players}

    @property
    def your_chip_delta(self) -> int:
        return self.own_player.chip_delta

    @property
    def session_key(self) -> tuple[str, int | None, str]:
        return (self.match_id, self.leg_number, self.table_rule)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{field} must be an object")
    return value


def _require_sequence(value: Any, field: str) -> Sequence[Any]:
    if not _is_sequence(value):
        raise ProtocolError(f"{field} must be an array")
    return value


def _integer(
    value: Any,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    # bool is an int subclass, but accepting it here makes malformed JSON very
    # difficult to diagnose and can produce illegal bet amounts.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ProtocolError(f"{field} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ProtocolError(f"{field} must be <= {maximum}")
    return value


def _optional_integer(
    value: Any,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    if value is None:
        return None
    return _integer(value, field, minimum=minimum, maximum=maximum)


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ProtocolError(f"{field} must be a boolean")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{field} must be a non-empty string")
    return value.strip()


def _round(value: Any, field: str) -> str:
    result = _text(value, field)
    if result not in ROUNDS:
        raise ProtocolError(f"{field} must be one of {sorted(ROUNDS)}")
    return result


def _parse_player(value: Any, index: int) -> Player:
    item = _require_mapping(value, f"players[{index}]")
    prefix = f"players[{index}]"
    return Player(
        seat=_integer(item.get("seat"), f"{prefix}.seat", minimum=0),
        name=_text(item.get("name"), f"{prefix}.name"),
        folded=_boolean(item.get("folded"), f"{prefix}.folded"),
        chip_delta=_integer(item.get("chip_delta"), f"{prefix}.chip_delta"),
        bet_this_round=_integer(
            item.get("bet_this_round"), f"{prefix}.bet_this_round", minimum=0
        ),
        stack=_integer(item.get("stack"), f"{prefix}.stack", minimum=0),
        all_in=_boolean(item.get("all_in"), f"{prefix}.all_in"),
        busted=_boolean(item.get("busted"), f"{prefix}.busted"),
    )


def _parse_action(value: Any, field: str) -> ActionRecord:
    item = _require_mapping(value, field)
    action = _text(item.get("action"), f"{field}.action").lower()
    if action not in ACTIONS:
        raise ProtocolError(f"{field}.action is not a recognised action")

    amount = _optional_integer(item.get("amount"), f"{field}.amount", minimum=0)
    # Bet and raise logs must state the round-total amount.  Calls in completed
    # histories commonly state it as well; check/fold correctly leave it null.
    if action in SIZED_ACTIONS and amount is None:
        raise ProtocolError(f"{field}.amount is required for {action}")

    position_value = item.get("position")
    position = None
    if position_value is not None:
        position = _text(position_value, f"{field}.position").lower()

    return ActionRecord(
        round=_round(item.get("round"), f"{field}.round"),
        seat=_integer(item.get("seat"), f"{field}.seat", minimum=0),
        action=action,
        amount=amount,
        pot_before=_optional_integer(
            item.get("pot_before"), f"{field}.pot_before", minimum=0
        ),
        to_call=_optional_integer(item.get("to_call"), f"{field}.to_call", minimum=0),
        live_players=_optional_integer(
            item.get("live_players"), f"{field}.live_players", minimum=1
        ),
        position=position,
    )


def _parse_actions(value: Any, field: str, *, tolerant: bool) -> tuple[ActionRecord, ...]:
    if value is None and tolerant:
        return ()
    sequence = _require_sequence(value, field)
    parsed: list[ActionRecord] = []
    for index, action in enumerate(sequence):
        try:
            parsed.append(_parse_action(action, f"{field}[{index}]"))
        except ProtocolError:
            if not tolerant:
                raise
    return tuple(parsed)


def _parse_recent_hand(value: Any, index: int) -> RecentHand:
    field = f"recent_hands[{index}]"
    item = _require_mapping(value, field)

    shown_raw = _require_mapping(item.get("shown_numbers", {}), f"{field}.shown_numbers")
    shown: dict[int, int] = {}
    for raw_seat, raw_number in shown_raw.items():
        try:
            # JSON object keys arrive as strings, while in-process test/replay
            # data sometimes already uses integer keys.
            if isinstance(raw_seat, bool):
                raise ProtocolError(f"{field}.shown_numbers has an invalid seat")
            seat = int(raw_seat)
            if str(seat) != str(raw_seat) and not isinstance(raw_seat, int):
                raise ProtocolError(f"{field}.shown_numbers has an invalid seat")
            if seat < 0:
                raise ProtocolError(f"{field}.shown_numbers has an invalid seat")
            shown[seat] = _integer(
                raw_number,
                f"{field}.shown_numbers[{raw_seat!r}]",
                minimum=1,
                maximum=13,
            )
        except (ProtocolError, TypeError, ValueError):
            # A bad historical reveal must not make the current legal move fail.
            continue

    winners_raw = _require_sequence(item.get("winners", ()), f"{field}.winners")
    winners: list[int] = []
    for winner_index, raw_winner in enumerate(winners_raw):
        try:
            winner = _integer(
                raw_winner, f"{field}.winners[{winner_index}]", minimum=0
            )
        except ProtocolError:
            continue
        if winner not in winners:
            winners.append(winner)

    return RecentHand(
        hand_number=_integer(
            item.get("hand_number"), f"{field}.hand_number", minimum=1
        ),
        community_number=_optional_integer(
            item.get("community_number"),
            f"{field}.community_number",
            minimum=1,
            maximum=13,
        ),
        winners=tuple(winners),
        pot=_integer(item.get("pot"), f"{field}.pot", minimum=0),
        shown_numbers=shown,
        actions=_parse_actions(item.get("actions", ()), f"{field}.actions", tolerant=True),
        button_seat=_optional_integer(
            item.get("button_seat"), f"{field}.button_seat", minimum=0
        ),
    )


def parse_payload(payload: Mapping[str, Any]) -> MoveRequest:
    """Parse a protocol-v2 ``/move`` JSON object.

    Unknown keys are intentionally ignored.  Historical entries are advisory:
    malformed individual recent hands/actions are skipped so a corrupt learning
    sample cannot prevent a legal response to the current turn.  Current-turn
    fields and actions are strict because the decision engine relies on them.
    """

    data = _require_mapping(payload, "payload")
    protocol_version = _integer(data.get("protocol_version"), "protocol_version")
    if protocol_version != 2:
        raise ProtocolError("protocol_version must be 2")

    round_name = _round(data.get("round"), "round")
    community_number = _optional_integer(
        data.get("community_number"), "community_number", minimum=1, maximum=13
    )
    if round_name == "post_reveal" and community_number is None:
        raise ProtocolError("community_number is required post_reveal")

    legal_raw = _require_sequence(data.get("legal_actions"), "legal_actions")
    legal: list[str] = []
    for index, value in enumerate(legal_raw):
        action = _text(value, f"legal_actions[{index}]").lower()
        if action not in ACTIONS:
            raise ProtocolError(f"legal_actions[{index}] is not recognised")
        if action not in legal:
            legal.append(action)
    if not legal:
        raise ProtocolError("legal_actions must contain at least one action")

    min_raise_to = _optional_integer(
        data.get("min_raise_to"), "min_raise_to", minimum=0
    )
    max_raise_to = _optional_integer(
        data.get("max_raise_to"), "max_raise_to", minimum=0
    )
    if any(action in SIZED_ACTIONS for action in legal):
        if min_raise_to is None or max_raise_to is None:
            raise ProtocolError("sized legal actions require min_raise_to/max_raise_to")
        if min_raise_to > max_raise_to:
            raise ProtocolError("min_raise_to cannot exceed max_raise_to")

    players_raw = _require_sequence(data.get("players"), "players")
    players = tuple(_parse_player(value, index) for index, value in enumerate(players_raw))
    if not players:
        raise ProtocolError("players must contain at least one seat")
    seats = [player.seat for player in players]
    if len(seats) != len(set(seats)):
        raise ProtocolError("players contains duplicate seats")

    your_seat = _integer(data.get("your_seat"), "your_seat", minimum=0)
    if your_seat not in set(seats):
        raise ProtocolError("players must contain your_seat")

    hand_number = _integer(data.get("hand_number"), "hand_number", minimum=1)
    total_hands = _integer(data.get("total_hands"), "total_hands", minimum=1)
    if hand_number > total_hands:
        raise ProtocolError("hand_number cannot exceed total_hands")

    leg_number = _optional_integer(data.get("leg_number"), "leg_number", minimum=1)
    total_legs = _optional_integer(data.get("total_legs"), "total_legs", minimum=1)
    if (leg_number is None) != (total_legs is None):
        raise ProtocolError("leg_number and total_legs must both be null or integers")
    if leg_number is not None and total_legs is not None and leg_number > total_legs:
        raise ProtocolError("leg_number cannot exceed total_legs")

    recent: list[RecentHand] = []
    recent_raw = data.get("recent_hands", ())
    if _is_sequence(recent_raw):
        for index, value in enumerate(recent_raw):
            try:
                recent.append(_parse_recent_hand(value, index))
            except ProtocolError:
                continue

    request = MoveRequest(
        protocol_version=protocol_version,
        match_id=_text(data.get("match_id"), "match_id"),
        phase=_integer(data.get("phase"), "phase", minimum=1),
        table_rule=_text(data.get("table_rule"), "table_rule"),
        small_blind=_integer(data.get("small_blind"), "small_blind", minimum=0),
        big_blind=_integer(data.get("big_blind"), "big_blind", minimum=0),
        starting_stack=_integer(
            data.get("starting_stack"), "starting_stack", minimum=1
        ),
        your_stack=_integer(data.get("your_stack"), "your_stack", minimum=0),
        hand_number=hand_number,
        total_hands=total_hands,
        round=round_name,
        your_number=_integer(
            data.get("your_number"), "your_number", minimum=1, maximum=13
        ),
        community_number=community_number,
        your_seat=your_seat,
        button_seat=_integer(data.get("button_seat"), "button_seat", minimum=0),
        pot=_integer(data.get("pot"), "pot", minimum=0),
        to_call=_integer(data.get("to_call"), "to_call", minimum=0),
        min_raise_to=min_raise_to,
        max_raise_to=max_raise_to,
        legal_actions=tuple(legal),
        players=players,
        current_hand_actions=_parse_actions(
            data.get("current_hand_actions", ()),
            "current_hand_actions",
            tolerant=False,
        ),
        recent_hands=tuple(recent),
        leg_number=leg_number,
        total_legs=total_legs,
    )

    # Cross-field checks catch impossible inputs without trying to reproduce the
    # coordinator's betting engine.  legal_actions remains authoritative.
    if request.own_player.stack != request.your_stack:
        raise ProtocolError("your_stack must match your player stack")
    if "check" in legal and request.to_call != 0:
        raise ProtocolError("check cannot be legal when to_call is non-zero")
    return request


def validate_response(
    request: MoveRequest, response: Mapping[str, Any]
) -> dict[str, int | str]:
    """Return a canonical, legal response or raise ``ProtocolError``.

    Unknown response keys are ignored.  Amounts on non-sized actions are omitted
    from the canonical result; bet/raise amounts must be JSON integers inside the
    coordinator-provided inclusive range.
    """

    data = _require_mapping(response, "response")
    action = _text(data.get("action"), "response.action").lower()
    if action not in request.legal_actions:
        raise ProtocolError(f"response action {action!r} is not legal")

    if action not in SIZED_ACTIONS:
        return {"action": action}

    amount = _integer(data.get("amount"), "response.amount")
    lower = request.min_raise_to
    upper = request.max_raise_to
    if lower is None or upper is None:
        raise ProtocolError("sized action has no legal amount range")
    if amount < lower or amount > upper:
        raise ProtocolError(f"response.amount must be in [{lower}, {upper}]")
    return {"action": action, "amount": amount}


def safe_fallback(
    value: MoveRequest | Mapping[str, Any] | None = None,
    request: MoveRequest | None = None,
) -> dict[str, int | str]:
    """Choose the safest usable legal response.

    ``value`` may be either a parsed request or the original JSON payload.  The
    optional ``request`` parameter lets an API handler retain a parsed request
    while also passing the raw payload (``safe_fallback(payload, request)``).
    Priority is check, fold, call, then a minimum-sized bet/raise.
    """

    source: MoveRequest | Mapping[str, Any] | None = request or value
    legal: tuple[str, ...]
    lower: int | None

    if isinstance(source, MoveRequest):
        legal = source.legal_actions
        lower = source.min_raise_to
    elif isinstance(source, Mapping):
        raw_legal = source.get("legal_actions", ())
        if not _is_sequence(raw_legal):
            raw_legal = ()
        legal = tuple(
            item.lower()
            for item in raw_legal
            if isinstance(item, str) and item.lower() in ACTIONS
        )
        raw_lower = source.get("min_raise_to")
        lower = (
            raw_lower
            if isinstance(raw_lower, int)
            and not isinstance(raw_lower, bool)
            and raw_lower >= 0
            else None
        )
    else:
        legal = ()
        lower = None

    for action in ("check", "fold", "call"):
        if action in legal:
            return {"action": action}
    for action in ("bet", "raise"):
        if action in legal and lower is not None:
            return {"action": action, "amount": lower}
    raise ProtocolError("payload contains no usable legal fallback action")


__all__ = [
    "ACTIONS",
    "SIZED_ACTIONS",
    "ROUNDS",
    "ProtocolError",
    "Player",
    "ActionRecord",
    "RecentHand",
    "MoveRequest",
    "parse_payload",
    "validate_response",
    "safe_fallback",
]
