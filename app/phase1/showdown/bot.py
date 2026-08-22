"""Fast, stateless decision policy for the standard SHOWDOWN ruleset.

The challenge server is authoritative about which actions and wager sizes are
legal.  This module deliberately keeps strategy separate from the HTTP layer so
it can be tested without running a web server.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping


Action = dict[str, Any]


@dataclass(frozen=True)
class OpponentProfile:
    """Small, deliberately conservative model derived from recent hands."""

    fold_to_pressure: float
    aggression: float
    observed_decisions: int


def showdown_equity(your_number: int, community_number: int | None) -> float:
    """Return expected pot share against one uniformly random opponent number.

    A tie contributes half a pot.  Before the reveal, this averages over all 13
    possible community numbers.  Draws are independent and with replacement.
    """

    if not 1 <= your_number <= 13:
        raise ValueError("your_number must be between 1 and 13")

    if community_number is None:
        return (11 * your_number + 7.5) / 169

    if not 1 <= community_number <= 13:
        raise ValueError("community_number must be between 1 and 13")

    if your_number == community_number:
        return 12.5 / 13

    if community_number < your_number:
        return (your_number - 1.5) / 13

    return (your_number - 0.5) / 13


def decide_move(payload: Mapping[str, Any]) -> Action:
    """Choose a legal SHOWDOWN move for a protocol-version 2 request."""

    legal = _legal_actions(payload)
    if not legal:
        # Valid challenge requests always provide legal_actions.  Returning the
        # safest generally useful action keeps malformed manual probes harmless.
        return {"action": "check"}

    your_number = _integer(payload.get("your_number"), 7)
    community = payload.get("community_number")
    community_number = None if community is None else _integer(community, 7)

    try:
        equity = showdown_equity(your_number, community_number)
    except ValueError:
        equity = 0.5

    profile = _opponent_profile(payload)
    to_call = max(0, _integer(payload.get("to_call"), 0))
    pot = max(0, _integer(payload.get("pot"), 0))
    stack = max(1, _integer(payload.get("your_stack"), 1))
    is_pair = community_number is not None and your_number == community_number

    if to_call > 0:
        candidate = _facing_bet(
            payload=payload,
            equity=equity,
            is_pair=is_pair,
            profile=profile,
            pot=pot,
            to_call=to_call,
            stack=stack,
        )
    else:
        candidate = _when_free_to_continue(
            payload=payload,
            equity=equity,
            is_pair=is_pair,
            profile=profile,
        )

    return _sanitize(payload, candidate)


def _facing_bet(
    *,
    payload: Mapping[str, Any],
    equity: float,
    is_pair: bool,
    profile: OpponentProfile,
    pot: int,
    to_call: int,
    stack: int,
) -> Action:
    legal = _legal_actions(payload)
    pot_odds = to_call / max(1, pot + to_call)

    # A larger fraction of the remaining stack needs a clearer edge.  Highly
    # aggressive opponents receive a modest bluff allowance, shrunk by priors
    # in _opponent_profile so a tiny sample cannot swing the policy wildly.
    risk_premium = min(0.08, 0.06 * (to_call / stack) ** 0.5)
    aggression_adjustment = (profile.aggression - 0.5) * 0.12
    round_margin = 0.04 if payload.get("round") == "pre_reveal" else 0.02
    repeated_pressure = max(0, _opponent_pressure_count(payload) - 1) * 0.02
    call_threshold = pot_odds + risk_premium + round_margin + repeated_pressure
    call_threshold -= aggression_adjustment

    # A pair cannot lose at a standard showdown; occasionally calling instead
    # of raising prevents the strongest hand from being completely transparent.
    if is_pair:
        if (
            "raise" in legal
            and ("call" not in legal or _roll(payload, "pair-trap") >= 0.20)
        ):
            return _wager(payload, "raise", pot_fraction=0.75)
        if "call" in legal:
            return {"action": "call"}
        return {"action": "raise"}

    post_reveal = payload.get("round") == "post_reveal"
    call_fraction = to_call / stack

    # A post-reveal re-raise is much stronger evidence than an ordinary bet.
    # Raw equity assumes every opponent number is equally likely, but after we
    # raise and the opponent raises again their range is heavily concentrated
    # on the community number (a pair).  Never risk a substantial part of the
    # stack with a non-pair in that line.  Only a proven extreme aggressor gets
    # called when the additional price is very small.
    if (
        post_reveal
        and _facing_reraise(payload)
        and "fold" in legal
        and (call_fraction > 0.15 or profile.aggression < 0.68)
    ):
        return {"action": "fold"}

    # The match score has a hard -200 bust outcome.  An unknown opponent's
    # large post-reveal shove is not a spot to stack off with a non-pair, even
    # when a high number has excellent unconditional showdown equity.
    if post_reveal and call_fraction >= 0.45 and "fold" in legal:
        return {"action": "fold"}

    has_calling_edge = equity >= call_threshold
    value_raise = (
        not post_reveal
        and equity >= 0.76
        and equity >= call_threshold + 0.10
    )
    if (
        value_raise
        and "raise" in legal
        and _minimum_wager_fraction(payload) <= 0.45
        and _roll(payload, "value-raise") < 0.58
    ):
        return _wager(payload, "raise", pot_fraction=0.65)

    if has_calling_edge and "call" in legal:
        return {"action": "call"}

    # Bluff-raises use the weakest hands, not medium-strength hands that still
    # have showdown value.  They are rare unless the observed opponent folds.
    bluff_chance = 0.025 + max(0.0, profile.fold_to_pressure - 0.5) * 0.30
    if (
        equity <= 0.25
        and "raise" in legal
        and _minimum_wager_fraction(payload) <= 0.20
        and _roll(payload, "bluff-raise") < bluff_chance
    ):
        return _wager(payload, "raise", pot_fraction=0.50)

    if "fold" in legal:
        return {"action": "fold"}
    if "call" in legal:
        return {"action": "call"}
    return {"action": "check"}


def _when_free_to_continue(
    *,
    payload: Mapping[str, Any],
    equity: float,
    is_pair: bool,
    profile: OpponentProfile,
) -> Action:
    legal = _legal_actions(payload)
    wager_action = "bet" if "bet" in legal else "raise" if "raise" in legal else None
    if wager_action is None:
        return {"action": "check"}

    checked_to_us = _opponent_checked_this_round(payload)
    in_position_post_reveal = (
        payload.get("round") == "post_reveal"
        and payload.get("your_seat") == payload.get("button_seat")
    )

    if is_pair:
        if "check" in legal and _roll(payload, "pair-open-trap") < 0.15:
            return {"action": "check"}
        return _wager(payload, wager_action, pot_fraction=0.70)

    value_threshold = 0.62 if checked_to_us or in_position_post_reveal else 0.68
    if (
        equity >= value_threshold
        and _minimum_wager_fraction(payload) <= 0.40
        and _roll(payload, "value-open") < 0.78
    ):
        return _wager(payload, wager_action, pot_fraction=0.60)

    bluff_chance = 0.06 + max(0.0, profile.fold_to_pressure - 0.5) * 0.35
    if checked_to_us or in_position_post_reveal:
        bluff_chance += 0.04

    if (
        equity <= 0.28
        and _minimum_wager_fraction(payload) <= 0.18
        and _roll(payload, "open-bluff") < bluff_chance
    ):
        return _wager(payload, wager_action, pot_fraction=0.50)

    return {"action": "check"}


def _wager(payload: Mapping[str, Any], action: str, pot_fraction: float) -> Action:
    minimum = _optional_integer(payload.get("min_raise_to"))
    maximum = _optional_integer(payload.get("max_raise_to"))
    if minimum is None or maximum is None or minimum > maximum:
        return {"action": action}

    contribution = _your_round_contribution(payload)
    to_call = max(0, _integer(payload.get("to_call"), 0))
    pot = max(1, _integer(payload.get("pot"), 1))

    if to_call:
        raise_extra = max(to_call, round(pot * pot_fraction))
        desired_addition = to_call + raise_extra
    else:
        desired_addition = max(1, round(pot * pot_fraction))

    desired_total = contribution + desired_addition
    amount = max(minimum, min(maximum, desired_total))
    return {"action": action, "amount": amount}


def _sanitize(payload: Mapping[str, Any], candidate: Mapping[str, Any]) -> Action:
    """Make illegal strategy output impossible on a valid server request."""

    legal = _legal_actions(payload)
    action = str(candidate.get("action", ""))

    if action in legal:
        if action not in {"bet", "raise"}:
            return {"action": action}

        minimum = _optional_integer(payload.get("min_raise_to"))
        maximum = _optional_integer(payload.get("max_raise_to"))
        amount = _optional_integer(candidate.get("amount"))
        if minimum is not None and maximum is not None and minimum <= maximum:
            if amount is None:
                amount = minimum
            return {"action": action, "amount": max(minimum, min(maximum, amount))}

    # Prefer a free continuation.  When facing a wager, folding is the safest
    # protocol fallback; it avoids the coordinator substituting a fold for us.
    if "check" in legal:
        return {"action": "check"}
    if "fold" in legal:
        return {"action": "fold"}
    if "call" in legal:
        return {"action": "call"}

    for wager_action in ("bet", "raise"):
        if wager_action in legal:
            minimum = _optional_integer(payload.get("min_raise_to"))
            if minimum is not None:
                return {"action": wager_action, "amount": minimum}

    return {"action": legal[0]}


def _opponent_profile(payload: Mapping[str, Any]) -> OpponentProfile:
    your_seat = payload.get("your_seat")
    pressure_opportunities = 0
    folds = 0
    aggressive_actions = 0
    decisions = 0

    recent_hands = payload.get("recent_hands")
    if not isinstance(recent_hands, list):
        recent_hands = []

    for hand in recent_hands:
        if not isinstance(hand, Mapping):
            continue
        actions = hand.get("actions")
        if not isinstance(actions, list):
            continue

        previous: Mapping[str, Any] | None = None
        for raw_action in actions:
            if not isinstance(raw_action, Mapping):
                continue

            seat = raw_action.get("seat")
            verb = raw_action.get("action")
            if seat != your_seat and verb in {"check", "call", "bet", "raise", "fold"}:
                decisions += 1
                if verb in {"bet", "raise"}:
                    aggressive_actions += 1

                if (
                    verb in {"fold", "call", "raise"}
                    and previous is not None
                    and previous.get("seat") == your_seat
                    and previous.get("action") in {"bet", "raise"}
                ):
                    pressure_opportunities += 1
                    if verb == "fold":
                        folds += 1

            previous = raw_action

    # Beta-style priors pull small samples toward a neutral 50% rather than
    # overfitting one visible showdown or fold.
    fold_rate = (folds + 2) / (pressure_opportunities + 4)
    aggression = (aggressive_actions + 2) / (decisions + 4)
    return OpponentProfile(fold_rate, aggression, decisions)


def _opponent_pressure_count(payload: Mapping[str, Any]) -> int:
    your_seat = payload.get("your_seat")
    current_round = payload.get("round")
    actions = payload.get("current_hand_actions")
    if not isinstance(actions, list):
        return 0
    return sum(
        1
        for action in actions
        if isinstance(action, Mapping)
        and action.get("round") == current_round
        and action.get("seat") != your_seat
        and action.get("action") in {"bet", "raise"}
    )


def _facing_reraise(payload: Mapping[str, Any]) -> bool:
    """Whether the opponent just raised after our bet or raise this round."""

    your_seat = payload.get("your_seat")
    current_round = payload.get("round")
    actions = payload.get("current_hand_actions")
    if not isinstance(actions, list):
        return False

    round_actions = [
        action
        for action in actions
        if isinstance(action, Mapping) and action.get("round") == current_round
    ]
    if len(round_actions) < 2:
        return False

    previous, latest = round_actions[-2:]
    return (
        previous.get("seat") == your_seat
        and previous.get("action") in {"bet", "raise"}
        and latest.get("seat") != your_seat
        and latest.get("action") == "raise"
    )


def _opponent_checked_this_round(payload: Mapping[str, Any]) -> bool:
    your_seat = payload.get("your_seat")
    current_round = payload.get("round")
    actions = payload.get("current_hand_actions")
    if not isinstance(actions, list):
        return False

    for action in reversed(actions):
        if not isinstance(action, Mapping) or action.get("round") != current_round:
            continue
        if action.get("seat") != your_seat:
            return action.get("action") == "check"
    return False


def _minimum_wager_fraction(payload: Mapping[str, Any]) -> float:
    minimum = _optional_integer(payload.get("min_raise_to"))
    if minimum is None:
        return 1.0
    contribution = _your_round_contribution(payload)
    stack = max(1, _integer(payload.get("your_stack"), 1))
    return max(0, minimum - contribution) / stack


def _your_round_contribution(payload: Mapping[str, Any]) -> int:
    your_seat = payload.get("your_seat")
    players = payload.get("players")
    if not isinstance(players, list):
        return 0
    for player in players:
        if isinstance(player, Mapping) and player.get("seat") == your_seat:
            return max(0, _integer(player.get("bet_this_round"), 0))
    return 0


def _roll(payload: Mapping[str, Any], purpose: str) -> float:
    """Stable pseudo-randomness: identical requests always get identical moves."""

    key = "|".join(
        str(value)
        for value in (
            payload.get("match_id", ""),
            payload.get("hand_number", ""),
            payload.get("round", ""),
            payload.get("your_seat", ""),
            len(payload.get("current_hand_actions", []))
            if isinstance(payload.get("current_hand_actions"), list)
            else 0,
            purpose,
        )
    )
    value = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    return value / 2**64


def _legal_actions(payload: Mapping[str, Any]) -> list[str]:
    raw = payload.get("legal_actions")
    if not isinstance(raw, list):
        return []
    return [action for action in raw if isinstance(action, str)]


def _integer(value: Any, default: int) -> int:
    parsed = _optional_integer(value)
    return default if parsed is None else parsed


def _optional_integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
