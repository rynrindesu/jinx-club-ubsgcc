"""Equity-driven, tight-aggressive policy for standard SHOWDOWN.

The policy is intentionally stateless: every decision is derived from the
request and its rolling ``recent_hands`` window.  Strategy decisions live here
instead of in the HTTP endpoint so they remain fast and easy to tune.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping


Action = dict[str, Any]
_DECISION_ACTIONS = {"check", "call", "bet", "raise", "fold"}


@dataclass(frozen=True)
class OpponentProfile:
    """Conservative opponent tendencies inferred from completed hands."""

    button_fold_rate: float
    button_reraise_rate: float
    post_fold_rate: float
    bet_after_check_rate: float
    aggression_rate: float
    shown_bluff_rate: float
    button_responses: int
    button_folds: int
    button_reraises: int
    post_responses: int
    post_folds: int
    checked_to: int
    decisions: int
    shown_bets: int

    @property
    def tight_folder(self) -> bool:
        return self.button_responses >= 3 and self.button_fold_rate > 0.45

    @property
    def calling_station(self) -> bool:
        return (
            self.button_responses >= 4 and self.button_fold_rate < 0.25
        ) or (self.post_responses >= 4 and self.post_fold_rate < 0.22)

    @property
    def frequent_reraiser(self) -> bool:
        return (
            self.button_responses >= 3 and self.button_reraise_rate > 0.25
        )

    @property
    def aggressive(self) -> bool:
        return self.decisions >= 5 and (
            self.aggression_rate > 0.58
            or (self.checked_to >= 3 and self.bet_after_check_rate > 0.65)
        )

    @property
    def passive(self) -> bool:
        return self.decisions >= 5 and self.aggression_rate < 0.30

    @property
    def demonstrated_bluffer(self) -> bool:
        # Shown hands are selection-biased, so require more than one example.
        return (
            self.shown_bets >= 2 and self.shown_bluff_rate > 0.35
        ) or self.aggressive


def showdown_equity(your_number: int, community_number: int | None) -> float:
    """Return pot-share equity against one uniformly random opponent.

    Independent draws and ties worth half a pot are both included.  Before the
    reveal the result is averaged over all possible community numbers.
    """

    if not 1 <= your_number <= 13:
        raise ValueError("your_number must be between 1 and 13")

    if community_number is None:
        return (22 * your_number + 15) / 338

    if not 1 <= community_number <= 13:
        raise ValueError("community_number must be between 1 and 13")

    if your_number == community_number:
        return 25 / 26
    if community_number < your_number:
        return (your_number - 1.5) / 13
    return (your_number - 0.5) / 13


def decide_move(payload: Mapping[str, Any]) -> Action:
    """Choose a legal protocol-version 2 move for standard SHOWDOWN."""

    legal = _legal_actions(payload)
    if not legal:
        # Valid coordinator requests always contain legal_actions.
        return {"action": "check"}

    your_number = _integer(payload.get("your_number"), 7)
    raw_community = payload.get("community_number")
    community = None if raw_community is None else _integer(raw_community, 7)

    try:
        equity = showdown_equity(your_number, community)
    except ValueError:
        your_number = 7
        community = None
        equity = 0.5

    profile = _opponent_profile(payload)
    if payload.get("round") == "post_reveal" or community is not None:
        candidate = _post_reveal_move(
            payload, your_number, community, equity, profile
        )
    else:
        candidate = _pre_reveal_move(payload, your_number, equity, profile)

    return _sanitize(payload, candidate)


def _pre_reveal_move(
    payload: Mapping[str, Any],
    your_number: int,
    equity: float,
    profile: OpponentProfile,
) -> Action:
    """Implement the positional pre-reveal playbook."""

    to_call = max(0, _integer(payload.get("to_call"), 0))
    stack = max(0, _integer(payload.get("your_stack"), 0))
    is_button = payload.get("your_seat") == payload.get("button_seat")
    actions = _round_actions(payload, "pre_reveal")

    # The button acts first.  Open 3-13 to four chips by default, widen to
    # every number against a proven folder, and value-size to five against a
    # calling station.  Numbers 1-2 retain a tiny readless bluff frequency.
    if is_button and not actions:
        if profile.calling_station:
            open_from = 6
            target = 5
        elif profile.tight_folder:
            open_from = 1
            target = 4
        else:
            open_from = 3
            target = 4

        rare_bluff = your_number <= 2 and _roll(payload, "button-open-bluff") < 0.03
        if your_number >= open_from or rare_bluff:
            return _fixed_raise(payload, target, fallback="call")
        return _fallback(payload, "fold")

    # A free decision before the reveal is normally the big blind facing a
    # limp.  Check the lower half and punish the limp with 8-13.
    if to_call == 0:
        raise_from = 6 if profile.calling_station else 8
        if your_number >= raise_from:
            return _fixed_raise(payload, 5, fallback="check")
        return _fallback(payload, "check")

    # Once we have opened, an opponent raise represents a much stronger range.
    # Do not start a raise war: continue with 11-13 at controlled prices, or
    # include 10 when repeated history shows excessive re-raising.
    if _we_raised_this_round(payload) or _facing_reraise(payload):
        continue_from = 10 if profile.frequent_reraiser else 11
        if your_number < continue_from or to_call >= stack:
            return _fallback(payload, "fold")

        pot = max(0, _integer(payload.get("pot"), 0))
        pot_odds = to_call / max(1, pot + to_call)
        starting_stack = max(1, _integer(payload.get("starting_stack"), 200))
        price_caps = {10: 0.04, 11: 0.06, 12: 0.09, 13: 0.12}
        price_cap = max(1, round(starting_stack * price_caps[your_number]))
        if to_call <= price_cap and equity >= pot_odds + 0.08:
            return _fallback(payload, "call")
        return _fallback(payload, "fold")

    # Big blind versus a small button open.  The explicit range is deliberately
    # wider than the old bot: fold 1-5, call 6-10, and re-raise 11-13 to ten.
    opponent_total = _opponent_round_contribution(payload)
    if opponent_total <= 5:
        if your_number <= 5 or to_call >= stack:
            return _fallback(payload, "fold")
        if your_number <= 10:
            return _fallback(payload, "call")
        return _fixed_raise(payload, 10, fallback="call")

    # Treat larger opens like re-raises.  Continue mainly with the top three
    # numbers, and only when both the incremental price and pot odds are sane.
    continue_from = 10 if profile.frequent_reraiser else 11
    if your_number < continue_from or to_call >= stack:
        return _fallback(payload, "fold")

    pot = max(0, _integer(payload.get("pot"), 0))
    pot_odds = to_call / max(1, pot + to_call)
    starting_stack = max(1, _integer(payload.get("starting_stack"), 200))
    price_cap = round(starting_stack * {10: 0.04, 11: 0.06, 12: 0.09, 13: 0.12}[your_number])
    if to_call <= max(1, price_cap) and equity >= pot_odds + 0.08:
        return _fallback(payload, "call")
    return _fallback(payload, "fold")


def _post_reveal_move(
    payload: Mapping[str, Any],
    your_number: int,
    community: int | None,
    equity: float,
    profile: OpponentProfile,
) -> Action:
    """Use revealed strength, position, and exact pot odds."""

    to_call = max(0, _integer(payload.get("to_call"), 0))
    is_pair = community is not None and your_number == community
    if to_call > 0:
        return _post_reveal_facing_bet(
            payload, your_number, community, equity, is_pair, profile
        )
    return _post_reveal_checked_to(payload, your_number, is_pair, profile)


def _post_reveal_checked_to(
    payload: Mapping[str, Any],
    your_number: int,
    is_pair: bool,
    profile: OpponentProfile,
) -> Action:
    legal = _legal_actions(payload)
    wager_action = "bet" if "bet" in legal else "raise" if "raise" in legal else None
    if wager_action is None:
        return _fallback(payload, "check")

    in_position = payload.get("your_seat") == payload.get("button_seat")
    opponent_checked = _opponent_checked_this_round(payload)

    if is_pair:
        trap_frequency = 0.30 if profile.aggressive else 0.06
        if "check" in legal and _roll(payload, "pair-trap") < trap_frequency:
            return {"action": "check"}
        fraction = 0.78 if profile.calling_station else 2 / 3
        return _pot_wager(
            payload, wager_action, fraction, allow_all_in=True, fallback="check"
        )

    if your_number == 13:
        if (
            profile.aggressive
            and "check" in legal
            and _roll(payload, "thirteen-trap") < 0.25
        ):
            return {"action": "check"}
        fraction = 0.70 if profile.calling_station else 0.60
        return _pot_wager(payload, wager_action, fraction, fallback="check")

    if your_number >= 10:
        fraction = 0.65 if profile.calling_station else 0.50
        return _pot_wager(payload, wager_action, fraction, fallback="check")

    if your_number >= 8 and in_position and opponent_checked:
        return _pot_wager(payload, wager_action, 1 / 3, fallback="check")

    # Bluff only the hands with the least showdown value.  A tiny baseline mix
    # is allowed in position; it grows only when a conservative fold estimate
    # clears the actual bet's break-even rate by five percentage points.
    if your_number <= 2 and in_position and opponent_checked:
        addition = _planned_open_addition(payload, 1 / 3)
        pot = max(1, _integer(payload.get("pot"), 1))
        break_even = addition / (pot + addition)
        lower_fold_rate = _conservative_rate(
            profile.post_folds, profile.post_responses
        )
        bluff_frequency = 0.02
        if profile.post_responses >= 3 and lower_fold_rate > break_even + 0.05:
            bluff_frequency += min(0.28, lower_fold_rate - break_even - 0.05)
        if _roll(payload, "post-reveal-bluff") < bluff_frequency:
            return _pot_wager(payload, wager_action, 1 / 3, fallback="check")

    return _fallback(payload, "check")


def _post_reveal_facing_bet(
    payload: Mapping[str, Any],
    your_number: int,
    community: int | None,
    equity: float,
    is_pair: bool,
    profile: OpponentProfile,
) -> Action:
    legal = _legal_actions(payload)
    to_call = max(0, _integer(payload.get("to_call"), 0))
    stack = max(0, _integer(payload.get("your_stack"), 0))
    pot = max(0, _integer(payload.get("pot"), 0))

    # A pair cannot lose under the standard rule (it can only split), so it is
    # the sole hand allowed to take an all-in value raise.
    if is_pair:
        if "raise" in legal:
            fraction = 0.80 if profile.calling_station else 2 / 3
            return _pot_wager(
                payload, "raise", fraction, allow_all_in=True, fallback="call"
            )
        return _fallback(payload, "call")

    # Never voluntarily call an all-in with a beatable hand.  Likewise, treat
    # a re-raise after our own post-reveal bet as pair-heavy; only a tiny call
    # with 13 survives against a demonstrated bluffer.
    if to_call >= stack:
        return _fallback(payload, "fold")

    pot_odds = to_call / max(1, pot + to_call)
    if _facing_reraise(payload):
        if (
            your_number == 13
            and profile.demonstrated_bluffer
            and to_call <= max(2, round(stack * 0.10))
            and equity >= pot_odds + 0.05
        ):
            return _fallback(payload, "call")
        return _fallback(payload, "fold")

    # Unknown/passive opponents' enormous bets are not treated as random
    # ranges.  This prevents a non-pair 13 from blindly stacking off into what
    # is very often the community number.
    if to_call >= 0.45 * stack and not profile.demonstrated_bluffer:
        return _fallback(payload, "fold")

    fresh_bet_fraction = _fresh_bet_fraction(pot, to_call)
    if fresh_bet_fraction <= 1 / 3 + 0.03:
        minimum_number = 7
    elif fresh_bet_fraction <= 0.50 + 0.05:
        minimum_number = 8
    elif fresh_bet_fraction <= 2 / 3 + 0.08:
        minimum_number = 9
    elif fresh_bet_fraction <= 1.10:
        minimum_number = 10
    else:
        minimum_number = 11

    if profile.demonstrated_bluffer:
        minimum_number = max(1, minimum_number - 1)
    elif profile.passive and fresh_bet_fraction >= 2 / 3:
        minimum_number = min(13, minimum_number + 1)

    # When the community is below us, one normally beaten number has become a
    # pair.  Tighten one tier only at the boundary rather than over-penalising
    # premium numbers that are well above the fallback threshold.
    if (
        community is not None
        and community < your_number
        and your_number <= minimum_number + 1
    ):
        minimum_number = min(13, minimum_number + 1)

    range_margin = 0.02
    if profile.demonstrated_bluffer:
        range_margin = 0.0
    elif profile.passive:
        range_margin = 0.06
    if to_call > 0.20 * stack:
        range_margin += 0.03

    if your_number >= minimum_number and equity >= pot_odds + range_margin:
        return _fallback(payload, "call")
    return _fallback(payload, "fold")


def _opponent_profile(payload: Mapping[str, Any]) -> OpponentProfile:
    """Build position-aware rates without overreacting to tiny samples."""

    your_seat = payload.get("your_seat")
    current_is_button = your_seat == payload.get("button_seat")
    button_responses = button_folds = button_reraises = 0
    post_responses = post_folds = 0
    checked_to = bets_after_check = 0
    decisions = aggressive_actions = 0
    shown_bets = shown_bluffs = 0

    recent_hands = payload.get("recent_hands")
    if not isinstance(recent_hands, list):
        recent_hands = []

    for hand in recent_hands:
        if not isinstance(hand, Mapping):
            continue
        raw_actions = hand.get("actions")
        if not isinstance(raw_actions, list):
            continue
        actions = [
            action
            for action in raw_actions
            if isinstance(action, Mapping)
            and action.get("action") in _DECISION_ACTIONS
        ]

        pre_actions = [a for a in actions if a.get("round") == "pre_reveal"]
        # The first pre-reveal actor holds the button.  Historical entries do
        # not carry button_seat, so use action order to keep post-reveal reads
        # separated by our current position.
        same_position = bool(pre_actions) and (
            (pre_actions[0].get("seat") == your_seat) == current_is_button
        )

        if same_position:
            for action in actions:
                if action.get("seat") != your_seat:
                    decisions += 1
                    if action.get("action") in {"bet", "raise"}:
                        aggressive_actions += 1

        if (
            pre_actions
            and pre_actions[0].get("seat") == your_seat
            and pre_actions[0].get("action") == "raise"
        ):
            response = _next_action_by_other_seat(pre_actions, 0, your_seat)
            if response is not None and response.get("action") in {"fold", "call", "raise"}:
                button_responses += 1
                if response.get("action") == "fold":
                    button_folds += 1
                elif response.get("action") == "raise":
                    button_reraises += 1

        post_actions = (
            [a for a in actions if a.get("round") == "post_reveal"]
            if same_position
            else []
        )
        for index, action in enumerate(post_actions):
            if action.get("seat") != your_seat:
                continue
            verb = action.get("action")
            response = _next_action_by_other_seat(post_actions, index, your_seat)
            if response is None:
                continue
            if verb in {"bet", "raise"} and response.get("action") in {
                "fold",
                "call",
                "raise",
            }:
                post_responses += 1
                if response.get("action") == "fold":
                    post_folds += 1
            elif verb == "check" and response.get("action") in {"check", "bet"}:
                checked_to += 1
                if response.get("action") == "bet":
                    bets_after_check += 1

        opponent_bet_post = any(
            action.get("seat") != your_seat
            and action.get("action") in {"bet", "raise"}
            for action in post_actions
        )
        opponent_number = _shown_opponent_number(hand, your_seat)
        community = _optional_integer(hand.get("community_number"))
        if opponent_bet_post and opponent_number is not None and community is not None:
            shown_bets += 1
            if opponent_number <= 4 and opponent_number != community:
                shown_bluffs += 1

    return OpponentProfile(
        button_fold_rate=_smoothed_rate(button_folds, button_responses, 0.35),
        button_reraise_rate=_smoothed_rate(button_reraises, button_responses, 0.18),
        post_fold_rate=_smoothed_rate(post_folds, post_responses, 0.35),
        bet_after_check_rate=_smoothed_rate(bets_after_check, checked_to, 0.45),
        aggression_rate=_smoothed_rate(aggressive_actions, decisions, 0.45),
        shown_bluff_rate=_smoothed_rate(shown_bluffs, shown_bets, 0.10),
        button_responses=button_responses,
        button_folds=button_folds,
        button_reraises=button_reraises,
        post_responses=post_responses,
        post_folds=post_folds,
        checked_to=checked_to,
        decisions=decisions,
        shown_bets=shown_bets,
    )


def _smoothed_rate(hits: int, trials: int, prior: float, weight: int = 4) -> float:
    return (hits + prior * weight) / (trials + weight)


def _conservative_rate(hits: int, trials: int) -> float:
    """Return a one-standard-error lower Wilson-style estimate."""

    if trials <= 0:
        return 0.0
    probability = (hits + 1) / (trials + 2)
    error = math.sqrt(probability * (1 - probability) / (trials + 2))
    return max(0.0, probability - error)


def _next_action_by_other_seat(
    actions: list[Mapping[str, Any]], index: int, your_seat: Any
) -> Mapping[str, Any] | None:
    for action in actions[index + 1 :]:
        if action.get("seat") != your_seat:
            return action
    return None


def _shown_opponent_number(hand: Mapping[str, Any], your_seat: Any) -> int | None:
    shown = hand.get("shown_numbers")
    if not isinstance(shown, Mapping):
        return None
    for seat, number in shown.items():
        if str(seat) != str(your_seat):
            parsed = _optional_integer(number)
            if parsed is not None and 1 <= parsed <= 13:
                return parsed
    return None


def _fixed_raise(
    payload: Mapping[str, Any], desired_total: int, *, fallback: str
) -> Action:
    """Raise to a small strategic target without accepting a huge forced size."""

    if "raise" not in _legal_actions(payload):
        return _fallback(payload, fallback)
    minimum = _optional_integer(payload.get("min_raise_to"))
    maximum = _optional_integer(payload.get("max_raise_to"))
    if minimum is None or maximum is None or minimum > maximum:
        return _fallback(payload, fallback)

    big_blind = max(1, _integer(payload.get("big_blind"), 2))
    if minimum > desired_total + 2 * big_blind:
        return _fallback(payload, fallback)

    contribution = _your_round_contribution(payload)
    stack = max(0, _integer(payload.get("your_stack"), 0))
    non_all_in_maximum = contribution + max(0, stack - 1)
    maximum = min(maximum, non_all_in_maximum)
    if maximum < minimum:
        return _fallback(payload, fallback)

    amount = max(minimum, min(maximum, desired_total))
    return {"action": "raise", "amount": amount}


def _pot_wager(
    payload: Mapping[str, Any],
    action: str,
    pot_fraction: float,
    *,
    allow_all_in: bool = False,
    fallback: str,
) -> Action:
    """Create a legal total-round wager from a desired fraction of the pot."""

    if action not in _legal_actions(payload):
        return _fallback(payload, fallback)
    minimum = _optional_integer(payload.get("min_raise_to"))
    maximum = _optional_integer(payload.get("max_raise_to"))
    if minimum is None or maximum is None or minimum > maximum:
        return _fallback(payload, fallback)

    contribution = _your_round_contribution(payload)
    stack = max(0, _integer(payload.get("your_stack"), 0))
    to_call = max(0, _integer(payload.get("to_call"), 0))
    pot = max(1, _integer(payload.get("pot"), 1))
    matched_total = contribution + to_call
    desired_total = matched_total + max(1, round(pot * pot_fraction))

    if not allow_all_in:
        maximum = min(maximum, contribution + max(0, stack - 1))
    if maximum < minimum:
        return _fallback(payload, fallback)

    amount = max(minimum, min(maximum, desired_total))
    return {"action": action, "amount": amount}


def _planned_open_addition(payload: Mapping[str, Any], pot_fraction: float) -> int:
    pot = max(1, _integer(payload.get("pot"), 1))
    desired = max(1, round(pot * pot_fraction))
    minimum = _optional_integer(payload.get("min_raise_to"))
    contribution = _your_round_contribution(payload)
    if minimum is not None:
        desired = max(desired, minimum - contribution)
    return desired


def _fresh_bet_fraction(pot: int, to_call: int) -> float:
    # On a first bet to_call is the new bet and pot already includes it.
    pot_before_bet = max(1, pot - to_call)
    return to_call / pot_before_bet


def _round_actions(
    payload: Mapping[str, Any], round_name: str | None = None
) -> list[Mapping[str, Any]]:
    if round_name is None:
        round_name = str(payload.get("round", ""))
    raw_actions = payload.get("current_hand_actions")
    if not isinstance(raw_actions, list):
        return []
    return [
        action
        for action in raw_actions
        if isinstance(action, Mapping) and action.get("round") == round_name
    ]


def _we_raised_this_round(payload: Mapping[str, Any]) -> bool:
    your_seat = payload.get("your_seat")
    return any(
        action.get("seat") == your_seat
        and action.get("action") in {"bet", "raise"}
        for action in _round_actions(payload)
    )


def _facing_reraise(payload: Mapping[str, Any]) -> bool:
    actions = _round_actions(payload)
    your_seat = payload.get("your_seat")
    if len(actions) < 2:
        return False
    previous, latest = actions[-2:]
    return (
        previous.get("seat") == your_seat
        and previous.get("action") in {"bet", "raise"}
        and latest.get("seat") != your_seat
        and latest.get("action") == "raise"
    )


def _opponent_checked_this_round(payload: Mapping[str, Any]) -> bool:
    your_seat = payload.get("your_seat")
    for action in reversed(_round_actions(payload)):
        if action.get("seat") != your_seat:
            return action.get("action") == "check"
    return False


def _your_round_contribution(payload: Mapping[str, Any]) -> int:
    player = _your_player(payload)
    if player is None:
        return 0
    return max(0, _integer(player.get("bet_this_round"), 0))


def _opponent_round_contribution(payload: Mapping[str, Any]) -> int:
    your_seat = payload.get("your_seat")
    players = payload.get("players")
    if not isinstance(players, list):
        return 0
    for player in players:
        if isinstance(player, Mapping) and player.get("seat") != your_seat:
            return max(0, _integer(player.get("bet_this_round"), 0))
    return 0


def _your_player(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    your_seat = payload.get("your_seat")
    players = payload.get("players")
    if not isinstance(players, list):
        return None
    for player in players:
        if isinstance(player, Mapping) and player.get("seat") == your_seat:
            return player
    for player in players:
        if isinstance(player, Mapping) and player.get("name") == "you":
            return player
    return None


def _fallback(payload: Mapping[str, Any], preferred: str) -> Action:
    legal = _legal_actions(payload)
    if preferred in legal:
        return {"action": preferred}
    if "check" in legal:
        return {"action": "check"}
    if preferred == "call" and "fold" in legal:
        return {"action": "fold"}
    if "fold" in legal:
        return {"action": "fold"}
    if "call" in legal:
        return {"action": "call"}
    return {"action": legal[0]} if legal else {"action": "check"}


def _sanitize(payload: Mapping[str, Any], candidate: Mapping[str, Any]) -> Action:
    """Make illegal action names and wager amounts impossible."""

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
            return {
                "action": action,
                "amount": max(minimum, min(maximum, amount)),
            }

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
    return {"action": legal[0]} if legal else {"action": "check"}


def _roll(payload: Mapping[str, Any], purpose: str) -> float:
    """Stable pseudo-randomness keeps retries deterministic."""

    actions = payload.get("current_hand_actions")
    action_count = len(actions) if isinstance(actions, list) else 0
    key = "|".join(
        str(value)
        for value in (
            payload.get("match_id", ""),
            payload.get("hand_number", ""),
            payload.get("round", ""),
            payload.get("your_seat", ""),
            action_count,
            purpose,
        )
    )
    value = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
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
    except (TypeError, ValueError, OverflowError):
        return None
