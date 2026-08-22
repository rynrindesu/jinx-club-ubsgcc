"""Confidence-aware Phase 2 policy for opaque SHOWDOWN table rules."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from .rules import EquityEstimate
from .state import OpponentProfile, Phase2State


Action = dict[str, Any]


@dataclass(frozen=True)
class RiskContext:
    delta: int
    hands_remaining: int
    secure_current_hand: bool
    protect: bool
    desperate: bool


class Phase2Engine:
    """Combine state observation, rule inference, and legal action selection."""

    def __init__(self, state: Phase2State | None = None) -> None:
        self.state = state or Phase2State()

    def decide(self, payload: Mapping[str, Any]) -> Action:
        legal = _legal_actions(payload)
        if not legal:
            return {"action": "check"}

        knowledge, profile = self.state.observe_payload(payload)
        your_number = _integer(payload.get("your_number"), 7)
        raw_community = payload.get("community_number")
        community = None if raw_community is None else _integer(raw_community, 7)
        if not 1 <= your_number <= 13:
            your_number = 7
        if community is not None and not 1 <= community <= 13:
            community = None

        estimate = knowledge.estimate(your_number, community)
        risk = _risk_context(payload)
        if risk.secure_current_hand:
            candidate = _fallback(payload, "check" if "check" in legal else "fold")
        elif payload.get("round") == "post_reveal" or community is not None:
            candidate = _post_reveal_move(payload, estimate, profile, risk)
        else:
            candidate = _pre_reveal_move(payload, estimate, profile, risk)
        return _sanitize(payload, candidate)

    def reset(self) -> None:
        self.state.reset()


_ENGINE = Phase2Engine()


def decide_move(payload: Mapping[str, Any]) -> Action:
    """Choose one legal Phase 2 move using process-persistent rule knowledge."""

    return _ENGINE.decide(payload)


def reset_state() -> None:
    """Reset the module engine; intended for tests and deliberate recalibration."""

    _ENGINE.reset()


def _pre_reveal_move(
    payload: Mapping[str, Any],
    estimate: EquityEstimate,
    profile: OpponentProfile,
    risk: RiskContext,
) -> Action:
    to_call = max(0, _integer(payload.get("to_call"), 0))
    stack = max(0, _integer(payload.get("your_stack"), 0))
    pot = max(0, _integer(payload.get("pot"), 0))
    big_blind = max(1, _integer(payload.get("big_blind"), 2))
    is_button = payload.get("your_seat") == payload.get("button_seat")
    actions = _round_actions(payload, "pre_reveal")

    # First contact with a codename is an information-gathering phase.  Limping
    # or checking reaches the reveal cheaply without assuming that high, low,
    # or a pair has any special value.
    if estimate.confidence == "unknown":
        if to_call == 0:
            return _fallback(payload, "check")
        if is_button and not actions and "call" in _legal_actions(payload):
            return {"action": "call"}
        if (
            _cheap_calibration_allowed(payload, to_call, big_blind)
            and to_call < stack
            and not risk.protect
        ):
            return _fallback(payload, "call")
        return _fallback(payload, "fold")

    # A partially identified rule must never recreate the calibration
    # attempt's 23-29 chip losses. Information has value up to a tightly capped
    # blind-plus-small-blind continuation; large bets wait for a trusted rule.
    if (
        estimate.confidence == "partial"
        and to_call > 0
        and not _partial_exposure_allowed(
            payload=payload,
            to_call=to_call,
            pot=pot,
            big_blind=big_blind,
            estimate=estimate,
        )
    ):
        return _fallback(payload, "fold")

    decision_consensus = _decision_consensus(estimate)
    if estimate.confidence == "partial":
        margin = 0.055 if decision_consensus else 0.08
    else:
        margin = 0.045
    if profile.passive:
        margin += 0.025
    if profile.aggressive:
        margin -= 0.02
    if risk.protect:
        margin += 0.10
    if risk.desperate:
        margin -= 0.035

    if estimate.confidence == "partial" and decision_consensus:
        conservative = estimate.mean - 0.15 * estimate.disagreement
    elif estimate.confidence == "partial":
        conservative = estimate.lower
    else:
        conservative = estimate.mean - 0.20 * estimate.disagreement

    if is_button and not actions:
        open_threshold = 0.50
        if profile.tight_folder:
            open_threshold -= 0.06
        if profile.calling_station:
            open_threshold += 0.04
        if risk.protect:
            open_threshold += 0.09
        if risk.desperate:
            open_threshold -= 0.04

        if conservative >= open_threshold:
            target = 5 if profile.calling_station else 4
            return _fixed_wager(payload, target, fallback="call")
        if (
            "call" in _legal_actions(payload)
            and estimate.mean >= open_threshold - 0.12
            and not risk.protect
        ):
            return {"action": "call"}
        return _fallback(payload, "fold")

    if to_call == 0:
        if conservative >= 0.67 + margin and not risk.protect:
            return _fixed_wager(payload, 5, fallback="check")
        return _fallback(payload, "check")

    pot_odds = to_call / max(1, pot + to_call)
    facing_reraise = _we_wagered_this_round(payload)
    reraise_floor = 0.90 if risk.protect else 0.80
    required = max(
        pot_odds + margin,
        reraise_floor if facing_reraise else 0.45,
    )
    if not _call_exposure_allowed(
        to_call=to_call,
        stack=stack,
        big_blind=big_blind,
        conservative=conservative,
        estimate=estimate,
        risk=risk,
    ):
        return _fallback(payload, "fold")
    if conservative >= required and to_call <= stack:
        if (
            conservative >= 0.76
            and not facing_reraise
            and not risk.protect
            and "raise" in _legal_actions(payload)
        ):
            return _pot_wager(
                payload,
                "raise",
                0.45,
                hand_exposure_cap=(
                    _calibration_exposure_cap(payload, big_blind)
                    if estimate.confidence == "partial"
                    else None
                ),
                allow_all_in=(
                    risk.desperate
                    and estimate.confidence == "learned"
                    and conservative >= 0.90
                ),
                fallback="call",
            )
        return _fallback(payload, "call")

    # A tiny call in partial mode buys a deterministic rule comparison.  It is
    # allowed only when average equity covers the literal pot price.
    if (
        estimate.confidence == "partial"
        and not risk.protect
        and to_call <= big_blind
        and estimate.mean >= pot_odds
        and to_call < stack
    ):
        return _fallback(payload, "call")
    return _fallback(payload, "fold")


def _post_reveal_move(
    payload: Mapping[str, Any],
    estimate: EquityEstimate,
    profile: OpponentProfile,
    risk: RiskContext,
) -> Action:
    to_call = max(0, _integer(payload.get("to_call"), 0))
    stack = max(0, _integer(payload.get("your_stack"), 0))
    pot = max(0, _integer(payload.get("pot"), 0))
    big_blind = max(1, _integer(payload.get("big_blind"), 2))

    if estimate.confidence == "unknown":
        if to_call == 0:
            return _fallback(payload, "check")
        # Calling normally closes heads-up action and exposes both numbers.
        # A three-chip continuation is still cheap enough to prevent an
        # opponent from auto-profiting every time we limp or check.
        if (
            _cheap_calibration_allowed(payload, to_call, big_blind)
            and to_call < stack
            and not risk.protect
        ):
            return _fallback(payload, "call")
        return _fallback(payload, "fold")

    if (
        estimate.confidence == "partial"
        and to_call > 0
        and not _partial_exposure_allowed(
            payload=payload,
            to_call=to_call,
            pot=pot,
            big_blind=big_blind,
            estimate=estimate,
        )
    ):
        return _fallback(payload, "fold")

    decision_consensus = _decision_consensus(estimate)
    if estimate.confidence == "partial" and decision_consensus:
        conservative = estimate.mean - 0.10 * estimate.disagreement
    elif estimate.confidence == "partial":
        conservative = estimate.lower
    else:
        conservative = estimate.mean - 0.15 * estimate.disagreement
    adjustment = 0.0
    if profile.aggressive:
        adjustment -= 0.035
    if profile.passive:
        adjustment += 0.04
    if risk.protect:
        adjustment += 0.12
    if risk.desperate:
        adjustment -= 0.04

    if to_call > 0:
        pot_odds = to_call / max(1, pot + to_call)
        fresh_fraction = to_call / max(1, pot - to_call)
        range_floor = 0.43
        if fresh_fraction > 0.35:
            range_floor = 0.64
        if fresh_fraction > 0.55:
            range_floor = 0.78
        if fresh_fraction > 1.05:
            range_floor = 0.88
        if _we_wagered_this_round(payload) and fresh_fraction > 0.35:
            range_floor = max(range_floor, 0.80)
        if risk.protect and fresh_fraction > 0.35:
            range_floor = max(range_floor, 0.86)
        required = max(
            pot_odds + 0.055 + adjustment,
            range_floor + max(0.0, adjustment),
        )

        if not _call_exposure_allowed(
            to_call=to_call,
            stack=stack,
            big_blind=big_blind,
            conservative=conservative,
            estimate=estimate,
            risk=risk,
        ):
            return _fallback(payload, "fold")

        if conservative >= required:
            if (
                "raise" in _legal_actions(payload)
                and conservative >= 0.90 + max(0.0, adjustment)
                and estimate.confidence == "learned"
                and not risk.protect
            ):
                return _pot_wager(
                    payload,
                    "raise",
                    0.55,
                    allow_all_in=risk.desperate and conservative >= 0.94,
                    fallback="call",
                )
            return _fallback(payload, "call")

        if (
            estimate.confidence == "partial"
            and not risk.protect
            and to_call <= big_blind
            and estimate.mean >= pot_odds + 0.02
            and to_call < stack
        ):
            return _fallback(payload, "call")
        return _fallback(payload, "fold")

    wager_action = _wager_action(payload)
    if wager_action is None:
        return _fallback(payload, "check")

    value_threshold = 0.61 + adjustment
    if profile.calling_station:
        value_threshold += 0.025
    if conservative >= value_threshold:
        fraction = 0.65 if profile.calling_station else 0.50
        if estimate.confidence == "partial" and not decision_consensus:
            fraction = 1 / 3
        if (
            risk.desperate
            and estimate.confidence == "learned"
            and conservative >= 0.72
        ):
            fraction = 0.75
        return _pot_wager(
            payload,
            wager_action,
            fraction,
            hand_exposure_cap=(
                _calibration_exposure_cap(payload, big_blind)
                if estimate.confidence == "partial"
                else None
            ),
            allow_all_in=(
                risk.desperate
                and estimate.confidence == "learned"
                and conservative >= 0.94
            ),
            fallback="check",
        )

    # Use only learned, very weak hands as occasional bluffs, and only after
    # the opponent has shown a profitable fold tendency.  Strength is an
    # inferred percentile here; raw private numbers never enter the test.
    if (
        estimate.confidence == "learned"
        and estimate.mean <= 0.20
        and profile.post_responses >= 4
        and profile.post_fold_rate > 0.48
        and not risk.protect
        and _opponent_checked_this_round(payload)
    ):
        addition = _planned_wager_addition(payload, 1 / 3)
        break_even = addition / max(1, pot + addition)
        excess = profile.post_fold_rate - break_even
        frequency = min(0.25, max(0.0, excess - 0.05))
        if _roll(payload, "phase2-post-bluff") < frequency:
            return _pot_wager(
                payload, wager_action, 1 / 3, fallback="check"
            )
    return _fallback(payload, "check")


def _risk_context(payload: Mapping[str, Any]) -> RiskContext:
    delta = _your_chip_delta(payload)
    hand_number = max(1, _integer(payload.get("hand_number"), 1))
    total_hands = max(hand_number, _integer(payload.get("total_hands"), hand_number))
    remaining = total_hands - hand_number + 1
    player = _your_player(payload)
    stack = max(0, _integer(player.get("stack"), 0)) if player else max(
        0, _integer(payload.get("your_stack"), 0)
    )
    starting = max(1, _integer(payload.get("starting_stack"), 200))
    committed = max(0, starting + delta - stack)
    future_blinds = _future_forced_bets(payload, remaining - 1)
    coast_floor = delta - committed - future_blinds
    secure_current = coast_floor >= 25
    protect = (
        secure_current
        or delta >= 30
        or (remaining <= 12 and delta >= 25)
    )
    desperate = (
        (remaining <= 6 and delta < 20)
        or (remaining <= 3 and delta < 24)
    ) and not protect
    return RiskContext(
        delta=delta,
        hands_remaining=remaining,
        secure_current_hand=secure_current,
        protect=protect,
        desperate=desperate,
    )


def _partial_exposure_allowed(
    *,
    payload: Mapping[str, Any],
    to_call: int,
    pot: int,
    big_blind: int,
    estimate: EquityEstimate,
) -> bool:
    """Permit only cheap calibration calls before a rule is trusted."""

    if not _within_calibration_cap(payload, to_call, big_blind):
        return False
    if _cheap_calibration_allowed(payload, to_call, big_blind):
        return True
    pot_before_wager = max(1, pot - to_call)
    if to_call <= 2 * big_blind and to_call / pot_before_wager <= 0.20:
        return True
    return (
        to_call <= 4 * big_blind
        and estimate.lower >= 0.90
        and estimate.coverage >= 0.95
    )


def _cheap_calibration_allowed(
    payload: Mapping[str, Any], to_call: int, big_blind: int
) -> bool:
    """Allow a blind-plus-small-blind call within the eight-chip hand cap."""

    small_blind = max(1, _integer(payload.get("small_blind"), 1))
    return (
        to_call <= big_blind + small_blind
        and _within_calibration_cap(payload, to_call, big_blind)
    )


def _within_calibration_cap(
    payload: Mapping[str, Any], to_call: int, big_blind: int
) -> bool:
    exposure_cap = _calibration_exposure_cap(payload, big_blind)
    return _current_hand_commitment(payload) + to_call <= exposure_cap


def _calibration_exposure_cap(
    payload: Mapping[str, Any], big_blind: int
) -> int:
    starting_stack = max(1, _integer(payload.get("starting_stack"), 200))
    return max(2 * big_blind, round(0.04 * starting_stack))


def _decision_consensus(estimate: EquityEstimate) -> bool:
    """Trust a partial model only when live candidates agree on this hand."""

    return (
        estimate.confidence == "learned"
        or (
            estimate.confidence == "partial"
            and estimate.candidate_count > 0
            and estimate.disagreement <= 0.10
        )
    )


def _call_exposure_allowed(
    *,
    to_call: int,
    stack: int,
    big_blind: int,
    conservative: float,
    estimate: EquityEstimate,
    risk: RiskContext,
) -> bool:
    """Reject stack-threatening calls unless the rule proves near-nut equity."""

    if to_call <= 0:
        return True
    proven_nuts = estimate.confidence == "learned" and conservative >= 0.94
    if to_call >= stack:
        return proven_nuts
    if to_call / max(1, stack) >= 0.35 and conservative < 0.90:
        return False
    score_cushion = max(big_blind, risk.delta - 25)
    if risk.delta >= 25 and to_call > score_cushion and not proven_nuts:
        return False
    return True


def _current_hand_commitment(payload: Mapping[str, Any]) -> int:
    starting = max(1, _integer(payload.get("starting_stack"), 200))
    delta = _your_chip_delta(payload)
    stack = max(0, _integer(payload.get("your_stack"), 0))
    return max(0, starting + delta - stack)


def _future_forced_bets(payload: Mapping[str, Any], future_hands: int) -> int:
    """Exact heads-up blind cost if every future bet is checked or folded."""

    if future_hands <= 0:
        return 0
    small = max(0, _integer(payload.get("small_blind"), 1))
    big = max(small, _integer(payload.get("big_blind"), 2))
    currently_button = str(payload.get("your_seat")) == str(
        payload.get("button_seat")
    )
    cost = 0
    for offset in range(1, future_hands + 1):
        future_button = currently_button if offset % 2 == 0 else not currently_button
        cost += small if future_button else big
    return cost


def _your_chip_delta(payload: Mapping[str, Any]) -> int:
    player = _your_player(payload)
    if player is not None and _optional_integer(player.get("chip_delta")) is not None:
        return _integer(player.get("chip_delta"), 0)
    return _integer(payload.get("chip_delta"), 0)


def _fixed_wager(
    payload: Mapping[str, Any], desired_total: int, *, fallback: str
) -> Action:
    action = _wager_action(payload)
    if action is None:
        return _fallback(payload, fallback)
    minimum = _optional_integer(payload.get("min_raise_to"))
    maximum = _optional_integer(payload.get("max_raise_to"))
    if minimum is None or maximum is None or minimum > maximum:
        return _fallback(payload, fallback)
    if minimum > desired_total + 2 * max(1, _integer(payload.get("big_blind"), 2)):
        return _fallback(payload, fallback)
    maximum = _non_all_in_maximum(payload, maximum)
    if maximum < minimum:
        return _fallback(payload, fallback)
    return {"action": action, "amount": max(minimum, min(maximum, desired_total))}


def _pot_wager(
    payload: Mapping[str, Any],
    action: str,
    fraction: float,
    *,
    hand_exposure_cap: int | None = None,
    allow_all_in: bool = False,
    fallback: str,
) -> Action:
    if action not in _legal_actions(payload):
        return _fallback(payload, fallback)
    minimum = _optional_integer(payload.get("min_raise_to"))
    maximum = _optional_integer(payload.get("max_raise_to"))
    if minimum is None or maximum is None or minimum > maximum:
        return _fallback(payload, fallback)

    contribution = _your_round_contribution(payload)
    to_call = max(0, _integer(payload.get("to_call"), 0))
    pot = max(1, _integer(payload.get("pot"), 1))
    desired = contribution + to_call + max(1, round(pot * fraction))
    if not allow_all_in:
        maximum = _non_all_in_maximum(payload, maximum)
    if hand_exposure_cap is not None:
        remaining_exposure = max(
            0, hand_exposure_cap - _current_hand_commitment(payload)
        )
        maximum = min(maximum, contribution + remaining_exposure)
    if maximum < minimum:
        return _fallback(payload, fallback)
    return {"action": action, "amount": max(minimum, min(maximum, desired))}


def _non_all_in_maximum(payload: Mapping[str, Any], maximum: int) -> int:
    contribution = _your_round_contribution(payload)
    stack = max(0, _integer(payload.get("your_stack"), 0))
    return min(maximum, contribution + max(0, stack - 1))


def _planned_wager_addition(payload: Mapping[str, Any], fraction: float) -> int:
    pot = max(1, _integer(payload.get("pot"), 1))
    desired = max(1, round(pot * fraction))
    minimum = _optional_integer(payload.get("min_raise_to"))
    if minimum is not None:
        desired = max(desired, minimum - _your_round_contribution(payload))
    return desired


def _wager_action(payload: Mapping[str, Any]) -> str | None:
    legal = _legal_actions(payload)
    if "bet" in legal:
        return "bet"
    if "raise" in legal:
        return "raise"
    return None


def _round_actions(
    payload: Mapping[str, Any], round_name: str | None = None
) -> list[Mapping[str, Any]]:
    if round_name is None:
        round_name = str(payload.get("round", ""))
    actions = payload.get("current_hand_actions")
    if not isinstance(actions, list):
        return []
    return [
        action
        for action in actions
        if isinstance(action, Mapping) and action.get("round") == round_name
    ]


def _we_wagered_this_round(payload: Mapping[str, Any]) -> bool:
    your_seat = payload.get("your_seat")
    return any(
        str(action.get("seat")) == str(your_seat)
        and action.get("action") in {"bet", "raise"}
        for action in _round_actions(payload)
    )


def _opponent_checked_this_round(payload: Mapping[str, Any]) -> bool:
    your_seat = payload.get("your_seat")
    for action in reversed(_round_actions(payload)):
        if str(action.get("seat")) != str(your_seat):
            return action.get("action") == "check"
    return False


def _your_round_contribution(payload: Mapping[str, Any]) -> int:
    player = _your_player(payload)
    return max(0, _integer(player.get("bet_this_round"), 0)) if player else 0


def _your_player(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    your_seat = payload.get("your_seat")
    players = payload.get("players")
    if not isinstance(players, list):
        return None
    for player in players:
        if isinstance(player, Mapping) and str(player.get("seat")) == str(your_seat):
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
    if "fold" in legal:
        return {"action": "fold"}
    if "call" in legal:
        return {"action": "call"}
    return {"action": legal[0]} if legal else {"action": "check"}


def _sanitize(payload: Mapping[str, Any], candidate: Mapping[str, Any]) -> Action:
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
    for wager in ("bet", "raise"):
        if wager in legal:
            minimum = _optional_integer(payload.get("min_raise_to"))
            if minimum is not None:
                return {"action": wager, "amount": minimum}
    return {"action": legal[0]} if legal else {"action": "check"}


def _roll(payload: Mapping[str, Any], purpose: str) -> float:
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
    legal = payload.get("legal_actions")
    if not isinstance(legal, list):
        return []
    return [action for action in legal if isinstance(action, str)]


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
