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
    future_blinds: int
    coast_floor: int
    target_gap: int
    secure_current_hand: bool
    tier: str
    call_caution: float
    open_adjustment: float
    value_adjustment: float
    continuation_floor: float

    @property
    def chasing(self) -> bool:
        return self.tier == "chase"

    @property
    def guarded(self) -> bool:
        return self.tier == "guarded"


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

        opponent_range = profile.range_for(
            payload=payload,
            rule_knowledge=knowledge,
        )
        estimate = knowledge.estimate(
            your_number,
            community,
            opponent_range=opponent_range,
        )
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
        if (
            is_button
            and not actions
            and "call" in _legal_actions(payload)
        ):
            return {"action": "call"}
        if (
            (
                _cheap_calibration_allowed(payload, to_call, big_blind)
                or _unknown_medium_probe(payload, to_call, big_blind)
            )
            and to_call < stack
            and not risk.guarded
            and (not profile.aggressive or to_call <= big_blind)
        ):
            return _fallback(payload, "call")
        return _fallback(payload, "fold")

    # Partial means this specific decision is not yet resolved.  Free reveals
    # are always useful, but a mature partial model must not pay a re-raise
    # merely because it fits under a fixed cap.  That created a deterministic
    # limp/call/fold pipeline against the Phase 2 opponent.
    if estimate.confidence == "partial":
        if to_call == 0:
            return _fallback(payload, "check")
        if (
            to_call < stack
            and _partial_call_allowed(
                payload=payload,
                to_call=to_call,
                pot=pot,
                big_blind=big_blind,
                estimate=estimate,
                profile=profile,
                risk=risk,
                post_reveal=False,
            )
        ):
            return _fallback(payload, "call")
        return _fallback(payload, "fold")

    margin = 0.045
    if profile.passive:
        margin += 0.025
    if profile.aggressive:
        # Raw aggression is pressure, not evidence of bluffing.  Any explicit
        # weak range learned from shown hands is already reflected in the
        # equity estimate, so the generic adjustment must remain cautious.
        margin += 0.025
    margin += risk.call_caution

    conservative = estimate.mean - 0.20 * estimate.disagreement

    if is_button and not actions:
        open_threshold = 0.34
        if profile.tight_folder:
            open_threshold = 0.22
        elif profile.calling_station:
            open_threshold = 0.42
        open_threshold += risk.open_adjustment

        # Once this opponent has punished an open, only open hands that also
        # clear the continuation threshold.  Against general aggression use a
        # smaller commitment premium so medium hands limp instead of creating
        # another raise-to-four/fold line.
        commitment_threshold = open_threshold
        if profile.punishes_opens:
            commitment_threshold = max(commitment_threshold, 0.80)
        elif profile.aggressive:
            commitment_threshold = max(commitment_threshold, 0.62)
        if risk.continuation_floor > 0:
            commitment_threshold = max(
                commitment_threshold, risk.continuation_floor
            )

        if conservative >= commitment_threshold:
            target = 5 if profile.calling_station else 4
            return _fixed_wager(payload, target, fallback="call")
        if (
            "call" in _legal_actions(payload)
            and estimate.mean >= 0.24
        ):
            return {"action": "call"}
        return _fallback(payload, "fold")

    if to_call == 0:
        free_raise_threshold = 0.67 + margin
        if risk.continuation_floor > 0:
            free_raise_threshold = max(
                free_raise_threshold, risk.continuation_floor
            )
        if conservative >= free_raise_threshold:
            return _fixed_wager(payload, 5, fallback="check")
        return _fallback(payload, "check")

    pot_odds = to_call / max(1, pot + to_call)
    facing_reraise = _we_wagered_this_round(payload)
    reraise_floor = max(0.80, risk.continuation_floor)
    required = max(
        pot_odds + margin,
        reraise_floor if facing_reraise else 0.45,
    )
    if not facing_reraise:
        required = max(
            required,
            _planned_two_street_requirement(
                payload=payload,
                to_call=to_call,
                pot=pot,
                stack=stack,
                profile=profile,
                risk=risk,
            ),
        )
    if not _call_exposure_allowed(
        to_call=to_call,
        stack=stack,
        conservative=conservative,
        estimate=estimate,
        risk=risk,
        required_equity=required,
    ):
        return _fallback(payload, "fold")
    # The observed opponent four-bet every exploratory three-bet, turning
    # raise-to-8/fold lines into the largest repeated pre-reveal leak.  Realize
    # learned equity by calling; value raises happen after reveal.
    return _fallback(payload, "call")


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
            (
                _cheap_calibration_allowed(payload, to_call, big_blind)
                or _unknown_medium_probe(payload, to_call, big_blind)
            )
            and to_call < stack
            and not risk.guarded
            and (not profile.aggressive or to_call <= big_blind)
        ):
            return _fallback(payload, "call")
        return _fallback(payload, "fold")

    if estimate.confidence == "partial":
        if to_call == 0:
            return _fallback(payload, "check")
        if (
            to_call < stack
            and _partial_call_allowed(
                payload=payload,
                to_call=to_call,
                pot=pot,
                big_blind=big_blind,
                estimate=estimate,
                profile=profile,
                risk=risk,
                post_reveal=True,
            )
        ):
            return _fallback(payload, "call")
        return _fallback(payload, "fold")

    conservative = estimate.mean - 0.15 * estimate.disagreement
    call_adjustment = risk.call_caution
    if profile.aggressive:
        # Aggression alone is not a bluff posterior.  A demonstrated weak
        # betting range raises `estimate.mean`; absent that evidence, require
        # more equity rather than rewarding the opponent's pressure.
        call_adjustment += 0.035
    if profile.passive:
        call_adjustment += 0.04

    if to_call > 0:
        pot_odds = to_call / max(1, pot + to_call)
        fresh_fraction = to_call / max(1, pot - to_call)
        wagered_this_round = _we_wagered_this_round(payload)
        required = pot_odds + 0.03 + call_adjustment
        if fresh_fraction > 1.0:
            required = max(required, 0.72)
        if wagered_this_round:
            reopen_floor = 0.78
            if (
                pot_odds <= 0.35
                and estimate.confidence == "learned"
                and estimate.candidate_count == 1
                and risk.tier in {"normal", "press", "chase"}
                and _we_bet_without_raising_this_round(payload)
            ):
                # The opponent's raise is less conclusive when we were the
                # sole pre-reveal aggressor: with an exact rule and a cheap
                # closing price, continue with majority equity.  When the
                # opponent also raised pre-reveal, retain the stronger floor
                # because both streets consistently represent strength.
                reopen_floor = (
                    0.55
                    if _we_were_sole_pre_reveal_raiser(payload)
                    else 0.70
                )
            required = max(
                required,
                reopen_floor,
                risk.continuation_floor,
            )

        if not _call_exposure_allowed(
            to_call=to_call,
            stack=stack,
            conservative=conservative,
            estimate=estimate,
            risk=risk,
            required_equity=required,
        ):
            return _fallback(payload, "fold")

        # Make at most one voluntary value wager per post-reveal round.  Once
        # the opponent reopens the action, realize our equity instead of
        # escalating into a raise war and later folding the closing price.
        if (
            conservative >= required
            and not wagered_this_round
            and "raise" in _legal_actions(payload)
            and conservative >= 0.89
            and estimate.confidence == "learned"
            and conservative >= risk.continuation_floor
        ):
            return _pot_wager(
                payload,
                "raise",
                0.55,
                allow_all_in=risk.chasing and conservative >= 0.94,
                fallback="call",
            )
        return _fallback(payload, "call")

    wager_action = _wager_action(payload)
    if wager_action is None:
        return _fallback(payload, "check")

    value_threshold = 0.59
    if profile.calling_station:
        value_threshold -= 0.05
    elif profile.passive:
        value_threshold -= 0.02
    value_threshold += risk.value_adjustment
    # A value bet is only coherent if the same hand can continue against the
    # response this opponent has already shown.  Checking retains all equity
    # and removes the repeated bet-five/fold-to-twenty-one loss.
    if profile.punishes_post_bets:
        value_threshold = max(value_threshold, 0.78)
    elif profile.aggressive:
        value_threshold = max(value_threshold, 0.68)
    if risk.continuation_floor > 0:
        value_threshold = max(value_threshold, risk.continuation_floor)
    if conservative >= value_threshold:
        fraction = 0.65 if profile.calling_station else 0.50
        if (
            risk.chasing
            and conservative >= 0.72
        ):
            fraction = 0.75
        return _pot_wager(
            payload,
            wager_action,
            fraction,
            allow_all_in=(
                risk.chasing
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
        and not risk.guarded
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
    target_gap = max(0, 25 - coast_floor)

    # The threshold creates four useful non-terminal states.  A late lead is
    # guarded, not frozen: strong hands still play when future blinds mean that
    # folding everything cannot score.  A deficit is pressed or chased through
    # response-consistent value lines; calls never become looser merely because
    # the clock is short.
    if secure_current:
        tier = "coast"
    elif remaining <= 12 and delta >= 25:
        tier = "guarded"
    elif remaining <= 8 and target_gap >= 2 * remaining:
        tier = "chase"
    elif remaining <= 8 and target_gap > 0:
        tier = "press"
    else:
        tier = "normal"

    call_caution = {
        "coast": 1.0,
        "guarded": 0.04,
        "normal": 0.0,
        "press": 0.01,
        "chase": 0.02,
    }[tier]
    open_adjustment = {
        "coast": 1.0,
        "guarded": 0.04,
        "normal": 0.0,
        "press": -0.02,
        "chase": -0.04,
    }[tier]
    value_adjustment = open_adjustment
    continuation_floor = {
        "coast": 1.0,
        # Scouted higher-family card 12 has a robust pre-reveal estimate just
        # above 0.82.  This keeps that response-ready open while filtering the
        # middling hands that created protection folds on the next action.
        "guarded": 0.82,
        "normal": 0.0,
        "press": 0.72,
        "chase": 0.68,
    }[tier]
    return RiskContext(
        delta=delta,
        hands_remaining=remaining,
        future_blinds=future_blinds,
        coast_floor=coast_floor,
        target_gap=target_gap,
        secure_current_hand=secure_current,
        tier=tier,
        call_caution=call_caution,
        open_adjustment=open_adjustment,
        value_adjustment=value_adjustment,
        continuation_floor=continuation_floor,
    )


def _partial_call_allowed(
    *,
    payload: Mapping[str, Any],
    to_call: int,
    pot: int,
    big_blind: int,
    estimate: EquityEstimate,
    profile: OpponentProfile,
    risk: RiskContext,
    post_reveal: bool,
) -> bool:
    """Buy partial-rule showdowns only when discovery or equity pays for it."""

    discovery = estimate.observation_count < 8
    exposure_cap = _calibration_exposure_cap(payload, big_blind)
    if discovery or post_reveal:
        exposure_cap = max(exposure_cap, 6 * big_blind)
    if _current_hand_commitment(payload) + to_call > exposure_cap:
        return False

    pot_odds = to_call / max(1, pot + to_call)
    caution = risk.call_caution + (0.03 if profile.aggressive else 0.0)
    if discovery:
        # Early evidence has genuine information value, but even then do not
        # pay when every live rule says the hand is already drawing dead.
        information_credit = 0.10 if post_reveal else 0.06
        return estimate.upper + information_credit >= pot_odds + caution

    disagreement_penalty = 0.20 if post_reveal else 0.25
    conservative = estimate.mean - disagreement_penalty * estimate.disagreement
    if post_reveal:
        return conservative >= pot_odds + 0.04 + caution

    # A pre-reveal call can still face another betting decision, so require a
    # real strength edge rather than paying only to observe the community.
    return conservative >= max(0.58 + caution, pot_odds + 0.10 + caution)


def _cheap_calibration_allowed(
    payload: Mapping[str, Any], to_call: int, big_blind: int
) -> bool:
    """Allow a blind-plus-small-blind call within the eight-chip hand cap."""

    small_blind = max(1, _integer(payload.get("small_blind"), 1))
    return (
        to_call <= big_blind + small_blind
        and _within_calibration_cap(payload, to_call, big_blind)
    )


def _unknown_medium_probe(
    payload: Mapping[str, Any], to_call: int, big_blind: int
) -> bool:
    """Occasionally defeat a four-to-six-chip anti-calibration stab."""

    return (
        to_call <= 3 * big_blind
        and _within_calibration_cap(payload, to_call, big_blind)
        and _roll(payload, "unknown-rule-probe") < 0.12
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


def _planned_two_street_requirement(
    *,
    payload: Mapping[str, Any],
    to_call: int,
    pot: int,
    stack: int,
    profile: OpponentProfile,
    risk: RiskContext,
) -> float:
    """Price a recurring pre-raise plus post-reveal continuation as one line.

    The Phase 2 opponent repeatedly raises to five and then bets seven.  Looking
    only at the first three-chip call makes middling hands appear cheap even
    though they predictably face another seven chips after the reveal.  Treat
    an opening total of at least five as planning to continue for roughly one
    more big blind above that total; strong hands enter, marginal hands do not.
    """

    if payload.get("round") != "pre_reveal" or _we_wagered_this_round(payload):
        return 0.0
    raise_total = _latest_opponent_raise_total(payload)
    big_blind = max(1, _integer(payload.get("big_blind"), 2))
    if raise_total is None or raise_total < 2 * big_blind + 1:
        return 0.0

    remaining_after_call = max(0, stack - to_call)
    if remaining_after_call == 0:
        # An all-in call closes our betting.  The caller's ordinary pot-odds
        # requirement already prices it; there is no second street to reserve.
        return 0.0
    planned_post_call = min(
        raise_total + big_blind,
        remaining_after_call,
    )
    planned_cost = to_call + planned_post_call
    final_pot = pot + to_call + 2 * planned_post_call
    planned_odds = planned_cost / max(1, final_pot)

    # The extra ten points account for entering a range that will face a
    # strength-conditioned second barrel, not a fresh uniformly random hand.
    entry_floor = 0.60 + risk.call_caution
    if profile.aggressive:
        entry_floor += 0.04
    elif profile.passive:
        entry_floor -= 0.03
    return max(entry_floor, planned_odds + 0.10 + risk.call_caution)


def _call_exposure_allowed(
    *,
    to_call: int,
    stack: int,
    conservative: float,
    estimate: EquityEstimate,
    risk: RiskContext,
    required_equity: float,
) -> bool:
    """Apply one incremental-price requirement to every learned-hand call."""

    if to_call <= 0:
        return True
    if stack <= 0 or to_call > stack:
        return False
    if estimate.confidence == "learned" and estimate.cannot_lose:
        return True

    # A guarded lead is not yet a coast: future blinds can still consume it.
    # Permit a strong response within the chips needed to close the target gap,
    # and demand progressively more certainty only beyond that gap.
    if risk.guarded and to_call > max(2, risk.target_gap):
        excess = to_call - risk.target_gap
        guarded_requirement = min(
            0.94,
            risk.continuation_floor + 0.01 * excess,
        )
        required_equity = max(required_equity, guarded_requirement)
    return conservative >= required_equity


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


def _we_bet_without_raising_this_round(payload: Mapping[str, Any]) -> bool:
    """True only for a single hero bet that the opponent has reopened."""

    your_seat = payload.get("your_seat")
    hero_wagers = [
        action.get("action")
        for action in _round_actions(payload)
        if str(action.get("seat")) == str(your_seat)
        and action.get("action") in {"bet", "raise"}
    ]
    return hero_wagers == ["bet"]


def _we_were_sole_pre_reveal_raiser(payload: Mapping[str, Any]) -> bool:
    """Whether hero made every raise before the reveal."""

    your_seat = payload.get("your_seat")
    raisers = [
        action.get("seat")
        for action in _round_actions(payload, "pre_reveal")
        if action.get("action") == "raise"
    ]
    return bool(raisers) and all(
        str(seat) == str(your_seat) for seat in raisers
    )


def _latest_opponent_raise_total(payload: Mapping[str, Any]) -> int | None:
    your_seat = payload.get("your_seat")
    for action in reversed(_round_actions(payload, "pre_reveal")):
        if (
            str(action.get("seat")) != str(your_seat)
            and action.get("action") == "raise"
        ):
            amount = _optional_integer(action.get("amount"))
            if amount is not None and amount >= 0:
                return amount
    return None


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
