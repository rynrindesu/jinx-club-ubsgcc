"""Leaderboard-aware, high-variance decision policy for six-seat SHOWDOWN."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
import time
from typing import Any, Iterable

from .equity import showdown_shares_by_subset


@dataclass(frozen=True, slots=True)
class Candidate:
    action: str
    amount: int | None = None

    def response(self) -> dict[str, str | int]:
        result: dict[str, str | int] = {"action": self.action}
        if self.amount is not None:
            result["amount"] = self.amount
        return result


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    scout_hands: int = 10
    scout_confidence: float = 0.90
    scout_call_budget: int = 15
    endgame_hands: int = 12
    decision_budget_seconds: float = 0.45
    # Value is intentionally convex on positive outcomes: retries reward tail wins.
    positive_tail_weight: float = 0.045
    leader_margin_weight: float = 1.8
    early_clear_bonus: float = 18.0


class HighVariancePolicy:
    """Choose legal moves by exact subset rollouts and terminal-score utility.

    Opponent response probabilities are deliberately shallow.  The expensive,
    important part--multiway showdown equity under uncertain rules--is exact.
    Equity values are cached by the set of continuing seats, so at most 32 are
    computed at a six-seat table regardless of the number of bet sizes.
    """

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()

    def decide(self, request: Any, session: Any, knowledge: Any) -> dict[str, str | int]:
        deadline = time.monotonic() + self.config.decision_budget_seconds
        rule_model = knowledge.get_rule(request.table_rule)
        opponents = [
            player
            for player in request.players
            if player.seat != request.your_seat
            and not player.folded
            and not player.busted
        ]
        responders = [player for player in opponents if not player.all_in]

        ranges = {
            player.seat: self._opponent_range(
                request, player, knowledge.get_opponent(player.name), rule_model
            )
            for player in opponents
        }
        range_strengths = {
            player.seat: self._range_strength(
                ranges[player.seat], request.community_number, rule_model
            )
            for player in opponents
        }
        equity_cache = showdown_shares_by_subset(
            request.your_number,
            request.community_number,
            ranges,
            rule_model,
        )

        def equity_for(seats: Iterable[int]) -> float:
            key = frozenset(seats)
            return equity_cache.get(key, 1.0 if not key else 0.0)

        candidates = self._candidates(request, responders)
        if (
            request.hand_number <= self.config.scout_hands
            and rule_model.confidence() < self.config.scout_confidence
            and equity_for(player.seat for player in opponents) < 0.75
        ):
            candidates = [
                candidate
                for candidate in candidates
                if candidate.action not in {"bet", "raise"}
            ]
        if not candidates:
            return {"action": "check"}

        scores: list[tuple[float, int, int, Candidate]] = []
        for candidate in candidates:
            if scores and time.monotonic() >= deadline:
                break
            score = self._score(
                request,
                session,
                rule_model,
                candidate,
                opponents,
                responders,
                ranges,
                range_strengths,
                equity_for,
                knowledge,
            )
            # Stable tie-breaking: prefer actions that invest more in high-variance
            # mode, then deterministic action order.
            investment = self._hero_extra(request, candidate)
            action_order = {
                "fold": 0,
                "check": 1,
                "call": 2,
                "bet": 3,
                "raise": 4,
            }.get(candidate.action, -1)
            scores.append((score, investment, action_order, candidate))

        return max(scores, key=lambda item: item[:3])[3].response()

    def _candidates(self, request: Any, responders: list[Any]) -> list[Candidate]:
        candidates = [
            Candidate(action)
            for action in ("fold", "check", "call")
            if action in request.legal_actions
        ]
        own_bet = request.own_player.bet_this_round
        pot_after_call = max(1, request.pot + request.to_call)
        max_matchable = max(
            (player.bet_this_round + player.stack for player in responders),
            default=own_bet,
        )

        for action in ("bet", "raise"):
            if action not in request.legal_actions:
                continue
            minimum = request.min_raise_to
            maximum = request.max_raise_to
            if minimum is None or maximum is None:
                continue

            targets = {minimum, maximum}
            call_target = own_bet + request.to_call
            for fraction in (1 / 3, 1 / 2, 2 / 3, 1.0, 1.5):
                targets.add(call_target + round(pot_after_call * fraction))
            if max_matchable >= minimum:
                targets.add(min(maximum, max_matchable))

            for amount in sorted({max(minimum, min(maximum, value)) for value in targets}):
                candidates.append(Candidate(action, amount))

        # A set also protects against min == max and clamped fractional duplicates.
        return list(dict.fromkeys(candidates))

    def _score(
        self,
        request: Any,
        session: Any,
        rule_model: Any,
        candidate: Candidate,
        opponents: list[Any],
        responders: list[Any],
        ranges: dict[int, list[float]],
        range_strengths: dict[int, float],
        equity_for: Any,
        knowledge: Any,
    ) -> float:
        hero_extra = self._hero_extra(request, candidate)
        fixed_all_ins = {player.seat for player in opponents if player.all_in}
        remaining_hands = max(0, request.total_hands - request.hand_number)

        if candidate.action == "fold":
            other_deltas = self._base_other_deltas(request, {})
            self._award_to_likely_opponent(request, opponents, other_deltas, request.pot)
            return self._outcome_utility(
                request.your_stack - request.starting_stack,
                other_deltas.values(),
                request,
            )

        if candidate.action in {"check", "call"}:
            continuing = {player.seat for player in opponents}
            final_pot = request.pot + hero_extra
            return self._showdown_branch_utility(
                request,
                opponents,
                continuing,
                ranges,
                equity_for(continuing),
                hero_extra,
                {},
                final_pot,
            ) + self._scout_adjustment(
                request,
                session,
                rule_model,
                candidate,
                len(continuing),
                equity_for(continuing),
            )

        target = candidate.amount or request.own_player.bet_this_round
        size_ratio = hero_extra / max(1.0, request.pot + request.to_call)
        probabilities: list[tuple[Any, float, int]] = []
        for player in responders:
            call_extra = min(
                player.stack,
                max(0, target - player.bet_this_round),
            )
            strength = range_strengths[player.seat]
            profile = knowledge.get_opponent(player.name)
            probability = self._continue_probability(
                profile,
                request,
                player,
                size_ratio,
                strength,
            )
            probabilities.append((player, probability, call_extra))

        total = 0.0
        for choices in product((False, True), repeat=len(probabilities)):
            branch_probability = 1.0
            continuing = set(fixed_all_ins)
            call_extras: dict[int, int] = {}
            for choice, (player, probability, call_extra) in zip(
                choices, probabilities, strict=True
            ):
                branch_probability *= probability if choice else (1.0 - probability)
                if choice:
                    continuing.add(player.seat)
                    call_extras[player.seat] = call_extra
            if branch_probability <= 0.0:
                continue

            final_pot = request.pot + hero_extra + sum(call_extras.values())
            if not continuing:
                hero_delta = (
                    request.your_stack
                    - hero_extra
                    + final_pot
                    - request.starting_stack
                )
                other_deltas = self._base_other_deltas(request, call_extras)
                branch_value = self._outcome_utility(
                    hero_delta, other_deltas.values(), request
                )
            else:
                branch_value = self._showdown_branch_utility(
                    request,
                    opponents,
                    continuing,
                    ranges,
                    equity_for(continuing),
                    hero_extra,
                    call_extras,
                    final_pot,
                )
            total += branch_probability * branch_value

        base_equity = equity_for(player.seat for player in opponents)
        total += self._scout_adjustment(
            request,
            session,
            rule_model,
            candidate,
            len(opponents),
            base_equity,
        )

        # Multiway air is a poor bluff.  Keep the high-variance exception for a
        # late deficit, when passivity cannot meet the terminal predicate.
        leader = max(
            (player.chip_delta for player in request.players if player.seat != request.your_seat),
            default=0,
        )
        desperate = remaining_hands < self.config.endgame_hands and (
            request.own_player.chip_delta < max(10, leader + 1)
        )
        if base_equity < 0.20 and len(responders) > 2 and not desperate:
            total -= 80.0 * (1.0 + size_ratio)
        return total

    def _showdown_branch_utility(
        self,
        request: Any,
        opponents: list[Any],
        continuing: set[int],
        ranges: dict[int, list[float]],
        equity: float,
        hero_extra: int,
        call_extras: dict[int, int],
        final_pot: int,
    ) -> float:
        base_other = self._base_other_deltas(request, call_extras)
        hero_win_delta = (
            request.your_stack - hero_extra + final_pot - request.starting_stack
        )
        win_value = self._outcome_utility(
            hero_win_delta, base_other.values(), request
        )

        eligible = [player for player in opponents if player.seat in continuing]
        if not eligible:
            return win_value
        weights = []
        for player in eligible:
            strength = self._range_strength(
                ranges[player.seat], request.community_number, None
            )
            # Targeting the current leader matters: if hero loses, a pot going to
            # that seat is the worst branch for the exact clearing objective.
            lead_weight = 1.0 + max(0.0, player.chip_delta) / 100.0
            weights.append(max(0.01, strength * lead_weight))
        weight_total = sum(weights)

        lose_value = 0.0
        hero_lose_delta = request.your_stack - hero_extra - request.starting_stack
        for player, weight in zip(eligible, weights, strict=True):
            outcome = dict(base_other)
            outcome[player.seat] += final_pot
            lose_value += (weight / weight_total) * self._outcome_utility(
                hero_lose_delta, outcome.values(), request
            )
        return equity * win_value + (1.0 - equity) * lose_value

    def _outcome_utility(
        self,
        hero_delta: float,
        other_deltas: Iterable[float],
        request: Any,
    ) -> float:
        others = list(other_deltas)
        leader = max(others, default=-200.0)
        margin = hero_delta - leader
        clear = hero_delta >= 10.0 and margin > 0.0
        remaining = max(0, request.total_hands - request.hand_number)
        target = max(10.0, leader + 1.0)

        if remaining == 0:
            return 10_000.0 if clear else -20.0 * max(1.0, target - hero_delta)
        if remaining <= self.config.endgame_hands:
            if clear:
                return 2_000.0 + 30.0 * margin + 2.0 * hero_delta
            return (
                2.0 * hero_delta
                - 12.0 * max(0.0, target - hero_delta)
                + 3.0 * max(0.0, margin)
            )

        trajectory = math.ceil(10.0 * request.hand_number / request.total_hands)
        on_trajectory = hero_delta >= trajectory and margin > 0.0
        positive = max(0.0, hero_delta)
        # Convex positive utility is the explicit high-variance posture selected
        # for retry-best scoring.  Negative outcomes receive only a light penalty.
        return (
            hero_delta
            + self.config.positive_tail_weight * positive * positive
            + self.config.leader_margin_weight * margin
            + (self.config.early_clear_bonus if on_trajectory else 0.0)
            - (12.0 if hero_delta <= -200.0 else 0.0)
        )

    def _scout_adjustment(
        self,
        request: Any,
        session: Any,
        rule_model: Any,
        candidate: Candidate,
        showdown_opponents: int,
        equity: float,
    ) -> float:
        confidence = float(rule_model.confidence())
        if request.hand_number > self.config.scout_hands or confidence >= self.config.scout_confidence:
            return 0.0
        spent = self._observed_scout_calls(request)
        investment = self._hero_extra(request, candidate)
        free_or_call = candidate.action in {"check", "call"}
        information = (1.0 - confidence) * min(3, showdown_opponents) * 4.0
        adjustment = information if free_or_call else -information
        if candidate.action == "call" and spent + investment > self.config.scout_call_budget:
            pot_odds = investment / max(1, request.pot + investment)
            if equity < pot_odds + 0.12:
                adjustment -= 60.0
        if candidate.action in {"bet", "raise"} and equity < 0.75:
            # Scouting is intentionally controlled even though the post-scout
            # strategy is high variance.  Do not burn the leg before learning
            # what the codename makes strong.
            adjustment -= 500.0 + 0.5 * investment
        return adjustment

    @staticmethod
    def _hero_extra(request: Any, candidate: Candidate) -> int:
        if candidate.action == "call":
            return min(request.your_stack, request.to_call)
        if candidate.action in {"bet", "raise"} and candidate.amount is not None:
            return max(0, candidate.amount - request.own_player.bet_this_round)
        return 0

    @staticmethod
    def _base_other_deltas(request: Any, extras: dict[int, int]) -> dict[int, float]:
        return {
            player.seat: player.stack
            - extras.get(player.seat, 0)
            - request.starting_stack
            for player in request.players
            if player.seat != request.your_seat
        }

    @staticmethod
    def _award_to_likely_opponent(
        request: Any,
        opponents: list[Any],
        deltas: dict[int, float],
        pot: int,
    ) -> None:
        if not opponents:
            return
        winner = max(opponents, key=lambda player: (player.chip_delta, -player.seat))
        deltas[winner.seat] += pot

    def _opponent_range(
        self, request: Any, player: Any, profile: Any, rule_model: Any
    ) -> list[float]:
        actions = [
            action
            for action in request.current_hand_actions
            if action.seat == player.seat
        ]
        position = self._position_bucket(request, player.seat)
        try:
            result = profile.range_for_action_history(
                actions=actions,
                seat=player.seat,
                round_name=request.round,
                community=request.community_number,
                rule_model=rule_model,
                live_count=sum(
                    not item.folded and not item.busted for item in request.players
                ),
                position_bucket=position,
            )
        except (AttributeError, TypeError, ValueError):
            result = [1.0 / 13.0] * 13
        if not isinstance(result, (list, tuple)) or len(result) != 13:
            return [1.0 / 13.0] * 13
        cleaned = [max(0.0, float(value)) for value in result]
        total = sum(cleaned)
        return [value / total for value in cleaned] if total > 0 else [1.0 / 13.0] * 13

    def _continue_probability(
        self,
        profile: Any,
        request: Any,
        player: Any,
        size_ratio: float,
        strength: float,
    ) -> float:
        size_bucket = (
            "small" if size_ratio <= 0.40 else "medium" if size_ratio <= 0.90 else "large"
        )
        try:
            probability = profile.continue_probability(
                round_name=request.round,
                facing="raise" if request.to_call else "bet",
                size_bucket=size_bucket,
                live_count=sum(
                    not item.folded and not item.busted for item in request.players
                ),
                position_bucket=self._position_bucket(request, player.seat),
                strength=strength,
            )
            return min(0.98, max(0.02, float(probability)))
        except (AttributeError, TypeError, ValueError):
            pressure = 0.17 * math.log2(1.0 + max(0.0, size_ratio))
            return min(0.94, max(0.04, 0.14 + 0.76 * strength - pressure))

    @staticmethod
    def _range_strength(
        card_range: list[float], community: int | None, rule_model: Any | None
    ) -> float:
        if rule_model is None:
            return sum((index + 1) / 13.0 * value for index, value in enumerate(card_range))
        if community is None:
            return sum(
                probability
                * sum(rule_model.strength(number, c) for c in range(1, 14))
                / 13.0
                for number, probability in enumerate(card_range, 1)
            )
        return sum(
            probability * rule_model.strength(number, community)
            for number, probability in enumerate(card_range, 1)
        )

    @staticmethod
    def _position_bucket(request: Any, seat: int) -> str:
        live = sorted(player.seat for player in request.players if not player.busted)
        if not live or seat not in live:
            return "unknown"
        after_button = [value for value in live if value > request.button_seat] + [
            value for value in live if value <= request.button_seat
        ]
        if request.round == "pre_reveal" and len(after_button) > 2:
            order = after_button[2:] + after_button[:2]
        else:
            order = after_button
        index = order.index(seat)
        third = max(1, math.ceil(len(order) / 3))
        if index < third:
            return "early"
        if index >= len(order) - third:
            return "late"
        return "middle"

    @staticmethod
    def _observed_scout_calls(request: Any) -> int:
        total = 0
        histories = list(request.recent_hands)
        for hand in histories:
            if hand.hand_number > 10:
                continue
            previous: dict[tuple[int, str], int] = {}
            for action in hand.actions:
                key = (action.seat, action.round)
                before = previous.get(key, 0)
                if action.amount is not None:
                    previous[key] = action.amount
                if action.seat == request.your_seat and action.action == "call":
                    total += max(0, (action.amount or before) - before)
        return total
