"""Deterministic, clean-room simulator for SHOWDOWN phase 3 policy work.

The simulator uses ``standard`` by default and requires an explicit ranking
hypothesis or ranker whenever an opaque nonstandard codename is configured.
Its move callbacks receive dictionaries shaped like protocol-v2 ``/move``
requests and return either an :class:`Action`, an action string, or a mapping
such as ``{"action": "raise", "amount": 18}``.

Assumptions made where the coordinator contract is not explicit:

* Blind and action order keep the six-seat rule after the table becomes
  heads-up: the small blind is first clockwise from the button and the big
  blind is next (therefore the button in heads-up).  There is no switch to the
  special two-seat button rule.
* A short all-in raise reopens action.  The legal range emitted by this module
  is authoritative for simulations, just as ``legal_actions`` is authoritative
  on the real server.
* Unmatched chips are represented as a one-player side pot and therefore
  returned.  Odd chips from a tied pot go to tied winners clockwise after the
  button.  These choices preserve chips and make every seeded run reproducible.

This is a tuning tool, not a coordinator clone.  It has no dependency on the
phase 1/2 implementations.  The production Phase-3 policy adapter is imported
only when explicitly requested.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Protocol, Sequence, TypeAlias

from .rules import RuleHypothesis, get_hypothesis


PLAYER_NAMES = ("you", "Dana", "Miles", "Theo", "Rhea", "Bram")


@dataclass(frozen=True, slots=True)
class Action:
    """A callback decision; ``amount`` is the total wager in this round."""

    action: str
    amount: int | None = None

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"action": self.action}
        if self.amount is not None:
            result["amount"] = self.amount
        return result


ActionLike: TypeAlias = Action | str | Mapping[str, object]
DecisionCallback: TypeAlias = Callable[[Mapping[str, Any]], ActionLike]
StrategyFactory: TypeAlias = Callable[[], DecisionCallback]
RankValue: TypeAlias = tuple[int | float, ...] | list[int | float] | int | float
Ranker: TypeAlias = Callable[[int, int], RankValue]


class Strategy(Protocol):
    """Structural interface implemented by hero and scripted opponents."""

    def __call__(self, request: Mapping[str, Any]) -> ActionLike: ...


@dataclass(slots=True)
class Player:
    seat: int
    name: str
    stack: int
    folded: bool = False
    all_in: bool = False
    busted: bool = False
    number: int | None = None
    round_bet: int = 0
    contribution: int = 0


@dataclass(frozen=True, slots=True)
class SidePot:
    cap: int
    amount: int
    eligible: tuple[int, ...]
    winners: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class HandResult:
    hand_number: int
    button_seat: int
    small_blind_seat: int
    big_blind_seat: int
    community_number: int | None
    pot: int
    winners: tuple[int, ...]
    shown_numbers: Mapping[int, int]
    actions: tuple[Mapping[str, object], ...]
    contributions: tuple[int, ...]
    payouts: tuple[int, ...]
    stacks_after: tuple[int, ...]
    side_pots: tuple[SidePot, ...]

    def recent_hand(self) -> dict[str, object]:
        """Return the protocol-shaped summary presented to later callbacks."""

        return {
            "hand_number": self.hand_number,
            "community_number": self.community_number,
            "winners": list(self.winners),
            "pot": self.pot,
            "shown_numbers": {
                str(seat): number for seat, number in self.shown_numbers.items()
            },
            "actions": [dict(action) for action in self.actions],
        }


@dataclass(frozen=True, slots=True)
class LegResult:
    seed: int
    hero_seat: int
    starting_stack: int
    hands: tuple[HandResult, ...]
    final_stacks: tuple[int, ...]
    invalid_actions: int

    @property
    def final_deltas(self) -> tuple[int, ...]:
        return tuple(stack - self.starting_stack for stack in self.final_stacks)

    @property
    def hero_delta(self) -> int:
        return self.final_deltas[self.hero_seat]

    @property
    def hero_busted(self) -> bool:
        return self.final_stacks[self.hero_seat] == 0

    @property
    def cleared(self) -> bool:
        hero = self.hero_delta
        return hero >= 10 and all(
            hero > delta
            for seat, delta in enumerate(self.final_deltas)
            if seat != self.hero_seat
        )


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    starting_stack: int = 200
    total_hands: int = 60
    small_blind: int = 1
    big_blind: int = 2
    hero_seat: int = 0
    initial_button: int = 0
    table_rule: str = "standard"
    # ``table_rule`` is the opaque codename sent to the policy.  The simulator
    # must separately know the true ranking formula used to settle the hand.
    # Known hypothesis names may be supplied directly; a custom ranker supports
    # synthetic/out-of-grammar rules without changing the production grammar.
    rule_hypothesis: str | RuleHypothesis | None = None
    ranker: Ranker | None = field(default=None, repr=False, compare=False)
    match_id_prefix: str = "phase3-sim"
    max_actions_per_round: int = 4096

    def __post_init__(self) -> None:
        if self.starting_stack <= 0 or self.total_hands <= 0:
            raise ValueError("starting_stack and total_hands must be positive")
        if not 0 < self.small_blind <= self.big_blind:
            raise ValueError("blinds must satisfy 0 < small_blind <= big_blind")
        if not 0 <= self.hero_seat < len(PLAYER_NAMES):
            raise ValueError("hero_seat must identify one of the six seats")
        if not isinstance(self.table_rule, str) or not self.table_rule.strip():
            raise ValueError("table_rule must be a non-empty codename")
        if self.ranker is not None and self.rule_hypothesis is not None:
            raise ValueError("provide either rule_hypothesis or ranker, not both")

        if self.ranker is None:
            truth = self.rule_hypothesis
            if truth is None:
                # ``standard`` is both a public rule label and a hypothesis.
                # Opaque nonstandard codenames cannot safely imply their truth.
                if self.table_rule != "standard":
                    raise ValueError(
                        "a nonstandard table_rule requires rule_hypothesis or ranker"
                    )
                truth = "standard"
            get_hypothesis(truth)
        self.rank(1, 1)

    def rank(self, number: int, community: int) -> tuple[int | float, ...]:
        """Return one comparable rank under the simulator's actual rule."""

        if self.ranker is not None:
            return _rank_key(self.ranker(number, community))
        truth = self.rule_hypothesis or "standard"
        return get_hypothesis(truth).rank(number, community)


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    trials: int
    clear_rate: float
    bust_rate: float
    upper_tail_delta: int
    mean_delta: float
    min_delta: int
    max_delta: int

    def as_dict(self) -> dict[str, int | float]:
        return {
            "trials": self.trials,
            "clear_rate": self.clear_rate,
            "bust_rate": self.bust_rate,
            "upper_tail_delta": self.upper_tail_delta,
            "mean_delta": self.mean_delta,
            "min_delta": self.min_delta,
            "max_delta": self.max_delta,
        }


def standard_rank(number: int, community_number: int) -> tuple[int, int]:
    """Rank one number under the documented standard rule."""

    return (int(number == community_number), number)


def _rank_key(value: RankValue) -> tuple[int | float, ...]:
    """Normalize a custom ranker result into a finite comparable tuple."""

    raw = value if isinstance(value, (tuple, list)) else (value,)
    if not raw:
        raise ValueError("ranker must return at least one numeric component")
    result: list[int | float] = []
    for component in raw:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise TypeError("ranker components must be numeric")
        if not math.isfinite(float(component)):
            raise ValueError("ranker components must be finite")
        result.append(component)
    return tuple(result)


def _rank_strength(
    number: int,
    community: int | None,
    ranker: Ranker,
) -> float:
    """Heads-up share against a uniform number under one known rule truth."""

    communities = range(1, 14) if community is None else (community,)
    total = 0.0
    comparisons = 0
    for revealed in communities:
        hero = _rank_key(ranker(number, revealed))
        for opponent in range(1, 14):
            rival = _rank_key(ranker(opponent, revealed))
            total += 1.0 if hero > rival else 0.5 if hero == rival else 0.0
            comparisons += 1
    return total / max(1, comparisons)


def cleared(deltas: Sequence[int], hero_seat: int = 0) -> bool:
    """Return the exact phase-3 per-leg scoring predicate."""

    hero = deltas[hero_seat]
    return hero >= 10 and all(
        hero > delta for seat, delta in enumerate(deltas) if seat != hero_seat
    )


def _unit_interval(request: Mapping[str, Any], salt: str) -> float:
    """Stable pseudo-random value derived solely from public decision context."""

    raw = "|".join(
        (
            str(request.get("match_id", "")),
            str(request.get("hand_number", "")),
            str(request.get("your_seat", "")),
            str(request.get("round", "")),
            str(len(request.get("current_hand_actions", ()))),
            salt,
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") / 2**64


def _sized_action(request: Mapping[str, Any], fraction: float) -> Action:
    legal = tuple(request["legal_actions"])
    aggressive = "raise" if "raise" in legal else "bet"
    minimum = int(request["min_raise_to"])
    maximum = int(request["max_raise_to"])
    own = next(
        player
        for player in request["players"]
        if int(player["seat"]) == int(request["your_seat"])
    )
    desired = int(own["bet_this_round"]) + max(
        1, round(int(request["pot"]) * fraction)
    )
    return Action(aggressive, max(minimum, min(maximum, desired)))


@dataclass(frozen=True, slots=True)
class ScriptedArchetype:
    """Small deterministic opponent useful for reproducible policy tuning.

    ``tightness`` raises the continue threshold, ``aggression`` controls how
    often a continuing hand bets or raises, and ``bluff_frequency`` injects
    stable context-derived bluffs.  No behavior is keyed to player names.
    """

    tightness: float = 0.5
    aggression: float = 0.5
    bluff_frequency: float = 0.06
    sizing: float = 0.65
    ranker: Ranker = field(default=standard_rank, repr=False, compare=False)

    def __post_init__(self) -> None:
        for value in (self.tightness, self.aggression, self.bluff_frequency):
            if not 0.0 <= value <= 1.0:
                raise ValueError("archetype rates must be in [0, 1]")
        if self.sizing <= 0:
            raise ValueError("sizing must be positive")
        _rank_key(self.ranker(1, 1))

    def __call__(self, request: Mapping[str, Any]) -> Action:
        number = int(request["your_number"])
        community = request.get("community_number")
        strength = _rank_strength(
            number,
            int(community) if community is not None else None,
            self.ranker,
        )
        premium = strength >= 0.92
        live = sum(
            not player["folded"] and not player["busted"]
            for player in request["players"]
        )
        threshold = min(0.96, 0.30 + self.tightness * 0.48 + 0.035 * (live - 2))
        bluff = _unit_interval(request, "bluff") < self.bluff_frequency
        aggressive = any(action in request["legal_actions"] for action in ("bet", "raise"))
        aggression_roll = _unit_interval(request, "aggression")

        if aggressive and (bluff or (strength >= threshold and aggression_roll < self.aggression)):
            return _sized_action(request, self.sizing * (1.35 if premium else 1.0))

        to_call = int(request["to_call"])
        if to_call:
            price = to_call / max(1, int(request["pot"]) + to_call)
            if premium or strength >= max(0.24, threshold - 0.17) or price < strength * 0.22:
                return Action("call")
            return Action("fold")
        return Action("check")


def _built_in_strategy_for_rule(
    request: Mapping[str, Any], ranker: Ranker
) -> Action:
    """Rule-aware implementation behind the public standard baseline."""

    number = int(request["your_number"])
    community = request.get("community_number")
    strength = _rank_strength(
        number,
        int(community) if community is not None else None,
        ranker,
    )
    premium = strength >= 0.92
    live_opponents = sum(
        int(player["seat"]) != int(request["your_seat"])
        and not player["folded"]
        and not player["busted"]
        for player in request["players"]
    )
    legal = tuple(request["legal_actions"])
    strong = strength >= (0.84 if live_opponents >= 4 else 0.70)
    late = int(request["hand_number"]) > int(request["total_hands"]) - 12
    deltas = [int(player["chip_delta"]) for player in request["players"]]
    hero_delta = deltas[int(request["your_seat"])]
    target = max(10, max(delta for seat, delta in enumerate(deltas) if seat != request["your_seat"]) + 1)
    desperate = late and hero_delta < target

    if strong and ("bet" in legal or "raise" in legal):
        return _sized_action(request, 1.25 if premium or desperate else 0.8)
    if desperate and live_opponents <= 2 and ("bet" in legal or "raise" in legal):
        return _sized_action(request, 1.1)
    if int(request["to_call"]):
        price = int(request["to_call"]) / max(
            1, int(request["pot"]) + int(request["to_call"])
        )
        if premium or strength >= 0.70 or (strength >= 0.52 and price <= 0.15):
            return Action("call")
        return Action("fold")
    return Action("check")


def built_in_strategy(request: Mapping[str, Any]) -> Action:
    """A deterministic, moderately high-variance standard-rule baseline."""

    return _built_in_strategy_for_rule(request, standard_rank)


def default_opponents(
    hero_seat: int = 0,
    *,
    ranker: Ranker = standard_rank,
) -> dict[int, ScriptedArchetype]:
    """Return five different, seat-assigned (not name-assigned) archetypes."""

    seats = [seat for seat in range(6) if seat != hero_seat]
    profiles = (
        ScriptedArchetype(tightness=0.82, aggression=0.30, sizing=0.55, ranker=ranker),
        ScriptedArchetype(tightness=0.30, aggression=0.25, sizing=0.45, ranker=ranker),
        ScriptedArchetype(
            tightness=0.46,
            aggression=0.88,
            bluff_frequency=0.13,
            sizing=0.95,
            ranker=ranker,
        ),
        ScriptedArchetype(tightness=0.63, aggression=0.55, sizing=0.70, ranker=ranker),
        ScriptedArchetype(
            tightness=0.20,
            aggression=0.72,
            bluff_frequency=0.18,
            sizing=1.15,
            ranker=ranker,
        ),
    )
    return dict(zip(seats, profiles, strict=True))


class ShowdownSimulator:
    """One deterministic six-seat, 60-hand standard-rule leg."""

    def __init__(self, seed: int, config: SimulationConfig | None = None) -> None:
        self.seed = seed
        self.config = config or SimulationConfig()
        self.rng = random.Random(seed)
        opponent_names = iter(PLAYER_NAMES[1:])
        self.players = [
            Player(
                seat,
                "you" if seat == self.config.hero_seat else next(opponent_names),
                self.config.starting_stack,
            )
            for seat in range(len(PLAYER_NAMES))
        ]
        self.button = self.config.initial_button
        self.history: list[HandResult] = []
        self.current_actions: list[dict[str, object]] = []
        self.invalid_actions = 0
        self._hand_start_stacks: tuple[int, ...] = tuple()
        self._community = 0
        self._community_revealed = False
        self._hand_number = 0
        self._strategies: dict[int, DecisionCallback] = {}

    def run(
        self,
        hero_strategy: DecisionCallback | None = None,
        opponent_strategies: Mapping[int, DecisionCallback] | None = None,
    ) -> LegResult:
        """Play the configured leg and return complete deterministic results."""

        if self.history:
            raise RuntimeError("a ShowdownSimulator instance can only run once")
        opponents = dict(
            opponent_strategies
            or default_opponents(
                self.config.hero_seat,
                ranker=self.config.rank,
            )
        )
        missing = [seat for seat in range(6) if seat != self.config.hero_seat and seat not in opponents]
        if missing:
            raise ValueError(f"missing opponent strategies for seats {missing}")
        self._strategies = opponents
        if hero_strategy is not None:
            self._strategies[self.config.hero_seat] = hero_strategy
        else:
            self._strategies[self.config.hero_seat] = lambda request: (
                _built_in_strategy_for_rule(request, self.config.rank)
            )

        self.button = self._first_live_at_or_after(self.button)
        for hand_number in range(1, self.config.total_hands + 1):
            if len(self._active_seats()) <= 1:
                break
            self._play_hand(hand_number)
            active = self._active_seats()
            if len(active) > 1:
                self.button = self._next_live(self.button)

        final_stacks = tuple(player.stack for player in self.players)
        expected = self.config.starting_stack * len(self.players)
        if sum(final_stacks) != expected:
            raise RuntimeError("chip conservation failed at leg end")
        return LegResult(
            seed=self.seed,
            hero_seat=self.config.hero_seat,
            starting_stack=self.config.starting_stack,
            hands=tuple(self.history),
            final_stacks=final_stacks,
            invalid_actions=self.invalid_actions,
        )

    def _active_seats(self) -> list[int]:
        return [player.seat for player in self.players if not player.busted]

    def _first_live_at_or_after(self, seat: int) -> int:
        if not 0 <= seat < len(self.players):
            raise ValueError("initial_button is not a table seat")
        if not self.players[seat].busted:
            return seat
        return self._next_live(seat)

    def _next_live(self, seat: int) -> int:
        for offset in range(1, len(self.players) + 1):
            candidate = (seat + offset) % len(self.players)
            if not self.players[candidate].busted:
                return candidate
        raise RuntimeError("table has no live seats")

    def _clockwise(self, start_after: int) -> list[int]:
        return [
            (start_after + offset) % len(self.players)
            for offset in range(1, len(self.players) + 1)
            if not self.players[(start_after + offset) % len(self.players)].busted
        ]

    def _play_hand(self, hand_number: int) -> None:
        self._hand_number = hand_number
        self.current_actions = []
        self._community_revealed = False
        self._hand_start_stacks = tuple(player.stack for player in self.players)
        for player in self.players:
            player.folded = player.busted
            player.all_in = False
            player.number = self.rng.randint(1, 13) if not player.busted else None
            player.round_bet = 0
            player.contribution = 0
        self._community = self.rng.randint(1, 13)

        small_seat = self._next_live(self.button)
        big_seat = self._next_live(small_seat)
        self._post_forced_bet(small_seat, self.config.small_blind)
        self._post_forced_bet(big_seat, self.config.big_blind)

        pre_order = self._clockwise(big_seat)
        self._betting_round("pre_reveal", pre_order)
        if len(self._unfolded_seats()) > 1:
            self._community_revealed = True
            for player in self.players:
                player.round_bet = 0
            post_order = self._clockwise(self.button)
            self._betting_round("post_reveal", post_order)

        result = self._settle(hand_number, small_seat, big_seat)
        self.history.append(result)
        for player in self.players:
            player.busted = player.stack == 0
            player.all_in = False

    def _post_forced_bet(self, seat: int, amount: int) -> None:
        player = self.players[seat]
        paid = min(player.stack, amount)
        player.stack -= paid
        player.round_bet += paid
        player.contribution += paid
        player.all_in = player.stack == 0

    def _unfolded_seats(self) -> list[int]:
        return [
            player.seat
            for player in self.players
            if not player.busted and not player.folded
        ]

    def _actionable(self, seat: int) -> bool:
        player = self.players[seat]
        return not player.busted and not player.folded and not player.all_in

    @staticmethod
    def _rotate_after(order: Sequence[int], seat: int) -> list[int]:
        index = order.index(seat)
        return list(order[index + 1 :]) + list(order[:index])

    def _betting_round(self, round_name: str, order: Sequence[int]) -> None:
        current_bet = max(self.players[seat].round_bet for seat in self._active_seats())
        last_raise = self.config.big_blind
        pending = deque(seat for seat in order if self._actionable(seat))
        action_count = 0

        while pending and len(self._unfolded_seats()) > 1:
            seat = pending.popleft()
            if not self._actionable(seat):
                continue
            other_actionable = [
                other
                for other in self._unfolded_seats()
                if other != seat and self._actionable(other)
            ]
            player = self.players[seat]
            if not other_actionable and player.round_bet >= current_bet:
                break
            action_count += 1
            if action_count > self.config.max_actions_per_round:
                raise RuntimeError("betting round exceeded action safety limit")

            legal, minimum, maximum, to_call = self._legal_options(
                seat, current_bet, last_raise
            )
            request = self._move_request(
                seat, round_name, legal, minimum, maximum, to_call
            )
            action = self._decision(seat, request, legal, minimum, maximum)
            prior_bet = current_bet
            current_bet = self._apply_action(seat, round_name, action, current_bet)
            if action.action in ("bet", "raise"):
                increase = current_bet - prior_bet
                if increase >= last_raise:
                    last_raise = increase
                pending = deque(
                    other
                    for other in self._rotate_after(order, seat)
                    if self._actionable(other)
                    and self.players[other].round_bet < current_bet
                )

    def _legal_options(
        self, seat: int, current_bet: int, last_raise: int
    ) -> tuple[tuple[str, ...], int | None, int | None, int]:
        player = self.players[seat]
        gap = max(0, current_bet - player.round_bet)
        to_call = min(gap, player.stack)
        legal: list[str] = ["fold", "call"] if gap else ["check"]
        maximum = player.round_bet + player.stack
        minimum: int | None = None
        sized_action = "raise" if current_bet else "bet"
        if maximum > current_bet:
            nominal = current_bet + last_raise if current_bet else self.config.big_blind
            minimum = min(maximum, nominal)
            legal.append(sized_action)
        else:
            maximum = None
        return tuple(legal), minimum, maximum, to_call

    def _move_request(
        self,
        seat: int,
        round_name: str,
        legal: Sequence[str],
        minimum: int | None,
        maximum: int | None,
        to_call: int,
    ) -> dict[str, Any]:
        player_rows = []
        for player in self.players:
            player_rows.append(
                {
                    "seat": player.seat,
                    "name": player.name,
                    "folded": player.folded,
                    "chip_delta": self._hand_start_stacks[player.seat]
                    - self.config.starting_stack,
                    "bet_this_round": player.round_bet,
                    "stack": player.stack,
                    "all_in": player.all_in,
                    "busted": player.busted,
                }
            )
        return {
            "protocol_version": 2,
            "match_id": f"{self.config.match_id_prefix}-{self.seed}",
            "phase": 3,
            "table_rule": self.config.table_rule,
            "small_blind": self.config.small_blind,
            "big_blind": self.config.big_blind,
            "starting_stack": self.config.starting_stack,
            "your_stack": self.players[seat].stack,
            "hand_number": self._hand_number,
            "total_hands": self.config.total_hands,
            "leg_number": 1,
            "total_legs": 4,
            "round": round_name,
            "your_number": self.players[seat].number,
            "community_number": self._community if self._community_revealed else None,
            "your_seat": seat,
            "button_seat": self.button,
            "pot": sum(player.contribution for player in self.players),
            "to_call": to_call,
            "min_raise_to": minimum,
            "max_raise_to": maximum,
            "legal_actions": list(legal),
            "players": player_rows,
            "current_hand_actions": [dict(action) for action in self.current_actions],
            "recent_hands": [hand.recent_hand() for hand in self.history[-20:]],
        }

    @staticmethod
    def _parse_action(raw: ActionLike) -> Action:
        if isinstance(raw, Action):
            return raw
        if isinstance(raw, str):
            return Action(raw)
        action = raw.get("action")
        amount = raw.get("amount")
        if not isinstance(action, str):
            return Action("")
        if amount is not None and (isinstance(amount, bool) or not isinstance(amount, int)):
            return Action(action, None)
        return Action(action, amount)

    def _decision(
        self,
        seat: int,
        request: Mapping[str, Any],
        legal: Sequence[str],
        minimum: int | None,
        maximum: int | None,
    ) -> Action:
        try:
            action = self._parse_action(self._strategies[seat](request))
        except Exception:
            action = Action("")
        valid = action.action in legal
        if action.action in ("bet", "raise"):
            valid = (
                valid
                and action.amount is not None
                and minimum is not None
                and maximum is not None
                and minimum <= action.amount <= maximum
            )
        elif action.amount is not None:
            valid = False
        if valid:
            return action
        self.invalid_actions += 1
        if "check" in legal:
            return Action("check")
        if "fold" in legal:
            return Action("fold")
        if "call" in legal:
            return Action("call")
        sized = "bet" if "bet" in legal else "raise"
        return Action(sized, minimum)

    def _apply_action(
        self, seat: int, round_name: str, action: Action, current_bet: int
    ) -> int:
        player = self.players[seat]
        record: dict[str, object] = {
            "round": round_name,
            "seat": seat,
            "action": action.action,
        }
        if action.action == "fold":
            player.folded = True
        elif action.action == "check":
            pass
        elif action.action == "call":
            paid = min(player.stack, max(0, current_bet - player.round_bet))
            player.stack -= paid
            player.round_bet += paid
            player.contribution += paid
            record["amount"] = player.round_bet
        else:
            assert action.amount is not None
            paid = action.amount - player.round_bet
            player.stack -= paid
            player.round_bet = action.amount
            player.contribution += paid
            current_bet = action.amount
            record["amount"] = action.amount
        player.all_in = player.stack == 0
        self.current_actions.append(record)
        return current_bet

    def _settle(
        self, hand_number: int, small_seat: int, big_seat: int
    ) -> HandResult:
        total_pot = sum(player.contribution for player in self.players)
        payouts = [0] * len(self.players)
        side_pots: list[SidePot] = []
        levels = sorted({player.contribution for player in self.players if player.contribution})
        previous = 0
        tie_order = self._clockwise(self.button)

        for cap in levels:
            contributors = [
                player.seat
                for player in self.players
                if player.contribution >= cap
            ]
            layer = (cap - previous) * len(contributors)
            eligible = [seat for seat in contributors if not self.players[seat].folded]
            if eligible:
                best = max(
                    self.config.rank(
                        self.players[seat].number or 0,
                        self._community,
                    )
                    for seat in eligible
                )
                winners = [
                    seat
                    for seat in eligible
                    if self.config.rank(
                        self.players[seat].number or 0,
                        self._community,
                    ) == best
                ]
                ordered_winners = [seat for seat in tie_order if seat in winners]
                share, remainder = divmod(layer, len(ordered_winners))
                for index, seat in enumerate(ordered_winners):
                    payouts[seat] += share + int(index < remainder)
            else:
                # Defensive refund for an impossible-in-normal-play folded-only layer.
                ordered_winners = contributors
                refund = cap - previous
                for seat in contributors:
                    payouts[seat] += refund
            side_pots.append(
                SidePot(cap, layer, tuple(eligible), tuple(ordered_winners))
            )
            previous = cap

        for seat, payout in enumerate(payouts):
            self.players[seat].stack += payout
        if sum(payouts) != total_pot:
            raise RuntimeError("side-pot settlement did not distribute the pot")
        expected = self.config.starting_stack * len(self.players)
        if sum(player.stack for player in self.players) != expected:
            raise RuntimeError("chip conservation failed after hand settlement")

        showdown = len(self._unfolded_seats()) > 1
        shown = (
            {
                seat: int(self.players[seat].number or 0)
                for seat in self._unfolded_seats()
            }
            if showdown
            else {}
        )
        winners = tuple(
            seat
            for seat, payout in enumerate(payouts)
            if payout and (showdown or not self.players[seat].folded)
        )
        return HandResult(
            hand_number=hand_number,
            button_seat=self.button,
            small_blind_seat=small_seat,
            big_blind_seat=big_seat,
            community_number=self._community if self._community_revealed else None,
            pot=total_pot,
            winners=winners,
            shown_numbers=shown,
            actions=tuple(dict(action) for action in self.current_actions),
            contributions=tuple(player.contribution for player in self.players),
            payouts=tuple(payouts),
            stacks_after=tuple(player.stack for player in self.players),
            side_pots=tuple(side_pots),
        )


def make_phase3_policy_strategy(
    *,
    policy_config: Any | None = None,
    seed_path: str | Path | None = None,
    knowledge: Any | Mapping[str, Any] | None = None,
) -> DecisionCallback:
    """Build one isolated production-policy callback for a simulated leg.

    The public engine intentionally owns process-global runtime learning.  A
    benchmark must not share that state across seeds or policy candidates, so
    this adapter clones seed knowledge and owns its own ``RuntimeStore``.  Call
    this factory once per simulated leg (or pass it as ``strategy_factory`` to
    :func:`benchmark`) while retaining normal within-leg learning.
    """

    from .learning import EventKnowledge, RuntimeStore
    from .policy import HighVariancePolicy
    from .protocol import parse_payload, validate_response

    if seed_path is not None and knowledge is not None:
        raise ValueError("provide either seed_path or knowledge, not both")

    if seed_path is not None:
        source = RuntimeStore(seed_path).knowledge
    elif knowledge is None:
        source = EventKnowledge()
    elif isinstance(knowledge, EventKnowledge):
        source = knowledge
    elif isinstance(knowledge, Mapping):
        source = EventKnowledge.from_dict(knowledge)
    else:
        raise TypeError("knowledge must be EventKnowledge or a seed mapping")

    # Serialization is the canonical, deterministic deep-copy boundary for
    # event knowledge.  The caller's seed remains read-only during simulation.
    isolated = EventKnowledge.from_dict(source.to_dict())
    store = RuntimeStore(knowledge=isolated)
    policy = HighVariancePolicy(policy_config)

    def decide(raw: Mapping[str, Any]) -> ActionLike:
        request = parse_payload(raw)
        with store.lock:
            session = store.ingest(request)
            proposed = policy.decide(request, session, store.knowledge)
        return validate_response(request, proposed)

    return decide


def simulate_leg(
    seed: int,
    hero_strategy: DecisionCallback | None = None,
    opponent_strategies: Mapping[int, DecisionCallback] | None = None,
    config: SimulationConfig | None = None,
) -> LegResult:
    """Convenience wrapper for one independently reproducible phase-3 leg."""

    return ShowdownSimulator(seed, config).run(hero_strategy, opponent_strategies)


def benchmark(
    strategy: DecisionCallback | None = None,
    *,
    strategy_factory: StrategyFactory | None = None,
    trials: int = 100,
    base_seed: int = 0,
    opponent_strategies: Mapping[int, DecisionCallback] | None = None,
    config: SimulationConfig | None = None,
) -> BenchmarkReport:
    """Measure clear rate, bust rate, and the 90th-percentile hero delta."""

    if trials <= 0:
        raise ValueError("trials must be positive")
    if strategy is not None and strategy_factory is not None:
        raise ValueError("provide either strategy or strategy_factory, not both")
    results: list[LegResult] = []
    for trial in range(trials):
        trial_strategy = strategy_factory() if strategy_factory is not None else strategy
        results.append(
            simulate_leg(
                base_seed + trial,
                trial_strategy,
                opponent_strategies,
                config,
            )
        )
    deltas = sorted(result.hero_delta for result in results)
    p90_index = max(0, math.ceil(0.90 * trials) - 1)
    return BenchmarkReport(
        trials=trials,
        clear_rate=sum(result.cleared for result in results) / trials,
        bust_rate=sum(result.hero_busted for result in results) / trials,
        upper_tail_delta=deltas[p90_index],
        mean_delta=sum(deltas) / trials,
        min_delta=deltas[0],
        max_delta=deltas[-1],
    )


def tune(
    candidates: Mapping[str, DecisionCallback],
    *,
    trials: int = 100,
    base_seed: int = 0,
    opponent_strategies: Mapping[int, DecisionCallback] | None = None,
    config: SimulationConfig | None = None,
) -> list[tuple[str, BenchmarkReport]]:
    """Benchmark named candidates on identical seeds, best clear rate first."""

    reports = [
        (
            name,
            benchmark(
                strategy,
                trials=trials,
                base_seed=base_seed,
                opponent_strategies=opponent_strategies,
                config=config,
            ),
        )
        for name, strategy in candidates.items()
    ]
    return sorted(
        reports,
        key=lambda item: (
            item[1].clear_rate,
            item[1].upper_tail_delta,
            -item[1].bust_rate,
        ),
        reverse=True,
    )


__all__ = [
    "Action",
    "BenchmarkReport",
    "DecisionCallback",
    "HandResult",
    "LegResult",
    "ScriptedArchetype",
    "ShowdownSimulator",
    "SidePot",
    "SimulationConfig",
    "StrategyFactory",
    "benchmark",
    "built_in_strategy",
    "cleared",
    "default_opponents",
    "make_phase3_policy_strategy",
    "simulate_leg",
    "standard_rank",
    "tune",
]
