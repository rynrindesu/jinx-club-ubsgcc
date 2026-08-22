"""In-memory learning and seed serialization for the phase 3 SHOWDOWN bot.

The live service learns only from completed ``recent_hands`` and never persists
runtime state.  Offline replay tooling can feed full dealt numbers to
``EventKnowledge.observe_hand``; those numbers train opponent behaviour, while
rule inference remains restricted to cards actually present at showdown.
"""

from __future__ import annotations

import json
import math
import threading
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .protocol import ACTIONS, ActionRecord, MoveRequest, Player, RecentHand


PROFILE_ACTIONS = ("fold", "check", "call", "bet", "raise")
_CONTEXT_SEPARATOR = "|"
_UNKNOWN = "unknown"


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, value))


def _normalise(values: Sequence[float]) -> list[float]:
    total = sum(value for value in values if math.isfinite(value) and value > 0.0)
    if total <= 0.0:
        return [1.0 / len(values)] * len(values) if values else []
    return [max(0.0, value) / total if math.isfinite(value) else 0.0 for value in values]


def _normalise_round(value: Any) -> str:
    return value if value in {"pre_reveal", "post_reveal"} else _UNKNOWN


def _normalise_facing(value: Any) -> str:
    if isinstance(value, bool):
        return "facing" if value else "free"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "facing" if value > 0 else "free"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {
            "facing",
            "facing_bet",
            "bet",
            "raise",
            "raised",
            "yes",
            "true",
        }:
            return "facing"
        if lowered in {"free", "checked_to", "none", "no", "false"}:
            return "free"
    return _UNKNOWN


def size_bucket(
    amount: int | float | None = None,
    pot: int | float | None = None,
    *,
    all_in: bool = False,
) -> str:
    """Bucket a wager by its incremental size relative to the pot."""

    if all_in:
        return "all_in"
    if amount is None or amount <= 0:
        return "none"
    if pot is None or pot <= 0:
        # Absolute fallback for sparse live-history records.
        if amount <= 3:
            return "small"
        if amount <= 12:
            return "medium"
        return "large"
    ratio = float(amount) / float(pot)
    if ratio <= 0.40:
        return "small"
    if ratio <= 0.85:
        return "medium"
    if ratio <= 1.50:
        return "large"
    return "all_in" if ratio >= 3.0 else "overbet"


def _normalise_size(value: Any) -> str:
    if isinstance(value, str):
        lowered = value.strip().lower()
        aliases = {"jam": "all_in", "all-in": "all_in", "pot": "large"}
        lowered = aliases.get(lowered, lowered)
        if lowered in {"none", "small", "medium", "large", "overbet", "all_in"}:
            return lowered
    return _UNKNOWN


def _live_bucket(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return _UNKNOWN
    return "6+" if value >= 6 else str(value)


def _normalise_position(value: Any) -> str:
    if isinstance(value, str):
        lowered = value.strip().lower()
        aliases = {
            "first": "early",
            "last": "late",
            "button": "late",
            "in_position": "late",
            "out_of_position": "early",
        }
        lowered = aliases.get(lowered, lowered)
        if lowered in {"early", "middle", "late"}:
            return lowered
    return _UNKNOWN


def strength_bucket(value: float | int | None) -> str:
    if value is None or isinstance(value, bool):
        return _UNKNOWN
    try:
        strength = _clamp(float(value))
    except (TypeError, ValueError):
        return _UNKNOWN
    if strength < 0.25:
        return "weak"
    if strength < 0.55:
        return "medium"
    if strength < 0.80:
        return "strong"
    return "premium"


def position_bucket(
    seat: int,
    button_seat: int | None,
    active_seats: Sequence[int],
    round_name: str,
) -> str:
    """Return early/middle/late for the known active seating ring."""

    seats = sorted(set(active_seats))
    if button_seat is None or seat not in seats or button_seat not in seats:
        return _UNKNOWN
    if len(seats) <= 1:
        return "late"
    button_index = seats.index(button_seat)
    clockwise = seats[button_index + 1 :] + seats[: button_index + 1]
    if round_name == "pre_reveal" and len(clockwise) > 2:
        # The first two seats after the button post the blinds; action opens at
        # the following active seat.  Rotate those two to the end.
        action_order = clockwise[2:] + clockwise[:2]
    elif round_name == "pre_reveal" and len(clockwise) == 2:
        # Heads-up: button/small blind acts first pre-reveal.
        action_order = [button_seat, next(s for s in seats if s != button_seat)]
    else:
        # Post-reveal action starts just after the button, leaving it last.
        action_order = clockwise
    try:
        fraction = action_order.index(seat) / max(1, len(action_order) - 1)
    except ValueError:
        return _UNKNOWN
    if fraction < 1.0 / 3.0:
        return "early"
    if fraction < 2.0 / 3.0:
        return "middle"
    return "late"


def _context_key(parts: Sequence[str]) -> str:
    return _CONTEXT_SEPARATOR.join(parts)


def _decode_context(key: str) -> tuple[str, str, str, str, str, str] | None:
    parts = tuple(key.split(_CONTEXT_SEPARATOR))
    if len(parts) != 6:
        return None
    return parts  # type: ignore[return-value]


def _rule_strength(
    rule_model: Any,
    number: int,
    community: int | None,
    round_name: str,
) -> float:
    # Pre-reveal decisions cannot depend on a community value that had not yet
    # been dealt.  Raw number percentile is a neutral, rule-agnostic proxy.
    if round_name != "post_reveal" or community is None:
        return (number - 1) / 12.0
    try:
        return _clamp(float(rule_model.strength(number, community)))
    except (AttributeError, TypeError, ValueError, ArithmeticError):
        return (number - 1) / 12.0


@dataclass(slots=True)
class OpponentProfile:
    """Smoothed action frequencies for one stable opponent name."""

    name: str
    alpha: float = 0.75
    counts: dict[str, dict[str, float]] = field(default_factory=dict)
    observations: float = 0.0

    def _parts(
        self,
        round_name: Any,
        facing: Any,
        wager_size: Any,
        live_count: Any,
        position: Any,
        strength: float | int | None,
    ) -> tuple[str, str, str, str, str, str]:
        return (
            _normalise_round(round_name),
            _normalise_facing(facing),
            _normalise_size(wager_size),
            _live_bucket(live_count),
            _normalise_position(position),
            strength_bucket(strength),
        )

    def observe(
        self,
        action: str | ActionRecord,
        *,
        round_name: str | None = None,
        facing: str | bool | int | None = None,
        size_bucket: str | None = None,
        live_count: int | None = None,
        position_bucket: str | None = None,
        strength: float | None = None,
        private_number: int | None = None,
        community_number: int | None = None,
        rule_model: Any = None,
        weight: float = 1.0,
    ) -> None:
        """Observe one action, optionally with the player's known dealt number."""

        record = action if isinstance(action, ActionRecord) else None
        action_name = record.action if record is not None else str(action).lower()
        if action_name not in ACTIONS:
            return
        if not math.isfinite(weight) or weight <= 0.0:
            return

        effective_round = round_name or (record.round if record is not None else None)
        effective_facing = facing
        if effective_facing is None:
            effective_facing = action_name in {"fold", "call", "raise"}
        effective_size = size_bucket
        if effective_size is None:
            effective_size = "none" if action_name in {"check", "bet"} else _UNKNOWN
        effective_live = live_count or (record.live_players if record is not None else None)
        effective_position = position_bucket or (
            record.position if record is not None else None
        )
        effective_strength = strength
        if effective_strength is None and private_number is not None:
            effective_strength = _rule_strength(
                rule_model, private_number, community_number, effective_round or _UNKNOWN
            )

        key = _context_key(
            self._parts(
                effective_round,
                effective_facing,
                effective_size,
                effective_live,
                effective_position,
                effective_strength,
            )
        )
        bucket = self.counts.setdefault(key, {})
        bucket[action_name] = bucket.get(action_name, 0.0) + float(weight)
        self.observations += float(weight)

    # Friendly alias used by replay/import callers.
    observe_action = observe

    def _aggregate(
        self,
        query: tuple[str, str, str, str, str, str],
        dimensions: tuple[int, ...],
    ) -> tuple[dict[str, float], float]:
        result: dict[str, float] = defaultdict(float)
        total = 0.0
        for key, action_counts in self.counts.items():
            observed = _decode_context(key)
            if observed is None:
                continue
            if any(observed[index] != query[index] for index in dimensions):
                continue
            for action, count in action_counts.items():
                if action in ACTIONS and math.isfinite(count) and count > 0.0:
                    result[action] += count
                    total += count
        return dict(result), total

    def _relevant_counts(
        self, query: tuple[str, str, str, str, str, str]
    ) -> tuple[dict[str, float], float, int]:
        # Progressively back off sparse dimensions, retaining strength for as
        # long as possible.  This learns useful tendencies after only one leg
        # without pretending a single exact context is conclusive.
        levels = (
            (0, 1, 2, 3, 4, 5),
            (0, 1, 2, 5),
            (0, 1, 5),
            (1, 5),
            (1,),
            (),
        )
        for backoff_level, dimensions in enumerate(levels):
            counts, total = self._aggregate(query, dimensions)
            if total > 0.0:
                return counts, total, backoff_level
        return {}, 0.0, len(levels)

    def action_likelihood(
        self,
        action: str,
        round_name: str,
        facing: str | bool | int,
        size_bucket: str,
        live_count: int,
        position_bucket: str,
        strength: float,
    ) -> float:
        """Smoothed likelihood of ``action`` in the supplied context."""

        action = action.lower()
        if action not in ACTIONS:
            return 1e-6
        query = self._parts(
            round_name, facing, size_bucket, live_count, position_bucket, strength
        )
        counts, total, backoff_level = self._relevant_counts(query)
        facing_bin = query[1]
        if facing_bin == "facing":
            choices = ("fold", "call", "raise")
            prior = {"fold": 0.34, "call": 0.48, "raise": 0.18}
        elif facing_bin == "free":
            choices = ("check", "bet")
            prior = {"check": 0.62, "bet": 0.38}
        else:
            choices = PROFILE_ACTIONS
            prior = {choice: 1.0 / len(choices) for choice in choices}
        if action not in choices:
            return 1e-6
        # A coarse fallback is deliberately shrunk more heavily than an exact
        # context.  That preserves broad opponent tendencies without letting one
        # revealed hand erase strength-dependent range information.
        effective_alpha = self.alpha * (1.0 + 0.75 * backoff_level)
        pseudo_total = effective_alpha * len(choices)
        return (counts.get(action, 0.0) + pseudo_total * prior[action]) / (
            total + pseudo_total
        )

    def continue_probability(
        self,
        round_name: str,
        facing: str | bool | int,
        size_bucket: str,
        live_count: int,
        position_bucket: str,
        strength: float,
    ) -> float:
        """Probability of not folding when facing a wager (call or raise)."""

        facing_bin = _normalise_facing(facing)
        if facing_bin == "free":
            return 1.0
        fold = self.action_likelihood(
            "fold",
            round_name,
            "facing",
            size_bucket,
            live_count,
            position_bucket,
            strength,
        )
        call = self.action_likelihood(
            "call",
            round_name,
            "facing",
            size_bucket,
            live_count,
            position_bucket,
            strength,
        )
        raise_probability = self.action_likelihood(
            "raise",
            round_name,
            "facing",
            size_bucket,
            live_count,
            position_bucket,
            strength,
        )
        denominator = fold + call + raise_probability
        return _clamp((call + raise_probability) / denominator) if denominator else 0.66

    def range_for_action_history(
        self,
        actions: Iterable[ActionRecord],
        seat: int | None = None,
        round_name: str = "pre_reveal",
        community: int | None = None,
        rule_model: Any = None,
        live_count: int = 2,
        position_bucket: str = _UNKNOWN,
        prior: Mapping[int, float] | Sequence[float] | None = None,
    ) -> list[float]:
        """Return posterior probabilities for numbers 1..13.

        The result is a 13-element list (index zero represents number 1).  All
        table actions may be supplied; ``seat`` filters the target opponent while
        still using other seats' folds/aggression to reconstruct what was faced.
        """

        if isinstance(prior, Mapping):
            weights = [float(prior.get(number, 0.0)) for number in range(1, 14)]
        elif prior is not None and len(prior) == 13:
            weights = [float(value) for value in prior]
        else:
            weights = [1.0] * 13
        weights = _normalise(weights)

        contributions: dict[str, dict[int, int]] = {
            "pre_reveal": defaultdict(int),
            "post_reveal": defaultdict(int),
        }
        current_max = {"pre_reveal": 2, "post_reveal": 0}
        last_size = {"pre_reveal": _UNKNOWN, "post_reveal": _UNKNOWN}
        live_now = max(1, live_count)

        for record in actions:
            if not isinstance(record, ActionRecord):
                continue
            street = _normalise_round(record.round)
            actor_contribution = contributions.setdefault(street, defaultdict(int)).get(
                record.seat, 0
            )
            inferred_facing = (
                record.to_call > 0
                if record.to_call is not None
                else record.action in {"fold", "call", "raise"}
            )
            faced_size = last_size.get(street, _UNKNOWN) if inferred_facing else "none"

            if seat is None or record.seat == seat:
                updated: list[float] = []
                for number, old_weight in enumerate(weights, start=1):
                    strength = _rule_strength(
                        rule_model,
                        number,
                        community if street == "post_reveal" else None,
                        street,
                    )
                    likelihood = self.action_likelihood(
                        record.action,
                        street if street != _UNKNOWN else round_name,
                        inferred_facing,
                        faced_size,
                        record.live_players or live_now,
                        record.position or position_bucket,
                        strength,
                    )
                    updated.append(old_weight * max(likelihood, 1e-6))
                weights = _normalise(updated)

            if record.amount is not None:
                increment = max(0, record.amount - actor_contribution)
                contributions[street][record.seat] = record.amount
            elif record.action == "call":
                increment = max(0, current_max.get(street, 0) - actor_contribution)
                contributions[street][record.seat] = current_max.get(street, 0)
            else:
                increment = 0
            if record.action in {"bet", "raise"}:
                previous_max = current_max.get(street, 0)
                new_total = record.amount if record.amount is not None else previous_max
                current_max[street] = max(previous_max, new_total)
                last_size[street] = size_bucket(
                    max(0, new_total - actor_contribution), record.pot_before
                )
            if record.action == "fold":
                live_now = max(1, live_now - 1)

        return weights

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "alpha": self.alpha,
            "observations": self.observations,
            "counts": {
                key: {action: action_counts[action] for action in sorted(action_counts)}
                for key, action_counts in sorted(self.counts.items())
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, name: str | None = None) -> "OpponentProfile":
        profile_name = name or str(data.get("name") or _UNKNOWN)
        try:
            alpha = float(data.get("alpha", 0.75))
        except (TypeError, ValueError):
            alpha = 0.75
        if not math.isfinite(alpha) or alpha <= 0.0:
            alpha = 0.75
        result = cls(name=profile_name, alpha=alpha)
        raw_counts = data.get("counts", {})
        if isinstance(raw_counts, Mapping):
            for key, value in raw_counts.items():
                if not isinstance(key, str) or _decode_context(key) is None:
                    continue
                if not isinstance(value, Mapping):
                    continue
                clean: dict[str, float] = {}
                for action, count in value.items():
                    try:
                        numeric = float(count)
                    except (TypeError, ValueError):
                        continue
                    if action in ACTIONS and math.isfinite(numeric) and numeric > 0.0:
                        clean[str(action)] = numeric
                if clean:
                    result.counts[key] = clean
        # Derive the observation total from counts: it is harder for a hand-edited
        # seed to leave a stale metadata value than to preserve every count.
        result.observations = sum(sum(bucket.values()) for bucket in result.counts.values())
        return result


@dataclass(slots=True)
class MatchState:
    match_id: str
    leg_number: int | None
    table_rule: str
    processed_hands: set[int] = field(default_factory=set)
    scout_spend: int = 0
    last_hand_number: int = 0

    @property
    def key(self) -> tuple[str, int | None, str]:
        return (self.match_id, self.leg_number, self.table_rule)

    @property
    def scout_budget_remaining(self) -> int:
        return max(0, 15 - self.scout_spend)

    def mark_processed(self, hand_number: int) -> bool:
        if hand_number in self.processed_hands:
            return False
        self.processed_hands.add(hand_number)
        self.last_hand_number = max(self.last_hand_number, hand_number)
        return True

    def record_scout_spend(self, amount: int) -> int:
        if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
            return self.scout_spend
        self.scout_spend += amount
        return self.scout_spend


def _new_rule_model(codename: str) -> Any:
    # Delayed import keeps protocol/state tests independent and avoids a circular
    # import if rules.py imports shared dataclasses.
    from .rules import RuleModel

    return RuleModel(codename)


def _rule_from_dict(codename: str, data: Mapping[str, Any]) -> Any:
    from .rules import RuleModel

    try:
        # The mapping key is authoritative for hand-edited/legacy seeds that did
        # not repeat the codename inside each rule object.
        enriched = dict(data)
        enriched.setdefault("codename", codename)
        return RuleModel.from_dict(enriched)
    except (AttributeError, TypeError, ValueError, KeyError):
        model = RuleModel(codename)
        return model


def _coerce_seat_number_map(value: Any) -> dict[int, int]:
    result: dict[int, int] = {}
    if not isinstance(value, Mapping):
        return result
    for raw_seat, raw_number in value.items():
        try:
            if isinstance(raw_seat, bool) or isinstance(raw_number, bool):
                continue
            seat = int(raw_seat)
            number = int(raw_number)
            if seat >= 0 and 1 <= number <= 13:
                result[seat] = number
        except (TypeError, ValueError):
            continue
    return result


def _player_name(value: Any, seat: int) -> str | None:
    if isinstance(value, Player):
        return value.name
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, Mapping):
        name = value.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return None


@dataclass(slots=True)
class EventKnowledge:
    """Knowledge shared across legs and attempts in one service process."""

    rules: dict[str, Any] = field(default_factory=dict)
    opponents: dict[str, OpponentProfile] = field(default_factory=dict)
    source_hashes: set[str] = field(default_factory=set)
    observation_keys: set[str] = field(default_factory=set)
    version: int = 1

    def get_rule(self, codename: str) -> Any:
        codename = str(codename).strip()
        if not codename:
            raise ValueError("table-rule codename cannot be empty")
        if codename not in self.rules:
            self.rules[codename] = _new_rule_model(codename)
        return self.rules[codename]

    def get_opponent(self, name: str) -> OpponentProfile:
        name = str(name).strip()
        if not name:
            name = _UNKNOWN
        if name not in self.opponents:
            self.opponents[name] = OpponentProfile(name=name)
        return self.opponents[name]

    def add_source_hash(self, source_hash: str) -> bool:
        """Register a replay source and return ``False`` when already imported."""

        value = str(source_hash).strip()
        if not value or value in self.source_hashes:
            return False
        self.source_hashes.add(value)
        return True

    def observe_opponent_action(
        self, name: str, action: str | ActionRecord, **context: Any
    ) -> OpponentProfile:
        profile = self.get_opponent(name)
        profile.observe(action, **context)
        return profile

    def observe_hand(
        self,
        table_rule: str,
        hand: RecentHand | Mapping[str, Any],
        players_by_seat: Mapping[int, Player | Mapping[str, Any] | str] | None = None,
        full_numbers: Mapping[int | str, int] | None = None,
        *,
        source_key: str | None = None,
        hand_key: str | int | None = None,
        your_seat: int | None = None,
        big_blind: int = 2,
    ) -> bool:
        """Learn from one completed hand.

        ``hand`` may be ``RecentHand`` or the equivalent mapping. ``full_numbers``
        is optional replay-only dealt-card data and is used solely for opponent
        action training.  Rule evidence always comes from ``shown_numbers``.
        Supplying both ``source_key`` and ``hand_key`` makes repeated calls
        idempotent; live deduplication is owned by ``MatchState`` instead.
        """

        dedupe_key = None
        if source_key is not None and hand_key is not None:
            dedupe_key = f"{source_key}:{hand_key}"
            if dedupe_key in self.observation_keys:
                return False

        raw_hand = hand if isinstance(hand, Mapping) else None
        if isinstance(hand, RecentHand):
            parsed = hand
        elif isinstance(hand, Mapping):
            # Tolerant replay conversion, intentionally independent of live
            # request validation.
            try:
                actions: list[ActionRecord] = []
                raw_actions = hand.get("actions", ())
                if isinstance(raw_actions, Sequence) and not isinstance(
                    raw_actions, (str, bytes, bytearray)
                ):
                    for entry in raw_actions:
                        if not isinstance(entry, Mapping):
                            continue
                        try:
                            action_name = str(entry.get("action", "")).lower()
                            street = str(entry.get("round", ""))
                            seat = int(entry.get("seat"))
                            if action_name not in ACTIONS or street not in {
                                "pre_reveal",
                                "post_reveal",
                            }:
                                continue
                            amount_value = entry.get("amount")
                            amount = (
                                int(amount_value)
                                if amount_value is not None
                                and not isinstance(amount_value, bool)
                                else None
                            )
                            actions.append(
                                ActionRecord(
                                    round=street,
                                    seat=seat,
                                    action=action_name,
                                    amount=amount,
                                    pot_before=(
                                        int(entry["pot_before"])
                                        if isinstance(entry.get("pot_before"), int)
                                        and not isinstance(entry.get("pot_before"), bool)
                                        else None
                                    ),
                                    to_call=(
                                        int(entry["to_call"])
                                        if isinstance(entry.get("to_call"), int)
                                        and not isinstance(entry.get("to_call"), bool)
                                        else None
                                    ),
                                    live_players=(
                                        int(entry["live_players"])
                                        if isinstance(entry.get("live_players"), int)
                                        and not isinstance(entry.get("live_players"), bool)
                                        else None
                                    ),
                                    position=(
                                        str(entry["position"])
                                        if isinstance(entry.get("position"), str)
                                        else None
                                    ),
                                )
                            )
                        except (TypeError, ValueError):
                            continue
                raw_winners = hand.get("winners", ())
                winner_values: list[int] = []
                if isinstance(raw_winners, Sequence) and not isinstance(
                    raw_winners, (str, bytes, bytearray)
                ):
                    for winner in raw_winners:
                        if isinstance(winner, Mapping):
                            winner = winner.get(
                                "seat", winner.get("player_id", winner.get("player"))
                            )
                        try:
                            if isinstance(winner, bool):
                                continue
                            parsed_winner = int(winner)
                        except (TypeError, ValueError):
                            continue
                        if parsed_winner >= 0 and parsed_winner not in winner_values:
                            winner_values.append(parsed_winner)
                winners = tuple(winner_values)
                community_raw = hand.get("community_number")
                community = (
                    int(community_raw)
                    if isinstance(community_raw, int)
                    and not isinstance(community_raw, bool)
                    and 1 <= community_raw <= 13
                    else None
                )
                parsed = RecentHand(
                    hand_number=int(hand.get("hand_number", 0)),
                    community_number=community,
                    winners=winners,
                    pot=max(0, int(hand.get("pot", 0))),
                    shown_numbers=_coerce_seat_number_map(
                        hand.get("shown_numbers", {})
                    ),
                    actions=tuple(actions),
                    button_seat=(
                        int(hand["button_seat"])
                        if isinstance(hand.get("button_seat"), int)
                        and not isinstance(hand.get("button_seat"), bool)
                        else None
                    ),
                )
            except (TypeError, ValueError):
                return False
        else:
            return False

        if full_numbers is None and raw_hand is not None:
            for candidate_key in (
                "full_numbers",
                "dealt_numbers",
                "private_numbers",
                "numbers",
            ):
                candidate = raw_hand.get(candidate_key)
                if isinstance(candidate, Mapping):
                    full_numbers = candidate  # type: ignore[assignment]
                    break
        known_numbers = dict(parsed.shown_numbers)
        known_numbers.update(_coerce_seat_number_map(full_numbers))

        rule = self.get_rule(table_rule)
        ambiguous = (
            parsed.community_number is None
            or len(parsed.winners) != 1
            or len(parsed.shown_numbers) < 2
            or parsed.winners[0] not in parsed.shown_numbers
        )
        if raw_hand is not None and (
            raw_hand.get("side_pots") or raw_hand.get("multiple_pots")
        ):
            ambiguous = True
        if not ambiguous:
            try:
                rule.observe_showdown(
                    parsed.community_number,
                    parsed.shown_numbers,
                    parsed.winners,
                    ambiguous=False,
                )
            except TypeError:
                # Accommodate minimal rule-model implementations used in tests.
                try:
                    rule.observe_showdown(
                        parsed.community_number,
                        parsed.shown_numbers,
                        parsed.winners,
                    )
                except (AttributeError, TypeError, ValueError):
                    pass
            except (AttributeError, ValueError, ArithmeticError):
                pass

        seat_names: dict[int, str] = {}
        if players_by_seat:
            for raw_seat, player in players_by_seat.items():
                try:
                    seat = int(raw_seat)
                except (TypeError, ValueError):
                    continue
                name = _player_name(player, seat)
                if name:
                    seat_names[seat] = name

        active_seats = sorted(seat_names) or sorted(known_numbers) or sorted(
            parsed.shown_numbers
        )
        live_now = max(2, len(active_seats))
        contributions: dict[str, dict[int, int]] = {
            "pre_reveal": defaultdict(int),
            "post_reveal": defaultdict(int),
        }
        current_max = {"pre_reveal": max(0, big_blind), "post_reveal": 0}
        last_aggression_size = {"pre_reveal": _UNKNOWN, "post_reveal": _UNKNOWN}
        estimated_pot = max(0, big_blind) + max(0, big_blind // 2)

        for action in parsed.actions:
            street = action.round
            round_contributions = contributions[street]
            before = round_contributions.get(action.seat, 0)
            facing = (
                action.to_call > 0
                if action.to_call is not None
                else action.action in {"fold", "call", "raise"}
            )
            faced_size = last_aggression_size[street] if facing else "none"
            player_name = seat_names.get(action.seat)
            private_number = known_numbers.get(action.seat)
            if (
                player_name
                and player_name.lower() != "you"
                and (your_seat is None or action.seat != your_seat)
            ):
                action_position = action.position or position_bucket(
                    action.seat, parsed.button_seat, active_seats, street
                )
                # Live fold winners do not reveal every private number.  Their
                # actions still teach the opponent's unconditional tendencies,
                # but only a shown/replay-supplied number may populate a
                # strength-specific bucket.
                strength = (
                    _rule_strength(
                        rule,
                        private_number,
                        parsed.community_number if street == "post_reveal" else None,
                        street,
                    )
                    if private_number is not None
                    else None
                )
                self.observe_opponent_action(
                    player_name,
                    action,
                    round_name=street,
                    facing=facing,
                    size_bucket=faced_size,
                    live_count=action.live_players or live_now,
                    position_bucket=action_position,
                    strength=strength,
                )

            if action.amount is not None:
                increment = max(0, action.amount - before)
                round_contributions[action.seat] = action.amount
            elif action.action == "call":
                increment = max(0, current_max[street] - before)
                round_contributions[action.seat] = current_max[street]
            else:
                increment = 0
            if action.action in {"bet", "raise"}:
                new_total = action.amount if action.amount is not None else current_max[street]
                last_aggression_size[street] = size_bucket(
                    max(0, new_total - before), action.pot_before or estimated_pot
                )
                current_max[street] = max(current_max[street], new_total)
            estimated_pot += increment
            if action.action == "fold":
                live_now = max(1, live_now - 1)

        if dedupe_key is not None:
            self.observation_keys.add(dedupe_key)
        return True

    def to_dict(self) -> dict[str, Any]:
        rules: dict[str, Any] = {}
        for codename, model in sorted(self.rules.items()):
            try:
                rules[codename] = model.to_dict()
            except (AttributeError, TypeError, ValueError):
                continue
        return {
            "version": self.version,
            "source_hashes": sorted(self.source_hashes),
            "observation_keys": sorted(self.observation_keys),
            "rules": rules,
            "opponents": {
                name: profile.to_dict()
                for name, profile in sorted(self.opponents.items())
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EventKnowledge":
        try:
            version = int(data.get("version", 1))
        except (TypeError, ValueError):
            version = 1
        result = cls(version=max(1, version))

        raw_hashes = data.get("source_hashes", ())
        if isinstance(raw_hashes, Sequence) and not isinstance(
            raw_hashes, (str, bytes, bytearray)
        ):
            result.source_hashes = {
                value for value in raw_hashes if isinstance(value, str) and value
            }
        raw_observations = data.get("observation_keys", ())
        if isinstance(raw_observations, Sequence) and not isinstance(
            raw_observations, (str, bytes, bytearray)
        ):
            result.observation_keys = {
                value for value in raw_observations if isinstance(value, str) and value
            }

        raw_rules = data.get("rules", {})
        if isinstance(raw_rules, Mapping):
            for codename, model_data in raw_rules.items():
                if not isinstance(codename, str) or not isinstance(model_data, Mapping):
                    continue
                try:
                    result.rules[codename] = _rule_from_dict(codename, model_data)
                except (ImportError, AttributeError, TypeError, ValueError):
                    continue

        raw_opponents = data.get("opponents", {})
        if isinstance(raw_opponents, Mapping):
            for name, profile_data in raw_opponents.items():
                if isinstance(name, str) and isinstance(profile_data, Mapping):
                    result.opponents[name] = OpponentProfile.from_dict(
                        profile_data, name=name
                    )
        return result


class RuntimeStore:
    """Thread-safe process-local session and event knowledge store."""

    def __init__(
        self,
        seed_path: str | Path | None = None,
        *,
        knowledge: EventKnowledge | None = None,
    ) -> None:
        if knowledge is not None:
            self.knowledge = knowledge
        elif seed_path is not None:
            self.knowledge = self._load_seed(seed_path)
        else:
            self.knowledge = EventKnowledge()
        self.sessions: dict[tuple[str, int | None, str], MatchState] = {}
        self.lock = threading.RLock()

    @staticmethod
    def _load_seed(path: str | Path) -> EventKnowledge:
        seed = Path(path)
        if not seed.is_file():
            return EventKnowledge()
        try:
            with seed.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, Mapping):
                return EventKnowledge.from_dict(data)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            pass
        return EventKnowledge()

    def get_rule(self, codename: str) -> Any:
        with self.lock:
            return self.knowledge.get_rule(codename)

    def ingest(self, request: MoveRequest) -> MatchState:
        """Ingest every newly visible completed hand exactly once."""

        with self.lock:
            key = request.session_key
            state = self.sessions.get(key)
            if state is None:
                state = MatchState(
                    match_id=request.match_id,
                    leg_number=request.leg_number,
                    table_rule=request.table_rule,
                )
                self.sessions[key] = state

            players_by_seat = request.players_by_seat
            for hand in request.recent_hands:
                if not state.mark_processed(hand.hand_number):
                    continue
                try:
                    self.knowledge.observe_hand(
                        request.table_rule,
                        hand,
                        players_by_seat,
                        your_seat=request.your_seat,
                        big_blind=request.big_blind,
                    )
                except (AttributeError, TypeError, ValueError, ArithmeticError):
                    # Learning is advisory.  A malformed historical observation
                    # must never prevent the API from returning a legal move.
                    continue
            state.last_hand_number = max(state.last_hand_number, request.hand_number)
            return state


__all__ = [
    "OpponentProfile",
    "MatchState",
    "EventKnowledge",
    "RuntimeStore",
    "size_bucket",
    "strength_bucket",
    "position_bucket",
]
