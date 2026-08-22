"""Thread-safe state scopes for Phase 2 SHOWDOWN play."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from threading import RLock
from typing import Iterable, Mapping

from .rules import (
    RuleKnowledge,
    build_candidate_rules,
    extract_observations,
)


@dataclass(frozen=True)
class RangeEvidence:
    """One revealed opponent number, expressed in rule-relative strength."""

    strength: float
    action: str
    size_bucket: str
    position: str
    round_name: str
    reliability: float


@dataclass(frozen=True)
class OpponentProfile:
    """Rule-independent tendencies accumulated across the current attempt."""

    fold_to_open_rate: float
    reraise_rate: float
    post_fold_rate: float
    bet_after_check_rate: float
    aggression_rate: float
    open_responses: int
    post_responses: int
    checked_to: int
    decisions: int
    range_samples: tuple[RangeEvidence, ...] = ()
    range_shown_hands: int = 0

    @property
    def tight_folder(self) -> bool:
        return self.open_responses >= 4 and self.fold_to_open_rate > 0.52

    @property
    def calling_station(self) -> bool:
        return (
            self.open_responses >= 4 and self.fold_to_open_rate < 0.22
        ) or (self.post_responses >= 4 and self.post_fold_rate < 0.20)

    @property
    def aggressive(self) -> bool:
        return self.decisions >= 6 and (
            self.aggression_rate > 0.58
            or (self.checked_to >= 3 and self.bet_after_check_rate > 0.65)
        )

    @property
    def passive(self) -> bool:
        return self.decisions >= 6 and self.aggression_rate < 0.28

    def range_for(
        self,
        *,
        payload: Mapping[str, object],
        rule_knowledge: RuleKnowledge,
    ) -> tuple[float, ...]:
        """Estimate the opponent's 13-number range for the live action."""

        uniform = (1 / 13,) * 13
        context = _live_range_context(payload)
        if (
            context is None
            or self.range_shown_hands < 6
            or not self.range_samples
            or rule_knowledge.observation_count == 0
        ):
            return uniform

        current_action, size_bucket, position, round_name = context
        contextual_samples: list[tuple[RangeEvidence, float]] = []
        for sample in self.range_samples:
            action_weight = _action_similarity(current_action, sample.action)
            round_weight = 1.6 if sample.round_name == round_name else 0.55
            position_weight = _context_similarity(position, sample.position)
            size_weight = (
                _context_similarity(size_bucket, sample.size_bucket)
                if current_action in {"bet", "raise"}
                else 1.0
            )
            weight = (
                action_weight
                * round_weight
                * position_weight
                * size_weight
                * sample.reliability
            )
            if weight > 0:
                contextual_samples.append((sample, weight))
        if not contextual_samples:
            return uniform

        denominator = sum(weight for _, weight in contextual_samples)
        raw_community = _integer(payload.get("community_number"))
        community = (
            raw_community
            if raw_community is not None and 1 <= raw_community <= 13
            else None
        )
        likelihoods: list[float] = []
        for number in range(1, 14):
            strength = rule_knowledge.estimate(number, community).mean
            similarity = sum(
                weight
                * math.exp(-0.5 * ((strength - sample.strength) / 0.18) ** 2)
                for sample, weight in contextual_samples
            ) / denominator
            # The uniform component prevents a small behavioral sample from
            # assigning zero probability to any legal private number.
            likelihoods.append(0.20 + similarity)
        total = sum(likelihoods)
        return tuple(likelihood / total for likelihood in likelihoods)


@dataclass
class AttemptState:
    """Opponent evidence whose lifetime is exactly one four-leg attempt."""

    first_match_id: str
    opponent_token: str = ""
    match_ids: set[str] = field(default_factory=set)
    seen_hands: set[tuple[str, int]] = field(default_factory=set)
    open_responses: int = 0
    open_folds: int = 0
    open_reraises: int = 0
    post_responses: int = 0
    post_folds: int = 0
    checked_to: int = 0
    bets_after_check: int = 0
    decisions: int = 0
    aggressive_actions: int = 0
    range_samples: list[RangeEvidence] = field(default_factory=list)
    range_shown_hands: int = 0

    def __post_init__(self) -> None:
        self.match_ids.add(self.first_match_id)

    def ingest_hands(
        self,
        match_id: str,
        your_seat: object,
        hands: Iterable[Mapping[str, object]],
        rule_knowledge: RuleKnowledge,
    ) -> None:
        self.match_ids.add(match_id)
        for hand in hands:
            hand_number = _integer(hand.get("hand_number"))
            if hand_number is None or (match_id, hand_number) in self.seen_hands:
                continue
            self.seen_hands.add((match_id, hand_number))
            actions = hand.get("actions")
            if not isinstance(actions, list):
                continue
            decisions = [
                action
                for action in actions
                if isinstance(action, Mapping)
                and action.get("action") in {"fold", "check", "call", "bet", "raise"}
            ]
            self._ingest_actions(decisions, your_seat)
            self._ingest_range_evidence(
                hand, decisions, your_seat, rule_knowledge
            )

    def _ingest_actions(
        self, actions: list[Mapping[str, object]], your_seat: object
    ) -> None:
        for action in actions:
            if str(action.get("seat")) == str(your_seat):
                continue
            self.decisions += 1
            if action.get("action") in {"bet", "raise"}:
                self.aggressive_actions += 1

        pre = [action for action in actions if action.get("round") == "pre_reveal"]
        for index, action in enumerate(pre):
            if (
                str(action.get("seat")) == str(your_seat)
                and action.get("action") == "raise"
                and not any(
                    earlier.get("action") == "raise" for earlier in pre[:index]
                )
            ):
                response = _next_other_action(pre, index, your_seat)
                if response is not None and response.get("action") in {
                    "fold",
                    "call",
                    "raise",
                }:
                    self.open_responses += 1
                    self.open_folds += response.get("action") == "fold"
                    self.open_reraises += response.get("action") == "raise"
                break

        post = [action for action in actions if action.get("round") == "post_reveal"]
        for index, action in enumerate(post):
            if str(action.get("seat")) != str(your_seat):
                continue
            response = _next_other_action(post, index, your_seat)
            if response is None:
                continue
            if action.get("action") in {"bet", "raise"} and response.get(
                "action"
            ) in {"fold", "call", "raise"}:
                self.post_responses += 1
                self.post_folds += response.get("action") == "fold"
            elif action.get("action") == "check" and response.get("action") in {
                "check",
                "bet",
            }:
                self.checked_to += 1
                self.bets_after_check += response.get("action") == "bet"

    def _ingest_range_evidence(
        self,
        hand: Mapping[str, object],
        actions: list[Mapping[str, object]],
        your_seat: object,
        rule_knowledge: RuleKnowledge,
    ) -> None:
        shown = hand.get("shown_numbers")
        community = _integer(hand.get("community_number"))
        if not isinstance(shown, Mapping) or community is None:
            return
        if not 1 <= community <= 13:
            return
        opponent_seat = opponent_number = None
        for seat, raw_number in shown.items():
            number = _integer(raw_number)
            if str(seat) != str(your_seat) and number is not None:
                opponent_seat, opponent_number = seat, number
                break
        if opponent_number is None or not 1 <= opponent_number <= 13:
            return

        position = _position(
            opponent_seat,
            hand.get("button_seat"),
        )
        pot = max(1, _integer(hand.get("pot")) or 1)
        added = False
        for action in actions:
            action_name = str(action.get("action", ""))
            if (
                str(action.get("seat")) != str(opponent_seat)
                or action_name not in {"check", "call", "bet", "raise"}
            ):
                continue
            round_name = str(action.get("round", ""))
            estimate = rule_knowledge.estimate(
                opponent_number,
                community if round_name == "post_reveal" else None,
            )
            reliability = max(
                0.20,
                min(
                    1.0,
                    estimate.coverage * (1.0 - estimate.disagreement)
                    + (0.20 if estimate.confidence == "learned" else 0.0),
                ),
            )
            self.range_samples.append(
                RangeEvidence(
                    strength=estimate.mean,
                    action=action_name,
                    size_bucket=_size_bucket(action, pot),
                    position=position,
                    round_name=round_name,
                    reliability=reliability,
                )
            )
            added = True
        self.range_shown_hands += added

    def profile(self) -> OpponentProfile:
        return OpponentProfile(
            fold_to_open_rate=_smoothed_rate(
                self.open_folds, self.open_responses, 0.35
            ),
            reraise_rate=_smoothed_rate(
                self.open_reraises, self.open_responses, 0.18
            ),
            post_fold_rate=_smoothed_rate(
                self.post_folds, self.post_responses, 0.35
            ),
            bet_after_check_rate=_smoothed_rate(
                self.bets_after_check, self.checked_to, 0.45
            ),
            aggression_rate=_smoothed_rate(
                self.aggressive_actions, self.decisions, 0.45
            ),
            open_responses=self.open_responses,
            post_responses=self.post_responses,
            checked_to=self.checked_to,
            decisions=self.decisions,
            range_samples=tuple(self.range_samples),
            range_shown_hands=self.range_shown_hands,
        )


class Phase2State:
    """Own the event, attempt, and leg state used by the live endpoint."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._candidates = build_candidate_rules()
        self._rules: dict[str, RuleKnowledge] = {}
        self._attempt: AttemptState | None = None

    def observe_payload(
        self, payload: Mapping[str, object]
    ) -> tuple[RuleKnowledge, OpponentProfile]:
        table_rule = str(payload.get("table_rule", ""))
        match_id = str(payload.get("match_id", ""))
        your_seat = payload.get("your_seat")
        recent_hands = payload.get("recent_hands")
        hands = (
            [hand for hand in recent_hands if isinstance(hand, Mapping)]
            if isinstance(recent_hands, list)
            else []
        )

        with self._lock:
            knowledge = self._rules.setdefault(
                table_rule, RuleKnowledge(self._candidates)
            )
            for observation in extract_observations(
                table_rule=table_rule,
                match_id=match_id,
                your_seat=your_seat,
                hands=hands,
            ):
                knowledge.ingest(observation)

            leg_number = _integer(payload.get("leg_number"))
            opponent_token = _opponent_token(payload)
            if self._attempt is None or (
                leg_number == 1
                and (
                    (
                        opponent_token
                        and opponent_token != self._attempt.opponent_token
                    )
                    or (
                        match_id not in self._attempt.match_ids
                        and _integer(payload.get("hand_number")) in {1, 2}
                    )
                )
            ):
                self._attempt = AttemptState(match_id, opponent_token)
            self._attempt.ingest_hands(match_id, your_seat, hands, knowledge)
            return knowledge, self._attempt.profile()

    def ingest_completed_hands(
        self,
        *,
        table_rule: str,
        match_id: str,
        your_seat: object,
        hands: Iterable[Mapping[str, object]],
    ) -> int:
        """Ingest a saved `/matches/<runId>` leg, including its final hand."""

        materialized = list(hands)
        with self._lock:
            knowledge = self._rules.setdefault(
                table_rule, RuleKnowledge(self._candidates)
            )
            added = 0
            for observation in extract_observations(
                table_rule=table_rule,
                match_id=match_id,
                your_seat=your_seat,
                hands=materialized,
            ):
                added += knowledge.ingest(observation)
            if self._attempt is not None:
                self._attempt.ingest_hands(
                    match_id, your_seat, materialized, knowledge
                )
            return added

    def knowledge(self, table_rule: str) -> RuleKnowledge:
        """Return/create codename knowledge; primarily useful to replay tools."""

        with self._lock:
            return self._rules.setdefault(
                table_rule, RuleKnowledge(self._candidates)
            )

    def profile(self) -> OpponentProfile:
        with self._lock:
            if self._attempt is None:
                return AttemptState("").profile()
            return self._attempt.profile()

    def reset(self) -> None:
        """Clear process state for deterministic tests or an explicit restart."""

        with self._lock:
            self._rules.clear()
            self._attempt = None


def _next_other_action(
    actions: list[Mapping[str, object]], index: int, your_seat: object
) -> Mapping[str, object] | None:
    for action in actions[index + 1 :]:
        if str(action.get("seat")) != str(your_seat):
            return action
    return None


def _live_range_context(
    payload: Mapping[str, object],
) -> tuple[str, str, str, str] | None:
    actions = payload.get("current_hand_actions")
    if not isinstance(actions, list):
        return None
    your_seat = payload.get("your_seat")
    round_name = str(payload.get("round", ""))
    pot = max(1, _integer(payload.get("pot")) or 1)
    for action in reversed(actions):
        if not isinstance(action, Mapping):
            continue
        action_name = str(action.get("action", ""))
        if (
            str(action.get("seat")) == str(your_seat)
            or action_name not in {"check", "call", "bet", "raise"}
        ):
            continue
        return (
            action_name,
            _size_bucket(action, pot),
            _position(action.get("seat"), payload.get("button_seat")),
            str(action.get("round", round_name)),
        )
    return None


def _action_similarity(current: str, observed: str) -> float:
    if current == observed:
        return 4.0
    if current in {"bet", "raise"} and observed in {"bet", "raise"}:
        return 1.5
    return 0.25


def _context_similarity(current: str, observed: str) -> float:
    if "unknown" in {current, observed}:
        return 1.0
    return 1.4 if current == observed else 0.7


def _position(seat: object, button_seat: object) -> str:
    if seat is None or button_seat is None:
        return "unknown"
    return "button" if str(seat) == str(button_seat) else "blind"


def _size_bucket(action: Mapping[str, object], pot: int) -> str:
    if action.get("action") not in {"bet", "raise"}:
        return "none"
    amount = _integer(action.get("amount"))
    if amount is None or amount <= 0:
        return "unknown"
    fraction = amount / max(1, pot)
    if fraction <= 0.35:
        return "small"
    if fraction <= 0.80:
        return "medium"
    return "large"


def _opponent_token(payload: Mapping[str, object]) -> str:
    """Use the fresh per-attempt name only as a boundary token, never a read."""

    your_seat = payload.get("your_seat")
    players = payload.get("players")
    if not isinstance(players, list):
        return ""
    names = [
        str(player.get("name", ""))
        for player in players
        if isinstance(player, Mapping)
        and str(player.get("seat")) != str(your_seat)
    ]
    return "|".join(names)


def _smoothed_rate(hits: int, trials: int, prior: float, weight: int = 4) -> float:
    return (hits + prior * weight) / (trials + weight)


def _integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
