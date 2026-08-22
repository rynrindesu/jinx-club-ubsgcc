"""Thread-safe state scopes for Phase 2 SHOWDOWN play."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Iterable, Mapping

from .rules import (
    RuleKnowledge,
    build_candidate_rules,
    extract_observations,
)


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

    def __post_init__(self) -> None:
        self.match_ids.add(self.first_match_id)

    def ingest_hands(
        self,
        match_id: str,
        your_seat: object,
        hands: Iterable[Mapping[str, object]],
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
            self._attempt.ingest_hands(match_id, your_seat, hands)
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
                self._attempt.ingest_hands(match_id, your_seat, materialized)
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
