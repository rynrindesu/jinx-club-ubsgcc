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
from .scouting import observations_for_leg


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
    post_reraise_rate: float
    bet_after_check_rate: float
    aggression_rate: float
    open_responses: int
    post_responses: int
    checked_to: int
    decisions: int
    range_samples: tuple[RangeEvidence, ...] = ()
    range_shown_hands: int = 0
    open_reraises: int = 0

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

    @property
    def punishes_opens(self) -> bool:
        """Return whether one observed open already drew a re-raise."""

        # The smoothed rate is deliberately slow to react before the first
        # re-raise, but it also falls back through the old threshold after a
        # handful of calls.  A demonstrated punish is more important than
        # that later dilution: keep it for the lifetime of this opponent
        # profile (one attempt).
        return self.open_reraises > 0 or (
            self.open_responses >= 1 and self.reraise_rate > 0.30
        )

    @property
    def punishes_post_bets(self) -> bool:
        """Return whether a value bet has already drawn a re-raise."""

        return self.post_responses >= 1 and self.post_reraise_rate > 0.29

    def range_for(
        self,
        *,
        payload: Mapping[str, object],
        rule_knowledge: RuleKnowledge,
    ) -> tuple[float, ...]:
        """Estimate the opponent's 13-number range for the live action."""

        uniform = (1 / 13,) * 13
        context = _live_range_context(payload)
        if context is None or rule_knowledge.observation_count == 0:
            return uniform

        current_action, size_bucket, position, round_name = context
        raw_community = _integer(payload.get("community_number"))
        community = (
            raw_community
            if raw_community is not None and 1 <= raw_community <= 13
            else None
        )
        prior = _action_range_prior(
            current_action,
            community=community,
            rule_knowledge=rule_knowledge,
        )
        contextual_samples: list[tuple[RangeEvidence, float]] = []
        for sample in self.range_samples:
            action_weight = _action_similarity(current_action, sample.action)
            round_weight = 1.6 if sample.round_name == round_name else 0.55
            position_weight = _context_similarity(position, sample.position)
            size_weight = (
                _context_similarity(size_bucket, sample.size_bucket)
                if current_action in {"bet", "raise", "reraise"}
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
            return prior

        denominator = sum(weight for _, weight in contextual_samples)
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
        observed = tuple(likelihood / total for likelihood in likelihoods)

        # Each shown hand contributes at most one effective sample even when
        # it contains actions on both streets.  Context and rule reliability
        # discount weakly related observations.  Four prior pseudo-hands make
        # the first reveal useful without letting it erase the conservative
        # action prior; observed behavior takes over as the sample grows.
        effective_samples = min(
            float(self.range_shown_hands),
            sum(min(1.0, weight / 4.0) for _, weight in contextual_samples),
        )
        observed_share = effective_samples / (effective_samples + 4.0)
        return tuple(
            (1.0 - observed_share) * prior_weight
            + observed_share * observed_weight
            for prior_weight, observed_weight in zip(prior, observed)
        )


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
    post_reraises: int = 0
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
                self.post_reraises += response.get("action") == "raise"
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
        action_contexts = _range_action_contexts(
            actions,
            final_pot=hand.get("pot"),
            button_seat=hand.get("button_seat"),
            small_blind=hand.get("small_blind"),
            big_blind=hand.get("big_blind"),
            seats=shown.keys(),
        )
        added = False
        for index, action in enumerate(actions):
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
                    action=action_contexts[index][0],
                    size_bucket=action_contexts[index][1],
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
            post_reraise_rate=_smoothed_rate(
                self.post_reraises, self.post_responses, 0.15
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
            open_reraises=self.open_reraises,
        )


class Phase2State:
    """Own the event, attempt, and leg state used by the live endpoint."""

    def __init__(self, *, use_scouted_priors: bool = True) -> None:
        self._lock = RLock()
        self._candidates = build_candidate_rules()
        self._rules: dict[str, RuleKnowledge] = {}
        self._attempt: AttemptState | None = None
        self._use_scouted_priors = use_scouted_priors

    def observe_payload(
        self, payload: Mapping[str, object]
    ) -> tuple[RuleKnowledge, OpponentProfile]:
        table_rule = str(payload.get("table_rule", ""))
        match_id = str(payload.get("match_id", ""))
        leg_number = _integer(payload.get("leg_number"))
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
            if self._use_scouted_priors and knowledge.observation_count == 0:
                for observation in observations_for_leg(leg_number):
                    knowledge.ingest(observation)
            for observation in extract_observations(
                table_rule=table_rule,
                match_id=match_id,
                your_seat=your_seat,
                hands=hands,
            ):
                knowledge.ingest(observation)

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


def _action_range_prior(
    action: str,
    *,
    community: int | None,
    rule_knowledge: RuleKnowledge,
) -> tuple[float, ...]:
    """Return a soft rule-relative prior for an opponent's live action.

    A first wager is modeled as roughly the strongest third of the private
    numbers and a re-raise as roughly the strongest fifth.  The likelihood
    floor is intentionally substantial: this is a cautious default, not a
    claim that an unknown opponent can never bluff.
    """

    uniform = (1 / 13,) * 13
    if action not in {"bet", "raise", "reraise"}:
        return uniform

    if action == "reraise":
        cutoff, transition, floor = 0.80, 0.075, 0.20
    else:
        cutoff, transition, floor = 0.66, 0.085, 0.45

    likelihoods: list[float] = []
    for number in range(1, 14):
        strength = rule_knowledge.estimate(number, community).mean
        selected = 1.0 / (1.0 + math.exp(-(strength - cutoff) / transition))
        likelihoods.append(floor + (1.0 - floor) * selected)
    total = sum(likelihoods)
    if total <= 0:
        return uniform
    return tuple(likelihood / total for likelihood in likelihoods)


def _range_action_contexts(
    actions: list[Mapping[str, object]],
    *,
    final_pot: object,
    button_seat: object,
    small_blind: object,
    big_blind: object,
    seats: Iterable[object] = (),
) -> list[tuple[str, str]]:
    """Classify and size actions using round increments and prior pot.

    SHOWDOWN action amounts are totals committed on that betting round.  The
    request pot is after every listed live action, while a saved hand's pot is
    after every listed historical action.  Walking the same contribution
    ledger for both lets us subtract later increments and recover the pot that
    existed immediately before each action.
    """

    parsed_small = _integer(small_blind)
    parsed_big = _integer(big_blind)
    small = max(0, parsed_small if parsed_small is not None else 1)
    big = max(small, parsed_big if parsed_big is not None else 2)

    seat_keys: list[str] = []
    for seat in seats:
        if seat is None:
            continue
        key = str(seat)
        if key not in seat_keys:
            seat_keys.append(key)
    for action in actions:
        seat = action.get("seat")
        if seat is None:
            continue
        key = str(seat)
        if key not in seat_keys:
            seat_keys.append(key)

    button_key = str(button_seat) if button_seat is not None else ""
    if not button_key:
        first_pre = next(
            (
                action
                for action in actions
                if str(action.get("round", "")) == "pre_reveal"
                and action.get("seat") is not None
            ),
            None,
        )
        if first_pre is not None:
            button_key = str(first_pre.get("seat"))
    if button_key and button_key not in seat_keys:
        seat_keys.append(button_key)

    pre_commitments: dict[str, int] = {}
    if button_key:
        pre_commitments[button_key] = small
        for key in seat_keys:
            if key != button_key:
                pre_commitments[key] = big

    commitments: dict[str, dict[str, int]] = {}
    increments: list[int] = []
    prior_totals: list[int] = []
    classified_actions: list[str] = []
    wager_seen: dict[str, bool] = {}
    for action in actions:
        round_name = str(action.get("round", ""))
        seat_key = str(action.get("seat", ""))
        if round_name not in commitments:
            commitments[round_name] = (
                dict(pre_commitments) if round_name == "pre_reveal" else {}
            )
        round_commitments = commitments[round_name]
        previous = max(0, round_commitments.get(seat_key, 0))
        prior_totals.append(previous)

        verb = str(action.get("action", ""))
        classified = verb
        if verb == "raise" and wager_seen.get(round_name, False):
            classified = "reraise"
        classified_actions.append(classified)

        amount = _integer(action.get("amount"))
        if verb == "call" and amount is None:
            amount = max(round_commitments.values(), default=previous)
        increment = 0
        if verb in {"call", "bet", "raise"} and amount is not None:
            increment = max(0, amount - previous)
            round_commitments[seat_key] = max(previous, amount)
        increments.append(increment)
        if verb in {"bet", "raise"}:
            wager_seen[round_name] = True

    parsed_final_pot = _integer(final_pot)
    if parsed_final_pot is None or parsed_final_pot <= 0:
        parsed_final_pot = max(1, small + big + sum(increments))
    pot_before = max(1, parsed_final_pot - sum(increments))
    contexts: list[tuple[str, str]] = []
    for action, classified, previous, increment in zip(
        actions, classified_actions, prior_totals, increments
    ):
        contexts.append((classified, _size_bucket(action, pot_before, previous)))
        pot_before += increment
    return contexts


def _live_range_context(
    payload: Mapping[str, object],
) -> tuple[str, str, str, str] | None:
    actions = payload.get("current_hand_actions")
    if not isinstance(actions, list):
        return None
    your_seat = payload.get("your_seat")
    round_name = str(payload.get("round", ""))
    materialized = [action for action in actions if isinstance(action, Mapping)]
    players = payload.get("players")
    seats = (
        [player.get("seat") for player in players if isinstance(player, Mapping)]
        if isinstance(players, list)
        else []
    )
    action_contexts = _range_action_contexts(
        materialized,
        final_pot=payload.get("pot"),
        button_seat=payload.get("button_seat"),
        small_blind=payload.get("small_blind"),
        big_blind=payload.get("big_blind"),
        seats=seats,
    )
    for index in range(len(materialized) - 1, -1, -1):
        action = materialized[index]
        action_name = str(action.get("action", ""))
        if (
            str(action.get("seat")) == str(your_seat)
            or action_name not in {"check", "call", "bet", "raise"}
        ):
            continue
        return (
            action_contexts[index][0],
            action_contexts[index][1],
            _position(action.get("seat"), payload.get("button_seat")),
            str(action.get("round", round_name)),
        )
    return None


def _action_similarity(current: str, observed: str) -> float:
    if current == observed:
        return 4.0
    if current in {"bet", "raise", "reraise"} and observed in {
        "bet",
        "raise",
        "reraise",
    }:
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


def _size_bucket(
    action: Mapping[str, object], pot: int, prior_round_total: int = 0
) -> str:
    """Bucket a round-total wager by its increment over the pot before it."""

    if action.get("action") not in {"bet", "raise"}:
        return "none"
    amount = _integer(action.get("amount"))
    if amount is None or amount <= 0:
        return "unknown"
    increment = amount - max(0, prior_round_total)
    if increment <= 0:
        return "unknown"
    fraction = increment / max(1, pot)
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
