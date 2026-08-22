"""Learn opaque SHOWDOWN table rules from completed showdowns.

The event deliberately does not describe the table rules.  This module models
them as deterministic comparisons, keeps every model still consistent with the
observed results, and falls back to a transitive pairwise table if the real
rule is outside the model library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import fmean
from typing import Iterable, Mapping, Sequence


Outcome = int
_AGREEMENT_CACHE: dict[
    tuple[tuple[tuple[str, tuple[tuple[str, int], ...]], ...], frozenset[int]],
    float,
] = {}


@dataclass(frozen=True)
class RuleCandidate:
    """One deterministic ordering of private numbers for a community."""

    name: str
    clauses: tuple[tuple[str, int], ...]

    def compare(self, first: int, second: int, community: int) -> Outcome:
        first_key = _candidate_key(self.clauses, first, community)
        second_key = _candidate_key(self.clauses, second, community)
        return (first_key > second_key) - (first_key < second_key)


@dataclass(frozen=True)
class ShowdownObservation:
    """A deterministic comparison learned from one completed hand."""

    key: tuple[str, int, str, str]
    community: int
    first_number: int
    second_number: int
    outcome: Outcome
    is_baseline: bool = False


@dataclass(frozen=True)
class EquityEstimate:
    """Equity interval produced by all currently plausible comparisons."""

    mean: float
    lower: float
    upper: float
    disagreement: float
    coverage: float
    candidate_count: int
    observation_count: int
    confidence: str


@dataclass
class RuleKnowledge:
    """Mutable evidence and surviving candidates for one codename."""

    candidates: tuple[RuleCandidate, ...]
    active_candidates: set[int] = field(init=False)
    observations: list[ShowdownObservation] = field(default_factory=list)
    seen_keys: set[tuple[str, int, str, str]] = field(default_factory=set)
    direct_results: dict[tuple[int, int, int], Outcome] = field(
        default_factory=dict
    )
    inconsistent_communities: set[int] = field(default_factory=set)
    tied_communities: set[int] = field(default_factory=set)
    _agreement_cache_key: frozenset[int] | None = field(
        init=False, default=None, repr=False
    )
    _agreement_cache: float = field(init=False, default=0.0, repr=False)
    _candidate_exhausted: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self.active_candidates = set(range(len(self.candidates)))

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    def ingest(self, observation: ShowdownObservation) -> bool:
        """Add one observation exactly once and eliminate contradictions."""

        if observation.key in self.seen_keys:
            return False
        if (
            observation.outcome not in {-1, 0, 1}
            or not 1 <= observation.community <= 13
            or not 1 <= observation.first_number <= 13
            or not 1 <= observation.second_number <= 13
        ):
            return False

        self.seen_keys.add(observation.key)
        self.observations.append(observation)
        comparison = (
            observation.community,
            observation.first_number,
            observation.second_number,
        )
        old_result = self.direct_results.get(comparison)
        if old_result is not None and old_result != observation.outcome:
            self.inconsistent_communities.add(observation.community)
        if (
            observation.outcome == 1
            and self._has_win_path(
                observation.community,
                observation.second_number,
                observation.first_number,
            )
        ) or (
            observation.outcome == -1
            and self._has_win_path(
                observation.community,
                observation.first_number,
                observation.second_number,
            )
        ):
            self.inconsistent_communities.add(observation.community)
        if observation.outcome == 0:
            if self._has_win_path(
                observation.community,
                observation.first_number,
                observation.second_number,
            ) or self._has_win_path(
                observation.community,
                observation.second_number,
                observation.first_number,
            ):
                self.inconsistent_communities.add(observation.community)
            self.tied_communities.add(observation.community)
        self.direct_results[comparison] = observation.outcome
        self.direct_results[
            (
                observation.community,
                observation.second_number,
                observation.first_number,
            )
        ] = -observation.outcome

        surviving = {
            index
            for index in self.active_candidates
            if self.candidates[index].compare(
                observation.first_number,
                observation.second_number,
                observation.community,
            )
            == observation.outcome
        }
        if surviving:
            self.active_candidates = surviving
            self._candidate_exhausted = False
        elif self.active_candidates or self._candidate_exhausted:
            # Filtering an already-empty set can never recover.  Re-evaluate
            # the whole library so live evidence can supersede a stale
            # baseline while retaining both in the observation history.
            self.active_candidates = self._recomputed_candidates()
            self._candidate_exhausted = not self.active_candidates
        self._agreement_cache_key = None
        return True

    def _recomputed_candidates(self) -> set[int]:
        """Rebuild candidates, giving live evidence priority over baselines."""

        if not self.candidates:
            return set()
        live = [
            observation
            for observation in self.observations
            if not observation.is_baseline
        ]
        baselines = [
            observation
            for observation in self.observations
            if observation.is_baseline
        ]
        pool = self._matching_candidates(live)
        if live and not pool:
            # Contradictory live showdowns are not safe to smooth away.
            return set()

        exact = self._matching_candidates(baselines, pool)
        if exact or not baselines:
            return exact

        # Baselines are scouting priors, not authority over a current
        # showdown.  If they conflict, keep the live-compatible candidates
        # contradicted by the fewest baseline comparisons.
        contradiction_counts = {
            index: sum(
                self.candidates[index].compare(
                    observation.first_number,
                    observation.second_number,
                    observation.community,
                )
                != observation.outcome
                for observation in baselines
            )
            for index in pool
        }
        fewest = min(contradiction_counts.values())
        return {
            index
            for index, count in contradiction_counts.items()
            if count == fewest
        }

    def _matching_candidates(
        self,
        observations: Sequence[ShowdownObservation],
        pool: Iterable[int] | None = None,
    ) -> set[int]:
        indices = range(len(self.candidates)) if pool is None else pool
        return {
            index
            for index in indices
            if all(
                self.candidates[index].compare(
                    observation.first_number,
                    observation.second_number,
                    observation.community,
                )
                == observation.outcome
                for observation in observations
            )
        }

    def estimate(
        self,
        your_number: int,
        community: int | None,
        opponent_range: Sequence[float] | None = None,
    ) -> EquityEstimate:
        if not 1 <= your_number <= 13:
            raise ValueError("your_number must be between 1 and 13")
        if community is not None and not 1 <= community <= 13:
            raise ValueError("community must be between 1 and 13")

        normalized_range = _normalized_opponent_range(opponent_range)
        agreement = self._candidate_agreement()
        if self.active_candidates:
            equities = [
                _candidate_equity(
                    self.candidates[index],
                    your_number,
                    community,
                    normalized_range,
                )
                for index in sorted(self.active_candidates)
            ]
            lower = min(equities)
            upper = max(equities)
            coverage = agreement
        else:
            mean, lower, upper, coverage = self._fallback_equity(
                your_number, community, normalized_range
            )
            equities = [mean]

        confidence = "unknown"
        if self.observation_count:
            confidence = "partial"
        distinct_communities = len(
            {observation.community for observation in self.observations}
        )
        disagreement = upper - lower
        locally_resolved = (
            len(self.active_candidates) <= 4
            and agreement >= 0.84
            and disagreement <= 0.10
        )
        if (
            self.observation_count >= 8
            and distinct_communities >= 4
            and self.active_candidates
            and (agreement >= 0.985 or locally_resolved)
        ):
            confidence = "learned"

        return EquityEstimate(
            mean=fmean(equities),
            lower=lower,
            upper=upper,
            disagreement=disagreement,
            coverage=coverage,
            candidate_count=len(self.active_candidates),
            observation_count=self.observation_count,
            confidence=confidence,
        )

    def _candidate_agreement(self) -> float:
        """Return how often all live candidates predict the same comparison."""

        cache_key = frozenset(self.active_candidates)
        if cache_key == self._agreement_cache_key:
            return self._agreement_cache
        library_key = tuple(
            (candidate.name, candidate.clauses) for candidate in self.candidates
        )
        shared_key = (library_key, cache_key)
        if shared_key in _AGREEMENT_CACHE:
            agreement = _AGREEMENT_CACHE[shared_key]
            self._agreement_cache_key = cache_key
            self._agreement_cache = agreement
            return agreement
        if not self.active_candidates:
            return 0.0
        if len(self.active_candidates) == 1:
            return 1.0

        unanimous = total = 0
        active = sorted(self.active_candidates)
        for community in range(1, 14):
            for first in range(1, 14):
                for second in range(first + 1, 14):
                    total += 1
                    outcomes = {
                        self.candidates[index].compare(
                            first, second, community
                        )
                        for index in active
                    }
                    unanimous += len(outcomes) == 1
        agreement = unanimous / total
        _AGREEMENT_CACHE[shared_key] = agreement
        self._agreement_cache_key = cache_key
        self._agreement_cache = agreement
        return agreement

    def _fallback_equity(
        self,
        your_number: int,
        community: int | None,
        opponent_range: tuple[float, ...],
    ) -> tuple[float, float, float, float]:
        communities = range(1, 14) if community is None else (community,)
        earned = guaranteed = possible = known = total = 0.0
        for shared in communities:
            for opponent in range(1, 14):
                weight = opponent_range[opponent - 1]
                total += weight
                outcome = self._inferred_result(shared, your_number, opponent)
                if outcome is None:
                    earned += 0.5 * weight
                    possible += weight
                    continue
                known += weight
                share = _pot_share(outcome)
                earned += share * weight
                guaranteed += share * weight
                possible += share * weight
        return (
            earned / total,
            guaranteed / total,
            possible / total,
            known / total,
        )

    def _inferred_result(
        self, community: int, first: int, second: int
    ) -> Outcome | None:
        if first == second:
            return 0
        if community in self.inconsistent_communities:
            return None
        direct = self.direct_results.get((community, first, second))
        if direct is not None:
            return direct
        if community in self.tied_communities:
            # Without merging equality classes, transitive inference across a
            # distinct-number tie can manufacture contradictory orderings.
            return None

        # Deterministic table rules are orderings.  A path of observed wins is
        # therefore enough to infer an unobserved comparison at this community.
        if self._has_win_path(community, first, second):
            return 1
        if self._has_win_path(community, second, first):
            return -1
        return None

    def _has_win_path(self, community: int, first: int, target: int) -> bool:
        frontier = [first]
        visited = {first}
        while frontier:
            current = frontier.pop()
            for candidate in range(1, 14):
                if candidate in visited:
                    continue
                if self.direct_results.get((community, current, candidate)) == 1:
                    if candidate == target:
                        return True
                    visited.add(candidate)
                    frontier.append(candidate)
        return False


def build_candidate_rules() -> tuple[RuleCandidate, ...]:
    """Build a diverse library of simple, event-shaped comparison rules.

    The explicitly fake odd-before-even/high illustration is intentionally not
    included.  The library instead covers raw order, pair priority, distance,
    direction around the community, and several simple grouping families.
    """

    specifications: list[tuple[str, tuple[tuple[str, int], ...]]] = [
        ("higher", (("number", 1),)),
        ("lower", (("number", -1),)),
        ("pair_then_higher", (("pair", 1), ("number", 1))),
        (
            "pair_then_seven_then_higher",
            (("pair", 1), ("private_seven", 1), ("number", 1)),
        ),
        ("pair_then_lower", (("pair", 1), ("number", -1))),
        ("pair_loses_then_higher", (("pair", -1), ("number", 1))),
        ("pair_loses_then_lower", (("pair", -1), ("number", -1))),
        ("closest_ties", (("distance", -1),)),
        ("closest_then_higher", (("distance", -1), ("number", 1))),
        ("closest_then_lower", (("distance", -1), ("number", -1))),
        ("furthest_ties", (("distance", 1),)),
        ("furthest_then_higher", (("distance", 1), ("number", 1))),
        ("furthest_then_lower", (("distance", 1), ("number", -1))),
        ("clockwise_nearest", (("clockwise", -1),)),
        ("clockwise_furthest", (("clockwise", 1),)),
        ("counterclockwise_nearest", (("counterclockwise", -1),)),
        ("counterclockwise_furthest", (("counterclockwise", 1),)),
        ("above_then_higher", (("above", 1), ("number", 1))),
        ("above_then_lower", (("above", 1), ("number", -1))),
        ("below_then_higher", (("below", 1), ("number", 1))),
        ("below_then_lower", (("below", 1), ("number", -1))),
        ("middle_first_ties", (("from_middle", -1),)),
        ("middle_then_higher", (("from_middle", -1), ("number", 1))),
        ("middle_then_lower", (("from_middle", -1), ("number", -1))),
        ("edges_then_higher", (("from_middle", 1), ("number", 1))),
        ("edges_then_lower", (("from_middle", 1), ("number", -1))),
    ]

    group_features = (
        "even",
        "prime",
        "fibonacci",
        "multiple_three",
        "face",
        "same_parity",
        "same_mod_three",
        "same_side_or_pair",
    )
    for feature in group_features:
        for preferred in (1, -1):
            for tiebreak, direction in (
                ("higher", 1),
                ("lower", -1),
                ("closest", -1),
                ("furthest", 1),
            ):
                # The guide says this exact example is not in play.
                if feature == "even" and preferred == -1 and tiebreak == "higher":
                    continue
                secondary = "number" if tiebreak in {"higher", "lower"} else "distance"
                specifications.append(
                    (
                        f"{feature}_{'first' if preferred == 1 else 'last'}_{tiebreak}",
                        ((feature, preferred), (secondary, direction)),
                    )
                )

    # Remove semantically duplicate candidates.  This keeps candidate counts
    # meaningful and lets confidence lock when two descriptions order every
    # possible showdown identically.
    unique: list[RuleCandidate] = []
    signatures: set[tuple[int, ...]] = set()
    for name, clauses in specifications:
        candidate = RuleCandidate(name=name, clauses=clauses)
        signature = tuple(
            candidate.compare(first, second, community)
            for community in range(1, 14)
            for first in range(1, 14)
            for second in range(first + 1, 14)
        )
        if signature not in signatures:
            signatures.add(signature)
            unique.append(candidate)
    return tuple(unique)


def extract_observations(
    *,
    table_rule: str,
    match_id: str,
    your_seat: object,
    hands: Iterable[Mapping[str, object]],
) -> list[ShowdownObservation]:
    """Extract every usable heads-up comparison from completed hand records."""

    del table_rule  # The registry owns the codename; evidence keys need not.
    observations: list[ShowdownObservation] = []
    hero_key = str(your_seat)
    for hand in hands:
        if not isinstance(hand, Mapping):
            continue
        actions = hand.get("actions")
        if isinstance(actions, list) and any(
            isinstance(action, Mapping) and action.get("action") == "fold"
            for action in actions
        ):
            # A folded hand is not a showdown even if a replay happens to
            # include private numbers for display or diagnostics.
            continue
        hand_number = _integer(hand.get("hand_number"))
        community = _integer(hand.get("community_number"))
        shown = hand.get("shown_numbers")
        winners = hand.get("winners")
        if (
            hand_number is None
            or community is None
            or not 1 <= community <= 13
            or not isinstance(shown, Mapping)
            or not isinstance(winners, Sequence)
            or isinstance(winners, (str, bytes))
        ):
            continue

        normalized_shown = {str(seat): number for seat, number in shown.items()}
        hero_number = _integer(normalized_shown.get(hero_key))
        if hero_number is None or not 1 <= hero_number <= 13:
            continue
        winner_keys = {str(seat) for seat in winners}
        for opponent_key, raw_number in normalized_shown.items():
            if opponent_key == hero_key:
                continue
            opponent_number = _integer(raw_number)
            if opponent_number is None or not 1 <= opponent_number <= 13:
                continue
            if hero_number == opponent_number:
                # Equal private numbers tie under every deterministic rule and
                # therefore carry no information about the opaque ordering.
                continue
            hero_won = hero_key in winner_keys
            opponent_won = opponent_key in winner_keys
            if hero_won == opponent_won:
                if not hero_won:
                    # In a later multiway phase two losing seats reveal no
                    # ordering relative to one another.
                    continue
                outcome = _shared_winner_outcome(
                    hand, hero_key, opponent_key
                )
                if outcome is None:
                    continue
            else:
                outcome = 1 if hero_won else -1
            observations.append(
                ShowdownObservation(
                    key=(match_id, hand_number, hero_key, opponent_key),
                    community=community,
                    first_number=hero_number,
                    second_number=opponent_number,
                    outcome=outcome,
                )
            )
    return observations


_PRIMES = {2, 3, 5, 7, 11, 13}
_FIBONACCI = {1, 2, 3, 5, 8, 13}


def _feature(name: str, number: int, community: int) -> int:
    features = {
        "number": number,
        "pair": int(number == community),
        "private_seven": int(number == 7),
        "distance": abs(number - community),
        "clockwise": (number - community) % 13,
        "counterclockwise": (community - number) % 13,
        "above": int(number > community),
        "below": int(number < community),
        "from_middle": abs(number - 7),
        "even": int(number % 2 == 0),
        "prime": int(number in _PRIMES),
        "fibonacci": int(number in _FIBONACCI),
        "multiple_three": int(number % 3 == 0),
        "face": int(number >= 11),
        "same_parity": int(number % 2 == community % 2),
        "same_mod_three": int(number % 3 == community % 3),
        "same_side_or_pair": int(
            number == community or (number - 7) * (community - 7) > 0
        ),
    }
    return features[name]


def _candidate_key(
    clauses: tuple[tuple[str, int], ...], number: int, community: int
) -> tuple[int, ...]:
    return tuple(
        direction * _feature(feature, number, community)
        for feature, direction in clauses
    )


def _candidate_equity(
    candidate: RuleCandidate,
    your_number: int,
    community: int | None,
    opponent_range: tuple[float, ...],
) -> float:
    communities = range(1, 14) if community is None else (community,)
    shares = [
        _pot_share(candidate.compare(your_number, opponent, shared))
        * opponent_range[opponent - 1]
        for shared in communities
        for opponent in range(1, 14)
    ]
    return sum(shares) / len(communities)


def _normalized_opponent_range(
    opponent_range: Sequence[float] | None,
) -> tuple[float, ...]:
    """Return a safe 13-number distribution, defaulting to uniform."""

    uniform = (1 / 13,) * 13
    if opponent_range is None or isinstance(opponent_range, (str, bytes)):
        return uniform
    if len(opponent_range) != 13:
        return uniform
    try:
        weights = tuple(float(weight) for weight in opponent_range)
    except (TypeError, ValueError, OverflowError):
        return uniform
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        return uniform
    total = sum(weights)
    if total <= 0:
        return uniform
    return tuple(weight / total for weight in weights)


def _pot_share(outcome: Outcome) -> float:
    if outcome > 0:
        return 1.0
    if outcome < 0:
        return 0.0
    return 0.5


def _integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None


def _shared_winner_outcome(
    hand: Mapping[str, object], hero_key: str, opponent_key: str
) -> Outcome | None:
    """Resolve two listed winners without mistaking a refund for a tie."""

    deltas, has_delta_data = _seat_values(
        hand,
        mapping_fields=("deltas", "chip_deltas"),
        player_fields=("delta", "chip_delta", "net_delta"),
    )
    if has_delta_data:
        hero_delta = deltas.get(hero_key)
        opponent_delta = deltas.get(opponent_key)
        if hero_delta is None or opponent_delta is None:
            return None
        if hero_delta > 0 > opponent_delta:
            return 1
        if hero_delta < 0 < opponent_delta:
            return -1
        if hero_delta == opponent_delta:
            return 0
        return None

    contributions, has_contribution_data = _seat_values(
        hand,
        mapping_fields=(
            "contributions",
            "total_contributions",
            "committed",
        ),
        player_fields=(
            "contribution",
            "contributed",
            "committed",
            "total_bet",
        ),
    )
    if has_contribution_data:
        hero_contribution = contributions.get(hero_key)
        opponent_contribution = contributions.get(opponent_key)
        if hero_contribution is None or opponent_contribution is None:
            return None
        return 0 if hero_contribution == opponent_contribution else None

    if _has_unequal_exposure_signal(hand, hero_key, opponent_key):
        return None
    # Preserve ordinary showdown compatibility: with no refund, all-in, or
    # unequal-contribution signal, two listed winners represent a split pot.
    return 0


def _seat_values(
    hand: Mapping[str, object],
    *,
    mapping_fields: Sequence[str],
    player_fields: Sequence[str],
) -> tuple[dict[str, int], bool]:
    """Read per-seat integers from replay mappings or player records."""

    values: dict[str, int] = {}
    present = False
    for field in mapping_fields:
        raw_values = hand.get(field)
        if raw_values is None:
            continue
        present = True
        if not isinstance(raw_values, Mapping):
            continue
        for seat, raw_value in raw_values.items():
            parsed = _integer(raw_value)
            if parsed is not None:
                values.setdefault(str(seat), parsed)

    players = hand.get("players")
    if isinstance(players, Sequence) and not isinstance(players, (str, bytes)):
        for player in players:
            if not isinstance(player, Mapping) or "seat" not in player:
                continue
            for field in player_fields:
                if field not in player:
                    continue
                present = True
                parsed = _integer(player.get(field))
                if parsed is not None:
                    values.setdefault(str(player.get("seat")), parsed)
                    break
    elif isinstance(players, Mapping):
        for seat, player in players.items():
            if not isinstance(player, Mapping):
                continue
            for field in player_fields:
                if field not in player:
                    continue
                present = True
                parsed = _integer(player.get(field))
                if parsed is not None:
                    values.setdefault(str(seat), parsed)
                    break
    return values, present


def _has_unequal_exposure_signal(
    hand: Mapping[str, object], hero_key: str, opponent_key: str
) -> bool:
    """Return whether an apparent split pot may include an uncalled refund."""

    for field in (
        "side_pots",
        "side_pot",
        "refunds",
        "refund",
        "uncalled_bets",
        "uncalled_amount",
    ):
        value = hand.get(field)
        if value not in (None, False, 0, (), [], {}):
            return True
    if any(
        hand.get(field) not in (None, False, 0, (), [], {})
        for field in ("all_in", "all_in_seats")
    ):
        return True

    players = hand.get("players")
    player_records: Iterable[object]
    if isinstance(players, Mapping):
        player_records = players.values()
    elif isinstance(players, Sequence) and not isinstance(players, (str, bytes)):
        player_records = players
    else:
        player_records = ()
    if any(
        isinstance(player, Mapping)
        and (player.get("all_in") is True or player.get("is_all_in") is True)
        for player in player_records
    ):
        return True

    actions = hand.get("actions")
    if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
        return False
    round_amounts: dict[str, dict[str, int]] = {}
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        name = str(action.get("action", "")).lower()
        if (
            action.get("all_in") is True
            or action.get("is_all_in") is True
            or "all_in" in name
            or "all-in" in name
            or "refund" in name
            or "uncalled" in name
            or "return" in name
        ):
            return True
        seat = str(action.get("seat"))
        amount = _integer(action.get("amount"))
        if seat not in {hero_key, opponent_key} or amount is None:
            continue
        round_key = str(action.get("round", ""))
        amounts = round_amounts.setdefault(round_key, {})
        amounts[seat] = max(amount, amounts.get(seat, amount))
    return any(
        hero_key in amounts
        and opponent_key in amounts
        and amounts[hero_key] != amounts[opponent_key]
        for amounts in round_amounts.values()
    )
