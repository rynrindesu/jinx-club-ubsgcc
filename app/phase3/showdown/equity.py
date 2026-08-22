"""Exact multiway showdown equity for Phase 3 SHOWDOWN.

Opponent private numbers are independent categorical ranges over 1..13.  Once
the community and deterministic rule are fixed, only three probabilities per
opponent matter: they lose to the hero, tie the hero, or beat the hero.  A tiny
dynamic program tracks the number of tying opponents while discarding every
branch in which anybody beats the hero.  This is exact and scales linearly in
the number of opponents instead of enumerating 13**N joint hands.
"""

from __future__ import annotations

import math
from typing import Hashable, Iterable, Mapping, Sequence

from .rules import (
    CARD_MAX,
    CARD_MIN,
    HYPOTHESIS_BY_NAME,
    RuleHypothesis,
    RuleModel,
    get_hypothesis,
)


RangeInput = Mapping[int | str, float] | Sequence[float]


def normalize_range(values: RangeInput | None) -> tuple[float, ...]:
    """Normalize a mapping or 13-value sequence into probabilities for 1..13.

    A missing/empty range becomes uniform, which is the safest live-play
    fallback.  Negative and non-finite weights are rejected so malformed model
    output cannot silently turn into invalid equity.
    """

    if values is None:
        return (1.0 / CARD_MAX,) * CARD_MAX
    if isinstance(values, Mapping):
        raw: list[float] = []
        for number in range(CARD_MIN, CARD_MAX + 1):
            value = values.get(number, values.get(str(number), 0.0))
            try:
                raw.append(float(value))
            except (TypeError, ValueError) as exc:
                raise ValueError("range weights must be numeric") from exc
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        if len(values) != CARD_MAX:
            raise ValueError("an opponent range must contain exactly 13 weights")
        try:
            raw = [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise ValueError("range weights must be numeric") from exc
    else:
        raise TypeError("an opponent range must be a mapping or 13-value sequence")

    if any(not math.isfinite(value) or value < 0.0 for value in raw):
        raise ValueError("range weights must be finite and non-negative")
    total = sum(raw)
    if total <= 0.0:
        return (1.0 / CARD_MAX,) * CARD_MAX
    return tuple(value / total for value in raw)


def _validated_card(value: int, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer from 1 to 13")
    try:
        card = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer from 1 to 13") from exc
    if card != value and not (isinstance(value, str) and str(card) == value.strip()):
        raise ValueError(f"{label} must be an integer from 1 to 13")
    if not CARD_MIN <= card <= CARD_MAX:
        raise ValueError(f"{label} must be between 1 and 13")
    return card


def _share_from_outcomes(outcomes: Iterable[tuple[float, float]]) -> float:
    """Combine per-opponent (loses_to_hero, ties_hero) probabilities."""

    # dp[k] is the probability that nobody has beaten the hero and exactly k
    # opponents tie.  Beat branches intentionally disappear from the update.
    dp = [1.0]
    for loses, ties in outcomes:
        loses = max(0.0, min(1.0, loses))
        ties = max(0.0, min(1.0 - loses, ties))
        updated = [0.0] * (len(dp) + 1)
        for tie_count, probability in enumerate(dp):
            updated[tie_count] += probability * loses
            updated[tie_count + 1] += probability * ties
        dp = updated
    return max(
        0.0,
        min(1.0, sum(probability / (tie_count + 1) for tie_count, probability in enumerate(dp))),
    )


def _shares_for_all_masks(
    outcomes: Sequence[tuple[float, float]],
) -> tuple[float, ...]:
    """Return exact shares for every subset of an ordered outcome sequence.

    Each mask extends the already-computed DP for ``mask`` without its least
    significant bit.  Thus every subset performs one opponent update instead
    of rebuilding its tie distribution from scratch.
    """

    size = 1 << len(outcomes)
    distributions: list[tuple[float, ...] | None] = [None] * size
    shares = [0.0] * size
    distributions[0] = (1.0,)
    shares[0] = 1.0
    for mask in range(1, size):
        bit = mask & -mask
        opponent_index = bit.bit_length() - 1
        previous = distributions[mask ^ bit]
        assert previous is not None
        loses, ties = outcomes[opponent_index]
        loses = max(0.0, min(1.0, loses))
        ties = max(0.0, min(1.0 - loses, ties))
        updated = [0.0] * (len(previous) + 1)
        for tie_count, probability in enumerate(previous):
            updated[tie_count] += probability * loses
            updated[tie_count + 1] += probability * ties
        distribution = tuple(updated)
        distributions[mask] = distribution
        shares[mask] = max(
            0.0,
            min(
                1.0,
                sum(
                    probability / (tie_count + 1)
                    for tie_count, probability in enumerate(distribution)
                ),
            ),
        )
    return tuple(shares)


def exact_share_for_hypothesis(
    hero_number: int,
    community: int,
    opponent_ranges: Iterable[RangeInput | None],
    hypothesis: RuleHypothesis | str,
) -> float:
    """Return exact expected pot share under one deterministic hypothesis."""

    hero_number = _validated_card(hero_number, "hero number")
    community = _validated_card(community, "community")
    hypothesis = get_hypothesis(hypothesis)
    hero_rank = hypothesis.rank(hero_number, community)

    outcomes: list[tuple[float, float]] = []
    for range_input in opponent_ranges:
        probabilities = normalize_range(range_input)
        loses = 0.0
        ties = 0.0
        for offset, probability in enumerate(probabilities):
            opponent_number = offset + CARD_MIN
            opponent_rank = hypothesis.rank(opponent_number, community)
            if hero_rank > opponent_rank:
                loses += probability
            elif hero_rank == opponent_rank:
                ties += probability
        outcomes.append((loses, ties))
    return _share_from_outcomes(outcomes)


def _posterior_tie_probability(
    hero_number: int,
    opponent_number: int,
    community: int,
    rule_model: RuleModel,
) -> float:
    probability = 0.0
    for name, weight in rule_model.posterior().items():
        hypothesis = HYPOTHESIS_BY_NAME.get(name)
        if hypothesis is None:
            continue
        if hypothesis.rank(hero_number, community) == hypothesis.rank(
            opponent_number, community
        ):
            probability += weight
    return max(0.0, min(1.0, probability))


def _empirical_fallback_share(
    hero_number: int,
    community: int,
    opponent_ranges: Sequence[tuple[float, ...]],
    rule_model: RuleModel,
) -> float:
    """Conservative graph-only equity used when formula fit is weak.

    Unseen comparisons start at 50/50.  Against five players that translates to
    only about a 1/32 chance of beating everyone, intentionally avoiding false
    confidence from sparse empirical edges.
    """

    outcomes: list[tuple[float, float]] = []
    for probabilities in opponent_ranges:
        loses_to_hero = 0.0
        ties_hero = 0.0
        for offset, range_probability in enumerate(probabilities):
            opponent_number = offset + CARD_MIN
            tie_probability = _posterior_tie_probability(
                hero_number, opponent_number, community, rule_model
            )
            comparison = rule_model.empirical_comparison_probability(
                hero_number,
                opponent_number,
                community,
                default=0.5,
            )
            # comparison counts ties as half.  Recover a proper three-outcome
            # distribution while keeping the empirical win/loss balance.
            win_probability = comparison - 0.5 * tie_probability
            win_probability = max(0.0, min(1.0 - tie_probability, win_probability))
            loses_to_hero += range_probability * win_probability
            ties_hero += range_probability * tie_probability
        outcomes.append((loses_to_hero, ties_hero))
    return _share_from_outcomes(outcomes)


def _revealed_showdown_share(
    hero_number: int,
    community: int,
    opponent_ranges: Sequence[tuple[float, ...]],
    rule_model: RuleModel,
) -> float:
    formula_share = 0.0
    posterior = rule_model.posterior()
    posterior_total = 0.0
    for name, weight in posterior.items():
        hypothesis = HYPOTHESIS_BY_NAME.get(name)
        if hypothesis is None or weight <= 0.0:
            continue
        formula_share += weight * exact_share_for_hypothesis(
            hero_number, community, opponent_ranges, hypothesis
        )
        posterior_total += weight
    if posterior_total > 0.0:
        formula_share /= posterior_total
    else:
        formula_share = 0.0

    fallback_weight = rule_model.fallback_weight(community)
    if fallback_weight <= 0.0:
        return formula_share
    empirical_share = _empirical_fallback_share(
        hero_number, community, opponent_ranges, rule_model
    )
    return (1.0 - fallback_weight) * formula_share + fallback_weight * empirical_share


def showdown_shares_by_subset(
    hero_number: int,
    community: int | None,
    opponent_ranges_by_key: Mapping[Hashable, RangeInput | None],
    rule_model: RuleModel,
) -> dict[frozenset[Hashable], float]:
    """Return equity against every subset of up to five named opponents.

    Ranges, ranks, and per-opponent outcomes are computed once.  Formula shares
    and the empirical fallback then reuse one dynamic-program transition per
    subset.  The returned mapping contains all ``2**N`` subsets, including the
    empty subset with a share of exactly ``1.0``.
    """

    hero_number = _validated_card(hero_number, "hero number")
    if not isinstance(rule_model, RuleModel):
        raise TypeError("rule_model must be a RuleModel")
    if not isinstance(opponent_ranges_by_key, Mapping):
        raise TypeError("opponent_ranges_by_key must be a mapping")

    keys = tuple(opponent_ranges_by_key)
    if len(keys) > 5:
        raise ValueError("subset equity supports at most five opponents")
    normalized = tuple(
        normalize_range(opponent_ranges_by_key[key]) for key in keys
    )
    subset_count = 1 << len(keys)
    subset_keys = tuple(
        frozenset(keys[index] for index in range(len(keys)) if mask & (1 << index))
        for mask in range(subset_count)
    )

    if community is None:
        communities = tuple(range(CARD_MIN, CARD_MAX + 1))
    else:
        communities = (_validated_card(community, "community"),)

    posterior_entries = [
        (weight, HYPOTHESIS_BY_NAME[name])
        for name, weight in rule_model.posterior().items()
        if name in HYPOTHESIS_BY_NAME and weight > 0.0
    ]
    posterior_total = sum(weight for weight, _hypothesis in posterior_entries)
    if posterior_total > 0.0:
        posterior_entries = [
            (weight / posterior_total, hypothesis)
            for weight, hypothesis in posterior_entries
        ]

    totals = [0.0] * subset_count
    for revealed in communities:
        formula_shares = [0.0] * subset_count
        ranked_hypotheses: list[
            tuple[float, tuple[int | float, ...], tuple[tuple[int | float, ...], ...]]
        ] = []

        for posterior_weight, hypothesis in posterior_entries:
            ranks = tuple(
                hypothesis.rank(number, revealed)
                for number in range(CARD_MIN, CARD_MAX + 1)
            )
            hero_rank = ranks[hero_number - CARD_MIN]
            ranked_hypotheses.append((posterior_weight, hero_rank, ranks))

            outcomes: list[tuple[float, float]] = []
            for probabilities in normalized:
                loses_to_hero = 0.0
                ties_hero = 0.0
                for probability, opponent_rank in zip(probabilities, ranks):
                    if hero_rank > opponent_rank:
                        loses_to_hero += probability
                    elif hero_rank == opponent_rank:
                        ties_hero += probability
                outcomes.append((loses_to_hero, ties_hero))
            hypothesis_shares = _shares_for_all_masks(outcomes)
            for mask, share in enumerate(hypothesis_shares):
                formula_shares[mask] += posterior_weight * share

        fallback_weight = rule_model.fallback_weight(revealed)
        if fallback_weight > 0.0:
            fallback_outcomes: list[tuple[float, float]] = []
            for probabilities in normalized:
                loses_to_hero = 0.0
                ties_hero = 0.0
                for offset, range_probability in enumerate(probabilities):
                    opponent_number = offset + CARD_MIN
                    tie_probability = sum(
                        posterior_weight
                        for posterior_weight, hero_rank, ranks in ranked_hypotheses
                        if hero_rank == ranks[offset]
                    )
                    comparison = rule_model.empirical_comparison_probability(
                        hero_number,
                        opponent_number,
                        revealed,
                        default=0.5,
                    )
                    win_probability = comparison - 0.5 * tie_probability
                    win_probability = max(
                        0.0, min(1.0 - tie_probability, win_probability)
                    )
                    loses_to_hero += range_probability * win_probability
                    ties_hero += range_probability * tie_probability
                fallback_outcomes.append((loses_to_hero, ties_hero))
            fallback_shares = _shares_for_all_masks(fallback_outcomes)
            for mask in range(subset_count):
                totals[mask] += (
                    (1.0 - fallback_weight) * formula_shares[mask]
                    + fallback_weight * fallback_shares[mask]
                )
        else:
            for mask in range(subset_count):
                totals[mask] += formula_shares[mask]

    divisor = float(len(communities))
    result = {
        subset_keys[mask]: totals[mask] / divisor for mask in range(subset_count)
    }
    # Preserve the exact mathematical identity rather than a nearly-one float
    # accumulated through posterior/community mixtures.
    result[frozenset()] = 1.0
    return result


def showdown_share(
    hero_number: int,
    community: int | None,
    opponent_ranges: Iterable[RangeInput | None],
    rule_model: RuleModel,
) -> float:
    """Return posterior-mixed exact multiway pot share.

    With no reveal, all 13 possible community values are integrated uniformly.
    This function consumes ``opponent_ranges`` once, so generators are supported.
    """

    hero_number = _validated_card(hero_number, "hero number")
    if not isinstance(rule_model, RuleModel):
        raise TypeError("rule_model must be a RuleModel")
    normalized = tuple(normalize_range(values) for values in opponent_ranges)
    if community is None:
        return sum(
            _revealed_showdown_share(hero_number, candidate, normalized, rule_model)
            for candidate in range(CARD_MIN, CARD_MAX + 1)
        ) / CARD_MAX
    community = _validated_card(community, "community")
    return _revealed_showdown_share(hero_number, community, normalized, rule_model)


__all__ = [
    "RangeInput",
    "exact_share_for_hypothesis",
    "normalize_range",
    "showdown_share",
    "showdown_shares_by_subset",
]
