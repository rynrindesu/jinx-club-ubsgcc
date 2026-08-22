"""Table-rule inference for the clean-room Phase 3 SHOWDOWN bot.

The challenge deliberately exposes table rules only by codename.  This module
therefore keeps two complementary representations:

* a small, generic Bayesian grammar of plausible deterministic rankings; and
* empirical winner-over-loser edges, keyed by the revealed community number.

The grammar intentionally contains no parity/odd-even rule (the guide calls
that example out as a rule which is *not* used).  Unknown rules can still be
learned locally through the empirical graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Iterable, Mapping, Sequence


CARD_MIN = 1
CARD_MAX = 13
CENTER = 7
SERIAL_VERSION = 1

Rank = tuple[int | float, ...]
RankEvaluator = Callable[[int, int], Rank | int | float]


def _card(value: Any, label: str = "number") -> int:
    """Return a validated integer card number (booleans are not cards)."""

    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer from 1 to 13")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer from 1 to 13") from exc
    if number != value and not (isinstance(value, str) and str(number) == value.strip()):
        raise ValueError(f"{label} must be an integer from 1 to 13")
    if not CARD_MIN <= number <= CARD_MAX:
        raise ValueError(f"{label} must be between 1 and 13")
    return number


def _as_rank(value: Rank | int | float) -> Rank:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


@dataclass(frozen=True, slots=True)
class RuleHypothesis:
    """A named deterministic showdown ranking.

    ``rank`` values are compared lexicographically and larger values win.  The
    evaluator is deliberately a field rather than an encoded enum so tests and
    the offline simulator can supply synthetic rules without modifying this
    module.
    """

    name: str
    complexity: float
    evaluator: RankEvaluator = field(repr=False, compare=False)

    def rank(self, number: int, community: int) -> Rank:
        number = _card(number)
        community = _card(community, "community")
        return _as_rank(self.evaluator(number, community))


def _raw_high(number: int, _community: int) -> Rank:
    return (number,)


def _raw_low(number: int, _community: int) -> Rank:
    return (-number,)


def _community_near_high(number: int, community: int) -> Rank:
    return (-abs(number - community), number)


def _community_near_low(number: int, community: int) -> Rank:
    return (-abs(number - community), -number)


def _community_far_high(number: int, community: int) -> Rank:
    return (abs(number - community), number)


def _community_far_low(number: int, community: int) -> Rank:
    return (abs(number - community), -number)


def _center_near_high(number: int, _community: int) -> Rank:
    return (-abs(number - CENTER), number)


def _center_near_low(number: int, _community: int) -> Rank:
    return (-abs(number - CENTER), -number)


def _center_far_high(number: int, _community: int) -> Rank:
    return (abs(number - CENTER), number)


def _center_far_low(number: int, _community: int) -> Rank:
    return (abs(number - CENTER), -number)


_BASE_EVALUATORS: tuple[tuple[str, float, RankEvaluator], ...] = (
    ("raw-high", 1.0, _raw_high),
    ("raw-low", 1.0, _raw_low),
    ("community-near-high", 1.6, _community_near_high),
    ("community-near-low", 1.6, _community_near_low),
    ("community-far-high", 1.6, _community_far_high),
    ("community-far-low", 1.6, _community_far_low),
    ("center-near-high", 1.8, _center_near_high),
    ("center-near-low", 1.8, _center_near_low),
    ("center-far-high", 1.8, _center_far_high),
    ("center-far-low", 1.8, _center_far_low),
)


def _with_equality(
    secondary: RankEvaluator,
    *,
    pair_first: bool,
) -> RankEvaluator:
    direction = 1 if pair_first else -1

    def evaluator(number: int, community: int) -> Rank:
        return (direction * int(number == community), *_as_rank(secondary(number, community)))

    return evaluator


def _ordering_signature(evaluator: RankEvaluator) -> tuple[int, ...]:
    """Canonical signature used to remove semantically duplicate formulas."""

    signature: list[int] = []
    for community in range(CARD_MIN, CARD_MAX + 1):
        values = [_as_rank(evaluator(number, community)) for number in range(1, 14)]
        ordered = {value: index for index, value in enumerate(sorted(set(values)))}
        signature.extend(ordered[value] for value in values)
    return tuple(signature)


def _build_hypotheses() -> tuple[RuleHypothesis, ...]:
    candidates: list[RuleHypothesis] = []

    # Keep the protocol's ordinary rule under a stable, obvious name.
    candidates.append(
        RuleHypothesis(
            "standard",
            1.2,
            _with_equality(_raw_high, pair_first=True),
        )
    )
    candidates.extend(
        RuleHypothesis(name, complexity, evaluator)
        for name, complexity, evaluator in _BASE_EVALUATORS
    )

    for pair_first, prefix in ((True, "pair-first"), (False, "pair-last")):
        for base_name, complexity, evaluator in _BASE_EVALUATORS:
            # pair-first-high is already represented by ``standard``.
            if pair_first and base_name == "raw-high":
                continue
            candidates.append(
                RuleHypothesis(
                    f"{prefix}-{base_name}",
                    complexity + 1.0,
                    _with_equality(evaluator, pair_first=pair_first),
                )
            )

    canonical: list[RuleHypothesis] = []
    seen: set[tuple[int, ...]] = set()
    for candidate in candidates:
        signature = _ordering_signature(candidate.evaluator)
        if signature not in seen:
            seen.add(signature)
            canonical.append(candidate)
    return tuple(canonical)


HYPOTHESES: tuple[RuleHypothesis, ...] = _build_hypotheses()
HYPOTHESIS_BY_NAME: dict[str, RuleHypothesis] = {
    hypothesis.name: hypothesis for hypothesis in HYPOTHESES
}
STANDARD = HYPOTHESIS_BY_NAME["standard"]


def hypotheses() -> tuple[RuleHypothesis, ...]:
    """Expose the immutable canonical hypothesis set."""

    return HYPOTHESES


def get_hypothesis(name: str | RuleHypothesis) -> RuleHypothesis:
    if isinstance(name, RuleHypothesis):
        return name
    try:
        return HYPOTHESIS_BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"unknown table-rule hypothesis: {name!r}") from exc


def _log_normalize(log_weights: Mapping[str, float]) -> dict[str, float]:
    finite = [value for value in log_weights.values() if math.isfinite(value)]
    if not finite:
        uniform = -math.log(len(HYPOTHESES))
        return {hypothesis.name: uniform for hypothesis in HYPOTHESES}
    maximum = max(finite)
    total = sum(
        math.exp(value - maximum)
        for value in log_weights.values()
        if math.isfinite(value)
    )
    normalizer = maximum + math.log(total)
    return {
        name: (value - normalizer if math.isfinite(value) else -math.inf)
        for name, value in log_weights.items()
    }


def _initial_log_weights() -> dict[str, float]:
    # An exponential Occam prior prevents elaborate equality combinations from
    # collectively drowning out the simpler rules merely because there are more
    # of them.  Standard receives a small, non-dominating prior preference.
    raw = {
        hypothesis.name: -0.72 * hypothesis.complexity
        + (0.35 if hypothesis.name == "standard" else 0.0)
        for hypothesis in HYPOTHESES
    }
    return _log_normalize(raw)


def _coerce_shown_numbers(shown_numbers: Any) -> dict[str, int]:
    """Accept replay-style mappings and common list encodings defensively."""

    shown: dict[str, int] = {}
    if isinstance(shown_numbers, Mapping):
        for player, value in shown_numbers.items():
            if isinstance(value, Mapping):
                value = value.get("number", value.get("card", value.get("hole_card")))
            try:
                shown[str(player)] = _card(value)
            except (TypeError, ValueError):
                continue
        return shown

    if not isinstance(shown_numbers, Sequence) or isinstance(shown_numbers, (str, bytes)):
        return shown
    for index, item in enumerate(shown_numbers):
        if isinstance(item, Mapping):
            player = item.get(
                "player",
                item.get("name", item.get("player_id", item.get("seat", index))),
            )
            value = item.get("number", item.get("card", item.get("hole_card")))
        else:
            player, value = index, item
        try:
            shown[str(player)] = _card(value)
        except (TypeError, ValueError):
            continue
    return shown


def _winner_tokens(winners: Any) -> list[Any]:
    if winners is None:
        return []
    if isinstance(winners, Mapping):
        # A winner object, or a mapping of player -> truthy/result metadata.
        identity = winners.get(
            "player",
            winners.get("name", winners.get("player_id", winners.get("seat"))),
        )
        if identity is not None:
            return [identity]
        return [key for key, value in winners.items() if value]
    if isinstance(winners, Sequence) and not isinstance(winners, (str, bytes)):
        result: list[Any] = []
        for winner in winners:
            if isinstance(winner, Mapping):
                winner = winner.get(
                    "player",
                    winner.get("name", winner.get("player_id", winner.get("seat"))),
                )
            if winner is not None:
                result.append(winner)
        return result
    return [winners]


def _coerce_winner_ids(winners: Any, shown: Mapping[str, int]) -> set[str]:
    result: set[str] = set()
    for token in _winner_tokens(winners):
        key = str(token)
        if key in shown:
            result.add(key)
            continue
        # Replays occasionally encode winners by their shown number rather than
        # player identity.  Preserve all matching players, which correctly makes
        # duplicate-number wins a tie rather than an invented sole winner.
        try:
            number = _card(token)
        except (TypeError, ValueError):
            continue
        result.update(player for player, shown_number in shown.items() if shown_number == number)
    return result


class RuleModel:
    """Bayesian rule inference plus a context-aware empirical fallback."""

    __slots__ = (
        "codename",
        "_log_weights",
        "_pairwise",
        "_observations",
        "_fit_sum",
        "_strength_cache",
    )

    def __init__(self, codename: str):
        self.codename = str(codename)
        self._log_weights = _initial_log_weights()
        # Directed weights: (community, winner_number, loser_number) -> evidence.
        # community=0 is a deliberately low-weight cross-community backstop.
        self._pairwise: dict[tuple[int, int, int], float] = {}
        self._observations = 0
        self._fit_sum = 0.0
        self._strength_cache: dict[tuple[int, int], float] = {}

    @property
    def observation_count(self) -> int:
        return self._observations

    def posterior(self) -> dict[str, float]:
        return {
            hypothesis.name: math.exp(self._log_weights[hypothesis.name])
            for hypothesis in HYPOTHESES
        }

    def confidence(self) -> float:
        """Confidence that a candidate formula explains the codename.

        Posterior concentration alone can be misleading for a rule outside the
        grammar, so confidence also requires observations and predictive fit.
        """

        if self._observations <= 0:
            return 0.0
        concentration = max(self.posterior().values(), default=0.0)
        maturity = 1.0 - math.exp(-self._observations / 1.8)
        average_fit = self._fit_sum / self._observations
        fit = min(1.0, average_fit / 0.86)
        return max(0.0, min(1.0, concentration * maturity * fit))

    def rank(
        self,
        number: int,
        community: int,
        hypothesis_name: str | RuleHypothesis = "standard",
    ) -> Rank:
        return get_hypothesis(hypothesis_name).rank(number, community)

    def _model_comparison_probability(self, a: int, b: int, community: int) -> float:
        a = _card(a, "first number")
        b = _card(b, "second number")
        community = _card(community, "community")
        probability = 0.0
        for name, weight in self.posterior().items():
            hypothesis = HYPOTHESIS_BY_NAME[name]
            rank_a = hypothesis.rank(a, community)
            rank_b = hypothesis.rank(b, community)
            if rank_a > rank_b:
                outcome = 1.0
            elif rank_a == rank_b:
                outcome = 0.5
            else:
                outcome = 0.0
            probability += weight * outcome
        return probability

    def _directed_evidence(self, community: int, winner: int, loser: int) -> float:
        return self._pairwise.get((community, winner, loser), 0.0)

    def empirical_comparison_probability(
        self,
        a: int,
        b: int,
        community: int,
        *,
        default: float = 0.5,
    ) -> float:
        """Smoothed P(a beats b), with a tie represented as one half.

        Context-specific observations dominate.  Weak global edges are used only
        until that community/number pair has direct evidence.
        """

        a = _card(a, "first number")
        b = _card(b, "second number")
        community = _card(community, "community")
        if a == b:
            return 0.5
        wins = self._directed_evidence(community, a, b)
        losses = self._directed_evidence(community, b, a)
        if wins + losses <= 0.0:
            wins = self._directed_evidence(0, a, b)
            losses = self._directed_evidence(0, b, a)
        if wins + losses <= 0.0:
            return max(0.0, min(1.0, float(default)))
        prior_strength = 1.2
        prior = max(0.0, min(1.0, float(default)))
        return (wins + prior * prior_strength) / (wins + losses + prior_strength)

    def pairwise_evidence(self) -> tuple[dict[str, int | float], ...]:
        return tuple(
            {
                "community": community,
                "winner": winner,
                "loser": loser,
                "weight": weight,
            }
            for (community, winner, loser), weight in sorted(self._pairwise.items())
            if weight > 0.0
        )

    def fallback_weight(self, community: int | None = None) -> float:
        """Return how much decisions should trust the empirical graph (0..0.9)."""

        if not self._pairwise:
            return 0.0
        if community is None:
            evidence = sum(
                value for (context, _a, _b), value in self._pairwise.items() if context != 0
            )
        else:
            context = _card(community, "community")
            evidence = sum(
                value for (seen, _a, _b), value in self._pairwise.items() if seen == context
            )
            if evidence <= 0.0:
                evidence = 0.2 * sum(
                    value
                    for (seen, _a, _b), value in self._pairwise.items()
                    if seen == 0
                )
        coverage = 1.0 - math.exp(-evidence / 18.0)
        average_fit = self._fit_sum / self._observations if self._observations else 0.0
        mismatch = 1.0 - min(1.0, average_fit / 0.86)
        return min(0.9, coverage * (0.20 + 0.72 * mismatch))

    def comparison_probability(self, a: int, b: int, community: int) -> float:
        """Posterior/fallback blend for P(a wins), counting a tie as one half."""

        model = self._model_comparison_probability(a, b, community)
        direct = self._directed_evidence(community, int(a), int(b)) + self._directed_evidence(
            community, int(b), int(a)
        )
        if direct <= 0.0 and not self._pairwise:
            return model
        empirical = self.empirical_comparison_probability(
            a, b, community, default=model
        )
        local_weight = 1.0 - math.exp(-direct / 4.0)
        weight = max(self.fallback_weight(community), 0.88 * local_weight)
        weight = min(0.92, weight)
        return (1.0 - weight) * model + weight * empirical

    def strength(self, number: int, community: int) -> float:
        """Heads-up equity against an independent uniform 1..13 range."""

        number = _card(number)
        community = _card(community, "community")
        key = (number, community)
        cached = self._strength_cache.get(key)
        if cached is not None:
            return cached
        result = sum(
            self.comparison_probability(number, opponent, community)
            for opponent in range(CARD_MIN, CARD_MAX + 1)
        ) / CARD_MAX
        self._strength_cache[key] = result
        return result

    def observe_showdown(
        self,
        community: int,
        shown_numbers: Mapping[Any, Any] | Sequence[Any],
        winners: Any,
        ambiguous: bool = False,
    ) -> bool:
        """Learn from a showdown and return whether usable evidence was applied.

        Ambiguous side-pot/multi-pot records are skipped.  A clear sole winner
        produces both a strong Bayesian update and strong pairwise graph edges;
        an unambiguous split pot updates only the formula posterior, more weakly.
        """

        if ambiguous:
            return False
        try:
            community = _card(community, "community")
        except ValueError:
            return False
        shown = _coerce_shown_numbers(shown_numbers)
        winner_ids = _coerce_winner_ids(winners, shown)
        if len(shown) < 2 or not winner_ids or not winner_ids.issubset(shown):
            return False

        sole_winner = len(winner_ids) == 1
        likelihoods: dict[str, float] = {}
        for hypothesis in HYPOTHESES:
            ranks = {
                player: hypothesis.rank(number, community)
                for player, number in shown.items()
            }
            best = max(ranks.values())
            predicted = {player for player, rank in ranks.items() if rank == best}
            if predicted == winner_ids:
                likelihood = 0.985 if sole_winner else 0.91
            elif predicted & winner_ids:
                # The observed winner appearing inside a predicted tie is useful,
                # but considerably weaker than predicting the complete winner set.
                overlap = len(predicted & winner_ids) / len(predicted | winner_ids)
                likelihood = 0.12 + 0.30 * overlap
            else:
                likelihood = 0.012 if sole_winner else 0.035
            likelihoods[hypothesis.name] = likelihood

        exponent = 1.75 if sole_winner else 0.55
        self._log_weights = _log_normalize(
            {
                name: self._log_weights[name] + exponent * math.log(likelihood)
                for name, likelihood in likelihoods.items()
            }
        )
        posterior = self.posterior()
        self._fit_sum += sum(
            posterior[name] * likelihood for name, likelihood in likelihoods.items()
        )
        self._observations += 1

        if sole_winner:
            winner_id = next(iter(winner_ids))
            winner_number = shown[winner_id]
            for loser_id, loser_number in shown.items():
                if loser_id == winner_id or loser_number == winner_number:
                    continue
                key = (community, winner_number, loser_number)
                self._pairwise[key] = self._pairwise.get(key, 0.0) + 3.0
                # Cross-community transfer is intentionally weak: it helps before
                # a community is seen but cannot override direct evidence.
                global_key = (0, winner_number, loser_number)
                self._pairwise[global_key] = self._pairwise.get(global_key, 0.0) + 0.30
        self._strength_cache.clear()
        return True

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-safe seed representation."""

        return {
            "version": SERIAL_VERSION,
            "codename": self.codename,
            "posterior": self.posterior(),
            # Stored for inspection/auditing; it is deliberately recomputed from
            # observations and fit_sum when loading rather than trusted as state.
            "confidence": self.confidence(),
            "observations": self._observations,
            "fit_sum": self._fit_sum,
            "pairwise": list(self.pairwise_evidence()),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuleModel":
        """Load current or partial seed data, ignoring malformed extra fields."""

        if not isinstance(data, Mapping):
            raise TypeError("rule model seed must be a mapping")
        model = cls(str(data.get("codename", "unknown")))

        raw_posterior = data.get("posterior")
        if isinstance(raw_posterior, Mapping):
            probabilities: dict[str, float] = {}
            for hypothesis in HYPOTHESES:
                try:
                    value = float(raw_posterior.get(hypothesis.name, 0.0))
                except (TypeError, ValueError):
                    value = 0.0
                probabilities[hypothesis.name] = (
                    value if math.isfinite(value) and value > 0.0 else 0.0
                )
            total = sum(probabilities.values())
            if total > 0.0:
                model._log_weights = _log_normalize(
                    {
                        name: math.log(value / total) if value > 0.0 else -math.inf
                        for name, value in probabilities.items()
                    }
                )

        try:
            observations = int(data.get("observations", 0))
        except (TypeError, ValueError):
            observations = 0
        model._observations = max(0, observations)
        try:
            fit_sum = float(data.get("fit_sum", 0.0))
        except (TypeError, ValueError):
            fit_sum = 0.0
        model._fit_sum = fit_sum if math.isfinite(fit_sum) and fit_sum >= 0.0 else 0.0

        pairwise = data.get("pairwise", ())
        if isinstance(pairwise, Mapping):
            # Tolerate an early-development ``"community:winner:loser"`` shape.
            pairwise = [
                {"key": key, "weight": value} for key, value in pairwise.items()
            ]
        if isinstance(pairwise, Sequence) and not isinstance(pairwise, (str, bytes)):
            for item in pairwise:
                if not isinstance(item, Mapping):
                    continue
                try:
                    if "key" in item:
                        context_text, winner_text, loser_text = str(item["key"]).split(":")
                        context, winner, loser = (
                            int(context_text),
                            _card(winner_text),
                            _card(loser_text),
                        )
                    else:
                        context = int(item.get("community", 0))
                        winner = _card(item.get("winner"))
                        loser = _card(item.get("loser"))
                    if context != 0:
                        context = _card(context, "community")
                    weight = float(item.get("weight", 0.0))
                except (TypeError, ValueError):
                    continue
                if winner != loser and math.isfinite(weight) and weight > 0.0:
                    model._pairwise[(context, winner, loser)] = (
                        model._pairwise.get((context, winner, loser), 0.0) + weight
                    )
        return model


__all__ = [
    "CARD_MIN",
    "CARD_MAX",
    "HYPOTHESES",
    "HYPOTHESIS_BY_NAME",
    "STANDARD",
    "RuleHypothesis",
    "RuleModel",
    "get_hypothesis",
    "hypotheses",
]
