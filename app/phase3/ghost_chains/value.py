"""Branch-local value-flow inference for Ghost Chains Phase 3."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation, localcontext
import json
import math
from typing import Iterable

from .models import PresentValue, Transaction


@dataclass(frozen=True)
class ValueConfig:
    """Deterministic limits and smooth value-signal parameters."""

    max_path_length: int = 8
    max_route_states: int = 50_000
    max_hypotheses: int = 64
    recency_half_life_seconds: float = 12 * 60 * 60
    large_drop_retention: float = 0.65
    zero_log_tolerance: float = 1e-8
    reversal_scale: float = 0.012
    coherence_scale: float = 0.03
    carry_weight: float = 0.035
    coherence_weight: float = 0.015
    direct_increase_weight: float = 0.42
    established_reversal_weight: float = 0.68
    ambiguity_discount: float = 0.35

    def __post_init__(self) -> None:
        for name in ("max_path_length", "max_route_states", "max_hypotheses"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.large_drop_retention < 1:
            raise ValueError("large_drop_retention must be between zero and one")
        for name in (
            "recency_half_life_seconds",
            "zero_log_tolerance",
            "reversal_scale",
            "coherence_scale",
            "carry_weight",
            "coherence_weight",
            "direct_increase_weight",
            "established_reversal_weight",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.ambiguity_discount <= 1:
            raise ValueError("ambiguity_discount must be in [0, 1]")


@dataclass(frozen=True)
class ValueEvent:
    """One active amount-bearing event with its arrival tie-breaker."""

    transaction: Transaction
    sequence: int

    @property
    def key(self) -> tuple[datetime, int]:
        return self.transaction.created_at, self.sequence


@dataclass(frozen=True)
class ValueHypothesisScore:
    """Explainable score for one branch-local transaction trajectory."""

    transaction_ids: tuple[str, ...]
    amounts: tuple[Decimal, ...]
    log_ratios: tuple[float, ...]
    weight: float
    risk: float
    carry_forward: float
    coherence: float
    direct_increase: float
    established_reversal: float
    decreasing_run: int
    branch_local: bool


@dataclass(frozen=True)
class ValueScore:
    """Weighted value evidence across credible sender-side hypotheses."""

    risk: float
    hypotheses: tuple[ValueHypothesisScore, ...]
    ambiguity: float
    reversal: float
    continuation: float
    branch_continuation: float

    @classmethod
    def zero(cls) -> ValueScore:
        return cls(
            risk=0.0,
            hypotheses=(),
            ambiguity=0.0,
            reversal=0.0,
            continuation=0.0,
            branch_continuation=0.0,
        )


class ValueFlowScorer:
    """Infer and score bounded value paths ending at a candidate sender.

    Paths are made from individual transactions in strict event-time order.
    Divergent outgoing branches start independent value segments, while
    convergence keeps one hypothesis per incoming path.
    """

    def __init__(self, config: ValueConfig | None = None):
        self.config = config or ValueConfig()

    def score(
        self,
        candidate: ValueEvent,
        active_events: Iterable[ValueEvent],
    ) -> ValueScore:
        active = tuple(active_events)
        candidate_amount = _amount_decimal(candidate.transaction.amount)
        if (
            candidate_amount is None
            or candidate_amount <= 0
            or candidate.transaction.sender == candidate.transaction.recipient
        ):
            return ValueScore.zero()

        paths = self._upstream_paths(candidate, active)
        if not paths:
            return ValueScore.zero()

        outgoing_recipients = self._causal_outgoing_recipients(
            candidate, active
        )
        scored: list[ValueHypothesisScore] = []
        seen_segments: set[tuple[int, ...]] = set()
        for path in paths:
            segment, branch_local = self._local_segment(
                (*path, candidate), outgoing_recipients
            )
            if len(segment) < 2:
                continue
            signature = tuple(event.sequence for event in segment)
            if signature in seen_segments:
                continue
            seen_segments.add(signature)
            hypothesis = self._score_hypothesis(segment, branch_local)
            if hypothesis is not None:
                scored.append(hypothesis)

        if not scored:
            return ValueScore.zero()

        total_weight = math.fsum(item.weight for item in scored)
        if total_weight <= 0 or not math.isfinite(total_weight):
            return ValueScore.zero()

        mean_risk = math.fsum(
            item.weight * item.risk for item in scored
        ) / total_weight
        ambiguity = math.fsum(
            item.weight * abs(item.risk - mean_risk) for item in scored
        ) / total_weight
        ambiguity = min(1.0, max(0.0, 2.0 * ambiguity))
        risk = mean_risk * (1.0 - self.config.ambiguity_discount * ambiguity)
        reversal = math.fsum(
            item.weight
            * _noisy_or(item.direct_increase, item.established_reversal)
            for item in scored
        ) / total_weight
        continuation = math.fsum(
            item.weight * _noisy_or(item.carry_forward, item.coherence)
            for item in scored
        ) / total_weight
        branch_continuation = math.fsum(
            item.weight
            * _noisy_or(item.carry_forward, item.coherence)
            * float(item.branch_local)
            for item in scored
        ) / total_weight

        return ValueScore(
            risk=_bounded(risk),
            hypotheses=tuple(scored),
            ambiguity=ambiguity,
            reversal=_bounded(reversal),
            continuation=_bounded(continuation),
            branch_continuation=_bounded(branch_continuation),
        )

    def _upstream_paths(
        self,
        candidate: ValueEvent,
        active: tuple[ValueEvent, ...],
    ) -> tuple[tuple[ValueEvent, ...], ...]:
        incoming: defaultdict[str, list[ValueEvent]] = defaultdict(list)
        for event in active:
            transaction = event.transaction
            if transaction.sender == transaction.recipient:
                continue
            if _amount_decimal(transaction.amount) is None:
                continue
            incoming[transaction.recipient].append(event)
        for events in incoming.values():
            events.sort(key=self._event_sort_key)

        # The reversed event tuple stores the immediate predecessor first.
        stack = [
            (
                candidate.transaction.sender,
                candidate.key,
                (candidate.transaction.sender, candidate.transaction.recipient),
                (),
            )
        ]
        completed: list[tuple[ValueEvent, ...]] = []
        explored = 0

        while stack and explored < self.config.max_route_states:
            explored += 1
            node, boundary, route, reversed_events = stack.pop()
            if reversed_events and (
                len(reversed_events) >= self.config.max_path_length - 1
                or route[0] == route[-1]
            ):
                completed.append(tuple(reversed(reversed_events)))
                continue

            extensions = []
            for event in reversed(incoming.get(node, ())):
                if event.key >= boundary:
                    continue
                predecessor = event.transaction.sender
                if predecessor in route and predecessor != route[-1]:
                    continue
                extensions.append((event, predecessor))

            if not extensions:
                if reversed_events:
                    completed.append(tuple(reversed(reversed_events)))
                continue

            # Push older alternatives first so the LIFO walk explores the
            # most recent hypotheses before reaching the route-state cap.
            for event, predecessor in reversed(extensions):
                stack.append(
                    (
                        predecessor,
                        event.key,
                        (predecessor, *route),
                        (*reversed_events, event),
                    )
                )

        # Prefer recent, deeper hypotheses when the safety cap is reached.
        unique = {tuple(event.sequence for event in path): path for path in completed}
        ranked = sorted(
            unique.values(),
            key=lambda path: (
                path[-1].key,
                len(path),
                tuple(event.sequence for event in path),
            ),
            reverse=True,
        )
        return tuple(ranked[: self.config.max_hypotheses])

    @staticmethod
    def _causal_outgoing_recipients(
        candidate: ValueEvent,
        active: tuple[ValueEvent, ...],
    ) -> dict[str, set[str]]:
        outgoing: defaultdict[str, set[str]] = defaultdict(set)
        for event in (*active, candidate):
            transaction = event.transaction
            if event is not candidate and event.key >= candidate.key:
                continue
            if transaction.sender == transaction.recipient:
                continue
            outgoing[transaction.sender].add(transaction.recipient)
        return dict(outgoing)

    def _local_segment(
        self,
        events: tuple[ValueEvent, ...],
        outgoing_recipients: dict[str, set[str]],
    ) -> tuple[tuple[ValueEvent, ...], bool]:
        """Cut a path at its latest structural or value-regime divergence."""

        start = 0
        branch_start: int | None = None
        for index, event in enumerate(events):
            if len(outgoing_recipients.get(event.transaction.sender, ())) > 1:
                start = index
                branch_start = index

        amounts = [_amount_decimal(event.transaction.amount) for event in events]
        for index, amount in enumerate(amounts):
            if amount is None or amount <= 0:
                start = max(start, index + 1)

        large_drop_log = math.log(self.config.large_drop_retention)
        # Earlier dramatic drops establish a new local regime. A dramatic
        # drop on the candidate itself remains a zero/low-risk hypothesis so
        # it can temper other converging hypotheses instead of disappearing
        # from the weighted mixture.
        for index in range(max(1, start + 1), len(events) - 1):
            previous = amounts[index - 1]
            current = amounts[index]
            if previous is None or current is None or previous <= 0 or current <= 0:
                continue
            if _log_ratio(current, previous) < large_drop_log:
                start = index

        branch_local = branch_start is not None and branch_start == start
        return events[start:], branch_local

    def _score_hypothesis(
        self,
        events: tuple[ValueEvent, ...],
        branch_local: bool,
    ) -> ValueHypothesisScore | None:
        amounts = tuple(_amount_decimal(event.transaction.amount) for event in events)
        if any(amount is None or amount <= 0 for amount in amounts):
            return None
        exact_amounts = tuple(amount for amount in amounts if amount is not None)
        ratios = tuple(
            _log_ratio(current, previous)
            for previous, current in zip(exact_amounts, exact_amounts[1:])
        )
        if not ratios:
            return None

        current_ratio = ratios[-1]
        past_ratios = ratios[:-1]
        previous_amount, current_amount = exact_amounts[-2:]
        tolerance = self.config.zero_log_tolerance

        decreasing_run = 0
        recent_decreases: list[float] = []
        for ratio in reversed(past_ratios):
            if ratio >= -tolerance:
                break
            decreasing_run += 1
            recent_decreases.append(ratio)

        carry_forward = 0.0
        coherence = 0.0
        direct_increase = 0.0
        established_reversal = 0.0

        if current_amount < previous_amount:
            loss = -current_ratio
            # A broad smooth peak around ordinary low-single-digit retention loss.
            carry_strength = math.exp(-((loss - 0.02) / 0.09) ** 2)
            carry_forward = self.config.carry_weight * carry_strength
        elif current_amount == previous_amount:
            carry_forward = self.config.carry_weight * 0.45

        if past_ratios:
            trend = _weighted_median_recent(past_ratios)
            if trend <= tolerance and current_ratio <= tolerance:
                coherence_strength = math.exp(
                    -abs(current_ratio - trend) / self.config.coherence_scale
                )
                coherence = self.config.coherence_weight * coherence_strength

        if current_amount > previous_amount:
            increase_strength = 1.0 - math.exp(
                -max(0.0, current_ratio) / self.config.reversal_scale
            )
            direct_increase = (
                self.config.direct_increase_weight * increase_strength
            )
            if decreasing_run:
                median = _median(recent_decreases)
                dispersion = _median(
                    [abs(ratio - median) for ratio in recent_decreases]
                )
                consistency = math.exp(
                    -dispersion / self.config.coherence_scale
                )
                run_strength = 1.0 - math.exp(-0.65 * decreasing_run)
                established_reversal = (
                    self.config.established_reversal_weight
                    * increase_strength
                    * run_strength
                    * consistency
                )

        risk = _noisy_or(
            carry_forward,
            coherence,
            direct_increase,
            established_reversal,
        )
        weight = self._hypothesis_weight(events)
        return ValueHypothesisScore(
            transaction_ids=tuple(event.transaction.tx_id for event in events),
            amounts=exact_amounts,
            log_ratios=ratios,
            weight=weight,
            risk=risk,
            carry_forward=carry_forward,
            coherence=coherence,
            direct_increase=direct_increase,
            established_reversal=established_reversal,
            decreasing_run=decreasing_run,
            branch_local=branch_local,
        )

    def _hypothesis_weight(self, events: tuple[ValueEvent, ...]) -> float:
        predecessor, candidate = events[-2:]
        age = max(
            0.0,
            (candidate.transaction.created_at - predecessor.transaction.created_at)
            .total_seconds(),
        )
        recency = 0.5 + 0.5 * math.exp(
            -age / self.config.recency_half_life_seconds
        )
        depth = 0.75 + 0.25 * (
            1.0 - math.exp(-0.7 * (len(events) - 1))
        )
        identity = self._identity_affinity(predecessor, candidate)
        return recency * depth * identity

    @staticmethod
    def _identity_affinity(previous: ValueEvent, candidate: ValueEvent) -> float:
        factors: list[float] = []
        for dimension in ("ip", "device"):
            old = _identity_key(getattr(previous.transaction, dimension))
            new = _identity_key(getattr(candidate.transaction, dimension))
            if old is None and new is None:
                factors.append(1.0)
            elif old == new:
                factors.append(1.15)
            elif new is None:
                factors.append(0.92)
            else:
                factors.append(0.85)
        return math.prod(factors)

    @staticmethod
    def _event_sort_key(event: ValueEvent) -> tuple[datetime, int, str, str]:
        transaction = event.transaction
        return (
            transaction.created_at,
            event.sequence,
            transaction.sender,
            transaction.recipient,
        )


def _amount_decimal(value: PresentValue) -> Decimal | None:
    """Recover a finite exact decimal representation of a supplied amount."""

    if not value.present or isinstance(value.value, bool):
        return None
    try:
        amount = Decimal(str(value.value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return amount if amount.is_finite() else None


def _log_ratio(current: Decimal, previous: Decimal) -> float:
    """Compute a scale-invariant ratio after exact decimal comparison."""

    try:
        with localcontext() as context:
            context.prec = 40
            result = (current / previous).ln()
        numeric = float(result)
    except (ArithmeticError, InvalidOperation, OverflowError, ValueError):
        numeric = math.log(float(current)) - math.log(float(previous))
    if not math.isfinite(numeric):
        return math.copysign(1_000_000.0, numeric)
    return numeric


def _identity_key(value: PresentValue) -> str | None:
    if not value.present or value.value is None:
        return None
    if isinstance(value.value, str) and not value.value.strip():
        return None
    try:
        encoded = json.dumps(
            value.value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        return None
    return f"{type(value.value).__name__}:{encoded}"


def _weighted_median_recent(values: tuple[float, ...]) -> float:
    weighted = [
        (value, 0.7 ** (len(values) - index - 1))
        for index, value in enumerate(values)
    ]
    ordered = sorted(weighted)
    midpoint = math.fsum(weight for _, weight in ordered) / 2
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= midpoint:
            return value
    return ordered[-1][0]


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _noisy_or(*signals: float) -> float:
    remaining = math.prod(1.0 - _bounded(signal) for signal in signals)
    return _bounded(1.0 - remaining)


def _bounded(value: float) -> float:
    if not math.isfinite(value):
        return 1.0 if value > 0 else 0.0
    return min(1.0, max(0.0, value))


__all__ = [
    "ValueConfig",
    "ValueEvent",
    "ValueFlowScorer",
    "ValueHypothesisScore",
    "ValueScore",
]
