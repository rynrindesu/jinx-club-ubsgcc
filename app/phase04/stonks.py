"""Planner for the Time Travelling Stonks Man challenge.

Every state produced by this module is deliberately liquid (it holds cash and
no shares).  A transition jumps to one year, buys one or more stocks there,
jumps to a later quoted year, and sells everything bought by that transition.
That makes every emitted prefix legal and lets the search safely stop at any
point before returning to 2037.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key
from typing import Any, Iterable


PRESENT_YEAR = 2037
_BEAM_WIDTH = 96
_MAX_DEPTH = 96
_MAX_GROUPS = 120
_MAX_TRANSITIONS = 64


@dataclass(frozen=True, slots=True)
class _Listing:
    year: int
    stock: str
    price: int
    qty: int


@dataclass(frozen=True, slots=True)
class _Opportunity:
    listing_index: int
    buy_year: int
    sell_year: int
    stock: str
    buy_price: int
    sell_price: int

    @property
    def profit(self) -> int:
        return self.sell_price - self.buy_price


@dataclass(frozen=True, slots=True)
class _State:
    cash: int
    year: int
    energy_used: int
    remaining: tuple[int, ...]
    actions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Transition:
    buy_year: int
    sell_year: int
    travel: int
    gain: int
    purchases: tuple[tuple[_Opportunity, int], ...]


def _opportunity_cmp(left: _Opportunity, right: _Opportunity) -> int:
    """Order by exact return on cost, then deterministic useful tie-breaks."""

    left_ratio = left.profit * right.buy_price
    right_ratio = right.profit * left.buy_price
    if left_ratio != right_ratio:
        return -1 if left_ratio > right_ratio else 1
    if left.profit != right.profit:
        return -1 if left.profit > right.profit else 1
    if left.buy_price != right.buy_price:
        return -1 if left.buy_price < right.buy_price else 1
    if left.stock != right.stock:
        return -1 if left.stock < right.stock else 1
    return 0


def _parse_case(
    case: dict[str, Any],
) -> tuple[int, int, list[_Listing], dict[int, dict[str, int]]]:
    energy = int(case["energy"])
    capital = int(case["capital"])
    raw_timeline = case["timeline"]

    prices: dict[int, dict[str, int]] = {}
    listings: list[_Listing] = []

    for raw_year, raw_stocks in raw_timeline.items():
        year = int(raw_year)
        year_prices: dict[str, int] = {}
        for stock, raw_quote in raw_stocks.items():
            price = int(raw_quote["price"])
            qty = int(raw_quote["qty"])
            stock_name = str(stock)
            year_prices[stock_name] = price
            if qty > 0 and year < PRESENT_YEAR:
                listings.append(_Listing(year, stock_name, price, qty))
        prices[year] = year_prices

    listings.sort(key=lambda item: (item.year, item.stock))
    return energy, capital, listings, prices


def _build_groups(
    listings: list[_Listing],
    prices: dict[int, dict[str, int]],
) -> list[tuple[tuple[int, int], tuple[_Opportunity, ...]]]:
    groups: dict[tuple[int, int], list[_Opportunity]] = {}

    for index, listing in enumerate(listings):
        for sell_year, sell_prices in prices.items():
            sell_price = sell_prices.get(listing.stock)
            if (
                sell_year <= listing.year
                or sell_year > PRESENT_YEAR
                or sell_price is None
                or sell_price <= listing.price
            ):
                continue
            opportunity = _Opportunity(
                listing_index=index,
                buy_year=listing.year,
                sell_year=sell_year,
                stock=listing.stock,
                buy_price=listing.price,
                sell_price=sell_price,
            )
            groups.setdefault((listing.year, sell_year), []).append(opportunity)

    result: list[tuple[tuple[int, int], tuple[_Opportunity, ...]]] = []
    for years, opportunities in groups.items():
        ordered = tuple(sorted(opportunities, key=cmp_to_key(_opportunity_cmp)))
        result.append((years, ordered))

    # High-return and high-capacity groups are the most useful if a very large
    # timeline forces the search to cap its branching factor.
    result.sort(
        key=lambda group: (
            -max(
                Fraction(opportunity.profit, opportunity.buy_price)
                for opportunity in group[1]
            ),
            -sum(
                listings[opportunity.listing_index].qty * opportunity.profit
                for opportunity in group[1]
            ),
            group[0],
        )
    )
    if len(result) > _MAX_GROUPS:
        nearest = sorted(
            result,
            key=lambda group: (
                PRESENT_YEAR - group[0][0],
                group[0][1] - group[0][0],
            ),
        )[: _MAX_GROUPS // 3]
        selected = result[: _MAX_GROUPS - len(nearest)] + nearest
        deduplicated = {years: opportunities for years, opportunities in selected}
        result = sorted(deduplicated.items())
    return result


def _orders(opportunities: tuple[_Opportunity, ...]) -> Iterable[tuple[_Opportunity, ...]]:
    """Yield a few bounded-knapsack orderings without exploding the search."""

    yield opportunities
    if len(opportunities) <= 1:
        return

    by_unit_profit = tuple(
        sorted(
            opportunities,
            key=lambda item: (-item.profit, item.buy_price, item.stock),
        )
    )
    if by_unit_profit != opportunities:
        yield by_unit_profit

    # Trying a small number of different first items repairs common integer
    # remainder cases where pure ratio-greedy cannot spend the final dollars.
    for pivot in opportunities[1:3]:
        yield (pivot,) + tuple(item for item in opportunities if item != pivot)


def _make_transition(
    state: _State,
    buy_year: int,
    sell_year: int,
    ordered: tuple[_Opportunity, ...],
    energy: int,
) -> _Transition | None:
    travel = abs(state.year - buy_year) + (sell_year - buy_year)
    if state.energy_used + travel + (PRESENT_YEAR - sell_year) > energy:
        return None

    budget = state.cash
    purchases: list[tuple[_Opportunity, int]] = []
    gain = 0
    for opportunity in ordered:
        available = state.remaining[opportunity.listing_index]
        quantity = min(available, budget // opportunity.buy_price)
        if quantity <= 0:
            continue
        budget -= quantity * opportunity.buy_price
        gain += quantity * opportunity.profit
        purchases.append((opportunity, quantity))

    if not purchases:
        return None
    return _Transition(
        buy_year=buy_year,
        sell_year=sell_year,
        travel=travel,
        gain=gain,
        purchases=tuple(purchases),
    )


def _candidate_transitions(
    state: _State,
    groups: list[tuple[tuple[int, int], tuple[_Opportunity, ...]]],
    energy: int,
) -> list[_Transition]:
    candidates: dict[
        tuple[int, int, tuple[tuple[int, int], ...]], _Transition
    ] = {}

    for (buy_year, sell_year), opportunities in groups:
        for ordered in _orders(opportunities):
            transition = _make_transition(
                state, buy_year, sell_year, ordered, energy
            )
            if transition is None:
                continue
            fingerprint = (
                buy_year,
                sell_year,
                tuple(
                    (opportunity.listing_index, quantity)
                    for opportunity, quantity in transition.purchases
                ),
            )
            previous = candidates.get(fingerprint)
            if previous is None or transition.gain > previous.gain:
                candidates[fingerprint] = transition

    transitions = list(candidates.values())
    if len(transitions) <= _MAX_TRANSITIONS:
        return transitions

    by_gain = sorted(
        transitions,
        key=lambda item: (-item.gain, item.travel, item.buy_year, item.sell_year),
    )
    by_efficiency = sorted(
        transitions,
        key=lambda item: (
            -Fraction(item.gain, max(1, item.travel)),
            -item.gain,
            item.travel,
        ),
    )
    chosen = by_gain[:48] + by_efficiency[:16]
    unique: dict[
        tuple[int, int, tuple[tuple[int, int], ...]], _Transition
    ] = {}
    for transition in chosen:
        key = (
            transition.buy_year,
            transition.sell_year,
            tuple(
                (opportunity.listing_index, quantity)
                for opportunity, quantity in transition.purchases
            ),
        )
        unique[key] = transition
    return list(unique.values())


def _apply_transition(state: _State, transition: _Transition) -> _State:
    remaining = list(state.remaining)
    actions = list(state.actions)

    if state.year != transition.buy_year:
        actions.append(f"j-{state.year}-{transition.buy_year}")
    for opportunity, quantity in transition.purchases:
        # A quote is a single-use lot: making any purchase permanently removes
        # that (year, stock) opportunity, including shares we could not afford.
        remaining[opportunity.listing_index] = 0
        actions.append(f"b-{opportunity.stock}-{quantity}")

    actions.append(f"j-{transition.buy_year}-{transition.sell_year}")
    for opportunity, quantity in transition.purchases:
        actions.append(f"s-{opportunity.stock}-{quantity}")

    return _State(
        cash=state.cash + transition.gain,
        year=transition.sell_year,
        energy_used=state.energy_used + transition.travel,
        remaining=tuple(remaining),
        actions=tuple(actions),
    )


def _is_better(candidate: _State, incumbent: _State) -> bool:
    if candidate.cash != incumbent.cash:
        return candidate.cash > incumbent.cash
    candidate_total_energy = candidate.energy_used + PRESENT_YEAR - candidate.year
    incumbent_total_energy = incumbent.energy_used + PRESENT_YEAR - incumbent.year
    if candidate_total_energy != incumbent_total_energy:
        return candidate_total_energy < incumbent_total_energy
    return len(candidate.actions) < len(incumbent.actions)


def _trim_frontier(
    states: Iterable[_State], listings: list[_Listing], max_profit: tuple[int, ...]
) -> list[_State]:
    deduplicated: dict[tuple[int, int, tuple[int, ...]], _State] = {}
    for state in states:
        key = (state.year, state.energy_used, state.remaining)
        previous = deduplicated.get(key)
        if previous is None or _is_better(state, previous):
            deduplicated[key] = state

    values = list(deduplicated.values())
    if len(values) <= _BEAM_WIDTH:
        return values

    by_cash = sorted(
        values,
        key=lambda state: (-state.cash, state.energy_used, len(state.actions)),
    )

    def optimistic_value(state: _State) -> int:
        return state.cash + sum(
            quantity * profit
            for quantity, profit in zip(state.remaining, max_profit, strict=True)
        )

    by_upside = sorted(
        values,
        key=lambda state: (
            -optimistic_value(state),
            -state.cash,
            state.energy_used,
        ),
    )
    selected = by_cash[:72] + by_upside[:24]
    unique: dict[tuple[int, int, tuple[int, ...]], _State] = {}
    for state in selected:
        unique[(state.year, state.energy_used, state.remaining)] = state
    return list(unique.values())


def solve_case(case: dict[str, Any]) -> list[str]:
    """Return a profitable, energy-safe action sequence for one test case."""

    energy, capital, listings, prices = _parse_case(case)
    groups = _build_groups(listings, prices)
    if not groups:
        return []

    start = _State(
        cash=capital,
        year=PRESENT_YEAR,
        energy_used=0,
        remaining=tuple(listing.qty for listing in listings),
        actions=(),
    )
    best = start
    frontier = [start]

    max_profit_by_listing = [0] * len(listings)
    for _, opportunities in groups:
        for opportunity in opportunities:
            max_profit_by_listing[opportunity.listing_index] = max(
                max_profit_by_listing[opportunity.listing_index],
                opportunity.profit,
            )
    max_profit = tuple(max_profit_by_listing)

    max_depth = min(_MAX_DEPTH, max(1, energy), max(1, len(listings) * 6))
    for _ in range(max_depth):
        next_states: list[_State] = []
        for state in frontier:
            for transition in _candidate_transitions(state, groups, energy):
                candidate = _apply_transition(state, transition)
                next_states.append(candidate)
                if _is_better(candidate, best):
                    best = candidate
        if not next_states:
            break
        frontier = _trim_frontier(next_states, listings, max_profit)

    if best.cash <= capital:
        return []
    actions = list(best.actions)
    if best.year != PRESENT_YEAR:
        actions.append(f"j-{best.year}-{PRESENT_YEAR}")
    return actions


def solve_cases(cases: list[dict[str, Any]]) -> list[list[str]]:
    """Solve every case in the root JSON array independently."""

    return [solve_case(case) for case in cases]
