"""Exact planner for the Time Travelling Stonks Man challenge.

The cheapest round trip that reaches year ``Y`` costs
``2 * (2037 - Y)``.  Consequently, an energy budget determines a contiguous
range of years that can be swept once while travelling backwards and once on
the return trip.  A stock can be bought on the outbound sweep and sold on the
return sweep, so routing and choosing the purchases separate cleanly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import gcd
from typing import Any


PRESENT_YEAR = 2037


@dataclass(frozen=True, slots=True)
class _Lot:
    """One historical stock listing with a bounded available quantity."""

    year: int
    stock: str
    price: int
    qty: int
    sell_year: int
    unit_profit: int


def _quoted_range(
    case: dict[str, Any],
) -> tuple[int, int, dict[int, dict[str, tuple[int, int]]]]:
    """Return capital and every quote reachable within a round-trip budget."""

    energy = int(case["energy"])
    capital = int(case["capital"])
    earliest_year = PRESENT_YEAR - energy // 2
    quotes: dict[int, dict[str, tuple[int, int]]] = {}

    for raw_year, raw_stocks in case["timeline"].items():
        year = int(raw_year)
        if not earliest_year <= year <= PRESENT_YEAR:
            continue
        year_quotes: dict[str, tuple[int, int]] = {}
        for raw_stock, raw_quote in raw_stocks.items():
            year_quotes[str(raw_stock)] = (
                int(raw_quote["price"]),
                int(raw_quote["qty"]),
            )
        if year_quotes:
            quotes[year] = year_quotes

    return capital, earliest_year, quotes


def _build_lots(quotes: dict[int, dict[str, tuple[int, int]]]) -> list[_Lot]:
    """Create every profitable buy listing with its best reachable sale price."""

    best_sales: dict[str, tuple[int, int]] = {}
    for year, year_quotes in quotes.items():
        for stock, (price, _qty) in year_quotes.items():
            previous = best_sales.get(stock)
            # The year tie-break makes equally priced outputs reproducible.
            if previous is None or (price, year) > previous:
                best_sales[stock] = (price, year)

    lots: list[_Lot] = []
    for year in sorted(quotes):
        for stock in sorted(quotes[year]):
            price, qty = quotes[year][stock]
            sell_price, sell_year = best_sales[stock]
            if qty > 0 and price < sell_price:
                lots.append(
                    _Lot(
                        year=year,
                        stock=stock,
                        price=price,
                        qty=qty,
                        sell_year=sell_year,
                        unit_profit=sell_price - price,
                    )
                )
    return lots


def _scaled_capacity(capital: int, lots: list[_Lot]) -> tuple[int, int]:
    """Reduce the DP axis by the common divisor of every purchase price."""

    max_spend = min(capital, sum(lot.price * lot.qty for lot in lots))
    price_divisor = 0
    for lot in lots:
        price_divisor = gcd(price_divisor, lot.price)
    return max_spend // price_divisor, price_divisor


def _bounded_values(
    lots: list[_Lot],
    start: int,
    stop: int,
    capacity: int,
    price_divisor: int,
) -> list[int]:
    """Compute exact bounded-knapsack values in O(lots * capacity) time.

    The standard binary-split form needs one traceback row for every split
    item.  Instead, this applies every bounded lot with the monotone-queue
    optimization, one residue class at a time.  It mutates one value row in
    place, so a DP pass uses O(capacity) memory irrespective of quantities.
    """

    values = [0] * (capacity + 1)
    for lot in lots[start:stop]:
        weight = lot.price // price_divisor
        limit = min(lot.qty, capacity // weight)
        if limit == 0:
            continue

        for residue in range(min(weight, capacity + 1)):
            final_index = (capacity - residue) // weight
            first_index = max(0, final_index - limit)
            indexes: deque[int] = deque()
            scores: deque[int] = deque()

            def add_candidate(index: int) -> None:
                score = values[residue + index * weight] - index * lot.unit_profit
                while scores and scores[-1] <= score:
                    scores.pop()
                    indexes.pop()
                indexes.append(index)
                scores.append(score)

            # Scanning from high to low keeps every source cell unchanged
            # until it has left the transition window, permitting in-place DP.
            for index in range(final_index, first_index - 1, -1):
                add_candidate(index)

            for index in range(final_index, -1, -1):
                values[residue + index * weight] = (
                    index * lot.unit_profit + scores[0]
                )
                next_index = index - 1
                while indexes and indexes[0] > next_index:
                    indexes.popleft()
                    scores.popleft()
                incoming_index = next_index - limit
                if incoming_index >= 0:
                    add_candidate(incoming_index)

    return values


def _reconstruct_quantities(
    lots: list[_Lot],
    start: int,
    stop: int,
    capacity: int,
    price_divisor: int,
    selected: list[int],
) -> None:
    """Recover an optimum without retaining a full traceback matrix.

    Each split recomputes one value row for its left and right halves, finds
    the best budget division, then recurses.  This is a Hirschberg-style
    trade: O(capacity) workspace rather than O(lots * capacity) storage while
    retaining the exact optimum.
    """

    if start == stop or capacity == 0:
        return
    if stop - start == 1:
        lot = lots[start]
        selected[start] = min(lot.qty, capacity // (lot.price // price_divisor))
        return

    middle = (start + stop) // 2
    left_values = _bounded_values(lots, start, middle, capacity, price_divisor)
    right_values = _bounded_values(lots, middle, stop, capacity, price_divisor)

    left_capacity = max(
        range(capacity + 1),
        key=lambda budget: left_values[budget] + right_values[capacity - budget],
    )
    del left_values
    del right_values

    _reconstruct_quantities(
        lots, start, middle, left_capacity, price_divisor, selected
    )
    _reconstruct_quantities(
        lots, middle, stop, capacity - left_capacity, price_divisor, selected
    )


def _choose_lots(capital: int, lots: list[_Lot]) -> list[int]:
    """Return optimal purchase quantities with an exact bounded-knapsack DP."""

    if capital <= 0 or not lots:
        return [0] * len(lots)

    capacity, price_divisor = _scaled_capacity(capital, lots)
    selected = [0] * len(lots)
    _reconstruct_quantities(
        lots, 0, len(lots), capacity, price_divisor, selected
    )
    return selected


def _append_actions(lots: list[_Lot], quantities: list[int]) -> list[str]:
    """Turn chosen lots into one energy-minimal down-and-up itinerary."""

    buys: dict[int, list[tuple[str, int]]] = {}
    sales: dict[int, dict[str, int]] = {}
    for lot, quantity in zip(lots, quantities, strict=True):
        if quantity == 0:
            continue
        buys.setdefault(lot.year, []).append((lot.stock, quantity))
        year_sales = sales.setdefault(lot.sell_year, {})
        year_sales[lot.stock] = year_sales.get(lot.stock, 0) + quantity

    if not buys:
        return []

    action_years = set(buys) | set(sales)
    turn_year = min(action_years)
    actions: list[str] = []
    current_year = PRESENT_YEAR

    # Buy while travelling to the past.  A buy at 2037 is legal before the
    # initial departure, and sales at the same year are held for the return.
    for year in sorted(buys, reverse=True):
        if current_year != year:
            actions.append(f"j-{current_year}-{year}")
            current_year = year
        for stock, quantity in sorted(buys[year]):
            actions.append(f"b-{stock}-{quantity}")

    # Continue to the lowest action year if it is only a sale location.  This
    # makes a stock's highest price usable even when it occurs before its buy
    # year on the calendar.
    if current_year != turn_year:
        actions.append(f"j-{current_year}-{turn_year}")
        current_year = turn_year

    # Sell only on the return leg, after every purchase has been made.
    for year in sorted(sales):
        if current_year != year:
            actions.append(f"j-{current_year}-{year}")
            current_year = year
        for stock, quantity in sorted(sales[year].items()):
            actions.append(f"s-{stock}-{quantity}")

    if current_year != PRESENT_YEAR:
        actions.append(f"j-{current_year}-{PRESENT_YEAR}")
    return actions


def solve_case(case: dict[str, Any]) -> list[str]:
    """Return a maximum-profit, capital-feasible, energy-safe trade plan."""

    capital, _earliest_year, quotes = _quoted_range(case)
    lots = _build_lots(quotes)
    quantities = _choose_lots(capital, lots)
    return _append_actions(lots, quantities)


def solve_cases(cases: list[dict[str, Any]]) -> list[list[str]]:
    """Solve the independent test cases supplied in the root JSON array."""

    return [solve_case(case) for case in cases]
