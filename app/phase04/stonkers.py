"""Exact planner for the Time Travelling Stonks Man challenge.

The cheapest round trip that reaches year ``Y`` costs
``2 * (2037 - Y)``.  Consequently, an energy budget determines a contiguous
range of years that can be swept once while travelling backwards and once on
the return trip.  A stock can be bought on the outbound sweep and sold on the
return sweep, so routing and choosing the purchases separate cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class _Item:
    """A binary-split 0/1 representation of part of a bounded lot."""

    lot_index: int
    qty: int
    cost: int
    profit: int


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


def _binary_items(lots: list[_Lot]) -> list[_Item]:
    """Convert bounded lots into 0/1 knapsack items in logarithmic space."""

    items: list[_Item] = []
    for lot_index, lot in enumerate(lots):
        remaining = lot.qty
        chunk = 1
        while remaining:
            quantity = min(chunk, remaining)
            items.append(
                _Item(
                    lot_index=lot_index,
                    qty=quantity,
                    cost=quantity * lot.price,
                    profit=quantity * lot.unit_profit,
                )
            )
            remaining -= quantity
            chunk *= 2
    return items


def _choose_lots(capital: int, lots: list[_Lot]) -> list[int]:
    """Return optimal purchase quantities by exact bounded-knapsack DP."""

    if capital <= 0 or not lots:
        return [0] * len(lots)

    # No solution can spend more than the available stock costs.  Capping the
    # table at that amount avoids allocating an enormous, all-unused tail when
    # capital is much greater than the available inventory.
    capacity = min(capital, sum(lot.price * lot.qty for lot in lots))
    items = _binary_items(lots)
    profits = [0] * (capacity + 1)
    # One byte per (item, budget) makes traceback compact even when an item
    # improves many budgets.  A linked object per improvement can otherwise
    # consume far more memory than the DP table itself.
    took_item = [bytearray(capacity + 1) for _ in items]

    for item_index, item in enumerate(items):
        if item.cost > capacity:
            continue
        for budget in range(capacity, item.cost - 1, -1):
            candidate = profits[budget - item.cost] + item.profit
            if candidate > profits[budget]:
                profits[budget] = candidate
                took_item[item_index][budget] = 1

    selected = [0] * len(lots)
    budget = capacity
    for item_index in range(len(items) - 1, -1, -1):
        if not took_item[item_index][budget]:
            continue
        item = items[item_index]
        selected[item.lot_index] += item.qty
        budget -= item.cost
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
