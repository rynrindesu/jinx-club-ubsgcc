"""Guarded exact search for compact Time Travelling Stonks cases.

The production planner deliberately uses bounded heuristics for large inputs.
This module covers small, adversarial cases exactly, including partial sales,
holding several lots at once, reinvestment, revisits, and travel in either
calendar direction.  It returns ``None`` when the input is outside conservative
size limits or when the search exceeds its state budget.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


PRESENT_YEAR = 2037

_MAX_LOTS = 10
_MAX_TOTAL_QUANTITY = 20
_MAX_STOCKS = 4
_MAX_YEARS = 8
_MAX_ENERGY = 14
_MAX_EXPANDED_STATES = 300_000
_MAX_SEARCH_NODES = 900_000


@dataclass(frozen=True, slots=True)
class _Lot:
    year: int
    stock_index: int
    stock: str
    price: int
    qty: int


_State = tuple[int, int, int, tuple[int, ...]]


def _normalize_case(
    case: dict[str, Any],
) -> tuple[
    int,
    int,
    dict[int, dict[str, int]],
    list[_Lot],
    list[int],
    list[str],
] | None:
    energy = int(case["energy"])
    capital = int(case["capital"])

    if energy > _MAX_ENERGY:
        return None

    prices: dict[int, dict[str, int]] = {}
    quantities: dict[int, dict[str, int]] = {}
    for raw_year, raw_stocks in case["timeline"].items():
        year = int(raw_year)
        prices[year] = {
            str(stock): int(quote["price"])
            for stock, quote in raw_stocks.items()
        }
        quantities[year] = {
            str(stock): int(quote["qty"])
            for stock, quote in raw_stocks.items()
        }

    # With linear travel cost, visiting a quoted year and returning can never
    # cost less than this round trip.  Unreachable years cannot affect a plan.
    reachable_years = sorted(
        {
            PRESENT_YEAR,
            *(
                year
                for year in prices
                if 2 * abs(PRESENT_YEAR - year) <= energy
            ),
        }
    )
    if len(reachable_years) > _MAX_YEARS:
        return None

    relevant_lots: list[tuple[int, str, int, int]] = []
    for buy_year in reachable_years:
        for stock, qty in quantities.get(buy_year, {}).items():
            if qty <= 0:
                continue
            buy_price = prices[buy_year][stock]

            # A non-increasing purchase only consumes cash and a one-use lot.
            # It cannot be part of a strictly better terminal plan.
            can_profit = any(
                sell_price > buy_price
                and (
                    abs(PRESENT_YEAR - buy_year)
                    + abs(sell_year - buy_year)
                    + abs(PRESENT_YEAR - sell_year)
                    <= energy
                )
                for sell_year in reachable_years
                for sell_price in [prices.get(sell_year, {}).get(stock)]
                if sell_price is not None
            )
            if not can_profit:
                continue

            relevant_lots.append((buy_year, stock, buy_price, qty))

    # Zero-quantity and never-profitable quotes do not need a holdings
    # dimension.  Filtering them before applying the stock guard lets the
    # exact solver cover compact cases with a wide but mostly inert market.
    stock_names = sorted({stock for _, stock, _, _ in relevant_lots})
    if len(stock_names) > _MAX_STOCKS:
        return None
    stock_indices = {stock: index for index, stock in enumerate(stock_names)}
    lots = [
        _Lot(
            year=year,
            stock_index=stock_indices[stock],
            stock=stock,
            price=price,
            qty=qty,
        )
        for year, stock, price, qty in relevant_lots
    ]

    if (
        len(lots) > _MAX_LOTS
        or sum(lot.qty for lot in lots) > _MAX_TOTAL_QUANTITY
    ):
        return None

    return energy, capital, prices, lots, reachable_years, stock_names


def _reconstruct(
    node_id: int, parents: list[int], node_actions: list[str | None]
) -> list[str]:
    actions: list[str] = []
    while node_id:
        action = node_actions[node_id]
        if action is not None:
            actions.append(action)
        node_id = parents[node_id]
    actions.reverse()
    return actions


def solve_case(case: dict[str, Any]) -> list[str] | None:
    """Return an optimal compact-case plan, or ``None`` to use a fallback."""

    normalized = _normalize_case(case)
    if normalized is None:
        return None
    energy, capital, prices, lots, years, stock_names = normalized

    if not lots:
        return []

    lots_at_year: dict[int, list[int]] = {}
    for lot_index, lot in enumerate(lots):
        lots_at_year.setdefault(lot.year, []).append(lot_index)

    start: _State = (
        PRESENT_YEAR,
        0,
        (1 << len(lots)) - 1,
        (0,) * len(stock_names),
    )

    # For an identical structural state, extra cash strictly dominates less
    # cash: every future action remains available and the difference is kept.
    best: dict[_State, tuple[int, int]] = {start: (capital, 0)}
    queue: deque[tuple[_State, int, int]] = deque([(start, capital, 0)])

    # Nodes are immutable parent links.  A state may later receive a better
    # cash value without invalidating paths already referenced by descendants.
    parents = [-1]
    node_actions: list[str | None] = [None]

    best_terminal_cash = capital
    best_terminal_node = 0
    expanded = 0

    def push(
        state: _State, cash: int, parent_node: int, action: str
    ) -> bool:
        previous = best.get(state)
        if previous is not None and previous[0] >= cash:
            return True
        if len(parents) >= _MAX_SEARCH_NODES:
            return False
        node_id = len(parents)
        parents.append(parent_node)
        node_actions.append(action)
        best[state] = (cash, node_id)
        queue.append((state, cash, node_id))
        return True

    while queue:
        state, cash, node_id = queue.popleft()
        if best.get(state) != (cash, node_id):
            continue

        expanded += 1
        if expanded > _MAX_EXPANDED_STATES:
            return None

        year, energy_used, unused_mask, holdings = state
        if (
            year == PRESENT_YEAR
            and not any(holdings)
            and cash > best_terminal_cash
        ):
            best_terminal_cash = cash
            best_terminal_node = node_id

        year_prices = prices.get(year, {})

        # Partial sales are essential: selling only enough shares to fund a
        # high-return lot can beat liquidating the entire holding too early.
        for stock_index, held_qty in enumerate(holdings):
            if held_qty <= 0:
                continue
            stock = stock_names[stock_index]
            sell_price = year_prices.get(stock)
            if sell_price is None:
                continue
            for qty in range(1, held_qty + 1):
                next_holdings = list(holdings)
                next_holdings[stock_index] -= qty
                if not push(
                    (
                        year,
                        energy_used,
                        unused_mask,
                        tuple(next_holdings),
                    ),
                    cash + qty * sell_price,
                    node_id,
                    f"s-{stock}-{qty}",
                ):
                    return None

        # Any positive purchase consumes the entire (year, stock) lot, even
        # when current cash only permits buying part of its quoted quantity.
        for lot_index in lots_at_year.get(year, ()):
            if not (unused_mask & (1 << lot_index)):
                continue
            lot = lots[lot_index]
            max_quantity = min(lot.qty, cash // lot.price)
            for qty in range(1, max_quantity + 1):
                next_holdings = list(holdings)
                next_holdings[lot.stock_index] += qty
                if not push(
                    (
                        year,
                        energy_used,
                        unused_mask & ~(1 << lot_index),
                        tuple(next_holdings),
                    ),
                    cash - qty * lot.price,
                    node_id,
                    f"b-{lot.stock}-{qty}",
                ):
                    return None

        for target_year in years:
            if target_year == year:
                continue
            next_energy = energy_used + abs(target_year - year)
            # Reserve enough energy for the mandatory final return to 2037.
            if next_energy + abs(PRESENT_YEAR - target_year) > energy:
                continue
            if not push(
                (target_year, next_energy, unused_mask, holdings),
                cash,
                node_id,
                f"j-{year}-{target_year}",
            ):
                return None

    return _reconstruct(best_terminal_node, parents, node_actions)
