"""Bounded greedy planner for repeated high-energy trade cycles.

The route sweep in :mod:`stonks` is strong when one or two broad trips are
enough.  This complementary planner targets a different shape: repeatedly
liquidating one profitable lot and reinvesting the proceeds into the next
lot.  Every intermediate state is all-cash, so the generated plan is simple
and always has enough energy reserved for the mandatory return to 2037.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from time import perf_counter
from typing import Any


PRESENT_YEAR = 2037
_MAX_TRADES = 40


@dataclass(frozen=True, slots=True)
class _Sale:
    year: int
    price: int


@dataclass(frozen=True, slots=True)
class _Lot:
    year: int
    stock: str
    price: int
    qty: int
    sales: tuple[_Sale, ...]


def _sale_shortlist(
    buy_year: int,
    buy_price: int,
    stock: str,
    prices: dict[int, dict[str, int]],
) -> tuple[_Sale, ...]:
    sales = [
        _Sale(year, year_prices[stock])
        for year, year_prices in prices.items()
        if year != buy_year
        and stock in year_prices
        and year_prices[stock] > buy_price
    ]
    if len(sales) <= 6:
        return tuple(sales)

    selected = {
        max(sales, key=lambda sale: sale.price),
        min(sales, key=lambda sale: abs(sale.year - buy_year)),
        max(
            sales,
            key=lambda sale: (
                (sale.price - buy_price) / max(1, abs(sale.year - buy_year)),
                sale.price,
            ),
        ),
        max(
            sales,
            key=lambda sale: (
                log(sale.price / buy_price)
                / max(1, abs(sale.year - buy_year)),
                sale.price,
            ),
        ),
    }
    home_sale = next((sale for sale in sales if sale.year == PRESENT_YEAR), None)
    if home_sale is not None:
        selected.add(home_sale)
    return tuple(sorted(selected, key=lambda sale: sale.year))


def _parse_case(case: dict[str, Any]) -> tuple[int, int, list[_Lot]]:
    energy = int(case["energy"])
    capital = int(case["capital"])
    prices: dict[int, dict[str, int]] = {}
    raw_quantities: list[tuple[int, str, int, int]] = []

    for raw_year, raw_stocks in case["timeline"].items():
        year = int(raw_year)
        year_prices: dict[str, int] = {}
        for raw_stock, raw_quote in raw_stocks.items():
            stock = str(raw_stock)
            price = int(raw_quote["price"])
            qty = int(raw_quote["qty"])
            year_prices[stock] = price
            if qty > 0:
                raw_quantities.append((year, stock, price, qty))
        prices[year] = year_prices

    lots: list[_Lot] = []
    for year, stock, price, qty in raw_quantities:
        sales = _sale_shortlist(year, price, stock, prices)
        if sales:
            lots.append(_Lot(year, stock, price, qty, sales))
    return energy, capital, lots


def _score_key(
    strategy: str,
    lot: _Lot,
    sale: _Sale,
    quantity: int,
    travel: int,
) -> tuple[float, ...]:
    unit_profit = sale.price - lot.price
    gain = quantity * unit_profit
    ratio = sale.price / lot.price
    if strategy == "gain":
        return (gain, ratio, -travel)
    if strategy == "unit_profit":
        return (unit_profit, ratio, gain, -travel)
    if strategy == "energy":
        return (gain / travel, gain, ratio)
    if strategy == "growth":
        return (log(ratio) / travel, ratio, gain)
    if strategy == "quick":
        return (ratio / travel, -travel, gain)
    if strategy == "cheap":
        return (1 / lot.price, ratio, gain, -travel)
    return (ratio, gain, -travel)


def _run_strategy(
    energy: int,
    capital: int,
    lots: list[_Lot],
    strategy: str,
    deadline: float | None = None,
) -> tuple[int, list[str]]:
    cash = capital
    year = PRESENT_YEAR
    energy_used = 0
    remaining = [lot.qty for lot in lots]
    actions: list[str] = []

    for _ in range(_MAX_TRADES):
        if deadline is not None and perf_counter() >= deadline:
            break
        best: tuple[tuple[float, ...], int, _Sale, int, int] | None = None
        for lot_index, lot in enumerate(lots):
            if deadline is not None and perf_counter() >= deadline:
                break
            available = remaining[lot_index]
            if available <= 0 or lot.price > cash:
                continue
            quantity = min(available, cash // lot.price)
            if quantity <= 0:
                continue
            for sale in lot.sales:
                travel = abs(year - lot.year) + abs(sale.year - lot.year)
                if travel <= 0:
                    continue
                if (
                    energy_used
                    + travel
                    + abs(PRESENT_YEAR - sale.year)
                    > energy
                ):
                    continue
                key = _score_key(strategy, lot, sale, quantity, travel)
                candidate = (key, lot_index, sale, quantity, travel)
                if best is None or candidate[0] > best[0]:
                    best = candidate

        if best is None:
            break

        _, lot_index, sale, quantity, travel = best
        lot = lots[lot_index]
        if year != lot.year:
            actions.append(f"j-{year}-{lot.year}")
        actions.append(f"b-{lot.stock}-{quantity}")
        actions.append(f"j-{lot.year}-{sale.year}")
        actions.append(f"s-{lot.stock}-{quantity}")

        cash += quantity * (sale.price - lot.price)
        energy_used += travel
        year = sale.year
        remaining[lot_index] -= quantity

    if year != PRESENT_YEAR:
        actions.append(f"j-{year}-{PRESENT_YEAR}")
    return cash, actions


def solve_case(
    case: dict[str, Any], deadline: float | None = None
) -> list[str]:
    """Return the best plan found across complementary cycle priorities."""

    energy, capital, lots = _parse_case(case)
    best_cash = capital
    best_actions: list[str] = []
    for strategy in (
        "roi",
        "gain",
        "unit_profit",
        "energy",
        "growth",
        "quick",
        "cheap",
    ):
        if deadline is not None and perf_counter() >= deadline:
            break
        cash, actions = _run_strategy(
            energy, capital, lots, strategy, deadline
        )
        if cash > best_cash or (
            cash == best_cash and len(actions) < len(best_actions)
        ):
            best_cash = cash
            best_actions = actions
    return best_actions
