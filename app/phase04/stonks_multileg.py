"""Bounded multi-leg planner for the Time Travelling Stonks challenge.

This planner complements the route sweep and single buy/sell planners.  Each
transition starts and ends with cash only, but may buy a basket at several
years along one monotone part of its route before liquidating the whole basket
at a common sale year.  A best-first search then chains those liquid legs and
reinvests their proceeds.

The challenge has an unusual inventory rule: buying even one share consumes
the complete ``(year, stock)`` lot.  States therefore track a consumed-lot bit
mask rather than the number of shares left in each lot.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Any, Iterable


PRESENT_YEAR = 2037

_MAX_EXPANSIONS = 1_000
_MAX_BRANCHES = 28
_MAX_STATIC_SPECS = 180
_MAX_ACTIVE_SPECS = 26
_MAX_ORDER_SCAN = 48
_MAX_SPARSE_STATES = 320


@dataclass(frozen=True, slots=True)
class _Lot:
    index: int
    year: int
    stock: str
    price: int
    qty: int
    max_unit_profit: int


@dataclass(frozen=True, slots=True)
class _Opportunity:
    lot_index: int
    buy_year: int
    stock: str
    buy_price: int
    qty: int
    sell_price: int

    @property
    def unit_profit(self) -> int:
        return self.sell_price - self.buy_price


@dataclass(frozen=True, slots=True)
class _LegBook:
    roi: tuple[_Opportunity, ...]
    unit_profit: tuple[_Opportunity, ...]
    total_profit: tuple[_Opportunity, ...]
    cheap: tuple[_Opportunity, ...]
    potential: int


@dataclass(frozen=True, slots=True)
class _Spec:
    turn_year: int
    sell_year: int
    travel: int


@dataclass(frozen=True, slots=True)
class _Basket:
    purchases: tuple[tuple[_Opportunity, int], ...]
    used_bits: int
    spend: int
    gain: int


@dataclass(frozen=True, slots=True)
class _Transition:
    cash: int
    year: int
    energy_used: int
    used_mask: int
    actions: tuple[str, ...]
    gain: int
    travel: int


@dataclass(frozen=True, slots=True)
class _State:
    cash: int
    year: int
    energy_used: int
    used_mask: int
    node_id: int
    depth: int


def _ratio(opportunity: _Opportunity) -> float:
    return opportunity.unit_profit / opportunity.buy_price


def _even_sample(values: list[int], count: int) -> list[int]:
    if len(values) <= count:
        return values
    if count <= 1:
        return [values[0]]
    indexes = {
        round(index * (len(values) - 1) / (count - 1))
        for index in range(count)
    }
    return [values[index] for index in sorted(indexes)]


class _Planner:
    def __init__(self, case: dict[str, Any]) -> None:
        self.energy = int(case["energy"])
        self.capital = int(case["capital"])
        self.prices: dict[int, dict[str, int]] = {}
        raw_quantities: dict[int, dict[str, int]] = {}

        for raw_year, raw_stocks in case["timeline"].items():
            year = int(raw_year)
            if year <= 0 or year > PRESENT_YEAR:
                continue
            self.prices[year] = {
                str(stock): int(quote["price"])
                for stock, quote in raw_stocks.items()
            }
            raw_quantities[year] = {
                str(stock): int(quote["qty"])
                for stock, quote in raw_stocks.items()
            }

        # Every visited year must fit inside a trip that starts and finishes at
        # 2037.  Removing unreachable quotes also keeps the search dimensions
        # stable on timelines containing very old distractors.
        reachable_years = {
            year
            for year in self.prices
            if 2 * abs(PRESENT_YEAR - year) <= self.energy
        }
        self.market_years = sorted(reachable_years)

        best_price: dict[str, int] = {}
        for year in reachable_years:
            for stock, price in self.prices[year].items():
                best_price[stock] = max(best_price.get(stock, 0), price)

        lots: list[_Lot] = []
        for year in self.market_years:
            for stock in sorted(raw_quantities.get(year, {})):
                qty = raw_quantities[year][stock]
                price = self.prices[year][stock]
                max_unit_profit = best_price.get(stock, price) - price
                if qty <= 0 or max_unit_profit <= 0:
                    continue
                lots.append(
                    _Lot(
                        index=len(lots),
                        year=year,
                        stock=stock,
                        price=price,
                        qty=qty,
                        max_unit_profit=max_unit_profit,
                    )
                )
        self.lots = lots
        self.lot_years = sorted({lot.year for lot in lots})

        self.buy_score: dict[int, int] = {}
        for lot in lots:
            self.buy_score[lot.year] = self.buy_score.get(lot.year, 0) + (
                lot.qty * lot.max_unit_profit
            )

        self.sale_total: dict[int, int] = {}
        self.sale_roi: dict[int, float] = {}
        for year in self.market_years:
            total = 0
            best_roi = 0.0
            year_prices = self.prices[year]
            for lot in lots:
                sell_price = year_prices.get(lot.stock)
                if sell_price is None or sell_price <= lot.price:
                    continue
                profit = sell_price - lot.price
                total += profit * lot.qty
                best_roi = max(best_roi, profit / lot.price)
            self.sale_total[year] = total
            self.sale_roi[year] = best_roi

        self._spec_cache: dict[int, tuple[_Spec, ...]] = {}
        self._book_cache: dict[tuple[int, int, int], _LegBook | None] = {}

    def _turn_pool(self, start_year: int) -> list[int]:
        years = self.lot_years
        if len(years) <= 22:
            return list(dict.fromkeys([start_year, *years]))

        by_inventory = sorted(
            years,
            key=lambda year: (
                -self.buy_score.get(year, 0),
                abs(start_year - year),
            ),
        )[:12]
        by_distance = sorted(years, key=lambda year: abs(start_year - year))[:5]
        sampled = _even_sample(years, 8)
        return list(dict.fromkeys([start_year, *by_inventory, *by_distance, *sampled]))

    def _sale_pool(self, start_year: int) -> list[int]:
        years = self.market_years
        if len(years) <= 28:
            return list(years)

        by_total = sorted(
            years,
            key=lambda year: (-self.sale_total.get(year, 0), year),
        )[:14]
        by_roi = sorted(
            years,
            key=lambda year: (-self.sale_roi.get(year, 0.0), year),
        )[:8]
        nearby = sorted(years, key=lambda year: abs(start_year - year))[:5]
        sampled = _even_sample(years, 6)
        return list(
            dict.fromkeys(
                [start_year, PRESENT_YEAR, *by_total, *by_roi, *nearby, *sampled]
            )
        )

    def _specs(self, start_year: int) -> tuple[_Spec, ...]:
        cached = self._spec_cache.get(start_year)
        if cached is not None:
            return cached

        specs: list[_Spec] = []
        for turn_year in self._turn_pool(start_year):
            for sell_year in self._sale_pool(start_year):
                travel = abs(start_year - turn_year) + abs(turn_year - sell_year)
                if travel <= 0:
                    continue
                if travel + abs(PRESENT_YEAR - sell_year) > self.energy:
                    continue
                specs.append(_Spec(turn_year, sell_year, travel))

        if len(specs) > _MAX_STATIC_SPECS:
            by_total = sorted(
                specs,
                key=lambda spec: (
                    -self.sale_total.get(spec.sell_year, 0),
                    spec.travel,
                ),
            )[:90]
            by_efficiency = sorted(
                specs,
                key=lambda spec: (
                    -self.sale_total.get(spec.sell_year, 0)
                    / max(1, spec.travel),
                    spec.travel,
                ),
            )[:55]
            by_short = sorted(specs, key=lambda spec: spec.travel)[:35]
            # Excursions that return to the same sale year are especially
            # useful for compounding without changing the search location.
            round_trips = [spec for spec in specs if spec.sell_year == start_year]
            direct = [spec for spec in specs if spec.turn_year == spec.sell_year]

            selected: dict[tuple[int, int], _Spec] = {}
            for spec in [*round_trips, *direct, *by_total, *by_efficiency, *by_short]:
                selected[(spec.turn_year, spec.sell_year)] = spec
                if len(selected) >= _MAX_STATIC_SPECS:
                    break
            specs = list(selected.values())

        specs.sort(key=lambda spec: (spec.travel, spec.turn_year, spec.sell_year))
        result = tuple(specs)
        self._spec_cache[start_year] = result
        return result

    def _leg_book(
        self, start_year: int, turn_year: int, sell_year: int
    ) -> _LegBook | None:
        key = (start_year, turn_year, sell_year)
        if key in self._book_cache:
            return self._book_cache[key]

        lower = min(start_year, turn_year)
        upper = max(start_year, turn_year)
        sell_prices = self.prices.get(sell_year, {})
        opportunities: list[_Opportunity] = []
        for lot in self.lots:
            if not lower <= lot.year <= upper:
                continue
            sell_price = sell_prices.get(lot.stock)
            if sell_price is None or sell_price <= lot.price:
                continue
            opportunities.append(
                _Opportunity(
                    lot_index=lot.index,
                    buy_year=lot.year,
                    stock=lot.stock,
                    buy_price=lot.price,
                    qty=lot.qty,
                    sell_price=sell_price,
                )
            )

        if not opportunities:
            self._book_cache[key] = None
            return None

        roi = tuple(
            sorted(
                opportunities,
                key=lambda item: (
                    -_ratio(item),
                    -item.unit_profit,
                    item.buy_price,
                    item.buy_year,
                    item.stock,
                ),
            )
        )
        unit_profit = tuple(
            sorted(
                opportunities,
                key=lambda item: (
                    -item.unit_profit,
                    -_ratio(item),
                    item.buy_price,
                    item.stock,
                ),
            )
        )
        total_profit = tuple(
            sorted(
                opportunities,
                key=lambda item: (
                    -(item.unit_profit * item.qty),
                    -_ratio(item),
                    item.stock,
                ),
            )
        )
        cheap = tuple(
            sorted(
                opportunities,
                key=lambda item: (
                    item.buy_price,
                    -_ratio(item),
                    -item.unit_profit,
                    item.stock,
                ),
            )
        )
        book = _LegBook(
            roi=roi,
            unit_profit=unit_profit,
            total_profit=total_profit,
            cheap=cheap,
            potential=sum(item.unit_profit * item.qty for item in opportunities),
        )
        self._book_cache[key] = book
        return book

    @staticmethod
    def _basket_from_purchases(
        purchases: Iterable[tuple[_Opportunity, int]],
    ) -> _Basket | None:
        normalized = tuple(
            sorted(
                (
                    (opportunity, quantity)
                    for opportunity, quantity in purchases
                    if quantity > 0
                ),
                key=lambda item: item[0].lot_index,
            )
        )
        if not normalized:
            return None
        used_bits = 0
        spend = 0
        gain = 0
        for opportunity, quantity in normalized:
            used_bits |= 1 << opportunity.lot_index
            spend += quantity * opportunity.buy_price
            gain += quantity * opportunity.unit_profit
        return _Basket(normalized, used_bits, spend, gain)

    def _greedy_basket(
        self,
        order: Iterable[_Opportunity],
        state: _State,
        gate: str,
        prefix: _Opportunity | None = None,
    ) -> _Basket | None:
        budget = state.cash
        purchases: list[tuple[_Opportunity, int]] = []
        scanned = 0
        seen_prefix = prefix is None

        ordered: Iterable[_Opportunity]
        if prefix is None:
            ordered = order
        else:
            ordered = (prefix, *order)

        for opportunity in ordered:
            if prefix is not None and opportunity.lot_index == prefix.lot_index:
                if seen_prefix:
                    continue
                seen_prefix = True
            if state.used_mask & (1 << opportunity.lot_index):
                continue
            if opportunity.buy_price > budget:
                continue
            scanned += 1
            if scanned > _MAX_ORDER_SCAN:
                break

            quantity = min(opportunity.qty, budget // opportunity.buy_price)
            if gate == "full" and quantity < opportunity.qty:
                continue
            if gate == "half" and quantity * 2 < opportunity.qty:
                continue
            if quantity <= 0:
                continue
            purchases.append((opportunity, quantity))
            budget -= quantity * opportunity.buy_price
            if budget <= 0:
                break
        return self._basket_from_purchases(purchases)

    def _sparse_basket(self, book: _LegBook, state: _State) -> _Basket | None:
        """Approximate bounded knapsack over a small diverse opportunity set."""

        candidates: list[_Opportunity] = []
        seen: set[int] = set()
        for order in (book.roi, book.unit_profit, book.total_profit, book.cheap):
            for opportunity in order:
                if len(candidates) >= 14:
                    break
                if opportunity.lot_index in seen:
                    continue
                if state.used_mask & (1 << opportunity.lot_index):
                    continue
                if opportunity.buy_price > state.cash:
                    continue
                seen.add(opportunity.lot_index)
                candidates.append(opportunity)
            if len(candidates) >= 14:
                break
        if not candidates:
            return None

        # spent -> (profit, purchases).  Only Pareto-optimal spend/profit
        # points survive each lot, with evenly sampled pruning at the cap.
        states: dict[int, tuple[int, tuple[tuple[_Opportunity, int], ...]]] = {
            0: (0, ())
        }
        for opportunity in candidates:
            max_quantity = min(
                opportunity.qty, state.cash // opportunity.buy_price
            )
            if max_quantity <= 12:
                quantities = range(max_quantity + 1)
            else:
                quantities = sorted(
                    {
                        0,
                        1,
                        max_quantity // 4,
                        max_quantity // 2,
                        (3 * max_quantity) // 4,
                        max_quantity,
                    }
                )

            next_states: dict[
                int, tuple[int, tuple[tuple[_Opportunity, int], ...]]
            ] = {}
            for spent, (profit, purchases) in states.items():
                for quantity in quantities:
                    new_spent = spent + quantity * opportunity.buy_price
                    if new_spent > state.cash:
                        break
                    new_profit = profit + quantity * opportunity.unit_profit
                    new_purchases = purchases
                    if quantity:
                        new_purchases = purchases + ((opportunity, quantity),)
                    previous = next_states.get(new_spent)
                    if previous is None or new_profit > previous[0]:
                        next_states[new_spent] = (new_profit, new_purchases)

            frontier: list[
                tuple[int, tuple[int, tuple[tuple[_Opportunity, int], ...]]]
            ] = []
            best_profit = -1
            for spent in sorted(next_states):
                value = next_states[spent]
                if value[0] > best_profit:
                    frontier.append((spent, value))
                    best_profit = value[0]
            if len(frontier) > _MAX_SPARSE_STATES:
                indexes = {
                    round(index * (len(frontier) - 1) / (_MAX_SPARSE_STATES - 1))
                    for index in range(_MAX_SPARSE_STATES)
                }
                frontier = [frontier[index] for index in sorted(indexes)]
            states = dict(frontier)

        _, purchases = max(
            states.values(),
            key=lambda value: (value[0], -sum(
                quantity * opportunity.buy_price
                for opportunity, quantity in value[1]
            )),
        )
        return self._basket_from_purchases(purchases)

    def _baskets(
        self, book: _LegBook, state: _State, allow_sparse: bool
    ) -> list[_Basket]:
        baskets: list[_Basket | None] = [
            self._greedy_basket(book.roi, state, "any"),
            self._greedy_basket(book.unit_profit, state, "any"),
            self._greedy_basket(book.total_profit, state, "any"),
            self._greedy_basket(book.cheap, state, "any"),
            self._greedy_basket(book.roi, state, "full"),
            self._greedy_basket(book.roi, state, "half"),
            self._greedy_basket(book.total_profit, state, "full"),
        ]

        # Force a few non-greedy first choices.  This repairs integer remainders
        # and preserves a high-capacity ratio leader for a later, richer visit.
        pivots: list[_Opportunity] = []
        seen_pivots: set[int] = set()
        for order in (book.unit_profit, book.total_profit, book.cheap):
            for opportunity in order:
                if opportunity.lot_index in seen_pivots:
                    continue
                if state.used_mask & (1 << opportunity.lot_index):
                    continue
                if opportunity.buy_price > state.cash:
                    continue
                seen_pivots.add(opportunity.lot_index)
                pivots.append(opportunity)
                break
        for pivot in pivots[:3]:
            baskets.append(
                self._greedy_basket(book.roi, state, "any", prefix=pivot)
            )
            quantity = min(pivot.qty, state.cash // pivot.buy_price)
            baskets.append(self._basket_from_purchases(((pivot, quantity),)))

        if allow_sparse:
            baskets.append(self._sparse_basket(book, state))

        unique: dict[tuple[tuple[int, int], ...], _Basket] = {}
        for basket in baskets:
            if basket is None or basket.spend > state.cash or basket.gain <= 0:
                continue
            fingerprint = tuple(
                (opportunity.lot_index, quantity)
                for opportunity, quantity in basket.purchases
            )
            previous = unique.get(fingerprint)
            if previous is None or basket.gain > previous.gain:
                unique[fingerprint] = basket
        return list(unique.values())

    @staticmethod
    def _rough_gain(book: _LegBook, state: _State) -> int:
        budget = state.cash
        gain = 0
        scanned = 0
        for opportunity in book.roi:
            if state.used_mask & (1 << opportunity.lot_index):
                continue
            if opportunity.buy_price > budget:
                continue
            scanned += 1
            if scanned > 28:
                break
            quantity = min(opportunity.qty, budget // opportunity.buy_price)
            gain += quantity * opportunity.unit_profit
            budget -= quantity * opportunity.buy_price
            if budget <= 0:
                break
        return gain

    def _active_specs(self, state: _State) -> list[tuple[_Spec, _LegBook, int]]:
        scored: list[tuple[_Spec, _LegBook, int]] = []
        for spec in self._specs(state.year):
            if (
                state.energy_used
                + spec.travel
                + abs(PRESENT_YEAR - spec.sell_year)
                > self.energy
            ):
                continue
            book = self._leg_book(state.year, spec.turn_year, spec.sell_year)
            if book is None:
                continue
            gain = self._rough_gain(book, state)
            if gain > 0:
                scored.append((spec, book, gain))

        if len(scored) <= _MAX_ACTIVE_SPECS:
            return scored

        by_gain = sorted(
            scored,
            key=lambda item: (-item[2], item[0].travel, -item[1].potential),
        )[:14]
        by_efficiency = sorted(
            scored,
            key=lambda item: (
                -item[2] / max(1, item[0].travel),
                -item[2],
            ),
        )[:8]
        by_short = sorted(scored, key=lambda item: (item[0].travel, -item[2]))[:4]
        unique: dict[tuple[int, int], tuple[_Spec, _LegBook, int]] = {}
        for item in [*by_gain, *by_efficiency, *by_short]:
            unique[(item[0].turn_year, item[0].sell_year)] = item
        return list(unique.values())

    def _make_transition(
        self, state: _State, spec: _Spec, basket: _Basket
    ) -> _Transition | None:
        purchases = basket.purchases
        if spec.turn_year < state.year:
            effective_turn = min(opportunity.buy_year for opportunity, _ in purchases)
            buy_years = sorted(
                {opportunity.buy_year for opportunity, _ in purchases}, reverse=True
            )
        elif spec.turn_year > state.year:
            effective_turn = max(opportunity.buy_year for opportunity, _ in purchases)
            buy_years = sorted(
                {opportunity.buy_year for opportunity, _ in purchases}
            )
        else:
            effective_turn = state.year
            buy_years = [state.year]

        travel = abs(state.year - effective_turn) + abs(
            effective_turn - spec.sell_year
        )
        energy_used = state.energy_used + travel
        if (
            travel <= 0
            or energy_used + abs(PRESENT_YEAR - spec.sell_year) > self.energy
        ):
            return None

        by_year: dict[int, list[tuple[_Opportunity, int]]] = {}
        sales: dict[str, int] = {}
        for opportunity, quantity in purchases:
            by_year.setdefault(opportunity.buy_year, []).append(
                (opportunity, quantity)
            )
            sales[opportunity.stock] = sales.get(opportunity.stock, 0) + quantity

        actions: list[str] = []
        current_year = state.year
        for year in buy_years:
            if year not in by_year:
                continue
            if current_year != year:
                actions.append(f"j-{current_year}-{year}")
                current_year = year
            for opportunity, quantity in sorted(
                by_year[year], key=lambda item: (item[0].stock, item[0].lot_index)
            ):
                actions.append(f"b-{opportunity.stock}-{quantity}")

        if current_year != spec.sell_year:
            actions.append(f"j-{current_year}-{spec.sell_year}")
        for stock, quantity in sorted(sales.items()):
            actions.append(f"s-{stock}-{quantity}")

        return _Transition(
            cash=state.cash + basket.gain,
            year=spec.sell_year,
            energy_used=energy_used,
            used_mask=state.used_mask | basket.used_bits,
            actions=tuple(actions),
            gain=basket.gain,
            travel=travel,
        )

    def _transitions(self, state: _State, expanded: int) -> list[_Transition]:
        candidates: dict[
            tuple[int, int, tuple[tuple[int, int], ...]], _Transition
        ] = {}
        active = self._active_specs(state)
        active.sort(key=lambda item: (-item[2], item[0].travel))
        for spec_index, (spec, book, _rough_gain) in enumerate(active):
            allow_sparse = expanded < 16 and spec_index < 8
            for basket in self._baskets(book, state, allow_sparse):
                transition = self._make_transition(state, spec, basket)
                if transition is None:
                    continue
                fingerprint = (
                    transition.year,
                    transition.energy_used,
                    tuple(
                        (opportunity.lot_index, quantity)
                        for opportunity, quantity in basket.purchases
                    ),
                )
                previous = candidates.get(fingerprint)
                if previous is None or transition.gain > previous.gain:
                    candidates[fingerprint] = transition

        transitions = list(candidates.values())
        if len(transitions) <= _MAX_BRANCHES:
            return transitions

        by_gain = sorted(
            transitions,
            key=lambda item: (-item.gain, item.travel, -item.cash),
        )[:14]
        by_efficiency = sorted(
            transitions,
            key=lambda item: (-item.gain / max(1, item.travel), -item.gain),
        )[:7]
        by_growth = sorted(
            transitions,
            key=lambda item: (-item.cash, item.energy_used, item.travel),
        )[:5]
        by_short = sorted(
            transitions,
            key=lambda item: (item.travel, -item.gain),
        )[:2]

        selected: dict[tuple[int, int, int], _Transition] = {}
        for transition in [*by_gain, *by_efficiency, *by_growth, *by_short]:
            key = (
                transition.year,
                transition.energy_used,
                transition.used_mask,
            )
            previous = selected.get(key)
            if previous is None or transition.cash > previous.cash:
                selected[key] = transition
        return list(selected.values())[:_MAX_BRANCHES]

    @staticmethod
    def _accept_pareto(
        pareto: dict[tuple[int, int], list[tuple[int, int]]],
        year: int,
        used_mask: int,
        energy_used: int,
        cash: int,
    ) -> bool:
        key = (year, used_mask)
        frontier = pareto.setdefault(key, [])
        if any(
            previous_energy <= energy_used and previous_cash >= cash
            for previous_energy, previous_cash in frontier
        ):
            return False
        frontier[:] = [
            (previous_energy, previous_cash)
            for previous_energy, previous_cash in frontier
            if not (
                energy_used <= previous_energy and cash >= previous_cash
            )
        ]
        frontier.append((energy_used, cash))
        return True

    def solve(self) -> list[str]:
        if not self.lots or not self.market_years:
            return []

        parents = [-1]
        segments: list[tuple[str, ...]] = [()]
        start = _State(self.capital, PRESENT_YEAR, 0, 0, 0, 0)
        queue: list[tuple[int, int, int, _State]] = []
        sequence = 0
        heapq.heappush(queue, (-start.cash, start.energy_used, sequence, start))

        pareto: dict[tuple[int, int], list[tuple[int, int]]] = {
            (PRESENT_YEAR, 0): [(0, self.capital)]
        }
        best_state = start
        expanded = 0

        while queue and expanded < _MAX_EXPANSIONS:
            _negative_cash, _energy, _sequence, state = heapq.heappop(queue)
            if (state.energy_used, state.cash) not in pareto.get(
                (state.year, state.used_mask), ()
            ):
                continue
            expanded += 1

            for transition in self._transitions(state, expanded):
                if not self._accept_pareto(
                    pareto,
                    transition.year,
                    transition.used_mask,
                    transition.energy_used,
                    transition.cash,
                ):
                    continue

                node_id = len(parents)
                parents.append(state.node_id)
                segments.append(transition.actions)
                candidate = _State(
                    cash=transition.cash,
                    year=transition.year,
                    energy_used=transition.energy_used,
                    used_mask=transition.used_mask,
                    node_id=node_id,
                    depth=state.depth + 1,
                )
                if candidate.cash > best_state.cash or (
                    candidate.cash == best_state.cash
                    and candidate.energy_used
                    + abs(PRESENT_YEAR - candidate.year)
                    < best_state.energy_used
                    + abs(PRESENT_YEAR - best_state.year)
                ):
                    best_state = candidate

                sequence += 1
                heapq.heappush(
                    queue,
                    (-candidate.cash, candidate.energy_used, sequence, candidate),
                )

        if best_state.cash <= self.capital:
            return []

        path_segments: list[tuple[str, ...]] = []
        node_id = best_state.node_id
        while node_id:
            path_segments.append(segments[node_id])
            node_id = parents[node_id]
        path_segments.reverse()
        actions = [action for segment in path_segments for action in segment]
        if best_state.year != PRESENT_YEAR:
            actions.append(f"j-{best_state.year}-{PRESENT_YEAR}")
        return actions


def _score_actions(case: dict[str, Any], actions: list[str]) -> int | None:
    """Small internal guard against emitting a malformed heuristic route."""

    timeline = {
        int(raw_year): {
            str(stock): (int(quote["price"]), int(quote["qty"]))
            for stock, quote in raw_stocks.items()
        }
        for raw_year, raw_stocks in case["timeline"].items()
    }
    cash = int(case["capital"])
    energy_left = int(case["energy"])
    year = PRESENT_YEAR
    holdings: dict[str, int] = {}
    consumed: set[tuple[int, str]] = set()

    for action in actions:
        if not isinstance(action, str) or len(action) < 3 or action[1] != "-":
            return None
        kind = action[0]
        payload = action[2:]
        if kind == "j":
            parts = payload.split("-")
            if len(parts) != 2:
                return None
            try:
                source, destination = map(int, parts)
            except ValueError:
                return None
            travel = abs(destination - source)
            if source != year or travel > energy_left:
                return None
            energy_left -= travel
            year = destination
            continue

        if kind not in {"b", "s"} or "-" not in payload:
            return None
        stock, raw_quantity = payload.rsplit("-", 1)
        try:
            quantity = int(raw_quantity)
        except ValueError:
            return None
        quote = timeline.get(year, {}).get(stock)
        if quote is None or quantity <= 0:
            return None
        price, available = quote
        if kind == "b":
            lot = (year, stock)
            cost = quantity * price
            if lot in consumed or quantity > available or cost > cash:
                return None
            consumed.add(lot)
            cash -= cost
            holdings[stock] = holdings.get(stock, 0) + quantity
        else:
            if holdings.get(stock, 0) < quantity:
                return None
            holdings[stock] -= quantity
            cash += quantity * price

    if year != PRESENT_YEAR or any(holdings.values()):
        return None
    return cash


def solve_case(case: dict[str, Any]) -> list[str]:
    """Return a legal bounded-search plan for one challenge case."""

    try:
        planner = _Planner(case)
        actions = planner.solve()
        final_cash = _score_actions(case, actions)
    except (KeyError, TypeError, ValueError, OverflowError):
        return []
    if final_cash is None or final_cash <= int(case["capital"]):
        return []
    return actions


def solve_cases(cases: list[dict[str, Any]]) -> list[list[str]]:
    """Solve each object in the root request array independently."""

    return [solve_case(case) for case in cases]
