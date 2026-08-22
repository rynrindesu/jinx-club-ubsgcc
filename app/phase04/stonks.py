"""Planner for the Time Travelling Stonks Man challenge.

Every state produced by this module is deliberately liquid (it holds cash and
no shares).  A transition jumps to one year, buys one or more stocks there,
jumps to another quoted year, and sells everything bought by that transition.
That makes every emitted prefix legal and lets the search safely stop at any
point before returning to 2037.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key
from math import log
from typing import Any, Iterable


PRESENT_YEAR = 2037
_BEAM_WIDTH = 96
_MAX_DEPTH = 96
_MAX_GROUPS = 400
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
            if qty > 0 and year <= PRESENT_YEAR:
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
                sell_year == listing.year
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
        slice_size = _MAX_GROUPS // 4
        nearest = sorted(
            result,
            key=lambda group: (
                PRESENT_YEAR - group[0][0],
                abs(group[0][1] - group[0][0]),
            ),
        )[:slice_size]
        cheapest = sorted(
            result,
            key=lambda group: min(
                opportunity.buy_price for opportunity in group[1]
            ),
        )[:slice_size]
        selected = result[: _MAX_GROUPS // 2] + nearest + cheapest
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
    travel = abs(state.year - buy_year) + abs(sell_year - buy_year)
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

        # Preserve every single-stock choice as well.  A lower-ratio stock can
        # be the best integer purchase when the higher-ratio price leaves cash
        # stranded, and it may preserve other one-use lots for later trips.
        for opportunity in opportunities:
            transition = _make_transition(
                state, buy_year, sell_year, (opportunity,), energy
            )
            if transition is None:
                continue
            fingerprint = (
                buy_year,
                sell_year,
                ((opportunity.listing_index, transition.purchases[0][1]),),
            )
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
        # Purchased shares are permanently removed from this historical
        # year-stock inventory; any unpurchased shares remain for a revisit.
        remaining[opportunity.listing_index] -= quantity
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


def _sweep_route(minimum_year: int, prices: dict[int, dict[str, int]]) -> list[int]:
    """Visit every quoted year on both sides of one pastward round trip."""

    interior = sorted(
        (year for year in prices if minimum_year <= year < PRESENT_YEAR),
        reverse=True,
    )
    if minimum_year not in interior:
        interior.append(minimum_year)
    inbound = sorted(year for year in interior if year > minimum_year)
    return [PRESENT_YEAR, *interior, *inbound, PRESENT_YEAR]


def _route_cost(route: list[int]) -> int:
    return sum(abs(right - left) for left, right in zip(route, route[1:]))


def _compact_route(years: Iterable[int]) -> list[int]:
    route: list[int] = []
    for year in years:
        if not route or route[-1] != year:
            route.append(year)
    if not route or route[0] != PRESENT_YEAR:
        route.insert(0, PRESENT_YEAR)
    if route[-1] != PRESENT_YEAR:
        route.append(PRESENT_YEAR)
    return route


def _promising_turn_years(
    reachable: list[int],
    listings: list[_Listing],
    prices: dict[int, dict[str, int]],
) -> list[int]:
    """Rank trip depths by the inventory profit they make reachable."""

    best_price: dict[str, int] = {}
    for year_prices in prices.values():
        for stock, price in year_prices.items():
            best_price[stock] = max(best_price.get(stock, 0), price)

    score_by_year = {year: 0 for year in reachable}
    for listing in listings:
        if listing.year not in score_by_year:
            continue
        profit = best_price.get(listing.stock, listing.price) - listing.price
        if profit > 0:
            score_by_year[listing.year] += profit * listing.qty
    return sorted(
        reachable,
        key=lambda year: (
            -score_by_year[year],
            PRESENT_YEAR - year,
        ),
    )


def _candidate_routes(
    energy: int,
    capital: int,
    listings: list[_Listing],
    prices: dict[int, dict[str, int]],
) -> list[list[int]]:
    reachable = sorted(
        year
        for year in prices
        if year < PRESENT_YEAR
        and 2 * (PRESENT_YEAR - year) <= energy
    )
    if not reachable:
        return []

    route_limit = 5 if len(listings) > 250 else 9 if len(listings) > 80 else 16
    if len(reachable) > route_limit:
        positions = {
            round(index * (len(reachable) - 1) / (route_limit - 1))
            for index in range(route_limit)
        }
        sampled = [reachable[index] for index in sorted(positions)]
    else:
        sampled = reachable

    promising = _promising_turn_years(reachable, listings, prices)
    trip_years = list(dict.fromkeys(promising[:10] + sampled))
    routes: list[list[int]] = [_sweep_route(year, prices) for year in sampled]
    two_trip_routes: list[list[int]] = []

    # A shallow first trip can multiply the starting cash before a deeper trip
    # consumes a large, otherwise only-partly-affordable lot.  Order matters.
    for first_year in trip_years[:8]:
        first = _sweep_route(first_year, prices)
        first_cost = _route_cost(first)
        for second_year in trip_years[:10]:
            second = _sweep_route(second_year, prices)
            if first_cost + _route_cost(second) <= energy:
                two_trip_routes.append(
                    _compact_route([*first, *second[1:]])
                )

    # Isolated pair tours avoid being distracted by marginal listings on the
    # way to an especially strong non-home sale.  Score with affordable gain.
    pair_routes: list[tuple[Fraction, int, list[int]]] = []
    for listing in listings:
        affordable = min(listing.qty, capital // listing.price)
        if affordable <= 0:
            continue
        for sell_year, year_prices in prices.items():
            sell_price = year_prices.get(listing.stock)
            if sell_price is None or sell_price <= listing.price:
                continue
            route = _compact_route(
                [PRESENT_YEAR, listing.year, sell_year, PRESENT_YEAR]
            )
            cost = _route_cost(route)
            if cost <= 0 or cost > energy:
                continue
            gain = affordable * (sell_price - listing.price)
            pair_routes.append(
                (
                    Fraction(gain, cost),
                    gain,
                    route,
                )
            )
    pair_routes.sort(key=lambda item: (item[0], item[1]), reverse=True)

    # When energy is plentiful, repeating a strong complete tour lets each
    # sale finance a larger basket on the next visit.  Try both short repeats
    # and the maximum affordable repeat count; historical inventory prevents
    # these routes from buying more shares than actually existed.
    repeated_routes: list[list[int]] = []
    for _, _, route in pair_routes[:24]:
        base_cost = _route_cost(route)
        repetitions = min(25, energy // base_cost)
        for count in dict.fromkeys((2, 3, repetitions)):
            if count > repetitions:
                continue
            repeated = [PRESENT_YEAR]
            for _ in range(count):
                repeated.extend(route[1:])
            repeated_routes.append(_compact_route(repeated))

    # Preserve route diversity before applying the global cap: broad two-trip
    # tours, reinvestment cycles, and isolated pair tours each catch different
    # capital constraints.
    routes.extend(two_trip_routes[:24])
    routes.extend(repeated_routes[:32])
    routes.extend(route for _, _, route in pair_routes[:24])
    routes.extend(two_trip_routes[24:])

    max_routes = 45 if len(listings) > 250 else 90 if len(listings) > 80 else 140
    unique: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for route in routes:
        key = tuple(route)
        if key in seen or _route_cost(route) > energy:
            continue
        seen.add(key)
        unique.append(route)
        if len(unique) >= max_routes:
            break
    return unique


def _future_sale(
    route: list[int],
    route_index: int,
    listing: _Listing,
    prices: dict[int, dict[str, int]],
    policy: str,
) -> tuple[int, int] | None:
    candidates: list[tuple[int, int, int]] = []
    distance = 0
    for target_index in range(route_index + 1, len(route)):
        distance += abs(route[target_index] - route[target_index - 1])
        sell_price = prices.get(route[target_index], {}).get(listing.stock)
        if sell_price is not None and sell_price > listing.price:
            candidates.append((target_index, sell_price, distance))
    if not candidates:
        return None

    if policy == "earliest":
        target_index, sell_price, _ = candidates[0]
    elif policy == "quick_profit":
        target_index, sell_price, _ = max(
            candidates,
            key=lambda item: (
                Fraction(item[1] - listing.price, item[2]),
                item[1],
                -item[0],
            ),
        )
    elif policy == "calendar_rate":
        target_index, sell_price, _ = max(
            candidates,
            key=lambda item: (
                (log(item[1]) - log(listing.price))
                / max(1, abs(route[item[0]] - listing.year)),
                item[1],
                -item[0],
            ),
        )
    elif policy == "home":
        home_candidates = [
            candidate
            for candidate in candidates
            if route[candidate[0]] == PRESENT_YEAR
        ]
        if not home_candidates:
            return None
        target_index, sell_price, _ = max(
            home_candidates,
            key=lambda item: (item[1], -item[0]),
        )
    else:  # peak
        target_index, sell_price, _ = max(
            candidates,
            key=lambda item: (item[1], -item[0]),
        )
    return target_index, sell_price


def _buy_order_key(
    candidate: tuple[int, _Listing, int, int],
    policy: str,
    route: list[int],
) -> tuple[Any, ...]:
    _, listing, target_index, sell_price = candidate
    profit = sell_price - listing.price
    if policy == "unit_profit":
        return (-profit, listing.price, listing.stock)
    if policy == "cheap":
        return (listing.price, -Fraction(profit, listing.price), listing.stock)
    if policy == "quick":
        return (
            target_index,
            -Fraction(profit, listing.price),
            listing.stock,
        )
    if policy == "total_profit":
        return (-profit * listing.qty, -Fraction(profit, listing.price), listing.stock)
    if policy == "rate":
        distance = max(1, abs(route[target_index] - listing.year))
        return (
            -(log(sell_price) - log(listing.price)) / distance,
            -Fraction(profit, listing.price),
            listing.stock,
        )
    return (-Fraction(profit, listing.price), -profit, listing.stock)


def _knapsack_quantities(
    candidates: list[tuple[int, _Listing, int, int]], budget: int
) -> list[int] | None:
    """Exactly allocate modest cash balances across one stop's stock lots."""

    affordable_units = sum(
        min(listing.qty, budget // listing.price)
        for _, listing, _, _ in candidates
    )
    if budget > 5_000 or len(candidates) > 12 or affordable_units > 120:
        return None

    # spent -> (eventual profit, quantities chosen for processed candidates)
    states: dict[int, tuple[int, tuple[int, ...]]] = {0: (0, ())}
    for _, listing, _, sell_price in candidates:
        next_states: dict[int, tuple[int, tuple[int, ...]]] = {}
        for spent, (profit, quantities) in states.items():
            max_quantity = min(
                listing.qty, (budget - spent) // listing.price
            )
            for quantity in range(max_quantity + 1):
                new_spent = spent + quantity * listing.price
                candidate_value = (
                    profit + quantity * (sell_price - listing.price),
                    quantities + (quantity,),
                )
                previous = next_states.get(new_spent)
                if previous is None or candidate_value[0] > previous[0]:
                    next_states[new_spent] = candidate_value
        states = next_states

    _, (_, quantities) = max(
        states.items(),
        key=lambda item: (item[1][0], -item[0]),
    )
    return list(quantities)


def _simulate_sweep(
    route: list[int],
    capital: int,
    listings: list[_Listing],
    prices: dict[int, dict[str, int]],
    sale_policy: str,
    buy_policy: str,
    gate_policy: str,
    target_cache: dict[tuple[str, int, int], tuple[int, int] | None],
) -> tuple[int, list[str]]:
    listings_by_year: dict[int, list[tuple[int, _Listing]]] = {}
    for index, listing in enumerate(listings):
        listings_by_year.setdefault(listing.year, []).append((index, listing))

    cash = capital
    remaining_lots = [listing.qty for listing in listings]
    scheduled_sales: dict[int, list[tuple[str, int, int]]] = {}
    actions: list[str] = []

    for route_index, year in enumerate(route):
        if route_index > 0 and route[route_index - 1] != year:
            actions.append(f"j-{route[route_index - 1]}-{year}")

        for stock, quantity, sell_price in scheduled_sales.get(route_index, ()):
            cash += quantity * sell_price
            actions.append(f"s-{stock}-{quantity}")

        candidates: list[tuple[int, _Listing, int, int]] = []
        for listing_index, listing in listings_by_year.get(year, ()):
            available = remaining_lots[listing_index]
            if available <= 0:
                continue
            cache_key = (sale_policy, route_index, listing_index)
            if cache_key not in target_cache:
                target_cache[cache_key] = _future_sale(
                    route, route_index, listing, prices, sale_policy
                )
            target = target_cache[cache_key]
            if target is None:
                continue
            target_index, sell_price = target
            available_listing = _Listing(
                listing.year,
                listing.stock,
                listing.price,
                available,
            )
            candidates.append(
                (listing_index, available_listing, target_index, sell_price)
            )

        candidates.sort(
            key=lambda item: _buy_order_key(item, buy_policy, route)
        )
        quantities = (
            _knapsack_quantities(candidates, cash)
            if buy_policy == "knapsack"
            else None
        )
        for candidate_index, (
            listing_index,
            listing,
            target_index,
            sell_price,
        ) in enumerate(candidates):
            quantity = (
                quantities[candidate_index]
                if quantities is not None
                else min(listing.qty, cash // listing.price)
            )
            if quantity <= 0:
                continue

            if gate_policy != "any" and quantity < listing.qty:
                has_later_chance = False
                for future_index in range(route_index + 1, len(route)):
                    if route[future_index] != listing.year:
                        continue
                    cache_key = (sale_policy, future_index, listing_index)
                    if cache_key not in target_cache:
                        target_cache[cache_key] = _future_sale(
                            route,
                            future_index,
                            listing,
                            prices,
                            sale_policy,
                        )
                    if target_cache[cache_key] is not None:
                        has_later_chance = True
                        break

                required_numerator = {
                    "wait_quarter": 4,
                    "wait_half": 2,
                    "wait_full": 1,
                    "full": 1,
                }.get(gate_policy, 0)
                below_threshold = (
                    quantity * required_numerator < listing.qty
                    if required_numerator > 1
                    else quantity < listing.qty
                )
                if below_threshold and (
                    gate_policy == "full" or has_later_chance
                ):
                    continue

            cash -= quantity * listing.price
            remaining_lots[listing_index] -= quantity
            actions.append(f"b-{listing.stock}-{quantity}")
            scheduled_sales.setdefault(target_index, []).append(
                (listing.stock, quantity, sell_price)
            )

    return cash, actions


def _best_sweep(
    energy: int,
    capital: int,
    listings: list[_Listing],
    prices: dict[int, dict[str, int]],
) -> tuple[int, list[str]]:
    routes = _candidate_routes(energy, capital, listings, prices)
    if not routes:
        return capital, []

    best_cash = capital
    best_actions: list[str] = []
    sale_policies = (
        ("peak", "earliest", "quick_profit", "calendar_rate")
        if len(listings) > 250
        else (
            "peak",
            "earliest",
            "quick_profit",
            "calendar_rate",
            "home",
        )
    )
    buy_policies = (
        ("roi", "rate", "cheap", "quick", "total_profit")
        if len(listings) > 80
        else (
            "roi",
            "rate",
            "unit_profit",
            "cheap",
            "quick",
            "total_profit",
            "knapsack",
        )
    )
    gate_policies = (
        ("any", "wait_full")
        if len(listings) > 250
        else ("any", "wait_full", "wait_half")
        if len(listings) > 80
        else ("any", "wait_full", "wait_half", "wait_quarter", "full")
    )
    for route in routes:
        target_cache: dict[
            tuple[str, int, int], tuple[int, int] | None
        ] = {}
        for sale_policy in sale_policies:
            for buy_policy in buy_policies:
                for gate_policy in gate_policies:
                    cash, actions = _simulate_sweep(
                        route,
                        capital,
                        listings,
                        prices,
                        sale_policy,
                        buy_policy,
                        gate_policy,
                        target_cache,
                    )
                    if cash > best_cash or (
                        cash == best_cash and len(actions) < len(best_actions)
                    ):
                        best_cash = cash
                        best_actions = actions
    return best_cash, best_actions


def _solve_case_primary(case: dict[str, Any]) -> list[str]:
    """Return a profitable, energy-safe action sequence for one test case."""

    energy, capital, listings, prices = _parse_case(case)

    if len(listings) > 60:
        sweep_cash, sweep_actions = _best_sweep(
            energy, capital, listings, prices
        )
        return sweep_actions if sweep_cash > capital else []

    groups = _build_groups(listings, prices)
    if not groups:
        return []

    # Large challenge batches need a predictable response time.  The full
    # beam is reserved for compact cases; multi-policy sweeps remain legal and
    # exploit every quoted year on a round trip for large market matrices.
    if len(groups) > 160:
        sweep_cash, sweep_actions = _best_sweep(
            energy, capital, listings, prices
        )
        return sweep_actions if sweep_cash > capital else []

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

    actions = list(best.actions)
    if best.year != PRESENT_YEAR:
        actions.append(f"j-{best.year}-{PRESENT_YEAR}")

    sweep_cash, sweep_actions = _best_sweep(
        energy, capital, listings, prices
    )
    if sweep_cash > best.cash:
        return sweep_actions
    if best.cash <= capital:
        return []
    return actions


def _score_actions(case: dict[str, Any], actions: list[str]) -> int | None:
    """Return final cash for a legal plan, or ``None`` if it is invalid."""

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
    remaining_inventory = {
        (quote_year, stock): qty
        for quote_year, year_quotes in timeline.items()
        for stock, (_, qty) in year_quotes.items()
    }

    for action in actions:
        if not isinstance(action, str) or len(action) < 3 or action[1] != "-":
            return None
        kind = action[0]
        rest = action[2:]
        if kind == "j":
            parts = rest.split("-")
            if len(parts) != 2:
                return None
            try:
                source, destination = map(int, parts)
            except ValueError:
                return None
            cost = abs(destination - source)
            if (
                source != year
                or destination <= 0
                or destination > PRESENT_YEAR
                or cost > energy_left
            ):
                return None
            energy_left -= cost
            year = destination
            continue

        if kind not in {"b", "s"} or "-" not in rest:
            return None
        stock, raw_quantity = rest.rsplit("-", 1)
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
            cost = price * quantity
            if quantity > remaining_inventory.get(lot, available) or cost > cash:
                return None
            remaining_inventory[lot] -= quantity
            cash -= cost
            holdings[stock] = holdings.get(stock, 0) + quantity
        else:
            if holdings.get(stock, 0) < quantity:
                return None
            holdings[stock] -= quantity
            cash += price * quantity

    if year != PRESENT_YEAR or any(holdings.values()):
        return None
    return cash


def solve_case(case: dict[str, Any]) -> list[str]:
    """Run exact and complementary planners, returning their best legal plan."""

    # Compact adversarial cases are finite enough to solve globally, including
    # partial sales and concurrent holdings that route heuristics can miss.
    try:
        from .stonks_exact import solve_case as solve_exact_case

        exact_actions = solve_exact_case(case)
    except (KeyError, TypeError, ValueError):
        exact_actions = None
    if exact_actions is not None and _score_actions(case, exact_actions) is not None:
        return exact_actions

    candidates = [_solve_case_primary(case)]
    _, capital, listings, _ = _parse_case(case)

    # High energy can be better spent on several complete buy/sell cycles
    # than on one broad V-shaped sweep.  Keep this independent planner in the
    # ensemble so its liquid-state reinvestment routes compete by final cash.
    from .stonks_cycles import solve_case as solve_cycle_case

    candidates.append(solve_cycle_case(case))

    # The independent one-sweep bounded-knapsack planner is occasionally
    # stronger when reserving the initial capital across many outbound lots.
    binary_blocks = sum(listing.qty.bit_length() for listing in listings)
    if capital <= 20_000 and len(listings) <= 80 and binary_blocks <= 180:
        from .stonkers import solve_case as solve_knapsack_sweep

        candidates.append(solve_knapsack_sweep(case))

    best_actions: list[str] = []
    best_cash = capital
    for actions in candidates:
        final_cash = _score_actions(case, actions)
        if final_cash is None:
            continue
        if final_cash > best_cash or (
            final_cash == best_cash and len(actions) < len(best_actions)
        ):
            best_cash = final_cash
            best_actions = actions
    return best_actions


def solve_cases(cases: list[dict[str, Any]]) -> list[list[str]]:
    """Solve every case in the root JSON array independently."""

    return [solve_case(case) for case in cases]
