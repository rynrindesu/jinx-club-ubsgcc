import unittest

from app.phase04.stonks import _score_actions, solve_case


class StonksTests(unittest.TestCase):
    def test_challenge_sample(self):
        case = {
            "energy": 2,
            "capital": 500,
            "timeline": {
                "2037": {"Apple": {"price": 100, "qty": 10}},
                "2036": {"Apple": {"price": 10, "qty": 50}},
            },
        }

        self.assertEqual(
            solve_case(case),
            [
                "j-2037-2036",
                "b-Apple-50",
                "j-2036-2037",
                "s-Apple-50",
            ],
        )

    def test_sell_quantity_is_not_limited_by_destination_supply(self):
        case = {
            "energy": 2,
            "capital": 100,
            "timeline": {
                "2037": {"A": {"price": 20, "qty": 0}},
                "2036": {"A": {"price": 10, "qty": 10}},
            },
        }

        self.assertEqual(solve_case(case)[-1], "s-A-10")

    def test_revisit_can_buy_only_the_historical_quantity_remaining(self):
        case = {
            "energy": 6,
            "capital": 10,
            "timeline": {
                "2037": {"A": {"price": 20, "qty": 0}},
                "2036": {"A": {"price": 10, "qty": 4}},
            },
        }

        actions = solve_case(case)

        buys = [action for action in actions if action.startswith("b-A-")]
        self.assertGreater(len(buys), 1)
        self.assertEqual(sum(int(action.rsplit("-", 1)[1]) for action in buys), 4)
        self.assertEqual(_score_actions(case, actions), 50)

    def test_no_profitable_trade_returns_empty_actions(self):
        case = {
            "energy": 10,
            "capital": 100,
            "timeline": {
                "2037": {"A": {"price": 5, "qty": 100}},
                "2036": {"A": {"price": 5, "qty": 100}},
            },
        }

        self.assertEqual(solve_case(case), [])

    def test_can_sell_in_an_earlier_calendar_year(self):
        case = {
            "energy": 4,
            "capital": 10,
            "timeline": {
                "2037": {"A": {"price": 5, "qty": 2}},
                "2035": {"A": {"price": 20, "qty": 0}},
            },
        }

        self.assertEqual(
            solve_case(case),
            [
                "b-A-2",
                "j-2037-2035",
                "s-A-2",
                "j-2035-2037",
            ],
        )

    def test_can_hold_multiple_stocks_during_one_sweep(self):
        case = {
            "energy": 4,
            "capital": 20,
            "timeline": {
                "2037": {
                    "A": {"price": 10, "qty": 0},
                    "B": {"price": 10, "qty": 0},
                },
                "2036": {"B": {"price": 5, "qty": 1}},
                "2035": {"A": {"price": 5, "qty": 1}},
            },
        }

        actions = solve_case(case)

        self.assertIn("b-A-1", actions)
        self.assertIn("b-B-1", actions)
        self.assertIn("s-A-1", actions)
        self.assertIn("s-B-1", actions)

    def test_sweep_can_turn_at_a_sell_only_year(self):
        case = {
            "energy": 4,
            "capital": 20,
            "timeline": {
                "2037": {"A": {"price": 5, "qty": 1}},
                "2036": {"B": {"price": 5, "qty": 1}},
                "2035": {
                    "A": {"price": 10, "qty": 0},
                    "B": {"price": 10, "qty": 0},
                },
            },
        }

        actions = solve_case(case)

        self.assertIn("b-A-1", actions)
        self.assertIn("b-B-1", actions)
        self.assertIn("s-A-1", actions)
        self.assertIn("s-B-1", actions)

    def test_compounds_across_a_repeated_year_bounce(self):
        case = {
            "energy": 7,
            "capital": 3,
            "timeline": {
                "2037": {
                    "A": {"price": 11, "qty": 2},
                    "B": {"price": 11, "qty": 4},
                    "C": {"price": 4, "qty": 1},
                },
                "2036": {
                    "A": {"price": 4, "qty": 4},
                    "B": {"price": 7, "qty": 4},
                    "C": {"price": 9, "qty": 2},
                },
                "2035": {
                    "A": {"price": 3, "qty": 4},
                    "B": {"price": 10, "qty": 3},
                    "C": {"price": 1, "qty": 2},
                },
            },
        }

        self.assertEqual(_score_actions(case, solve_case(case)), 78)

    def test_large_market_reinvests_across_repeated_round_trips(self):
        case = {
            "energy": 20,
            "capital": 1,
            "timeline": {"2037": {}, "2036": {}},
        }
        for index in range(70):
            stock = f"S{index}"
            case["timeline"]["2037"][stock] = {"price": 2, "qty": 0}
            case["timeline"]["2036"][stock] = {"price": 1, "qty": 1}

        self.assertEqual(_score_actions(case, solve_case(case)), 71)

if __name__ == "__main__":
    unittest.main()
