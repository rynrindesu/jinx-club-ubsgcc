import unittest

from app.phase04.stonks import solve_case


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

    def test_buying_once_consumes_the_entire_year_stock_lot(self):
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
        self.assertEqual(buys, ["b-A-1"])

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

if __name__ == "__main__":
    unittest.main()
