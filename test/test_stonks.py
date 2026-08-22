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
                "2036": {"A": {"price": 10, "qty": 100}},
            },
        }

        self.assertEqual(solve_case(case), [])

if __name__ == "__main__":
    unittest.main()
