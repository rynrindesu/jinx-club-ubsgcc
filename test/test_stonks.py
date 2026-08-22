import unittest

from fastapi.testclient import TestClient

from app.main import app
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

    def test_inventory_does_not_reset_between_trips(self):
        case = {
            "energy": 6,
            "capital": 10,
            "timeline": {
                "2037": {"A": {"price": 20, "qty": 0}},
                "2036": {"A": {"price": 10, "qty": 4}},
            },
        }

        actions = solve_case(case)

        bought = sum(
            int(action.rsplit("-", 1)[1])
            for action in actions
            if action.startswith("b-A-")
        )
        self.assertEqual(bought, 4)
        self.assertNotIn("b-A-4", actions)

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

    def test_endpoint_accepts_root_array(self):
        client = TestClient(app)
        response = client.post(
            "/stonks",
            json=[
                {
                    "energy": 2,
                    "capital": 500,
                    "timeline": {
                        "2037": {"Apple": {"price": 100, "qty": 10}},
                        "2036": {"Apple": {"price": 10, "qty": 50}},
                    },
                }
            ],
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            [[
                "j-2037-2036",
                "b-Apple-50",
                "j-2036-2037",
                "s-Apple-50",
            ]],
        )


if __name__ == "__main__":
    unittest.main()
