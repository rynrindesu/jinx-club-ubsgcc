import unittest

from fastapi.testclient import TestClient

from app.main import app
from test.test_showdown import sample_request


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_showdown_runtime_identifies_phase_three_engine(self):
        response = self.client.get("/showdown/runtime")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["router"], "phase-aware-v3")
        self.assertEqual(
            response.json()["phase3_engine"],
            "app.phase3.showdown.engine",
        )

    def test_move_endpoint_accepts_showdown_payload(self):
        request = sample_request()

        response = self.client.post("/move", json=request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"action": "fold"})

    def test_move_endpoint_ignores_unknown_future_fields(self):
        request = sample_request()
        request["new_protocol_field"] = {"enabled": True}

        response = self.client.post("/move", json=request)

        self.assertEqual(response.status_code, 200)
        self.assertIn(response.json()["action"], request["legal_actions"])

    def test_stonks_endpoint_accepts_root_array(self):
        response = self.client.post(
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

    def test_stonks_endpoint_uses_reinvestment_optimizer(self):
        response = self.client.post(
            "/stonks",
            json=[
                {
                    "energy": 5,
                    "capital": 10,
                    "timeline": {
                        "2037": {
                            "A": {"price": 11, "qty": 0},
                            "B": {"price": 12, "qty": 0},
                        },
                        "2036": {
                            "A": {"price": 2, "qty": 2},
                            "B": {"price": 1, "qty": 2},
                        },
                        "2035": {
                            "A": {"price": 9, "qty": 0},
                            "B": {"price": 6, "qty": 2},
                        },
                    },
                }
            ],
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            [[
                "j-2037-2036",
                "b-A-2",
                "b-B-2",
                "j-2036-2035",
                "s-A-1",
                "b-B-2",
                "j-2035-2037",
                "s-A-1",
                "s-B-4",
            ]],
        )


if __name__ == "__main__":
    unittest.main()
