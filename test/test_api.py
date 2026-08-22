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


if __name__ == "__main__":
    unittest.main()
