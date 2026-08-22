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


if __name__ == "__main__":
    unittest.main()
