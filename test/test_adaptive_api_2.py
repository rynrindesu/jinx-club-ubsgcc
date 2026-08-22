import base64
import json
import unittest

from app.phase2.adaptive.solution import solve


def encode_payload(body):
    return base64.b64encode(json.dumps(body).encode("utf-8")).decode("utf-8")


class AdaptiveGatewayPhase2Tests(unittest.TestCase):
    def setUp(self):
        self.body = {
            "adaptInput": {
                "user": {"id": "U42", "fullName": "Jane Doe"},
                "action": "CREATE",
                "metadata": {"priority": "HIGH"},
            },
            "heartbeats": [
                {
                    "service": "auth",
                    "timestamp": 1710000123,
                    "latencyMs": 120,
                    "status": "OK",
                },
                {
                    "service": "auth",
                    "timestamp": 1710000125,
                    "latencyMs": 180,
                    "status": "FAIL",
                },
                {
                    "service": "auth",
                    "timestamp": 1710000121,
                    "latencyMs": 95,
                    "status": "OK",
                },
                {
                    "service": "billing",
                    "timestamp": 1710000130,
                    "latencyMs": 300,
                    "status": "OK",
                },
            ],
            "sloQuery": {"service": "auth", "since": 1710000123},
        }

    def test_returns_the_combined_adaptation_and_slo_result(self):
        self.assertEqual(
            solve(encode_payload(self.body)),
            {
                "adaptOutput": {
                    "id": "U42",
                    "name": "Jane Doe",
                    "action": "create",
                    "priority": 3,
                },
                "sloOutput": {"availability": 0.5, "p95LatencyMs": 180},
            },
        )

    def test_empty_query_window_has_zero_metrics(self):
        self.body["sloQuery"] = {"service": "inventory", "since": 1710000123}

        result = solve(encode_payload(self.body))

        self.assertEqual(result["sloOutput"], {"availability": 0.0, "p95LatencyMs": 0})


if __name__ == "__main__":
    unittest.main()
