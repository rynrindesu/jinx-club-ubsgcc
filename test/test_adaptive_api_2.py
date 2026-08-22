import base64
import copy
import json
import unittest

from app.phase2.adaptive.solution import PayloadValidationError, solve


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

    def test_consolidates_any_requested_service_after_the_timestamp(self):
        self.body["heartbeats"].extend(
            [
                {
                    "service": "billing",
                    "timestamp": 1710000131,
                    "latencyMs": 90,
                    "status": "FAIL",
                },
                {
                    "service": "billing",
                    "timestamp": 1710000122,
                    "latencyMs": 15,
                    "status": "OK",
                },
            ]
        )
        self.body["sloQuery"] = {"service": "billing", "since": 1710000130}

        result = solve(encode_payload(self.body))

        self.assertEqual(result["sloOutput"], {"availability": 0.5, "p95LatencyMs": 300})

    def test_all_ok_payments_service_has_full_availability(self):
        self.body["heartbeats"].extend(
            [
                {
                    "service": "payments",
                    "timestamp": 1710000130,
                    "latencyMs": 40,
                    "status": "OK",
                },
                {
                    "service": "payments",
                    "timestamp": 1710000131,
                    "latencyMs": 60,
                    "status": "OK",
                },
            ]
        )
        self.body["sloQuery"] = {"service": "payments", "since": 1710000130}

        result = solve(encode_payload(self.body))

        self.assertEqual(result["sloOutput"], {"availability": 1.0, "p95LatencyMs": 60})

    def test_all_failed_shipping_service_has_zero_availability(self):
        self.body["heartbeats"].extend(
            [
                {
                    "service": "shipping",
                    "timestamp": 1710000130,
                    "latencyMs": 110,
                    "status": "FAIL",
                },
                {
                    "service": "shipping",
                    "timestamp": 1710000131,
                    "latencyMs": 240,
                    "status": "FAIL",
                },
            ]
        )
        self.body["sloQuery"] = {"service": "shipping", "since": 1710000130}

        result = solve(encode_payload(self.body))

        self.assertEqual(result["sloOutput"], {"availability": 0.0, "p95LatencyMs": 240})

    def test_since_boundary_is_inclusive(self):
        self.body["sloQuery"] = {"service": "auth", "since": 1710000124}

        result = solve(encode_payload(self.body))

        self.assertEqual(result["sloOutput"], {"availability": 0.0, "p95LatencyMs": 180})

    def test_unknown_status_counts_as_unavailable(self):
        self.body["heartbeats"].extend(
            [
                {
                    "service": "notifications",
                    "timestamp": 1710000130,
                    "latencyMs": 30,
                    "status": "OK",
                },
                {
                    "service": "notifications",
                    "timestamp": 1710000131,
                    "latencyMs": 100,
                    "status": "DEGRADED",
                },
            ]
        )
        self.body["sloQuery"] = {"service": "notifications", "since": 1710000130}

        result = solve(encode_payload(self.body))

        self.assertEqual(result["sloOutput"], {"availability": 0.5, "p95LatencyMs": 100})

    def test_duplicate_heartbeats_are_counted_as_observations(self):
        self.body["heartbeats"].append(
            {
                "service": "auth",
                "timestamp": 1710000125,
                "latencyMs": 180,
                "status": "FAIL",
            }
        )

        result = solve(encode_payload(self.body))

        self.assertEqual(
            result["sloOutput"], {"availability": 1 / 3, "p95LatencyMs": 180}
        )

    def test_p95_uses_nearest_rank_for_multiple_latencies(self):
        self.body["heartbeats"].extend(
            [
                {
                    "service": "search",
                    "timestamp": 1710000130,
                    "latencyMs": 500,
                    "status": "OK",
                },
                {
                    "service": "search",
                    "timestamp": 1710000131,
                    "latencyMs": 10,
                    "status": "OK",
                },
                {
                    "service": "search",
                    "timestamp": 1710000132,
                    "latencyMs": 70,
                    "status": "OK",
                },
            ]
        )
        self.body["sloQuery"] = {"service": "search", "since": 1710000130}

        result = solve(encode_payload(self.body))

        self.assertEqual(result["sloOutput"], {"availability": 1.0, "p95LatencyMs": 500})

    def test_p95_uses_the_latency_range_that_ends_at_95_percent(self):
        self.body["heartbeats"] = [
            {
                "service": "metrics",
                "timestamp": 1710000130 + latency,
                "latencyMs": latency,
                "status": "OK",
            }
            for latency in range(1, 21)
        ]
        self.body["sloQuery"] = {"service": "metrics", "since": 1710000130}

        result = solve(encode_payload(self.body))

        # With 20 records, each range is 5%. 95% is in the 19th range.
        self.assertEqual(result["sloOutput"], {"availability": 1.0, "p95LatencyMs": 19})

    def test_rejects_malformed_payloads_and_invalid_field_values(self):
        invalid_cases = [
            ("missing priority", lambda body: body["adaptInput"]["metadata"].pop("priority")),
            ("unsupported priority", lambda body: body["adaptInput"]["metadata"].update(priority="URGENT")),
            ("non-string action", lambda body: body["adaptInput"].update(action=1)),
            ("missing heartbeat service", lambda body: body["heartbeats"][0].pop("service")),
            ("non-numeric timestamp", lambda body: body["heartbeats"][0].update(timestamp="now")),
            ("negative latency", lambda body: body["heartbeats"][0].update(latencyMs=-1)),
            ("non-string status", lambda body: body["heartbeats"][0].update(status=True)),
            ("non-numeric since", lambda body: body["sloQuery"].update(since="today")),
        ]

        for name, mutate in invalid_cases:
            with self.subTest(name=name):
                body = copy.deepcopy(self.body)
                mutate(body)
                with self.assertRaises(PayloadValidationError):
                    solve(encode_payload(body))

        invalid_json = base64.b64encode(b"{").decode("utf-8")
        for malformed_payload in ("not valid Base64!", invalid_json, None):
            with self.subTest(payload=malformed_payload):
                with self.assertRaises(PayloadValidationError):
                    solve(malformed_payload)


if __name__ == "__main__":
    unittest.main()
