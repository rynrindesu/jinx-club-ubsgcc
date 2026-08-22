import base64
import json
import unittest

from app.phase1.adaptive_api_gateway_1.solution import solve


def encode_payload(adapt_input):
    """Build the Base64-encoded request body expected by solve()."""
    body = json.dumps({"adaptInput": adapt_input})
    return base64.b64encode(body.encode("utf-8")).decode("utf-8")


class SolveTests(unittest.TestCase):
    def test_transforms_a_complete_high_priority_request(self):
        payload = encode_payload(
            {
                "user": {"id": "user-42", "fullName": "Ada Lovelace"},
                "action": "CREATE",
                "metadata": {"priority": "HIGH"},
            }
        )

        self.assertEqual(
            solve(payload),
            {
                "id": "user-42",
                "name": "Ada Lovelace",
                "action": "create",
                "priority": 3,
            },
        )

    def test_maps_every_supported_priority(self):
        expected_priorities = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}

        for source_priority, expected_priority in expected_priorities.items():
            with self.subTest(priority=source_priority):
                payload = encode_payload(
                    {
                        "user": {"id": 7, "fullName": "Grace Hopper"},
                        "action": "UPDATE",
                        "metadata": {"priority": source_priority},
                    }
                )

                result = solve(payload)

                self.assertEqual(result["priority"], expected_priority)
                self.assertEqual(result["action"], "update")

    def test_preserves_unicode_names_and_lowercases_mixed_case_actions(self):
        payload = encode_payload(
            {
                "user": {"id": "u-8", "fullName": "李小龍"},
                "action": "ReViEw",
                "metadata": {"priority": "MEDIUM"},
            }
        )

        result = solve(payload)

        self.assertEqual(result["id"], "u-8")
        self.assertEqual(result["name"], "李小龍")
        self.assertEqual(result["action"], "review")
        self.assertEqual(result["priority"], 2)


if __name__ == "__main__":
    unittest.main()
