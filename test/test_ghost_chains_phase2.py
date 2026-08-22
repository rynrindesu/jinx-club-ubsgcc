from datetime import datetime, timedelta, timezone
import unittest

from fastapi.testclient import TestClient

from app.main import app
from app.phase1.ghost_chains import GhostChainsEngine as Phase1Engine
from app.phase2.ghost_chains import GhostChainsEngine


BASE_TIME = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)


def transaction(
    tx_id,
    sender,
    recipient,
    *,
    minute=0,
    when=None,
    **identity,
):
    created_at = when or BASE_TIME + timedelta(minutes=minute)
    return {
        "txId": tx_id,
        "fromUserId": sender,
        "toUserId": recipient,
        "amount": 100.0,
        "createdAt": created_at.isoformat().replace("+00:00", "Z"),
        **identity,
    }


class CumulativeStructuralTests(unittest.TestCase):
    def test_identity_free_stream_is_exactly_phase_one(self):
        payloads = [
            transaction("one", "A", "B"),
            transaction("two", "B", "C", minute=1),
            transaction("three", "C", "A", minute=2),
            transaction("four", "B", "D", minute=3),
        ]

        self.assertEqual(
            GhostChainsEngine().score_batch(payloads),
            Phase1Engine().score_batch(payloads),
        )

    def test_structural_ordering_is_preserved_without_identity(self):
        def final_score(edges):
            engine = GhostChainsEngine()
            return engine.score_batch(
                [
                    transaction(str(index), sender, recipient, minute=index)
                    for index, (sender, recipient) in enumerate(edges)
                ]
            )[-1]

        extension = final_score([("A", "B"), ("B", "C")])
        convergence = final_score(
            [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
        )
        returning = final_score(
            [("A", "B"), ("B", "C"), ("C", "D"), ("D", "B")]
        )

        self.assertLess(extension, convergence)
        self.assertLess(convergence, returning)


class IdentityPathTests(unittest.TestCase):
    def test_consistent_device_reinforces_a_connected_flow(self):
        plain = GhostChainsEngine()
        consistent = GhostChainsEngine()
        divergent = GhostChainsEngine()

        plain.score_transaction(transaction("p1", "A", "B"))
        plain_score = plain.score_transaction(transaction("p2", "B", "C", minute=1))

        consistent.score_transaction(
            transaction("c1", "A", "B", deviceId="device-a")
        )
        consistent_score = consistent.score_transaction(
            transaction("c2", "B", "C", minute=1, deviceId="device-a")
        )

        divergent.score_transaction(
            transaction("d1", "A", "B", deviceId="device-a")
        )
        divergent_score = divergent.score_transaction(
            transaction("d2", "B", "C", minute=1, deviceId="device-b")
        )

        self.assertGreater(consistent_score, divergent_score)
        self.assertGreater(divergent_score, plain_score)

    def test_a_mid_path_shift_remains_visible_farther_downstream(self):
        consistent = GhostChainsEngine()
        shifted = GhostChainsEngine()
        consistent_payloads = [
            transaction("c1", "A", "B", deviceId="one"),
            transaction("c2", "B", "C", minute=1, deviceId="one"),
            transaction("c3", "C", "D", minute=2, deviceId="one"),
            transaction("c4", "D", "E", minute=3, deviceId="one"),
        ]
        shifted_payloads = [
            transaction("s1", "A", "B", deviceId="one"),
            transaction("s2", "B", "C", minute=1, deviceId="one"),
            transaction("s3", "C", "D", minute=2, deviceId="two"),
            transaction("s4", "D", "E", minute=3, deviceId="two"),
        ]

        consistent_final = consistent.score_batch(consistent_payloads)[-1]
        shifted_final = shifted.score_batch(shifted_payloads)[-1]

        self.assertGreater(consistent_final, shifted_final)
        self.assertGreater(shifted_final, Phase1Engine().score_batch(shifted_payloads)[-1])

    def test_missing_identity_only_scores_when_a_connected_trail_drops_it(self):
        connected = GhostChainsEngine()
        connected.score_transaction(
            transaction("first", "A", "B", deviceId="device-a")
        )
        dropout = connected.score_transaction(
            transaction("second", "B", "C", minute=1)
        )

        no_trail = GhostChainsEngine()
        plain_extension = no_trail.score_batch(
            [
                transaction("plain-1", "A", "B"),
                transaction("plain-2", "B", "C", minute=1),
            ]
        )[-1]
        isolated_missing = GhostChainsEngine().score_transaction(
            transaction("isolated", "X", "Y")
        )

        self.assertGreater(dropout, plain_extension)
        self.assertEqual(isolated_missing, 0.0)

    def test_ip_and_device_are_independent_additive_dimensions(self):
        device_only = GhostChainsEngine()
        device_only.score_transaction(
            transaction("d1", "A", "B", deviceId="device-a")
        )
        one_dimension = device_only.score_transaction(
            transaction("d2", "B", "C", minute=1, deviceId="device-a")
        )

        both = GhostChainsEngine()
        both.score_transaction(
            transaction(
                "b1", "A", "B", deviceId="device-a", ipAddress="10.0.0.1"
            )
        )
        two_dimensions = both.score_transaction(
            transaction(
                "b2",
                "B",
                "C",
                minute=1,
                deviceId="device-a",
                ipAddress="10.0.0.1",
            )
        )

        self.assertGreater(two_dimensions, one_dimension)


class DisconnectedIdentityTests(unittest.TestCase):
    def test_reuse_across_disconnected_components_is_low_but_accumulates(self):
        engine = GhostChainsEngine()

        first = engine.score_transaction(
            transaction("one", "A", "B", ipAddress="10.0.0.1")
        )
        second = engine.score_transaction(
            transaction("two", "C", "D", minute=1, ipAddress="10.0.0.1")
        )
        third = engine.score_transaction(
            transaction("three", "E", "F", minute=2, ipAddress="10.0.0.1")
        )

        self.assertEqual(first, 0.0)
        self.assertGreater(second, first)
        self.assertGreater(third, second)
        self.assertLess(third, 0.2)

    def test_unrelated_unique_identity_stays_neutral(self):
        engine = GhostChainsEngine()
        engine.score_transaction(
            transaction("one", "A", "B", ipAddress="10.0.0.1")
        )

        score = engine.score_transaction(
            transaction("two", "C", "D", minute=1, ipAddress="10.0.0.2")
        )

        self.assertEqual(score, 0.0)

    def test_expired_identity_cannot_link_a_new_component(self):
        engine = GhostChainsEngine()
        engine.score_transaction(
            transaction("old", "A", "B", ipAddress="10.0.0.1")
        )

        score = engine.score_transaction(
            transaction(
                "new",
                "C",
                "D",
                when=BASE_TIME + timedelta(hours=24, microseconds=1),
                ipAddress="10.0.0.1",
            )
        )

        self.assertEqual(score, 0.0)


class IdentityStateTests(unittest.TestCase):
    def test_retry_is_exact_and_does_not_add_identity_evidence(self):
        engine = GhostChainsEngine()
        payload = transaction("same", "A", "B", deviceId="device-a")
        original = engine.score_transaction(payload)
        before = engine.snapshot()

        retried = engine.score_transaction(dict(payload))

        self.assertEqual(retried, original)
        self.assertEqual(engine.snapshot(), before)

    def test_reset_clears_identity_evidence(self):
        engine = GhostChainsEngine()
        engine.score_transaction(
            transaction("one", "A", "B", ipAddress="10.0.0.1")
        )
        engine.reset()

        score = engine.score_transaction(
            transaction("two", "C", "D", minute=1, ipAddress="10.0.0.1")
        )

        self.assertEqual(score, 0.0)


class GhostChainsPhase2ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        response = self.client.post(
            "/ghost-chains/reset", json={"clearTransactions": True}
        )
        self.assertEqual(response.status_code, 200)

    def test_public_endpoint_uses_phase_two_identity_scoring(self):
        payloads = [
            transaction("one", "A", "B", deviceId="device-a"),
            transaction("two", "C", "D", minute=1, deviceId="device-a"),
        ]

        response = self.client.post(
            "/ghost-chains/transactions", json={"transactions": payloads}
        )

        self.assertEqual(response.status_code, 200)
        scores = [item["riskScore"] for item in response.json()["transactions"]]
        self.assertEqual(scores[0], 0.0)
        self.assertGreater(scores[1], 0.0)


if __name__ == "__main__":
    unittest.main()
