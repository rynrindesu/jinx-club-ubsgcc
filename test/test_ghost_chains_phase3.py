from datetime import datetime, timedelta, timezone
from decimal import Decimal
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.phase2.ghost_chains import GhostChainsEngine as Phase2Engine
from app.phase3.ghost_chains import GhostChainsEngine


BASE_TIME = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)


def transaction(
    tx_id,
    sender,
    recipient,
    amount,
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
        "amount": amount,
        "createdAt": created_at.isoformat().replace("+00:00", "Z"),
        **identity,
    }


def score_flow(edges, *, prefix="flow", identity=None):
    identity = identity or {}
    engine = GhostChainsEngine()
    payloads = [
        transaction(
            f"{prefix}-{index}",
            sender,
            recipient,
            amount,
            minute=index,
            **identity,
        )
        for index, (sender, recipient, amount) in enumerate(edges)
    ]
    return engine.score_batch(payloads)


class Phase3ExampleTests(unittest.TestCase):
    def test_documented_final_transaction_ordering(self):
        examples = {
            "decay": [
                ("meridian", "apex", 10000),
                ("apex", "cascade", 9910),
                ("cascade", "horizon", 9820.81),
                ("horizon", "nimbus", 9732.42),
            ],
            "branches": [
                ("meridian", "apex", 10000),
                ("apex", "cascade", 9800),
                ("apex", "sterling", 5000),
                ("cascade", "horizon", 9700),
                ("sterling", "oakridge", 4900),
            ],
            "reversal": [
                ("meridian", "apex", 10000),
                ("apex", "cascade", 9950),
                ("cascade", "horizon", 9800),
                ("horizon", "nimbus", 9950),
            ],
            "convergence": [
                ("meridian", "apex", 10000),
                ("apex", "cascade", 9800),
                ("apex", "sterling", 5000),
                ("cascade", "horizon", 9700),
                ("sterling", "horizon", 4950),
            ],
        }
        final = {
            name: score_flow(edges, prefix=name)[-1]
            for name, edges in examples.items()
        }

        self.assertLess(final["decay"], final["branches"])
        self.assertLess(final["decay"], final["convergence"])
        self.assertGreater(final["reversal"], final["branches"])
        self.assertGreater(final["reversal"], final["convergence"])

    def test_return_plus_reversal_reinforces_both_signals(self):
        consistent_return = score_flow(
            [
                ("A", "B", 10000),
                ("B", "C", 9800),
                ("C", "D", 9700),
                ("D", "B", 9600),
            ],
            prefix="return-decay",
        )[-1]
        reversing_return = score_flow(
            [
                ("A", "B", 10000),
                ("B", "C", 9800),
                ("C", "D", 9700),
                ("D", "B", 9850),
            ],
            prefix="return-reverse",
        )[-1]

        self.assertGreater(reversing_return, consistent_return)


class ValueFlowInvariantTests(unittest.TestCase):
    def test_scaling_every_amount_does_not_change_scores(self):
        base = [
            ("A", "B", Decimal("10000")),
            ("B", "C", Decimal("9950")),
            ("C", "D", Decimal("9800")),
            ("D", "E", Decimal("9950")),
        ]
        scaled = [
            (sender, recipient, amount * Decimal("10"))
            for sender, recipient, amount in base
        ]

        base_scores = score_flow(base, prefix="scale-one")
        scaled_scores = score_flow(scaled, prefix="scale-ten")

        for first, second in zip(base_scores, scaled_scores, strict=True):
            self.assertAlmostEqual(first, second, places=14)

    def test_isolated_amount_size_has_no_value_signal(self):
        small = GhostChainsEngine().score_transaction(
            transaction("small", "A", "B", Decimal("1"))
        )
        large = GhostChainsEngine().score_transaction(
            transaction("large", "A", "B", Decimal("1000000000"))
        )

        self.assertEqual(small, 0.0)
        self.assertEqual(large, small)

    def test_reversal_strength_increases_smoothly_with_magnitude(self):
        def final(amount):
            return score_flow(
                [
                    ("A", "B", 10000),
                    ("B", "C", 9900),
                    ("C", "D", 9800),
                    ("D", "E", amount),
                ],
                prefix=f"magnitude-{amount}",
            )[-1]

        tiny = final(9800.1)
        moderate = final(9900)
        large = final(10500)

        self.assertLess(tiny, moderate)
        self.assertLess(moderate, large)

    def test_established_decline_makes_reversal_stronger(self):
        short = score_flow(
            [("B", "C", 9800), ("C", "D", 9900)],
            prefix="short-reversal",
        )[-1]
        established = score_flow(
            [
                ("A", "B", 10000),
                ("B", "C", 9900),
                ("C", "D", 9800),
                ("D", "E", 9900),
            ],
            prefix="established-reversal",
        )[-1]

        self.assertGreater(established, short)

    def test_candidate_that_starts_a_branch_is_not_compared_globally(self):
        def branch_score(amount):
            engine = GhostChainsEngine()
            engine.score_batch(
                [
                    transaction("root", "A", "B", 10000),
                    transaction("first-child", "B", "C", 9800, minute=1),
                ]
            )
            return engine.score_transaction(
                transaction("second-child", "B", "D", amount, minute=2)
            )

        self.assertEqual(branch_score(5000), branch_score(9800))

    def test_unrelated_sibling_amount_does_not_change_candidate(self):
        def final(sibling_amount):
            return score_flow(
                [
                    ("A", "B", 10000),
                    ("B", "C", sibling_amount),
                    ("B", "D", 5000),
                    ("C", "E", sibling_amount * Decimal("0.99")),
                    ("D", "F", 4900),
                ],
                prefix=f"sibling-{sibling_amount}",
            )[-1]

        self.assertAlmostEqual(
            final(Decimal("9800")), final(Decimal("2000")), places=14
        )

    def test_other_converging_amount_does_not_contaminate_sender_path(self):
        def final(other_amount):
            return score_flow(
                [
                    ("A", "B", 10000),
                    ("B", "C", 9800),
                    ("B", "D", 5000),
                    ("C", "E", other_amount),
                    ("D", "E", 4950),
                ],
                prefix=f"converge-{other_amount}",
            )[-1]

        self.assertAlmostEqual(final(9700), final(100), places=14)

    def test_disagreeing_convergence_hypotheses_reduce_value_confidence(self):
        def final(first_amount, second_amount):
            return score_flow(
                [
                    ("A", "H", first_amount),
                    ("B", "H", second_amount),
                    ("H", "N", 7000),
                ],
                prefix=f"mixed-{first_amount}-{second_amount}",
            )[-1]

        reversal_under_both = final(4950, 5000)
        disagreeing = final(9700, 5000)

        self.assertGreater(reversal_under_both, disagreeing)

    def test_parallel_transactions_remain_separate_value_hypotheses(self):
        ambiguous = score_flow(
            [
                ("A", "B", 100),
                ("A", "B", 1000),
                ("B", "C", 110),
            ],
            prefix="parallel-ambiguous",
        )[-1]
        consistent = score_flow(
            [
                ("A", "B", 100),
                ("A", "B", 100),
                ("B", "C", 110),
            ],
            prefix="parallel-consistent",
        )[-1]

        self.assertGreater(consistent, ambiguous)


class CumulativeAndStateTests(unittest.TestCase):
    def test_no_upstream_value_context_is_exactly_phase_two(self):
        payloads = [
            transaction("one", "A", "B", 10000, ipAddress="10.0.0.1"),
            transaction(
                "two", "C", "D", 1, minute=1, ipAddress="10.0.0.1"
            ),
        ]

        self.assertEqual(
            GhostChainsEngine().score_batch(payloads),
            Phase2Engine().score_batch(payloads),
        )

    def test_disconnected_shared_identity_never_manufactures_value_path(self):
        phase_two = Phase2Engine()
        phase_three = GhostChainsEngine()
        history = transaction(
            "history", "A", "B", 100, ipAddress="10.0.0.1"
        )
        candidate = transaction(
            "candidate",
            "X",
            "Y",
            1000,
            minute=1,
            ipAddress="10.0.0.1",
        )
        phase_two.score_transaction(history)
        phase_three.score_transaction(history)

        self.assertEqual(
            phase_three.score_transaction(candidate),
            phase_two.score_transaction(candidate),
        )

    def test_expired_predecessor_stops_affecting_value_immediately(self):
        active = GhostChainsEngine()
        active.score_transaction(transaction("active-old", "A", "B", 100))
        active_score = active.score_transaction(
            transaction(
                "active-new",
                "B",
                "C",
                110,
                when=BASE_TIME + timedelta(hours=24, microseconds=-1),
            )
        )

        boundary = GhostChainsEngine()
        boundary.score_transaction(transaction("boundary-old", "A", "B", 100))
        boundary_score = boundary.score_transaction(
            transaction(
                "boundary-new",
                "B",
                "C",
                110,
                when=BASE_TIME + timedelta(hours=24),
            )
        )

        expired = GhostChainsEngine()
        expired.score_transaction(transaction("expired-old", "A", "B", 100))
        expired_score = expired.score_transaction(
            transaction(
                "expired-new",
                "B",
                "C",
                110,
                when=BASE_TIME + timedelta(hours=24, microseconds=1),
            )
        )

        self.assertAlmostEqual(active_score, boundary_score, places=12)
        self.assertGreater(boundary_score, expired_score)
        self.assertEqual(expired_score, 0.0)

    def test_retry_does_not_add_an_amount_hypothesis(self):
        first = transaction("same", "A", "B", 100)
        candidate = transaction("next", "B", "C", 110, minute=1)

        retried = GhostChainsEngine()
        original_score = retried.score_transaction(first)
        before_retry = retried.snapshot()
        self.assertEqual(retried.score_transaction(dict(first)), original_score)
        self.assertEqual(retried.snapshot(), before_retry)
        retried_final = retried.score_transaction(candidate)

        clean = GhostChainsEngine()
        clean.score_transaction(first)
        clean_final = clean.score_transaction(candidate)

        self.assertEqual(retried_final, clean_final)

    def test_identity_dropout_and_continuing_value_reinforce_each_other(self):
        payloads = [
            transaction("one", "A", "B", 10000, deviceId="device-a"),
            transaction("two", "B", "C", 9900, minute=1),
        ]

        self.assertGreater(
            GhostChainsEngine().score_batch(payloads)[-1],
            Phase2Engine().score_batch(payloads)[-1],
        )


class GhostChainsPhase3ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.phase_environment = patch.object(
            main_module, "_GHOST_CHAINS_PHASE", "3"
        )
        cls.phase_environment.start()
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        cls.phase_environment.stop()

    def setUp(self):
        response = self.client.post(
            "/ghost-chains/reset", json={"clearTransactions": True}
        )
        self.assertEqual(response.status_code, 200)

    def test_public_endpoint_uses_phase_three_value_scoring(self):
        payloads = [
            transaction("one", "A", "B", 10000),
            transaction("two", "B", "C", 9900, minute=1),
            transaction("three", "C", "D", 9800, minute=2),
            transaction("four", "D", "E", 9950, minute=3),
        ]

        response = self.client.post(
            "/ghost-chains/transactions", json={"transactions": payloads}
        )

        self.assertEqual(response.status_code, 200)
        scores = [item["riskScore"] for item in response.json()["transactions"]]
        self.assertGreater(scores[-1], scores[-2])
        runtime = self.client.get("/ghost-chains/runtime").json()
        self.assertEqual(runtime["phase"], "3")
        self.assertEqual(runtime["model"], "segmented-value-flow-v2")


if __name__ == "__main__":
    unittest.main()
