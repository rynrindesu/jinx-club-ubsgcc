from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import math
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.phase1.ghost_chains import (
    DiscountedWalkScorer,
    GhostChainsEngine,
    ScoreConfig,
    TemporalEdge,
    Transaction,
    TransactionConflictError,
    TransactionValidationError,
)


BASE_TIME = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)


def transaction(
    tx_id,
    sender,
    recipient,
    *,
    when=BASE_TIME,
    amount=100.0,
    **extra,
):
    return {
        "txId": tx_id,
        "fromUserId": sender,
        "toUserId": recipient,
        "amount": amount,
        "createdAt": when.isoformat().replace("+00:00", "Z"),
        **extra,
    }


def score_edges(edges, *, prefix="tx"):
    engine = GhostChainsEngine()
    payloads = [
        transaction(
            f"{prefix}-{index}",
            sender,
            recipient,
            when=BASE_TIME + timedelta(minutes=index),
        )
        for index, (sender, recipient) in enumerate(edges, start=1)
    ]
    return engine.score_batch(payloads)


class StructuralScoringTests(unittest.TestCase):
    def test_phase_one_examples_have_the_intended_ordering(self):
        examples = {
            "isolated": [("meridian", "apex")],
            "extension": [("meridian", "apex"), ("apex", "cascade")],
            "convergence": [
                ("meridian", "apex"),
                ("meridian", "horizon"),
                ("apex", "sterling"),
                ("horizon", "sterling"),
            ],
            "return": [
                ("meridian", "apex"),
                ("apex", "cascade"),
                ("cascade", "oakridge"),
                ("oakridge", "apex"),
            ],
            "multi_loop": [
                ("meridian", "apex"),
                ("apex", "cascade"),
                ("cascade", "meridian"),
                ("apex", "nimbus"),
                ("nimbus", "meridian"),
            ],
        }

        final_scores = {
            name: score_edges(edges, prefix=name)[-1]
            for name, edges in examples.items()
        }

        self.assertLess(final_scores["isolated"], final_scores["extension"])
        self.assertLess(final_scores["extension"], final_scores["convergence"])
        self.assertLess(final_scores["convergence"], final_scores["return"])
        self.assertLess(final_scores["return"], final_scores["multi_loop"])
        self.assertGreater(
            final_scores["return"] - final_scores["extension"], 0.25
        )
        self.assertGreater(
            final_scores["multi_loop"] - final_scores["return"], 0.05
        )

    def test_shortcuts_are_stronger_than_plain_extension(self):
        extension = score_edges([("A", "B"), ("B", "C")])[-1]
        short_shortcut = score_edges(
            [("A", "B"), ("B", "C"), ("A", "C")], prefix="short-2"
        )[-1]
        long_path = [(f"N{index}", f"N{index + 1}") for index in range(7)]
        long_shortcut = score_edges(
            [*long_path, ("N0", "N7")], prefix="short-7"
        )[-1]

        self.assertGreater(short_shortcut, extension)
        self.assertGreater(long_shortcut, extension)
        self.assertGreater(long_shortcut, short_shortcut)

    def test_unconnected_fan_in_stays_neutral(self):
        isolated = score_edges([("A", "C")], prefix="fan-isolated")[-1]
        fan_in = score_edges(
            [("A", "C"), ("B", "C")], prefix="fan-distinct"
        )[-1]

        self.assertEqual(isolated, 0.0)
        self.assertEqual(fan_in, isolated)

    def test_repeated_edge_is_neutral_but_reverse_edge_is_a_return(self):
        engine = GhostChainsEngine()

        first = engine.score_transaction(transaction("one", "A", "B"))
        repeat = engine.score_transaction(
            transaction("two", "A", "B", when=BASE_TIME + timedelta(minutes=1))
        )
        reverse = engine.score_transaction(
            transaction("three", "B", "A", when=BASE_TIME + timedelta(minutes=2))
        )

        self.assertEqual(first, 0.0)
        self.assertEqual(repeat, 0.0)
        self.assertGreater(reverse, 0.4)

    def test_self_transfer_is_bounded_and_does_not_amplify_the_graph(self):
        engine = GhostChainsEngine()

        first = engine.score_transaction(transaction("self-1", "A", "A"))
        repeated = engine.score_transaction(
            transaction("self-2", "A", "A", when=BASE_TIME + timedelta(minutes=1))
        )
        isolated = engine.score_transaction(
            transaction("edge", "A", "B", when=BASE_TIME + timedelta(minutes=2))
        )

        self.assertGreater(first, 0.0)
        self.assertLess(first, 0.2)
        self.assertEqual(repeated, 0.0)
        self.assertEqual(isolated, 0.0)

    def test_entity_renaming_does_not_change_scores(self):
        topology = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
        renamed = [("willow", "xenon"), ("willow", "yarrow"),
                   ("xenon", "zephyr"), ("yarrow", "zephyr")]

        self.assertEqual(score_edges(topology), score_edges(renamed))

    def test_overlapping_cycle_scores_do_not_depend_on_identifier_sort_order(self):
        topology = [
            ("A", "B"),
            ("B", "C"),
            ("C", "A"),
            ("B", "D"),
            ("D", "E"),
            ("E", "B"),
            ("C", "D"),
        ]
        mappings = (
            {"A": "z", "B": "a", "C": "y", "D": "x", "E": "w"},
            {"A": "a", "B": "z", "C": "b", "D": "c", "E": "d"},
        )
        expected = score_edges(topology, prefix="cycle-name-base")

        for index, mapping in enumerate(mappings):
            renamed = [
                (mapping[sender], mapping[recipient])
                for sender, recipient in topology
            ]
            with self.subTest(mapping=index):
                self.assertEqual(
                    expected,
                    score_edges(renamed, prefix=f"cycle-name-{index}"),
                )

    def test_temporal_route_cap_does_not_depend_on_identifier_sort_order(self):
        topology = [
            ("A", "B"),
            ("A", "C"),
            ("B", "D"),
            ("C", "D"),
            ("D", "E"),
            ("B", "E"),
            ("E", "A"),
            ("C", "F"),
            ("F", "A"),
        ]
        renamed = {
            "A": "z",
            "B": "q",
            "C": "a",
            "D": "x",
            "E": "b",
            "F": "y",
        }

        def capped_scores(mapping):
            scorer = DiscountedWalkScorer(
                ScoreConfig(max_route_signatures=5)
            )
            engine = GhostChainsEngine(scorer=scorer)
            return engine.score_batch(
                [
                    transaction(
                        f"cap-{index}",
                        mapping[sender],
                        mapping[recipient],
                        when=BASE_TIME + timedelta(minutes=index),
                    )
                    for index, (sender, recipient) in enumerate(topology)
                ]
            )

        self.assertEqual(
            capped_scores({node: node for node in "ABCDEF"}),
            capped_scores(renamed),
        )

    def test_disconnected_history_does_not_change_candidate_score(self):
        base_engine = GhostChainsEngine()
        base_engine.score_transaction(transaction("base-1", "A", "B"))
        base_score = base_engine.score_transaction(
            transaction("base-2", "B", "C", when=BASE_TIME + timedelta(minutes=3))
        )

        augmented_engine = GhostChainsEngine()
        augmented_engine.score_transaction(transaction("aug-1", "A", "B"))
        augmented_engine.score_transaction(
            transaction("aug-2", "X", "Y", when=BASE_TIME + timedelta(minutes=1))
        )
        augmented_engine.score_transaction(
            transaction("aug-3", "Q", "R", when=BASE_TIME + timedelta(minutes=2))
        )
        augmented_score = augmented_engine.score_transaction(
            transaction("aug-4", "B", "C", when=BASE_TIME + timedelta(minutes=3))
        )

        self.assertEqual(base_score, augmented_score)

    def test_large_acyclic_hub_does_not_outrank_a_return_loop(self):
        hub = GhostChainsEngine()
        for index in range(50):
            hub.score_transaction(
                transaction(
                    f"hub-{index}",
                    f"payer-{index}",
                    "merchant",
                    when=BASE_TIME + timedelta(seconds=index),
                )
            )
        hub_extension = hub.score_transaction(
            transaction(
                "hub-out",
                "merchant",
                "supplier",
                when=BASE_TIME + timedelta(minutes=1),
            )
        )

        return_loop = score_edges(
            [("A", "B"), ("B", "C"), ("C", "D"), ("D", "B")],
            prefix="return-over-hub",
        )[-1]

        self.assertLess(hub_extension, return_loop)

    def test_return_signal_decays_coherently_with_route_length(self):
        return_scores = []
        for cycle_length in range(2, 9):
            path = [
                (f"N{index}", f"N{index + 1}")
                for index in range(cycle_length - 1)
            ]
            returning = score_edges(
                [*path, (f"N{cycle_length - 1}", "N0")],
                prefix=f"cycle-{cycle_length}",
            )[-1]
            return_scores.append(returning)

        self.assertTrue(all(score > 0 for score in return_scores))
        self.assertTrue(
            all(
                shorter > longer
                for shorter, longer in zip(
                    return_scores, return_scores[1:]
                )
            )
        )

    def test_return_and_shortcut_remain_visible_beyond_enumeration_bound(self):
        path = [(f"N{index}", f"N{index + 1}") for index in range(12)]
        extension = score_edges(path, prefix="long-extension")[-1]
        shortcut = score_edges(
            [*path, ("N0", "N12")], prefix="long-shortcut"
        )[-1]
        returning = score_edges(
            [*path, ("N12", "N0")], prefix="long-return"
        )[-1]

        self.assertGreater(shortcut, extension)
        self.assertGreater(returning, shortcut)

    def test_every_bounded_return_outranks_acyclic_evidence(self):
        convergence = score_edges(
            [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")],
            prefix="return-floor-convergence",
        )[-1]
        shortcut = score_edges(
            [(f"P{index}", f"P{index + 1}") for index in range(7)]
            + [("P0", "P7")],
            prefix="return-floor-shortcut",
        )[-1]
        acyclic_ceiling = max(convergence, shortcut, 0.2)

        for cycle_length in range(2, 9):
            path = [
                (f"N{index}", f"N{index + 1}")
                for index in range(cycle_length - 1)
            ]
            returning = score_edges(
                [*path, (f"N{cycle_length - 1}", "N0")],
                prefix=f"return-floor-{cycle_length}",
            )[-1]
            with self.subTest(cycle_length=cycle_length):
                self.assertGreater(returning, acyclic_ceiling)

    def test_overlapping_second_return_outranks_a_disjoint_first_return(self):
        first_loop = [("A", "B"), ("B", "C"), ("C", "A")]
        overlapping = score_edges(
            [*first_loop, ("A", "D"), ("D", "E"), ("E", "A")],
            prefix="overlapping-return",
        )[-1]
        disjoint = score_edges(
            [*first_loop, ("X", "Y"), ("Y", "Z"), ("Z", "X")],
            prefix="disjoint-return",
        )[-1]

        self.assertGreater(overlapping, disjoint)

    def test_acyclic_evidence_accumulates_but_stays_below_a_return(self):
        branches = 5
        acyclic_history = [
            edge
            for index in range(branches)
            for edge in ((f"U{index}", "S"), (f"U{index}", "R"))
        ]
        acyclic_history.extend(("R", f"D{index}") for index in range(branches))
        dense_acyclic = score_edges(
            [*acyclic_history, ("S", "R")], prefix="acyclic-cap"
        )[-1]
        returning = score_edges(
            [("A", "B"), ("B", "C"), ("C", "D"), ("D", "B")],
            prefix="acyclic-cap-return",
        )[-1]

        simple_extension = score_edges(
            [("X", "Y"), ("Y", "Z")], prefix="acyclic-simple"
        )[-1]
        self.assertGreater(dense_acyclic, simple_extension)
        self.assertGreater(returning, dense_acyclic)

    def test_stronger_recurring_history_does_not_lower_candidate_risk(self):
        history = [
            ("0", "3"),
            ("1", "0"),
            ("2", "0"),
            ("3", "0"),
            ("3", "4"),
            ("4", "2"),
        ]
        base = score_edges([*history, ("0", "1")], prefix="monotone-base")[-1]
        reinforced = score_edges(
            [*history, ("2", "1"), ("0", "1")],
            prefix="monotone-reinforced",
        )[-1]

        self.assertGreater(reinforced, base)

    def test_amount_identity_and_unknown_fields_do_not_affect_phase_one(self):
        plain = GhostChainsEngine()
        enriched = GhostChainsEngine()
        plain_scores = plain.score_batch(
            [
                transaction("plain-1", "A", "B", amount=1),
                transaction(
                    "plain-2", "B", "C", when=BASE_TIME + timedelta(minutes=1)
                ),
            ]
        )
        enriched_scores = enriched.score_batch(
            [
                transaction(
                    "rich-1",
                    "A",
                    "B",
                    amount=999_999,
                    ipAddress="203.0.113.1",
                    deviceId="device-a",
                    futureSignal={"anything": True},
                ),
                transaction(
                    "rich-2",
                    "B",
                    "C",
                    when=BASE_TIME + timedelta(minutes=1),
                    amount=0.01,
                    ipAddress=None,
                    anotherUnknown="accepted",
                ),
            ]
        )

        self.assertEqual(plain_scores, enriched_scores)

    def test_dense_graph_scores_remain_finite_and_bounded(self):
        nodes = [f"N{index}" for index in range(7)]
        edges = [
            (sender, recipient)
            for sender in nodes
            for recipient in nodes
            if sender != recipient
        ]

        scores = score_edges(edges, prefix="dense")

        self.assertTrue(all(math.isfinite(score) for score in scores))
        self.assertTrue(all(0.0 <= score <= 1.0 for score in scores))


class TemporalStateTests(unittest.TestCase):
    def test_temporal_hypothesis_cap_never_drops_observed_edges(self):
        scorer = DiscountedWalkScorer(
            ScoreConfig(max_route_signatures=1)
        )
        events = (
            TemporalEdge("A", "B", BASE_TIME, 1),
            TemporalEdge(
                "B", "C", BASE_TIME + timedelta(minutes=1), 2
            ),
            TemporalEdge(
                "X", "Y", BASE_TIME + timedelta(minutes=2), 3
            ),
        )

        state = scorer._temporal_route_state(events)

        self.assertEqual(
            state.direct_pairs,
            frozenset({("A", "B"), ("B", "C"), ("X", "Y")}),
        )

    def test_twenty_four_hour_window_includes_the_exact_boundary(self):
        inside = GhostChainsEngine()
        inside.score_transaction(transaction("inside-1", "A", "B"))
        inside_score = inside.score_transaction(
            transaction(
                "inside-2",
                "B",
                "A",
                when=BASE_TIME + timedelta(hours=24, microseconds=-1),
            )
        )

        boundary = GhostChainsEngine()
        boundary.score_transaction(transaction("boundary-1", "A", "B"))
        boundary_score = boundary.score_transaction(
            transaction(
                "boundary-2", "B", "A", when=BASE_TIME + timedelta(hours=24)
            )
        )

        outside = GhostChainsEngine()
        outside.score_transaction(transaction("outside-1", "A", "B"))
        outside_score = outside.score_transaction(
            transaction(
                "outside-2",
                "B",
                "A",
                when=BASE_TIME + timedelta(hours=24, microseconds=1),
            )
        )

        self.assertGreater(inside_score, 0.4)
        self.assertGreater(boundary_score, 0.4)
        self.assertEqual(outside_score, 0.0)

    def test_timezone_offset_is_normalized_at_the_window_boundary(self):
        inside = GhostChainsEngine()
        first = transaction("offset-1", "A", "B")
        first["createdAt"] = "2026-06-08T20:00:00+08:00"
        inside.score_transaction(first)
        inside_score = inside.score_transaction(
            transaction(
                "offset-2",
                "B",
                "A",
                when=BASE_TIME + timedelta(hours=24, microseconds=-1),
            )
        )

        boundary = GhostChainsEngine()
        boundary.score_transaction(first | {"txId": "offset-boundary-1"})
        boundary_score = boundary.score_transaction(
            transaction(
                "offset-boundary-2",
                "B",
                "A",
                when=BASE_TIME + timedelta(hours=24),
            )
        )

        self.assertGreater(inside_score, 0.4)
        self.assertGreater(boundary_score, 0.4)

    def test_causal_chain_outranks_reversed_event_time(self):
        causal = GhostChainsEngine()
        causal.score_transaction(
            transaction("causal-1", "A", "B", when=BASE_TIME)
        )
        causal_score = causal.score_transaction(
            transaction(
                "causal-2", "B", "C", when=BASE_TIME + timedelta(minutes=1)
            )
        )

        reversed_time = GhostChainsEngine()
        reversed_time.score_transaction(
            transaction(
                "reversed-1", "A", "B", when=BASE_TIME + timedelta(minutes=1)
            )
        )
        reversed_score = reversed_time.score_transaction(
            transaction("reversed-2", "B", "C", when=BASE_TIME)
        )

        self.assertGreater(causal_score, reversed_score)
        self.assertGreater(reversed_score, 0.0)

    def test_causal_return_outranks_reversed_event_time(self):
        causal = GhostChainsEngine()
        causal_score = causal.score_batch(
            [
                transaction("cycle-causal-1", "A", "B", when=BASE_TIME),
                transaction(
                    "cycle-causal-2",
                    "B",
                    "C",
                    when=BASE_TIME + timedelta(minutes=1),
                ),
                transaction(
                    "cycle-causal-3",
                    "C",
                    "A",
                    when=BASE_TIME + timedelta(minutes=2),
                ),
            ]
        )[-1]

        reversed_time = GhostChainsEngine()
        reversed_score = reversed_time.score_batch(
            [
                transaction(
                    "cycle-reversed-1",
                    "A",
                    "B",
                    when=BASE_TIME + timedelta(minutes=2),
                ),
                transaction(
                    "cycle-reversed-2",
                    "B",
                    "C",
                    when=BASE_TIME + timedelta(minutes=1),
                ),
                transaction("cycle-reversed-3", "C", "A", when=BASE_TIME),
            ]
        )[-1]

        self.assertGreater(causal_score, reversed_score)
        self.assertGreater(reversed_score, 0.4)

    def test_equal_timestamps_use_arrival_sequence(self):
        causal = GhostChainsEngine()
        causal.score_transaction(transaction("tie-1", "A", "B"))
        causal_score = causal.score_transaction(transaction("tie-2", "B", "C"))

        reversed_arrival = GhostChainsEngine()
        reversed_arrival.score_transaction(transaction("tie-r1", "B", "C"))
        reversed_score = reversed_arrival.score_transaction(
            transaction("tie-r2", "A", "B")
        )

        self.assertGreater(causal_score, reversed_score)
        self.assertGreater(reversed_score, 0.0)

    def test_late_event_can_bridge_to_a_later_active_event(self):
        engine = GhostChainsEngine()
        engine.score_transaction(
            transaction(
                "later-edge", "B", "C", when=BASE_TIME + timedelta(minutes=2)
            )
        )

        bridge_score = engine.score_transaction(
            transaction(
                "late-bridge", "A", "B", when=BASE_TIME + timedelta(minutes=1)
            )
        )

        self.assertGreater(bridge_score, 0.0)

    def test_active_graph_bridge_is_not_erased_by_timestamp_order(self):
        engine = GhostChainsEngine()
        engine.score_transaction(
            transaction("downstream-first", "B", "C", when=BASE_TIME)
        )

        bridge_score = engine.score_transaction(
            transaction(
                "upstream-later",
                "A",
                "B",
                when=BASE_TIME + timedelta(minutes=1),
            )
        )

        self.assertGreater(bridge_score, 0.0)

    def test_temporal_support_decays_smoothly_inside_the_window(self):
        def extension_score(gap: timedelta) -> float:
            engine = GhostChainsEngine()
            engine.score_transaction(transaction(f"first-{gap}", "A", "B"))
            return engine.score_transaction(
                transaction(
                    f"second-{gap}",
                    "B",
                    "C",
                    when=BASE_TIME + gap,
                )
            )

        rapid = extension_score(timedelta(minutes=1))
        medium = extension_score(timedelta(hours=12))
        slow = extension_score(timedelta(hours=23))

        self.assertGreater(rapid, medium)
        self.assertGreater(medium, slow)
        self.assertGreater(slow, 0.0)

    def test_temporal_shortcut_support_uses_the_route_completion_time(self):
        def shortcut_score(candidate_gap: timedelta) -> float:
            engine = GhostChainsEngine()
            engine.score_transaction(transaction("path-1", "A", "B"))
            engine.score_transaction(
                transaction(
                    "path-2",
                    "B",
                    "C",
                    when=BASE_TIME + timedelta(minutes=1),
                )
            )
            return engine.score_transaction(
                transaction(
                    f"shortcut-{candidate_gap}",
                    "A",
                    "C",
                    when=BASE_TIME + candidate_gap,
                )
            )

        rapid = shortcut_score(timedelta(minutes=2))
        stale = shortcut_score(timedelta(hours=23))
        retroactive = shortcut_score(timedelta(seconds=30))

        self.assertGreater(rapid, stale)
        self.assertGreater(rapid, retroactive)
        self.assertGreater(stale, 0.0)
        self.assertGreater(retroactive, 0.0)

    def test_repeated_edge_can_enable_a_new_causal_route(self):
        engine = GhostChainsEngine()
        engine.score_transaction(transaction("repeat-old", "A", "B"))
        engine.score_transaction(
            transaction(
                "repeat-prefix",
                "X",
                "A",
                when=BASE_TIME + timedelta(minutes=1),
            )
        )

        enabling_repeat = engine.score_transaction(
            transaction(
                "repeat-new",
                "A",
                "B",
                when=BASE_TIME + timedelta(minutes=2),
            )
        )

        self.assertGreater(enabling_repeat, 0.0)

    def test_out_of_order_transaction_inside_window_is_inserted(self):
        engine = GhostChainsEngine()
        engine.score_transaction(
            transaction("watermark", "X", "Y", when=BASE_TIME + timedelta(hours=23))
        )
        late_first_leg = engine.score_transaction(
            transaction("late-1", "A", "B", when=BASE_TIME + timedelta(hours=1))
        )
        late_return = engine.score_transaction(
            transaction("late-2", "B", "A", when=BASE_TIME + timedelta(hours=2))
        )

        self.assertEqual(late_first_leg, 0.0)
        self.assertGreater(late_return, 0.4)
        self.assertEqual(engine.snapshot().watermark, BASE_TIME + timedelta(hours=23))

    def test_unsorted_batch_matches_separate_arrival_order(self):
        payloads = [
            transaction(
                "unordered-1", "X", "Y", when=BASE_TIME + timedelta(hours=23)
            ),
            transaction(
                "unordered-2", "A", "B", when=BASE_TIME + timedelta(hours=1)
            ),
            transaction(
                "unordered-3", "B", "A", when=BASE_TIME + timedelta(hours=2)
            ),
        ]
        batched = GhostChainsEngine()
        separate = GhostChainsEngine()

        batch_scores = batched.score_batch(payloads)
        separate_scores = [
            separate.score_transaction(payload) for payload in payloads
        ]

        self.assertEqual(batch_scores, separate_scores)
        self.assertEqual(batched.snapshot(), separate.snapshot())
        self.assertGreater(batch_scores[-1], 0.4)

    def test_reset_clears_a_high_watermark(self):
        engine = GhostChainsEngine()
        engine.score_transaction(
            transaction(
                "future", "X", "Y", when=BASE_TIME + timedelta(days=30)
            )
        )

        engine.reset()
        engine.score_transaction(transaction("old-1", "A", "B"))
        return_score = engine.score_transaction(
            transaction(
                "old-2", "B", "A", when=BASE_TIME + timedelta(minutes=1)
            )
        )

        self.assertGreater(return_score, 0.4)
        self.assertEqual(
            engine.snapshot().watermark, BASE_TIME + timedelta(minutes=1)
        )

    def test_out_of_order_transaction_just_before_cutoff_is_not_inserted(self):
        engine = GhostChainsEngine()
        engine.score_transaction(
            transaction("watermark", "X", "Y", when=BASE_TIME + timedelta(hours=24))
        )

        stale = engine.score_transaction(
            transaction(
                "stale", "A", "B", when=BASE_TIME - timedelta(microseconds=1)
            )
        )
        would_be_return = engine.score_transaction(
            transaction(
                "later", "B", "A", when=BASE_TIME + timedelta(hours=23)
            )
        )

        self.assertEqual(stale, 0.0)
        self.assertEqual(would_be_return, 0.0)
        self.assertNotIn(("A", "B", 1), engine.snapshot().active_edges)

    def test_out_of_order_transaction_at_cutoff_remains_active(self):
        engine = GhostChainsEngine()
        engine.score_transaction(
            transaction("watermark", "X", "Y", when=BASE_TIME + timedelta(hours=24))
        )

        boundary = engine.score_transaction(transaction("boundary", "A", "B"))
        returning = engine.score_transaction(
            transaction(
                "return", "B", "A", when=BASE_TIME + timedelta(hours=23)
            )
        )

        self.assertEqual(boundary, 0.0)
        self.assertGreater(returning, 0.4)
        self.assertIn(("A", "B", 1), engine.snapshot().active_edges)

    def test_parallel_edge_reference_count_survives_first_expiry(self):
        engine = GhostChainsEngine()
        engine.score_transaction(transaction("parallel-1", "A", "B"))
        engine.score_transaction(
            transaction(
                "parallel-2", "A", "B", when=BASE_TIME + timedelta(hours=12)
            )
        )

        engine.score_transaction(
            transaction(
                "tick-1",
                "X",
                "Y",
                when=BASE_TIME + timedelta(hours=24, microseconds=1),
            )
        )
        self.assertIn(("A", "B", 1), engine.snapshot().active_edges)

        engine.score_transaction(
            transaction(
                "tick-2",
                "Y",
                "Z",
                when=BASE_TIME + timedelta(hours=36, microseconds=1),
            )
        )
        self.assertNotIn(("A", "B", 1), engine.snapshot().active_edges)

    def test_expiry_breaks_a_return_loop(self):
        active = GhostChainsEngine()
        active.score_transaction(transaction("a-1", "A", "B"))
        active.score_transaction(
            transaction("a-2", "B", "C", when=BASE_TIME + timedelta(hours=1))
        )
        active_return = active.score_transaction(
            transaction("a-3", "C", "A", when=BASE_TIME + timedelta(hours=2))
        )

        expired = GhostChainsEngine()
        expired.score_transaction(transaction("e-1", "A", "B"))
        expired.score_transaction(
            transaction("e-2", "B", "C", when=BASE_TIME + timedelta(hours=1))
        )
        expired_return = expired.score_transaction(
            transaction(
                "e-3",
                "C",
                "A",
                when=BASE_TIME + timedelta(hours=24, microseconds=1),
            )
        )

        self.assertGreater(active_return, expired_return)
        self.assertLess(expired_return, 0.2)


class IdempotencyAndBatchTests(unittest.TestCase):
    def test_absent_and_explicit_null_identity_remain_distinct_states(self):
        absent = Transaction.from_mapping(transaction("absent", "A", "B"))
        explicit_null = Transaction.from_mapping(
            transaction("null", "A", "B", ipAddress=None, deviceId=None)
        )

        self.assertFalse(absent.ip.present)
        self.assertFalse(absent.device.present)
        self.assertTrue(explicit_null.ip.present)
        self.assertIsNone(explicit_null.ip.value)
        self.assertTrue(explicit_null.device.present)
        self.assertIsNone(explicit_null.device.value)

    def test_huge_numeric_amount_is_a_validation_error(self):
        payload = transaction("huge", "A", "B", amount=10**10_000)

        with self.assertRaises(TransactionValidationError):
            GhostChainsEngine().score_transaction(payload)

    def test_identical_duplicate_returns_exact_score_without_mutation(self):
        engine = GhostChainsEngine()
        engine.score_transaction(transaction("upstream", "A", "B"))
        payload = transaction(
            "return", "B", "A", when=BASE_TIME + timedelta(minutes=1)
        )
        original = engine.score_transaction(payload)
        before_retry = engine.snapshot()

        retried = engine.score_transaction(dict(payload))

        self.assertEqual(retried, original)
        self.assertEqual(engine.snapshot(), before_retry)

    def test_conflicting_duplicate_is_rejected_without_mutation(self):
        engine = GhostChainsEngine()
        engine.score_transaction(transaction("same-id", "A", "B"))
        before_conflict = engine.snapshot()

        with self.assertRaises(TransactionConflictError):
            engine.score_transaction(
                transaction(
                    "same-id",
                    "A",
                    "C",
                    when=BASE_TIME + timedelta(days=365),
                )
            )

        self.assertEqual(engine.snapshot(), before_conflict)

    def test_duplicate_keeps_original_score_after_graph_expiry(self):
        engine = GhostChainsEngine()
        engine.score_transaction(transaction("leg", "A", "B"))
        returning_payload = transaction(
            "return", "B", "A", when=BASE_TIME + timedelta(hours=1)
        )
        original_score = engine.score_transaction(returning_payload)
        engine.score_transaction(
            transaction("advance", "X", "Y", when=BASE_TIME + timedelta(hours=30))
        )
        after_expiry = engine.snapshot()

        retry_score = engine.score_transaction(returning_payload)

        self.assertEqual(retry_score, original_score)
        self.assertEqual(engine.snapshot(), after_expiry)

    def test_batch_conflict_is_atomic_even_after_an_earlier_new_item(self):
        engine = GhostChainsEngine()
        engine.score_transaction(transaction("existing", "A", "B"))
        before_batch = engine.snapshot()
        batch = [
            transaction("new", "B", "C", when=BASE_TIME + timedelta(minutes=1)),
            transaction("existing", "A", "D"),
        ]

        with self.assertRaises(TransactionConflictError):
            engine.score_batch(batch)

        self.assertEqual(engine.snapshot(), before_batch)

    def test_invalid_batch_is_atomic(self):
        engine = GhostChainsEngine()
        before_batch = engine.snapshot()
        invalid = transaction("invalid", "B", "C")
        del invalid["amount"]

        with self.assertRaises(TransactionValidationError):
            engine.score_batch([transaction("valid", "A", "B"), invalid])

        self.assertEqual(engine.snapshot(), before_batch)

    def test_batch_and_separate_requests_produce_identical_results(self):
        payloads = [
            transaction(
                f"tx-{index}",
                sender,
                recipient,
                when=BASE_TIME + timedelta(minutes=index),
            )
            for index, (sender, recipient) in enumerate(
                [("A", "B"), ("B", "C"), ("C", "A"), ("B", "D")],
                start=1,
            )
        ]
        batched = GhostChainsEngine()
        separate = GhostChainsEngine()

        batch_scores = batched.score_batch(payloads)
        separate_scores = [
            separate.score_transaction(payload) for payload in payloads
        ]

        self.assertEqual(batch_scores, separate_scores)
        self.assertEqual(batched.snapshot(), separate.snapshot())

    def test_duplicate_inside_batch_is_processed_only_once(self):
        engine = GhostChainsEngine()
        engine.score_transaction(transaction("upstream", "A", "B"))
        returning_payload = transaction(
            "return", "B", "A", when=BASE_TIME + timedelta(minutes=1)
        )

        scores = engine.score_batch([returning_payload, dict(returning_payload)])

        self.assertEqual(scores[0], scores[1])
        self.assertGreater(scores[0], 0.4)
        self.assertEqual(engine.snapshot().active_transactions, 2)

    def test_reset_and_replay_is_deterministic(self):
        engine = GhostChainsEngine()
        payloads = [
            transaction("one", "A", "B"),
            transaction("two", "B", "C", when=BASE_TIME + timedelta(minutes=1)),
            transaction("three", "C", "A", when=BASE_TIME + timedelta(minutes=2)),
        ]
        original = engine.score_batch(payloads)

        engine.reset()
        replayed = engine.score_batch(payloads)

        self.assertEqual(replayed, original)
        self.assertEqual(engine.snapshot().remembered_transactions, 3)

    def test_concurrent_identical_requests_are_linearizable(self):
        engine = GhostChainsEngine()
        engine.score_transaction(transaction("leg", "A", "B"))
        returning_payload = transaction(
            "return", "B", "A", when=BASE_TIME + timedelta(minutes=1)
        )

        with ThreadPoolExecutor(max_workers=8) as executor:
            scores = list(
                executor.map(
                    lambda _: engine.score_transaction(dict(returning_payload)),
                    range(24),
                )
            )

        self.assertTrue(all(score == scores[0] for score in scores))
        self.assertGreater(scores[0], 0.4)
        snapshot = engine.snapshot()
        self.assertEqual(snapshot.active_transactions, 2)
        self.assertEqual(snapshot.remembered_transactions, 2)


class GhostChainsApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.phase_environment = patch.object(
            main_module, "_GHOST_CHAINS_PHASE", "1"
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

    def test_health_contract(self):
        response = self.client.get("/ghost-chains/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_runtime_reports_the_phase_one_temporal_model(self):
        with patch.dict(
            main_module.os.environ,
            {"RENDER_GIT_COMMIT": "", "RENDER_INSTANCE_ID": ""},
        ):
            response = self.client.get("/ghost-chains/runtime")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {"phase": "1", "model": "temporal-routes-v1"}
        )

    def test_runtime_exposes_the_deployed_render_artifact(self):
        with patch.dict(
            main_module.os.environ,
            {
                "RENDER_GIT_COMMIT": "abc123",
                "RENDER_INSTANCE_ID": "instance-1",
            },
        ):
            response = self.client.get("/ghost-chains/runtime")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["revision"], "abc123")
        self.assertEqual(response.json()["instance"], "instance-1")

    def test_phase_one_endpoint_does_not_add_identity_evidence(self):
        payloads = [
            transaction("phase-one-id-1", "A", "B", deviceId="device-a"),
            transaction(
                "phase-one-id-2",
                "B",
                "C",
                when=BASE_TIME + timedelta(minutes=1),
                deviceId="device-a",
            ),
        ]
        expected = GhostChainsEngine().score_batch(payloads)

        response = self.client.post(
            "/ghost-chains/transactions", json={"transactions": payloads}
        )

        self.assertEqual(response.status_code, 200)
        observed = [item["riskScore"] for item in response.json()["transactions"]]
        self.assertEqual(observed, expected)

    def test_documented_unrelated_transactions_both_score_zero(self):
        payloads = [
            transaction(
                "tx_meridian_001", "meridian_holdings", "apex_logistics", amount=370
            ),
            transaction(
                "tx_cascade_014",
                "cascade_payments",
                "horizon_capital",
                when=BASE_TIME + timedelta(minutes=1),
            ),
        ]

        response = self.client.post(
            "/ghost-chains/transactions", json={"transactions": payloads}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "transactions": [
                    {"txId": "tx_meridian_001", "riskScore": 0.0},
                    {"txId": "tx_cascade_014", "riskScore": 0.0},
                ]
            },
        )

    def test_empty_batch_returns_empty_result(self):
        response = self.client.post(
            "/ghost-chains/transactions", json={"transactions": []}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"transactions": []})

    def test_transaction_contract_preserves_order_and_accepts_unknown_fields(self):
        payloads = [
            transaction(
                "tx_meridian_001",
                "meridian_holdings",
                "apex_logistics",
                futurePhaseField="ignored",
            ),
            transaction(
                "tx_apex_002",
                "apex_logistics",
                "cascade_payments",
                when=BASE_TIME + timedelta(minutes=1),
                deviceId="device-7",
            ),
        ]

        response = self.client.post(
            "/ghost-chains/transactions", json={"transactions": payloads}
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            [item["txId"] for item in body["transactions"]],
            ["tx_meridian_001", "tx_apex_002"],
        )
        self.assertEqual(body["transactions"][0]["riskScore"], 0.0)
        self.assertGreater(body["transactions"][1]["riskScore"], 0.0)

    def test_conflicting_transaction_id_returns_409(self):
        first = transaction("same", "A", "B")
        conflict = transaction("same", "A", "C")
        self.client.post("/ghost-chains/transactions", json={"transactions": [first]})

        response = self.client.post(
            "/ghost-chains/transactions", json={"transactions": [conflict]}
        )

        self.assertEqual(response.status_code, 409)

    def test_reset_restores_clean_graph(self):
        first_leg = transaction("one", "A", "B")
        return_leg = transaction(
            "two", "B", "A", when=BASE_TIME + timedelta(minutes=1)
        )
        self.client.post(
            "/ghost-chains/transactions", json={"transactions": [first_leg]}
        )
        before_reset = self.client.post(
            "/ghost-chains/transactions", json={"transactions": [return_leg]}
        ).json()["transactions"][0]["riskScore"]

        reset_response = self.client.post(
            "/ghost-chains/reset", json={"clearTransactions": True}
        )
        after_reset = self.client.post(
            "/ghost-chains/transactions", json={"transactions": [return_leg]}
        ).json()["transactions"][0]["riskScore"]

        self.assertEqual(reset_response.json(), {"clearTransactions": True})
        self.assertGreater(before_reset, 0.4)
        self.assertEqual(after_reset, 0.0)

    def test_reset_rejects_false_flag(self):
        response = self.client.post(
            "/ghost-chains/reset", json={"clearTransactions": False}
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
