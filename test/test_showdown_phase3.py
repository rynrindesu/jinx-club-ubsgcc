"""Independent tests for the clean-room SHOWDOWN Phase 3 service."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest

from fastapi.testclient import TestClient

from app.phase3.showdown.api import app
from app.phase3.showdown.engine import (
    decide_move,
    reset_runtime_for_tests,
    runtime_snapshot,
)
from app.phase3.showdown.equity import exact_share_for_hypothesis, showdown_share
from app.phase3.showdown.learning import EventKnowledge, OpponentProfile, RuntimeStore
from app.phase3.showdown.protocol import (
    ActionRecord,
    ProtocolError,
    parse_payload,
    safe_fallback,
    validate_response,
)
from app.phase3.showdown.replay import fit_replays, write_seed
from app.phase3.showdown.rules import RuleModel, get_hypothesis
from app.phase3.showdown.simulator import (
    SimulationConfig,
    built_in_strategy,
    simulate_leg,
)


NAMES = ("you", "Dana", "Miles", "Theo", "Rhea", "Bram")


def player(
    seat: int,
    *,
    stack: int = 200,
    bet: int = 0,
    delta: int = 0,
    folded: bool = False,
    all_in: bool = False,
    busted: bool = False,
) -> dict[str, object]:
    return {
        "seat": seat,
        "name": NAMES[seat],
        "folded": folded,
        "chip_delta": delta,
        "bet_this_round": bet,
        "stack": stack,
        "all_in": all_in,
        "busted": busted,
    }


def payload(**changes: object) -> dict[str, object]:
    players = [
        player(0),
        player(1, stack=199, bet=1),
        player(2, stack=198, bet=2),
        player(3),
        player(4),
        player(5),
    ]
    result: dict[str, object] = {
        "protocol_version": 2,
        "match_id": "phase3-test-match",
        "phase": 3,
        "table_rule": "test-codename",
        "small_blind": 1,
        "big_blind": 2,
        "starting_stack": 200,
        "your_stack": 200,
        "hand_number": 1,
        "total_hands": 60,
        "round": "pre_reveal",
        "your_number": 11,
        "community_number": None,
        "your_seat": 0,
        "button_seat": 0,
        "pot": 3,
        "to_call": 2,
        "min_raise_to": 4,
        "max_raise_to": 200,
        "legal_actions": ["fold", "call", "raise"],
        "players": players,
        "current_hand_actions": [],
        "recent_hands": [],
        "leg_number": 1,
        "total_legs": 4,
    }
    result.update(changes)
    return result


class ProtocolTests(unittest.TestCase):
    def test_tolerates_unknown_fields_and_filters_live_players(self) -> None:
        players = [
            player(0),
            player(1, folded=True),
            player(2, all_in=True, stack=0),
            player(3, busted=True, stack=0, delta=-200),
            player(4),
            player(5),
        ]
        request = parse_payload(payload(players=players, future_field={"x": 1}))
        self.assertEqual([item.seat for item in request.live_opponents], [2, 4, 5])
        self.assertEqual(request.own_player.name, "you")

    def test_fallback_covers_every_legal_action_shape(self) -> None:
        cases = (
            ({"legal_actions": ["check"]}, {"action": "check"}),
            ({"legal_actions": ["fold", "call"]}, {"action": "fold"}),
            ({"legal_actions": ["call"]}, {"action": "call"}),
            (
                {"legal_actions": ["bet"], "min_raise_to": 17},
                {"action": "bet", "amount": 17},
            ),
            (
                {"legal_actions": ["raise"], "min_raise_to": 31},
                {"action": "raise", "amount": 31},
            ),
        )
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(safe_fallback(raw), expected)

    def test_validate_response_enforces_total_amount_bounds(self) -> None:
        request = parse_payload(payload(min_raise_to=50, max_raise_to=50))
        self.assertEqual(
            validate_response(request, {"action": "raise", "amount": 50}),
            {"action": "raise", "amount": 50},
        )
        with self.assertRaises(ProtocolError):
            validate_response(request, {"action": "raise", "amount": 49})


class RuleAndEquityTests(unittest.TestCase):
    def test_standard_pair_share_against_five_uniform_opponents(self) -> None:
        model = RuleModel("standard-test")
        # Lock the model to standard through its stable seed representation.
        seeded = model.to_dict()
        seeded["posterior"] = {
            name: float(name == "standard") for name in model.posterior()
        }
        model = RuleModel.from_dict(seeded)
        equity = showdown_share(7, 7, [None] * 5, model)
        self.assertAlmostEqual(equity, 0.8263128221287941, places=12)
        self.assertAlmostEqual(
            exact_share_for_hypothesis(7, 7, [None] * 5, "standard"),
            equity,
            places=12,
        )

    def test_candidate_rule_converges_from_unambiguous_showdowns(self) -> None:
        model = RuleModel("synthetic-near")
        hypothesis = get_hypothesis("community-near-high")
        for _ in range(3):
            for community in range(1, 14):
                for first, second in ((1, 5), (3, 12), (6, 13)):
                    rank_first = hypothesis.rank(first, community)
                    rank_second = hypothesis.rank(second, community)
                    if rank_first == rank_second:
                        continue
                    winner = 0 if rank_first > rank_second else 1
                    model.observe_showdown(
                        community, {0: first, 1: second}, [winner]
                    )
        top = max(model.posterior(), key=model.posterior().get)  # type: ignore[arg-type]
        self.assertEqual(top, "community-near-high")
        self.assertGreater(model.confidence(), 0.90)

    def test_out_of_grammar_evidence_populates_pairwise_fallback(self) -> None:
        model = RuleModel("scrambled")
        # A deliberately scrambled global order is not one of the formula rules.
        order = {7: 13, 2: 12, 12: 11, 1: 10, 9: 9, 4: 8}
        cards = list(order)
        for _ in range(3):
            for community in range(1, 14):
                for index, first in enumerate(cards):
                    for second in cards[index + 1 :]:
                        winner = 0 if order[first] > order[second] else 1
                        model.observe_showdown(
                            community, {0: first, 1: second}, [winner]
                        )
        self.assertTrue(model.pairwise_evidence())
        self.assertGreater(
            model.empirical_comparison_probability(7, 2, 5), 0.5
        )
        self.assertGreater(model.fallback_weight(5), 0.0)


class LearningTests(unittest.TestCase):
    def test_recent_hand_is_ingested_once(self) -> None:
        recent = {
            "hand_number": 1,
            "community_number": 8,
            "winners": [1],
            "pot": 12,
            "shown_numbers": {"0": 6, "1": 8},
            "actions": [
                {"round": "post_reveal", "seat": 1, "action": "bet", "amount": 4},
                {"round": "post_reveal", "seat": 0, "action": "call", "amount": 4},
            ],
        }
        request = parse_payload(
            payload(hand_number=2, recent_hands=[recent], match_id="dedupe")
        )
        store = RuntimeStore(knowledge=EventKnowledge())
        first = store.ingest(request)
        second = store.ingest(request)
        self.assertIs(first, second)
        self.assertEqual(first.processed_hands, {1})
        self.assertEqual(store.knowledge.get_rule("test-codename").observation_count, 1)

    def test_action_conditioned_range_is_normalized_and_directional(self) -> None:
        profile = OpponentProfile("Dana")
        for _ in range(30):
            profile.observe(
                "raise",
                round_name="pre_reveal",
                facing=True,
                size_bucket="unknown",
                live_count=6,
                position_bucket="late",
                strength=1.0,
            )
            profile.observe(
                "fold",
                round_name="pre_reveal",
                facing=True,
                size_bucket="unknown",
                live_count=6,
                position_bucket="late",
                strength=0.0,
            )
        action = ActionRecord(
            round="pre_reveal",
            seat=1,
            action="raise",
            amount=8,
            to_call=2,
            live_players=6,
            position="late",
        )
        result = profile.range_for_action_history(
            [action], seat=1, live_count=6, position_bucket="late"
        )
        self.assertAlmostEqual(sum(result), 1.0)
        self.assertGreater(result[12], result[0])

    def test_seed_round_trip(self) -> None:
        knowledge = EventKnowledge()
        knowledge.source_hashes.add("abc")
        knowledge.get_rule("opal").observe_showdown(4, {0: 4, 1: 13}, [0])
        knowledge.get_opponent("Rhea").observe("check", round_name="post_reveal")
        restored = EventKnowledge.from_dict(knowledge.to_dict())
        self.assertEqual(restored.source_hashes, {"abc"})
        self.assertEqual(restored.get_rule("opal").observation_count, 1)
        self.assertGreater(restored.get_opponent("Rhea").observations, 0)


class ReplayTests(unittest.TestCase):
    def test_replay_fit_and_duplicate_source_hash(self) -> None:
        document = {
            "table_rule": "opal",
            "your_seat": 0,
            "players": [
                {"seat": seat, "name": name} for seat, name in enumerate(NAMES)
            ],
            "hands": [
                {
                    "hand_number": 1,
                    "community_number": 6,
                    "winners": [2],
                    "pot": 30,
                    "shown_numbers": {"0": 9, "2": 6},
                    "full_numbers": {
                        "0": 9,
                        "1": 2,
                        "2": 6,
                        "3": 5,
                        "4": 11,
                        "5": 3,
                    },
                    "actions": [
                        {"round": "pre_reveal", "seat": 1, "action": "fold"},
                        {"round": "pre_reveal", "seat": 2, "action": "raise", "amount": 7},
                        {"round": "pre_reveal", "seat": 0, "action": "call", "amount": 7},
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            replay = Path(directory) / "match.json"
            seed = Path(directory) / "seed.json"
            replay.write_text(json.dumps(document), encoding="utf-8")
            knowledge, report = fit_replays([replay])
            self.assertEqual(report.hands_added, 1)
            self.assertEqual(knowledge.get_rule("opal").observation_count, 1)
            # Dana folded, but the raw replay exposes her completed-hand card.
            self.assertGreater(knowledge.get_opponent("Dana").observations, 0)
            write_seed(knowledge, seed)
            _again, duplicate = fit_replays([replay], seed_path=seed)
            self.assertEqual(duplicate.duplicate_files, 1)
            self.assertEqual(duplicate.hands_added, 0)


class ServiceAndSimulatorTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_runtime_for_tests()

    def test_standalone_http_contract(self) -> None:
        client = TestClient(app)
        self.assertEqual(client.get("/health").status_code, 200)
        response = client.post("/move", json=payload())
        self.assertEqual(response.status_code, 200)
        request = parse_payload(payload())
        self.assertEqual(validate_response(request, response.json()), response.json())

    def test_decision_is_deterministic_and_within_deadline(self) -> None:
        raw = payload()
        started = time.perf_counter()
        first = decide_move(raw)
        elapsed = time.perf_counter() - started
        second = decide_move(raw)
        self.assertEqual(first, second)
        self.assertLess(elapsed, 1.5)
        self.assertEqual(validate_response(parse_payload(raw), first), first)

    def test_exact_all_in_bounds_and_safe_fallback(self) -> None:
        raw = payload(
            legal_actions=["raise"],
            min_raise_to=200,
            max_raise_to=200,
        )
        result = decide_move(raw)
        self.assertEqual(result, {"action": "raise", "amount": 200})

        malformed = dict(raw)
        malformed["your_number"] = 99
        self.assertEqual(
            decide_move(malformed), {"action": "raise", "amount": 200}
        )

    def test_duplicate_history_does_not_double_train_service(self) -> None:
        recent = {
            "hand_number": 1,
            "community_number": 5,
            "winners": [1],
            "pot": 8,
            "shown_numbers": {"0": 3, "1": 5},
            "actions": [],
        }
        raw = payload(hand_number=2, match_id="service-dedupe", recent_hands=[recent])
        decide_move(raw)
        first = runtime_snapshot()["rules"]["test-codename"]["observations"]
        decide_move(raw)
        second = runtime_snapshot()["rules"]["test-codename"]["observations"]
        self.assertEqual((first, second), (1, 1))

    def test_simulator_is_deterministic_and_conserves_chips(self) -> None:
        config = SimulationConfig(total_hands=4)
        first = simulate_leg(77, built_in_strategy, config=config)
        second = simulate_leg(77, built_in_strategy, config=config)
        self.assertEqual(first, second)
        self.assertEqual(sum(first.final_stacks), 6 * config.starting_stack)
        self.assertEqual(first.invalid_actions, 0)
        for hand in first.hands:
            self.assertEqual(sum(hand.stacks_after), 6 * config.starting_stack)

    def test_live_policy_completes_simulated_hand_legally(self) -> None:
        result = simulate_leg(
            91,
            decide_move,
            config=SimulationConfig(total_hands=1),
        )
        self.assertEqual(result.invalid_actions, 0)
        self.assertEqual(sum(result.final_stacks), 1200)


if __name__ == "__main__":
    unittest.main()
