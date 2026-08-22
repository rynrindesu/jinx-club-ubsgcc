"""Independent tests for the clean-room SHOWDOWN Phase 3 service."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.phase3.showdown.api import app
from app.phase3.showdown.engine import (
    decide_move,
    reset_runtime_for_tests,
    runtime_snapshot,
)
from app.phase3.showdown.equity import (
    exact_share_for_hypothesis,
    showdown_metrics_by_subset,
    showdown_share,
)
from app.phase3.showdown.learning import EventKnowledge, OpponentProfile, RuntimeStore
from app.phase3.showdown.policy import HighVariancePolicy
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
    ScriptedArchetype,
    SimulationConfig,
    benchmark,
    built_in_strategy,
    make_phase3_policy_strategy,
    simulate_leg,
)
from app.showdown import decide_move as dispatch_move


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


def locked_rule_knowledge(codename: str, hypothesis: str) -> EventKnowledge:
    """Return mature seed knowledge concentrated on one synthetic truth."""

    model = RuleModel(codename)
    seeded = model.to_dict()
    seeded["posterior"] = {
        name: float(name == hypothesis) for name in model.posterior()
    }
    seeded["observations"] = 100
    seeded["fit_sum"] = 98.5
    knowledge = EventKnowledge()
    knowledge.rules[codename] = RuleModel.from_dict(seeded)
    return knowledge


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

        opponents = {seat: None for seat in range(1, 6)}
        metrics = showdown_metrics_by_subset(7, 7, opponents, model)[
            frozenset(opponents)
        ]
        self.assertAlmostEqual(metrics.expected_share, equity, places=12)
        self.assertAlmostEqual(
            metrics.sole_win_probability, (12.0 / 13.0) ** 5, places=12
        )
        self.assertAlmostEqual(metrics.loss_probability, 0.0, places=12)
        self.assertAlmostEqual(
            sum(metrics.split_probabilities), 1.0, places=12
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

    def test_reused_match_identifier_resets_on_hand_one(self) -> None:
        store = RuntimeStore(knowledge=EventKnowledge())
        old = store.ingest(parse_payload(payload(match_id="reused", hand_number=20)))
        fresh = store.ingest(parse_payload(payload(match_id="reused", hand_number=1)))
        self.assertIsNot(old, fresh)
        self.assertEqual(fresh.last_hand_number, 1)
        self.assertFalse(fresh.processed_hands)

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

    def test_identical_number_split_is_safe_rule_evidence(self) -> None:
        knowledge = EventKnowledge()
        applied = knowledge.observe_hand(
            "split-rule",
            {
                "hand_number": 4,
                "community_number": 9,
                "winners": [1, 2],
                "pot": 20,
                "shown_numbers": {"0": 8, "1": 9, "2": 9},
                "actions": [],
            },
        )
        self.assertTrue(applied)
        self.assertEqual(knowledge.get_rule("split-rule").observation_count, 1)


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

    def test_nested_camelcase_replay_inherits_hero_and_mapped_reveals(self) -> None:
        document = {
            "yourSeat": 0,
            "players": [
                {"seat": seat, "name": "Contestant" if seat == 0 else NAMES[seat]}
                for seat in range(6)
            ],
            "legs": [
                {
                    "tableRule": "amber",
                    "handHistory": [
                        {
                            "handNumber": 1,
                            "communityNumber": 4,
                            "winners": [1],
                            "pot": 14,
                            "players": {
                                "0": {"number": 9, "shown": True},
                                "1": {"number": 4, "shown": True},
                            },
                            "actions": [
                                {"street": "pre_reveal", "seat": 0, "type": "raise", "amount": 6},
                                {"street": "pre_reveal", "seat": 1, "type": "call", "amount": 6},
                            ],
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            replay = Path(directory) / "nested.json"
            replay.write_text(json.dumps(document), encoding="utf-8")
            knowledge, report = fit_replays([replay])
        self.assertEqual(report.hands_added, 1)
        self.assertEqual(knowledge.get_rule("amber").observation_count, 1)
        self.assertNotIn("Contestant", knowledge.opponents)
        self.assertGreater(knowledge.get_opponent("Dana").observations, 0)

    def test_boolean_cards_are_not_coerced_to_number_one(self) -> None:
        document = {
            "table_rule": "boolean",
            "players": [{"seat": 0, "name": "you"}, {"seat": 1, "name": "Dana"}],
            "hands": [
                {
                    "hand_number": 1,
                    "community_number": 5,
                    "winners": [1],
                    "pot": 8,
                    "shown_numbers": {"0": True, "1": 5},
                    "actions": [],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            replay = Path(directory) / "booleans.json"
            replay.write_text(json.dumps(document), encoding="utf-8")
            knowledge, _report = fit_replays([replay])
        self.assertEqual(knowledge.get_rule("boolean").observation_count, 0)

    def test_replay_channels_can_recover_rule_after_profile_only_import(self) -> None:
        base = {
            "match_id": "recoverable",
            "table_rule": "amber",
            "your_seat": 0,
            "players": [
                {"seat": seat, "name": name} for seat, name in enumerate(NAMES)
            ],
            "hands": [
                {
                    "hand_number": 1,
                    "community_number": 4,
                    "winners": [{"unknown_winner_field": 1}],
                    "pot": 12,
                    "shown_numbers": {"0": 9, "1": 4},
                    "full_numbers": {"0": 9, "1": 4},
                    "actions": [
                        {
                            "round": "pre_reveal",
                            "seat": 1,
                            "action": "call",
                            "amount": 2,
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            corrected = Path(directory) / "corrected.json"
            seed = Path(directory) / "seed.json"
            first.write_text(json.dumps(base), encoding="utf-8")
            knowledge, initial = fit_replays([first])
            self.assertEqual(initial.hands_added, 1)
            self.assertEqual(knowledge.get_rule("amber").observation_count, 0)
            write_seed(knowledge, seed)

            base["hands"][0]["winners"] = [{"player_id": 1}]
            corrected.write_text(json.dumps(base), encoding="utf-8")
            recovered, report = fit_replays([corrected], seed_path=seed)
        self.assertEqual(report.hands_added, 1)
        self.assertEqual(recovered.get_rule("amber").observation_count, 1)

    def test_replay_rejects_boolean_action_seat_and_identity_conflicts(self) -> None:
        first = {
            "match_id": "same-id",
            "table_rule": "opal",
            "players": [{"seat": 0, "name": "you"}, {"seat": 1, "name": "Dana"}],
            "hands": [
                {
                    "hand_number": 1,
                    "community_number": 5,
                    "winners": [{"player": 1}],
                    "pot": 8,
                    "shown_numbers": {"0": 2, "1": 5},
                    "actions": [
                        {"round": "pre_reveal", "seat": True, "action": "call", "amount": 2}
                    ],
                }
            ],
        }
        second = json.loads(json.dumps(first))
        second["hands"][0]["shown_numbers"]["0"] = 3
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.json"
            second_path = Path(directory) / "second.json"
            first_path.write_text(json.dumps(first), encoding="utf-8")
            second_path.write_text(json.dumps(second), encoding="utf-8")
            learned, _report = fit_replays([first_path])
            self.assertEqual(learned.get_rule("opal").observation_count, 1)
            self.assertNotIn("Dana", learned.opponents)
            with self.assertRaises(ValueError):
                fit_replays([first_path, second_path])


class SimulatorFoundationTests(unittest.TestCase):
    @staticmethod
    def passive(request: dict[str, object]) -> dict[str, str]:
        legal = request["legal_actions"]
        assert isinstance(legal, list)
        if "check" in legal:
            return {"action": "check"}
        if "call" in legal:
            return {"action": "call"}
        return {"action": "fold"}

    def test_nonstandard_codename_requires_explicit_rule_truth(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires rule_hypothesis or ranker"):
            SimulationConfig(table_rule="obsidian")

    def test_rule_hypothesis_and_custom_ranker_control_settlement(self) -> None:
        opponents = {seat: self.passive for seat in range(1, 6)}
        standard = simulate_leg(
            0,
            self.passive,
            opponents,
            SimulationConfig(total_hands=1),
        )
        low_truth = simulate_leg(
            0,
            self.passive,
            opponents,
            SimulationConfig(
                total_hands=1,
                table_rule="obsidian",
                rule_hypothesis="pair-last-raw-low",
            ),
        )
        custom_low = simulate_leg(
            0,
            self.passive,
            opponents,
            SimulationConfig(
                total_hands=1,
                table_rule="synthetic-low",
                ranker=lambda number, _community: (-number,),
            ),
        )

        # Seed zero deals [7, 13, 7, 1, 5, 9] with community 8.
        self.assertEqual(standard.hands[0].winners, (1,))
        self.assertEqual(low_truth.hands[0].winners, (3,))
        self.assertEqual(custom_low.hands[0].winners, (3,))
        for result in (standard, low_truth, custom_low):
            self.assertEqual(sum(result.final_stacks), 1200)

    def test_scripted_archetype_uses_configured_rule_strength(self) -> None:
        raw = payload(
            round="post_reveal",
            your_number=13,
            community_number=13,
            pot=12,
            to_call=0,
            legal_actions=["check", "bet"],
            min_raise_to=2,
            max_raise_to=200,
            current_hand_actions=[],
        )
        standard = ScriptedArchetype(
            tightness=0.0,
            aggression=1.0,
            bluff_frequency=0.0,
        )
        pair_loses = ScriptedArchetype(
            tightness=0.0,
            aggression=1.0,
            bluff_frequency=0.0,
            ranker=get_hypothesis("pair-last-raw-low").rank,
        )
        self.assertEqual(standard(raw).action, "bet")
        self.assertEqual(pair_loses(raw).action, "check")

    def test_isolated_seeded_policy_folds_observed_obsidian_traps(self) -> None:
        knowledge = locked_rule_knowledge("obsidian", "pair-last-raw-low")
        before = knowledge.to_dict()
        strategy = make_phase3_policy_strategy(knowledge=knowledge)

        pre_players = [
            player(0),
            player(1),
            player(2, stack=199, bet=1),
            player(3, stack=198, bet=2),
            player(4, folded=True),
            player(5, stack=198, bet=2),
        ]
        worst_pre_reveal = payload(
            match_id="obsidian-worst-open",
            table_rule="obsidian",
            hand_number=2,
            your_number=13,
            button_seat=1,
            pot=5,
            to_call=2,
            min_raise_to=4,
            max_raise_to=200,
            players=pre_players,
            current_hand_actions=[
                {"round": "pre_reveal", "seat": 4, "action": "fold"},
                {"round": "pre_reveal", "seat": 5, "action": "call"},
            ],
        )
        self.assertEqual(strategy(worst_pre_reveal), {"action": "fold"})

        pair_players = [
            player(0, stack=183, delta=-17),
            player(1, stack=17, bet=183),
            player(2, folded=True),
            player(3, folded=True),
            player(4, folded=True),
            player(5, folded=True),
        ]
        losing_pair_all_in = payload(
            match_id="obsidian-pair-all-in",
            table_rule="obsidian",
            hand_number=20,
            round="post_reveal",
            your_number=11,
            community_number=11,
            your_stack=183,
            pot=183,
            to_call=183,
            min_raise_to=None,
            max_raise_to=None,
            legal_actions=["fold", "call"],
            players=pair_players,
            current_hand_actions=[
                {
                    "round": "post_reveal",
                    "seat": 1,
                    "action": "bet",
                    "amount": 183,
                }
            ],
        )
        self.assertEqual(strategy(losing_pair_all_in), {"action": "fold"})
        self.assertEqual(knowledge.to_dict(), before)

    def test_positive_second_place_is_not_mistaken_for_a_safe_lead(self) -> None:
        strategy = make_phase3_policy_strategy(
            knowledge=locked_rule_knowledge("cinnabar", "standard")
        )
        standings = [
            player(0, stack=386, delta=186),
            player(1, stack=420, delta=220),
            player(2, stack=130, delta=-70),
            player(3, stack=110, delta=-90),
            player(4, stack=90, delta=-110),
            player(5, stack=64, delta=-136),
        ]
        raw = payload(
            match_id="positive-second",
            table_rule="cinnabar",
            hand_number=60,
            round="post_reveal",
            your_number=13,
            community_number=13,
            your_stack=386,
            pot=3,
            to_call=0,
            legal_actions=["check", "bet"],
            min_raise_to=2,
            max_raise_to=386,
            players=standings,
        )
        self.assertEqual(strategy(raw)["action"], "bet")

    def test_benchmark_constructs_fresh_strategy_for_each_trial(self) -> None:
        calls = 0

        def factory():
            nonlocal calls
            calls += 1
            return self.passive

        opponents = {seat: self.passive for seat in range(1, 6)}
        report = benchmark(
            strategy_factory=factory,
            trials=3,
            opponent_strategies=opponents,
            config=SimulationConfig(total_hands=1),
        )
        self.assertEqual((calls, report.trials), (3, 3))
        with self.assertRaisesRegex(ValueError, "either strategy or strategy_factory"):
            benchmark(
                self.passive,
                strategy_factory=factory,
                trials=1,
                opponent_strategies=opponents,
                config=SimulationConfig(total_hands=1),
            )


class ServiceAndSimulatorTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_runtime_for_tests()

    def test_standalone_http_contract(self) -> None:
        client = TestClient(app)
        self.assertEqual(client.get("/health").status_code, 200)
        runtime = client.get("/showdown/runtime")
        self.assertEqual(runtime.status_code, 200)
        self.assertEqual(
            runtime.json()["phase3_engine"],
            "app.phase3.showdown.engine",
        )
        response = client.post("/move", json=payload())
        self.assertEqual(response.status_code, 200)
        request = parse_payload(payload())
        self.assertEqual(validate_response(request, response.json()), response.json())

    def test_shared_gateway_dispatches_phase_three_to_clean_room_engine(self) -> None:
        raw = payload()
        expected = {"action": "fold"}
        with patch("app.showdown.decide_phase3_move", return_value=expected) as routed:
            self.assertEqual(dispatch_move(raw), expected)
        routed.assert_called_once_with(raw)

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

    def test_final_hand_uses_exact_leaderboard_target(self) -> None:
        leading_players = [
            player(0, stack=210, delta=10),
            player(1, stack=204, delta=5, bet=1),
            player(2, stack=197, delta=-1, bet=2),
            player(3, stack=198, delta=-2),
            player(4, stack=197, delta=-3),
            player(5, stack=194, delta=-9),
        ]
        protect = payload(
            players=leading_players,
            your_stack=210,
            hand_number=60,
            round="post_reveal",
            community_number=7,
            your_number=1,
            pot=3,
            to_call=0,
            legal_actions=["check", "bet"],
            min_raise_to=2,
            max_raise_to=210,
        )
        self.assertEqual(decide_move(protect), {"action": "check"})

        trailing_players = [
            player(0, stack=200, delta=0),
            player(1, stack=219, delta=20, bet=1),
            player(2, stack=198, delta=-1, bet=2),
            player(3, stack=195, delta=-5),
            player(4, stack=195, delta=-5),
            player(5, stack=193, delta=-9),
        ]
        attack = payload(
            players=trailing_players,
            your_stack=200,
            hand_number=60,
            round="post_reveal",
            community_number=13,
            your_number=6,
            pot=3,
            to_call=0,
            legal_actions=["check", "bet"],
            min_raise_to=2,
            max_raise_to=200,
        )
        self.assertEqual(decide_move(attack)["action"], "bet")

    def test_penultimate_hand_protects_a_safe_unique_lead(self) -> None:
        players = [
            player(0, stack=220, delta=20),
            player(1, stack=205, delta=5),
            player(2, stack=198, delta=-2),
            player(3, stack=195, delta=-5),
            player(4, stack=192, delta=-8),
            player(5, stack=190, delta=-10),
        ]
        raw = payload(
            players=players,
            your_stack=220,
            hand_number=59,
            round="post_reveal",
            community_number=7,
            your_number=13,
            pot=0,
            to_call=0,
            legal_actions=["check", "bet"],
            min_raise_to=2,
            max_raise_to=220,
        )
        self.assertEqual(decide_move(raw), {"action": "check"})

    def test_scout_budget_counts_calls_in_current_hand(self) -> None:
        players = [
            player(0, stack=185, bet=15, delta=0),
            player(1, stack=175, bet=25, delta=0),
            player(2, stack=198, bet=2),
            player(3),
            player(4),
            player(5),
        ]
        raw = payload(
            players=players,
            your_stack=185,
            your_number=1,
            pot=42,
            to_call=10,
            current_hand_actions=[
                {"round": "pre_reveal", "seat": 0, "action": "call", "amount": 15},
                {"round": "pre_reveal", "seat": 1, "action": "raise", "amount": 25},
            ],
        )
        self.assertEqual(decide_move(raw)["action"], "fold")

    def test_sidepot_components_cap_hero_eligibility_and_refund_excess(self) -> None:
        request = parse_payload(payload())
        guaranteed, contestable = HighVariancePolicy._hero_pot_components(
            request,
            {1},
            {0: 100, 1: 50, 2: 100},
            250,
        )
        self.assertEqual((guaranteed, contestable), (100.0, 150.0))

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
