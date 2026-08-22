import copy
import unittest

from app.phase2.showdown import (
    Phase2Engine,
    Phase2State,
    RuleKnowledge,
    ShowdownObservation,
    build_candidate_rules,
)
from app.showdown import decide_move as dispatch_move


def phase2_request(
    number=7,
    *,
    table_rule="amber-test",
    match_id="attempt-a-leg-1",
    leg_number=1,
    hand_number=1,
    delta=0,
):
    starting_stack = 200
    return {
        "protocol_version": 2,
        "match_id": match_id,
        "phase": 2,
        "table_rule": table_rule,
        "leg_number": leg_number,
        "total_legs": 4,
        "small_blind": 1,
        "big_blind": 2,
        "starting_stack": starting_stack,
        "your_stack": starting_stack + delta - 1,
        "hand_number": hand_number,
        "total_hands": 40,
        "round": "pre_reveal",
        "your_number": number,
        "community_number": None,
        "your_seat": 0,
        "button_seat": 0,
        "pot": 3,
        "to_call": 1,
        "min_raise_to": 4,
        "max_raise_to": starting_stack + delta,
        "legal_actions": ["fold", "call", "raise"],
        "players": [
            {
                "seat": 0,
                "name": "you",
                "folded": False,
                "chip_delta": delta,
                "bet_this_round": 1,
                "stack": starting_stack + delta - 1,
                "all_in": False,
                "busted": False,
            },
            {
                "seat": 1,
                "name": "Ada",
                "folded": False,
                "chip_delta": -delta,
                "bet_this_round": 2,
                "stack": starting_stack - delta - 2,
                "all_in": False,
                "busted": False,
            },
        ],
        "current_hand_actions": [],
        "recent_hands": [],
    }


def post_bet_request(number, community, **kwargs):
    request = phase2_request(number, **kwargs)
    request.update(
        round="post_reveal",
        community_number=community,
        your_stack=request["starting_stack"]
        + request["players"][0]["chip_delta"]
        - 2,
        pot=6,
        to_call=2,
        min_raise_to=4,
        max_raise_to=request["starting_stack"]
        + request["players"][0]["chip_delta"],
        legal_actions=["fold", "call", "raise"],
        current_hand_actions=[
            {"round": "pre_reveal", "seat": 0, "action": "call", "amount": 2},
            {"round": "pre_reveal", "seat": 1, "action": "check"},
            {"round": "post_reveal", "seat": 1, "action": "bet", "amount": 2},
        ],
    )
    request["players"][0].update(stack=request["your_stack"], bet_this_round=0)
    request["players"][1].update(bet_this_round=2)
    return request


def completed_hand(
    hand_number,
    hero_number,
    opponent_number,
    community,
    outcome,
    *,
    actions=None,
):
    winners = [0] if outcome > 0 else [1] if outcome < 0 else [0, 1]
    return {
        "hand_number": hand_number,
        "community_number": community,
        "winners": winners,
        "pot": 4,
        "shown_numbers": {"0": hero_number, "1": opponent_number},
        "actions": actions or [],
    }


def teach_candidate(state, codename, candidate_name):
    candidate = next(
        candidate
        for candidate in build_candidate_rules()
        if candidate.name == candidate_name
    )
    knowledge = state.knowledge(codename)
    hand = 0
    for community in range(1, 14):
        for first in range(1, 14):
            for second in range(first + 1, 14):
                hand += 1
                knowledge.ingest(
                    ShowdownObservation(
                        key=("training", hand, "0", "1"),
                        community=community,
                        first_number=first,
                        second_number=second,
                        outcome=candidate.compare(first, second, community),
                    )
                )
    return knowledge


class RuleInferenceTests(unittest.TestCase):
    def test_explicitly_fake_odd_first_high_rule_is_not_a_candidate(self):
        candidates = build_candidate_rules()

        def fake(first, second, _community):
            first_key = (first % 2, first)
            second_key = (second % 2, second)
            return (first_key > second_key) - (first_key < second_key)

        for candidate in candidates:
            identical = all(
                candidate.compare(first, second, community)
                == fake(first, second, community)
                for community in range(1, 14)
                for first in range(1, 14)
                for second in range(first + 1, 14)
            )
            self.assertFalse(identical, candidate.name)

    def test_observation_ingestion_is_repeat_safe_and_skips_equal_numbers(self):
        state = Phase2State()
        request = phase2_request()
        request["recent_hands"] = [completed_hand(1, 4, 9, 6, -1)]

        knowledge, _ = state.observe_payload(request)
        state.observe_payload(copy.deepcopy(request))
        self.assertEqual(knowledge.observation_count, 1)

        equal = copy.deepcopy(request)
        equal["recent_hands"] = [completed_hand(2, 7, 7, 3, 0)]
        state.observe_payload(equal)
        self.assertEqual(knowledge.observation_count, 1)

    def test_same_number_has_opposite_strength_under_learned_rules(self):
        state = Phase2State()
        high = teach_candidate(state, "high-rule", "pair_then_higher")
        low = teach_candidate(state, "low-rule", "lower")

        high_estimate = high.estimate(2, 10)
        low_estimate = low.estimate(2, 10)

        self.assertEqual(high_estimate.confidence, "learned")
        self.assertEqual(low_estimate.confidence, "learned")
        self.assertLess(high_estimate.mean, 0.20)
        self.assertGreater(low_estimate.mean, 0.80)

    def test_pairwise_fallback_uses_transitive_results(self):
        knowledge = RuleKnowledge(())
        knowledge.ingest(
            ShowdownObservation(("m", 1, "0", "1"), 5, 3, 2, 1)
        )
        knowledge.ingest(
            ShowdownObservation(("m", 2, "0", "1"), 5, 2, 1, 1)
        )

        estimate = knowledge.estimate(3, 5)

        self.assertAlmostEqual(estimate.lower, 2.5 / 13)
        self.assertAlmostEqual(estimate.coverage, 3 / 13)

    def test_fallback_never_projects_one_community_onto_another(self):
        knowledge = RuleKnowledge(())
        knowledge.ingest(
            ShowdownObservation(("m", 1, "0", "1"), 1, 13, 1, 1)
        )

        estimate = knowledge.estimate(13, 7)

        self.assertAlmostEqual(estimate.lower, 0.5 / 13)
        self.assertAlmostEqual(estimate.coverage, 1 / 13)

    def test_tie_that_contradicts_a_win_path_forces_unknown_fallback(self):
        knowledge = RuleKnowledge(())
        knowledge.ingest(
            ShowdownObservation(("m", 1, "0", "1"), 5, 3, 2, 1)
        )
        knowledge.ingest(
            ShowdownObservation(("m", 2, "0", "1"), 5, 2, 1, 1)
        )
        knowledge.ingest(
            ShowdownObservation(("m", 3, "0", "1"), 5, 3, 1, 0)
        )

        estimate = knowledge.estimate(3, 5)

        self.assertAlmostEqual(estimate.lower, 0.5 / 13)


class Phase2PolicyTests(unittest.TestCase):
    def test_unknown_rule_limps_and_never_assumes_a_pair_is_strong(self):
        engine = Phase2Engine()
        self.assertEqual(engine.decide(phase2_request(13)), {"action": "call"})

        pair = post_bet_request(5, 5)
        pair.update(
            your_stack=100,
            pot=204,
            to_call=100,
            min_raise_to=None,
            max_raise_to=None,
            legal_actions=["fold", "call"],
        )
        pair["players"][0].update(stack=100, bet_this_round=0)
        self.assertEqual(engine.decide(pair), {"action": "fold"})

    def test_learned_codenames_produce_different_moves_for_same_numbers(self):
        state = Phase2State()
        teach_candidate(state, "high-rule", "pair_then_higher")
        teach_candidate(state, "low-rule", "lower")
        engine = Phase2Engine(state)

        high_move = engine.decide(
            post_bet_request(2, 10, table_rule="high-rule")
        )
        low_move = engine.decide(
            post_bet_request(2, 10, table_rule="low-rule")
        )

        self.assertEqual(high_move, {"action": "fold"})
        self.assertEqual(low_move, {"action": "call"})

    def test_learned_top_hand_can_call_for_its_entire_stack(self):
        state = Phase2State()
        teach_candidate(state, "low-rule", "lower")
        engine = Phase2Engine(state)
        request = phase2_request(1, table_rule="low-rule")
        request.update(
            your_stack=20,
            pot=220,
            to_call=20,
            min_raise_to=None,
            max_raise_to=None,
            legal_actions=["fold", "call"],
            button_seat=1,
            current_hand_actions=[
                {
                    "round": "pre_reveal",
                    "seat": 1,
                    "action": "raise",
                    "amount": 22,
                }
            ],
        )
        request["players"][0].update(stack=20, bet_this_round=2)

        self.assertEqual(engine.decide(request), {"action": "call"})

    def test_desperate_learned_nuts_can_take_only_legal_all_in_raise(self):
        state = Phase2State()
        teach_candidate(state, "low-rule", "lower")
        engine = Phase2Engine(state)
        request = post_bet_request(
            1,
            10,
            table_rule="low-rule",
            hand_number=40,
            delta=-180,
        )
        request.update(
            your_stack=20,
            pot=365,
            to_call=5,
            min_raise_to=20,
            max_raise_to=20,
            legal_actions=["fold", "call", "raise"],
        )
        request["players"][0].update(stack=20, bet_this_round=0)

        self.assertEqual(
            engine.decide(request), {"action": "raise", "amount": 20}
        )

    def test_mathematically_secured_leg_coasts(self):
        request = phase2_request(13, hand_number=40, delta=26)

        self.assertEqual(Phase2Engine().decide(request), {"action": "fold"})

    def test_policy_matrix_always_emits_a_legal_well_shaped_action(self):
        engine = Phase2Engine()
        legal_shapes = [
            (["check"], 0, None, None),
            (["check", "bet"], 0, 2, 190),
            (["check", "raise"], 0, 4, 190),
            (["fold", "call"], 5, None, None),
            (["fold", "call", "raise"], 5, 10, 190),
            (["call"], 5, None, None),
        ]
        for number in range(1, 14):
            for community in (None, 1, 7, 13):
                for legal, to_call, minimum, maximum in legal_shapes:
                    request = phase2_request(
                        number,
                        table_rule=f"matrix-{number}-{community}",
                    )
                    request.update(
                        round="pre_reveal" if community is None else "post_reveal",
                        community_number=community,
                        legal_actions=legal,
                        to_call=to_call,
                        min_raise_to=minimum,
                        max_raise_to=maximum,
                    )

                    move = engine.decide(request)

                    self.assertIn(move["action"], legal)
                    if move["action"] in {"bet", "raise"}:
                        self.assertIs(type(move.get("amount")), int)
                        self.assertGreaterEqual(move["amount"], minimum)
                        self.assertLessEqual(move["amount"], maximum)
                    else:
                        self.assertNotIn("amount", move)


class StateScopeTests(unittest.TestCase):
    def test_rule_knowledge_survives_attempt_reset(self):
        state = Phase2State()
        first = phase2_request(table_rule="persistent-rule")
        first["recent_hands"] = [completed_hand(1, 9, 3, 7, 1)]
        knowledge, _ = state.observe_payload(first)

        retry = phase2_request(
            table_rule="another-rule", match_id="attempt-b-leg-1"
        )
        state.observe_payload(retry)

        self.assertEqual(knowledge.observation_count, 1)
        self.assertIs(state.knowledge("persistent-rule"), knowledge)

    def test_opponent_profile_carries_across_legs_but_not_attempts(self):
        state = Phase2State()
        response_actions = [
            {"round": "pre_reveal", "seat": 0, "action": "raise", "amount": 4},
            {"round": "pre_reveal", "seat": 1, "action": "fold"},
        ]
        first = phase2_request()
        first["recent_hands"] = [
            completed_hand(1, 8, 3, 7, 1, actions=response_actions)
        ]
        _, first_profile = state.observe_payload(first)

        second_leg = phase2_request(
            table_rule="blue-test",
            match_id="attempt-a-leg-2",
            leg_number=2,
        )
        _, second_profile = state.observe_payload(second_leg)
        self.assertEqual(first_profile.open_responses, 1)
        self.assertEqual(second_profile.open_responses, 1)

        retry = phase2_request(match_id="attempt-b-leg-1")
        _, retry_profile = state.observe_payload(retry)
        self.assertEqual(retry_profile.open_responses, 0)

    def test_three_bet_response_is_not_counted_as_fold_to_open(self):
        state = Phase2State()
        actions = [
            {"round": "pre_reveal", "seat": 1, "action": "raise", "amount": 4},
            {"round": "pre_reveal", "seat": 0, "action": "raise", "amount": 10},
            {"round": "pre_reveal", "seat": 1, "action": "fold"},
        ]
        request = phase2_request()
        request["recent_hands"] = [
            completed_hand(1, 8, 3, 7, 1, actions=actions)
        ]

        _, profile = state.observe_payload(request)

        self.assertEqual(profile.open_responses, 0)

    def test_completed_hand_ingestion_captures_a_terminal_observation(self):
        state = Phase2State()
        added = state.ingest_completed_hands(
            table_rule="terminal-rule",
            match_id="finished-leg",
            your_seat=0,
            hands=[completed_hand(40, 11, 4, 8, 1)],
        )

        self.assertEqual(added, 1)
        self.assertEqual(state.knowledge("terminal-rule").observation_count, 1)


class DispatcherTests(unittest.TestCase):
    def test_phase_two_uses_new_policy_and_phase_one_keeps_legacy_policy(self):
        request = phase2_request(7, table_rule="dispatch-rule")
        phase_two = dispatch_move(request)
        request["phase"] = 1
        phase_one = dispatch_move(request)

        self.assertEqual(phase_two, {"action": "call"})
        self.assertEqual(phase_one, {"action": "raise", "amount": 4})


if __name__ == "__main__":
    unittest.main()
