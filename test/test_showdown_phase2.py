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


def teach_partial_high_family(state, codename):
    knowledge = state.knowledge(codename)
    candidates = build_candidate_rules()
    knowledge.active_candidates = {
        index
        for index, candidate in enumerate(candidates)
        if candidate.name in {"higher", "pair_then_higher"}
    }
    knowledge.ingest(
        ShowdownObservation(
            key=(f"partial-{codename}", 1, "0", "1"),
            community=5,
            first_number=9,
            second_number=7,
            outcome=1,
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

    def test_equity_accepts_an_opponent_number_distribution(self):
        state = Phase2State()
        knowledge = teach_candidate(state, "range-rule", "higher")

        uniform = knowledge.estimate(7, 5)
        weak_only = knowledge.estimate(7, 5, opponent_range=[1] + [0] * 12)
        strong_only = knowledge.estimate(7, 5, opponent_range=[0] * 12 + [1])

        self.assertAlmostEqual(uniform.mean, 0.5)
        self.assertEqual(weak_only.mean, 1.0)
        self.assertEqual(strong_only.mean, 0.0)

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

    def test_folded_hand_never_creates_showdown_rule_evidence(self):
        state = Phase2State()
        request = phase2_request(table_rule="fold-rule")
        request["recent_hands"] = [
            completed_hand(
                1,
                13,
                1,
                7,
                1,
                actions=[
                    {
                        "round": "post_reveal",
                        "seat": 1,
                        "action": "fold",
                    }
                ],
            )
        ]

        knowledge, _ = state.observe_payload(request)

        self.assertEqual(knowledge.observation_count, 0)

    def test_distinct_number_tie_is_recorded_as_rule_evidence(self):
        state = Phase2State()
        request = phase2_request(table_rule="tie-rule")
        request["recent_hands"] = [completed_hand(1, 4, 9, 7, 0)]

        knowledge, _ = state.observe_payload(request)

        self.assertEqual(knowledge.observation_count, 1)
        self.assertEqual(knowledge.direct_results[(7, 4, 9)], 0)
        self.assertIn(7, knowledge.tied_communities)

    def test_contradiction_downgrades_a_learned_rule(self):
        state = Phase2State()
        knowledge = teach_candidate(state, "contradiction-rule", "lower")
        self.assertEqual(knowledge.estimate(3, 5).confidence, "learned")

        knowledge.ingest(
            ShowdownObservation(
                key=("contradiction", 1, "0", "1"),
                community=5,
                first_number=3,
                second_number=2,
                outcome=1,
            )
        )

        self.assertEqual(knowledge.estimate(3, 5).confidence, "partial")
        self.assertEqual(knowledge.active_candidates, set())


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

    def test_unknown_rule_keeps_buying_three_chip_showdowns_when_down(self):
        request = phase2_request(
            7,
            table_rule="drawdown-rule",
            hand_number=12,
            delta=-10,
        )
        request.update(
            your_stack=188,
            pot=7,
            to_call=3,
            min_raise_to=8,
            max_raise_to=188,
            legal_actions=["fold", "call", "raise"],
            current_hand_actions=[
                {
                    "round": "pre_reveal",
                    "seat": 0,
                    "action": "call",
                    "amount": 2,
                },
                {
                    "round": "pre_reveal",
                    "seat": 1,
                    "action": "raise",
                    "amount": 5,
                },
            ],
        )
        request["players"][0].update(stack=188, bet_this_round=2)
        request["players"][1].update(bet_this_round=5)

        self.assertEqual(Phase2Engine().decide(request), {"action": "call"})

    def test_unknown_rule_occasionally_probes_a_four_chip_raise(self):
        request = phase2_request(7, match_id="probe-16")
        request.update(
            your_stack=198,
            pot=8,
            to_call=4,
            min_raise_to=12,
            max_raise_to=198,
            legal_actions=["fold", "call", "raise"],
            current_hand_actions=[
                {
                    "round": "pre_reveal",
                    "seat": 0,
                    "action": "call",
                    "amount": 2,
                },
                {
                    "round": "pre_reveal",
                    "seat": 1,
                    "action": "raise",
                    "amount": 6,
                },
            ],
        )
        request["players"][0].update(stack=198, bet_this_round=2)
        request["players"][1].update(bet_this_round=6)

        self.assertEqual(Phase2Engine().decide(request), {"action": "call"})

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

    def test_learned_rule_opens_a_profitable_button_range(self):
        state = Phase2State()
        teach_candidate(state, "open-rule", "higher")

        move = Phase2Engine(state).decide(
            phase2_request(6, table_rule="open-rule")
        )

        self.assertEqual(move, {"action": "raise", "amount": 4})

    def test_calling_station_receives_more_post_reveal_value_bets(self):
        state = Phase2State()
        teach_candidate(state, "station-rule", "lower")
        responses = []
        for hand_number in range(1, 5):
            responses.append(
                completed_hand(
                    hand_number,
                    7,
                    7,
                    10,
                    0,
                    actions=[
                        {
                            "round": "pre_reveal",
                            "seat": 0,
                            "action": "call",
                            "amount": 2,
                        },
                        {
                            "round": "pre_reveal",
                            "seat": 1,
                            "action": "check",
                        },
                        {
                            "round": "post_reveal",
                            "seat": 0,
                            "action": "bet",
                            "amount": 2,
                        },
                        {
                            "round": "post_reveal",
                            "seat": 1,
                            "action": "call",
                            "amount": 2,
                        },
                    ],
                )
            )
        request = phase2_request(
            6,
            table_rule="station-rule",
            hand_number=5,
        )
        request.update(
            round="post_reveal",
            community_number=10,
            pot=6,
            to_call=0,
            min_raise_to=2,
            max_raise_to=199,
            legal_actions=["check", "bet"],
            current_hand_actions=[
                {"round": "post_reveal", "seat": 1, "action": "check"}
            ],
            recent_hands=responses,
        )
        request["players"][0].update(bet_this_round=0)
        request["players"][1].update(bet_this_round=0)

        self.assertEqual(
            Phase2Engine(state).decide(request),
            {"action": "bet", "amount": 4},
        )

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

    def test_learned_second_best_hand_cannot_call_off_entire_stack(self):
        state = Phase2State()
        teach_candidate(state, "low-rule", "lower")
        engine = Phase2Engine(state)
        request = phase2_request(2, table_rule="low-rule")
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

        self.assertEqual(engine.decide(request), {"action": "fold"})

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

    def test_partial_rule_caps_total_hand_exposure_at_eight_chips(self):
        state = Phase2State()
        knowledge = teach_partial_high_family(state, "partial-risk-rule")
        self.assertEqual(knowledge.estimate(13, None).confidence, "partial")
        engine = Phase2Engine(state)

        reraised = phase2_request(
            13,
            table_rule="partial-risk-rule",
            hand_number=11,
        )
        reraised.update(
            your_stack=196,
            pot=14,
            to_call=6,
            min_raise_to=16,
            max_raise_to=194,
            legal_actions=["fold", "call", "raise"],
            current_hand_actions=[
                {
                    "round": "pre_reveal",
                    "seat": 0,
                    "action": "raise",
                    "amount": 4,
                },
                {
                    "round": "pre_reveal",
                    "seat": 1,
                    "action": "raise",
                    "amount": 10,
                },
            ],
        )
        reraised["players"][0].update(stack=196, bet_this_round=4)
        reraised["players"][1].update(stack=190, bet_this_round=10)
        self.assertEqual(engine.decide(reraised), {"action": "fold"})

    def test_partial_consensus_defends_limp_and_check_from_small_stabs(self):
        state = Phase2State()
        knowledge = teach_partial_high_family(state, "partial-stab-rule")
        estimate = knowledge.estimate(13, 5)
        self.assertLessEqual(estimate.disagreement, 0.10)
        engine = Phase2Engine(state)

        pre = phase2_request(
            13,
            table_rule="partial-stab-rule",
            hand_number=12,
            delta=-20,
        )
        pre.update(
            your_stack=178,
            pot=7,
            to_call=3,
            min_raise_to=8,
            max_raise_to=178,
            legal_actions=["fold", "call", "raise"],
            current_hand_actions=[
                {
                    "round": "pre_reveal",
                    "seat": 0,
                    "action": "call",
                    "amount": 2,
                },
                {
                    "round": "pre_reveal",
                    "seat": 1,
                    "action": "raise",
                    "amount": 5,
                },
            ],
        )
        pre["players"][0].update(stack=178, bet_this_round=2)
        pre["players"][1].update(bet_this_round=5)

        post = post_bet_request(
            13,
            5,
            table_rule="partial-stab-rule",
            hand_number=13,
            delta=-20,
        )
        post.update(your_stack=178, pot=7, to_call=3)
        post["current_hand_actions"][-1]["amount"] = 3
        post["players"][0].update(stack=178, bet_this_round=0)
        post["players"][1].update(bet_this_round=3)

        self.assertEqual(engine.decide(pre), {"action": "raise", "amount": 8})
        self.assertEqual(engine.decide(post), {"action": "call"})

    def test_large_post_reveal_raise_needs_top_tier_learned_equity(self):
        state = Phase2State()
        teach_candidate(state, "safe-low-rule", "lower")
        engine = Phase2Engine(state)

        def raised_post(number, community):
            request = post_bet_request(
                number,
                community,
                table_rule="safe-low-rule",
                hand_number=14,
            )
            request.update(
                your_stack=171,
                pot=40,
                to_call=18,
                min_raise_to=42,
                max_raise_to=195,
                legal_actions=["fold", "call", "raise"],
                current_hand_actions=[
                    {
                        "round": "pre_reveal",
                        "seat": 1,
                        "action": "raise",
                        "amount": 5,
                    },
                    {
                        "round": "pre_reveal",
                        "seat": 0,
                        "action": "call",
                        "amount": 5,
                    },
                    {
                        "round": "post_reveal",
                        "seat": 0,
                        "action": "bet",
                        "amount": 6,
                    },
                    {
                        "round": "post_reveal",
                        "seat": 1,
                        "action": "raise",
                        "amount": 24,
                    },
                ],
            )
            request["players"][0].update(stack=171, bet_this_round=6)
            request["players"][1].update(bet_this_round=24)
            return request

        self.assertEqual(engine.decide(raised_post(4, 12)), {"action": "fold"})
        self.assertEqual(engine.decide(raised_post(3, 10)), {"action": "call"})

    def test_early_thirty_chip_result_stays_active_but_late_cushion_protects(self):
        state = Phase2State()
        teach_candidate(state, "protected-low-rule", "lower")
        engine = Phase2Engine(state)
        request = phase2_request(
            3,
            table_rule="protected-low-rule",
            hand_number=12,
            delta=33,
        )
        request.update(
            button_seat=1,
            your_stack=212,
            pot=29,
            to_call=13,
            min_raise_to=34,
            max_raise_to=225,
            legal_actions=["fold", "call", "raise"],
            current_hand_actions=[
                {
                    "round": "pre_reveal",
                    "seat": 1,
                    "action": "raise",
                    "amount": 5,
                },
                {
                    "round": "pre_reveal",
                    "seat": 0,
                    "action": "raise",
                    "amount": 8,
                },
                {
                    "round": "pre_reveal",
                    "seat": 1,
                    "action": "raise",
                    "amount": 21,
                },
            ],
        )
        request["players"][0].update(
            chip_delta=33, stack=212, bet_this_round=8
        )
        request["players"][1].update(
            chip_delta=-33, stack=212, bet_this_round=21
        )

        self.assertEqual(engine.decide(request), {"action": "call"})

        late = copy.deepcopy(request)
        late["hand_number"] = 30
        self.assertEqual(engine.decide(late), {"action": "fold"})

    def test_desperate_partial_consensus_uses_half_pot_not_three_quarters(self):
        state = Phase2State()
        teach_partial_high_family(state, "partial-sizing-rule")
        engine = Phase2Engine(state)
        request = phase2_request(
            13,
            table_rule="partial-sizing-rule",
            hand_number=40,
            delta=-10,
        )
        request.update(
            round="post_reveal",
            community_number=5,
            pot=6,
            to_call=0,
            min_raise_to=2,
            max_raise_to=190,
            legal_actions=["check", "bet"],
            current_hand_actions=[
                {
                    "round": "pre_reveal",
                    "seat": 0,
                    "action": "call",
                    "amount": 2,
                },
                {"round": "pre_reveal", "seat": 1, "action": "check"},
                {"round": "post_reveal", "seat": 1, "action": "check"},
            ],
        )
        request["players"][0].update(stack=188, bet_this_round=0)
        request["players"][1].update(bet_this_round=0)

        self.assertEqual(
            engine.decide(request), {"action": "bet", "amount": 3}
        )

    def test_partial_consensus_value_bet_stays_inside_eight_chip_cap(self):
        state = Phase2State()
        teach_partial_high_family(state, "partial-value-cap-rule")
        engine = Phase2Engine(state)
        request = phase2_request(
            13,
            table_rule="partial-value-cap-rule",
            hand_number=20,
        )
        request.update(
            round="post_reveal",
            community_number=5,
            your_stack=195,
            pot=10,
            to_call=0,
            min_raise_to=2,
            max_raise_to=195,
            legal_actions=["check", "bet"],
            current_hand_actions=[
                {
                    "round": "pre_reveal",
                    "seat": 0,
                    "action": "raise",
                    "amount": 5,
                },
                {
                    "round": "pre_reveal",
                    "seat": 1,
                    "action": "call",
                    "amount": 5,
                },
                {"round": "post_reveal", "seat": 1, "action": "check"},
            ],
        )
        request["players"][0].update(stack=195, bet_this_round=0)
        request["players"][1].update(bet_this_round=0)

        self.assertEqual(
            engine.decide(request), {"action": "bet", "amount": 3}
        )

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

    def test_revealed_actions_condition_range_after_six_shown_hands(self):
        state = Phase2State()
        knowledge = teach_candidate(state, "behavior-rule", "lower")
        hands = []
        for hand_number in range(1, 7):
            hand = completed_hand(
                hand_number,
                2,
                1,
                hand_number + 1,
                -1,
                actions=[
                    {
                        "round": "pre_reveal",
                        "seat": 1,
                        "action": "raise",
                        "amount": 6,
                    },
                    {
                        "round": "pre_reveal",
                        "seat": 0,
                        "action": "call",
                        "amount": 6,
                    },
                ],
            )
            hand["button_seat"] = 1
            hand["pot"] = 10
            hands.append(hand)

        request = phase2_request(
            4,
            table_rule="behavior-rule",
            hand_number=7,
        )
        request.update(
            button_seat=1,
            pot=9,
            to_call=4,
            current_hand_actions=[
                {
                    "round": "pre_reveal",
                    "seat": 1,
                    "action": "raise",
                    "amount": 6,
                }
            ],
            recent_hands=hands[:5],
        )
        _, early_profile = state.observe_payload(request)
        early_range = early_profile.range_for(
            payload=request,
            rule_knowledge=knowledge,
        )
        self.assertTrue(all(abs(weight - 1 / 13) < 1e-12 for weight in early_range))

        request["recent_hands"] = hands
        _, learned_profile = state.observe_payload(request)
        opponent_range = learned_profile.range_for(
            payload=request,
            rule_knowledge=knowledge,
        )
        uniform_equity = knowledge.estimate(4, None).mean
        conditioned_equity = knowledge.estimate(
            4,
            None,
            opponent_range=opponent_range,
        ).mean

        self.assertAlmostEqual(sum(opponent_range), 1.0)
        self.assertGreater(opponent_range[0], 2 * opponent_range[-1])
        self.assertLess(conditioned_equity, uniform_equity - 0.15)

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
