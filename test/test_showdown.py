import copy
import unittest

from app.phase1.showdown import decide_move, showdown_equity


def sample_request():
    """Protocol example from the challenge statement."""

    return {
        "protocol_version": 2,
        "match_id": "phase1-seed7",
        "phase": 1,
        "table_rule": "standard",
        "small_blind": 1,
        "big_blind": 2,
        "starting_stack": 200,
        "your_stack": 185,
        "hand_number": 6,
        "total_hands": 100,
        "round": "post_reveal",
        "your_number": 3,
        "community_number": 5,
        "your_seat": 0,
        "button_seat": 1,
        "pot": 32,
        "to_call": 18,
        "min_raise_to": 36,
        "max_raise_to": 185,
        "legal_actions": ["fold", "call", "raise"],
        "players": [
            {
                "seat": 0,
                "name": "you",
                "folded": False,
                "chip_delta": -8,
                "bet_this_round": 0,
                "stack": 185,
                "all_in": False,
                "busted": False,
            },
            {
                "seat": 1,
                "name": "Gaston",
                "folded": False,
                "chip_delta": 8,
                "bet_this_round": 18,
                "stack": 183,
                "all_in": False,
                "busted": False,
            },
        ],
        "current_hand_actions": [
            {"round": "pre_reveal", "seat": 1, "action": "raise", "amount": 7},
            {"round": "pre_reveal", "seat": 0, "action": "call", "amount": 7},
            {"round": "post_reveal", "seat": 0, "action": "check"},
            {"round": "post_reveal", "seat": 1, "action": "bet", "amount": 18},
        ],
        "recent_hands": [],
    }


def pre_request(
    number,
    *,
    delta=0,
    contribution=2,
    opponent_total=6,
    to_call=4,
    already_raised=False,
):
    """Build a consistent pre-reveal decision with two live players."""

    starting_stack = 200
    your_start = starting_stack + delta
    opponent_start = starting_stack - delta
    actions = [
        {
            "round": "pre_reveal",
            "seat": 1,
            "action": "raise",
            "amount": opponent_total,
        }
    ]
    if already_raised:
        actions = [
            {
                "round": "pre_reveal",
                "seat": 0,
                "action": "raise",
                "amount": contribution,
            },
            {
                "round": "pre_reveal",
                "seat": 1,
                "action": "raise",
                "amount": opponent_total,
            },
        ]

    return {
        "protocol_version": 2,
        "match_id": "pre-policy",
        "phase": 1,
        "table_rule": "standard",
        "small_blind": 1,
        "big_blind": 2,
        "starting_stack": starting_stack,
        "your_stack": your_start - contribution,
        "hand_number": 1,
        "total_hands": 100,
        "round": "pre_reveal",
        "your_number": number,
        "community_number": None,
        "your_seat": 0,
        "button_seat": 1,
        "pot": contribution + opponent_total,
        "to_call": to_call,
        "min_raise_to": max(opponent_total + to_call, 10),
        "max_raise_to": your_start,
        "legal_actions": ["fold", "call", "raise"],
        "players": [
            {
                "seat": 0,
                "name": "you",
                "chip_delta": delta,
                "bet_this_round": contribution,
                "stack": your_start - contribution,
            },
            {
                "seat": 1,
                "name": "Gaston",
                "chip_delta": -delta,
                "bet_this_round": opponent_total,
                "stack": opponent_start - opponent_total,
            },
        ],
        "current_hand_actions": actions,
        "recent_hands": [],
    }


def set_frozen_state(request, *, delta=None, opponent_commitment=None):
    """Adjust frozen deltas/live stacks while preserving current commitments."""

    starting = request["starting_stack"]
    you, opponent = request["players"]
    old_delta = you.get("chip_delta", 0)
    own_commitment = starting + old_delta - you["stack"]
    old_opponent_delta = opponent.get("chip_delta", 0)
    current_opponent_commitment = (
        starting + old_opponent_delta - opponent["stack"]
    )

    if delta is None:
        delta = old_delta
    if opponent_commitment is None:
        opponent_commitment = current_opponent_commitment

    you["chip_delta"] = delta
    you["stack"] = starting + delta - own_commitment
    request["your_stack"] = you["stack"]

    opponent["chip_delta"] = -delta
    opponent["stack"] = starting - delta - opponent_commitment


class EquityTests(unittest.TestCase):
    def test_pre_reveal_anchor_points(self):
        self.assertAlmostEqual(showdown_equity(1, None), 18.5 / 169)
        self.assertAlmostEqual(showdown_equity(7, None), 0.5)
        self.assertAlmostEqual(showdown_equity(13, None), 150.5 / 169)

    def test_pair_and_non_pair_equity(self):
        self.assertAlmostEqual(showdown_equity(5, 5), 12.5 / 13)
        self.assertAlmostEqual(showdown_equity(3, 5), 2.5 / 13)
        self.assertAlmostEqual(showdown_equity(13, 5), 11.5 / 13)


class RegressionTests(unittest.TestCase):
    def test_challenge_example_folds_weak_hand(self):
        self.assertEqual(decide_move(sample_request()), {"action": "fold"})

    def test_latest_replay_eight_folds_before_reveal_while_down_98(self):
        request = pre_request(8, delta=-98)

        self.assertEqual(decide_move(request), {"action": "fold"})

    def test_latest_replay_eight_folds_post_reveal_while_down_98(self):
        request = sample_request()
        request.update(your_number=8, community_number=13, pot=24, to_call=12)
        set_frozen_state(request, delta=-98)

        self.assertEqual(decide_move(request), {"action": "fold"})

    def test_final_hand_thirteen_folds_after_opponent_commits_over_half(self):
        request = sample_request()
        request.update(
            your_number=13,
            community_number=11,
            your_stack=131,
            pot=269,
            to_call=131,
            min_raise_to=None,
            max_raise_to=None,
            legal_actions=["fold", "call"],
        )
        request["players"][0].update(bet_this_round=54, stack=131)
        request["players"][1].update(bet_this_round=195, stack=13)

        self.assertEqual(decide_move(request), {"action": "fold"})


class StartingRangeTests(unittest.TestCase):
    def test_normal_ten_and_eleven_continue_only_for_modest_prices(self):
        self.assertEqual(decide_move(pre_request(10, to_call=4)), {"action": "call"})
        self.assertEqual(decide_move(pre_request(10, to_call=5)), {"action": "fold"})
        self.assertEqual(decide_move(pre_request(11, to_call=8)), {"action": "call"})
        self.assertEqual(decide_move(pre_request(11, to_call=9)), {"action": "fold"})

    def test_down_20_boundary_keeps_eleven_but_down_21_removes_it(self):
        self.assertEqual(
            decide_move(pre_request(11, delta=-20, to_call=6)),
            {"action": "call"},
        )
        self.assertEqual(
            decide_move(pre_request(11, delta=-21, to_call=6)),
            {"action": "fold"},
        )

    def test_one_through_nine_fold_to_pressure(self):
        for number in range(1, 10):
            with self.subTest(number=number):
                self.assertEqual(
                    decide_move(pre_request(number, to_call=1)),
                    {"action": "fold"},
                )

    def test_weak_number_checks_when_continuation_is_free(self):
        request = pre_request(6, contribution=2, opponent_total=2, to_call=0)
        request.update(
            legal_actions=["check", "raise"],
            min_raise_to=4,
            max_raise_to=200,
        )

        self.assertEqual(decide_move(request), {"action": "check"})


class PremiumTests(unittest.TestCase):
    def test_twelve_and_thirteen_make_explicit_big_pre_reveal_raises(self):
        twelve = pre_request(12)
        thirteen = pre_request(13)

        self.assertEqual(decide_move(twelve), {"action": "raise", "amount": 20})
        self.assertEqual(decide_move(thirteen), {"action": "raise", "amount": 25})

    def test_raise_amount_is_round_total_not_additional_chips(self):
        request = pre_request(
            12,
            contribution=6,
            opponent_total=10,
            to_call=4,
        )

        self.assertEqual(decide_move(request), {"action": "raise", "amount": 23})

    def test_premium_calls_instead_of_starting_repeated_raise_war(self):
        request = pre_request(
            13,
            contribution=25,
            opponent_total=40,
            to_call=15,
            already_raised=True,
        )

        self.assertEqual(decide_move(request), {"action": "call"})

    def test_unsafe_minimum_raise_falls_back_to_call(self):
        request = pre_request(12)
        request.update(min_raise_to=31, max_raise_to=198)

        self.assertEqual(decide_move(request), {"action": "call"})

    def test_twelve_uses_strict_half_buyin_deficit_boundary(self):
        allowed = pre_request(12, delta=-100)
        critical = pre_request(
            12,
            delta=-101,
            contribution=2,
            opponent_total=13,
            to_call=11,
        )

        self.assertNotEqual(decide_move(allowed)["action"], "fold")
        self.assertEqual(decide_move(critical), {"action": "fold"})

    def test_critical_twelve_still_calls_a_cheap_non_all_in_price(self):
        request = pre_request(12, delta=-101, to_call=4)

        self.assertEqual(decide_move(request), {"action": "call"})

    def test_critical_twelve_checks_when_free(self):
        request = pre_request(
            12,
            delta=-101,
            contribution=2,
            opponent_total=2,
            to_call=0,
        )
        request.update(legal_actions=["check", "raise"], min_raise_to=4)

        self.assertEqual(decide_move(request), {"action": "check"})

    def test_thirteen_plays_at_exactly_half_opponent_commitment(self):
        request = sample_request()
        request.update(
            your_number=13,
            community_number=12,
            to_call=10,
            min_raise_to=None,
            max_raise_to=None,
            legal_actions=["fold", "call"],
        )
        set_frozen_state(request, delta=0, opponent_commitment=100)

        self.assertEqual(decide_move(request), {"action": "call"})

    def test_thirteen_folds_above_half_opponent_commitment(self):
        request = sample_request()
        request.update(
            your_number=13,
            community_number=12,
            to_call=10,
            min_raise_to=None,
            max_raise_to=None,
            legal_actions=["fold", "call"],
        )
        set_frozen_state(request, delta=0, opponent_commitment=101)

        self.assertEqual(decide_move(request), {"action": "fold"})

    def test_half_buyin_threshold_scales_with_starting_stack(self):
        request = sample_request()
        request.update(
            starting_stack=300,
            your_number=13,
            community_number=12,
            to_call=10,
            min_raise_to=None,
            max_raise_to=None,
            legal_actions=["fold", "call"],
        )
        request["players"][0].update(chip_delta=0, stack=293)
        request["your_stack"] = 293
        request["players"][1].update(chip_delta=0, stack=150)
        self.assertEqual(decide_move(request), {"action": "call"})

        request["players"][1]["stack"] = 149
        self.assertEqual(decide_move(request), {"action": "fold"})

    def test_non_pair_twelve_also_respects_extreme_opponent_pressure(self):
        request = sample_request()
        request.update(
            your_number=12,
            community_number=13,
            to_call=10,
            min_raise_to=None,
            max_raise_to=None,
            legal_actions=["fold", "call"],
        )
        set_frozen_state(request, delta=0, opponent_commitment=101)

        self.assertEqual(decide_move(request), {"action": "fold"})

    def test_total_hand_commitment_not_just_to_call_drives_pressure(self):
        request = sample_request()
        request.update(
            your_number=13,
            community_number=12,
            to_call=11,
            legal_actions=["fold", "call"],
            min_raise_to=None,
            max_raise_to=None,
            current_hand_actions=[
                {
                    "round": "pre_reveal",
                    "seat": 1,
                    "action": "raise",
                    "amount": 70,
                },
                {
                    "round": "pre_reveal",
                    "seat": 0,
                    "action": "call",
                    "amount": 70,
                },
                {
                    "round": "post_reveal",
                    "seat": 1,
                    "action": "bet",
                    "amount": 31,
                },
            ],
        )
        request["players"][0].update(chip_delta=0, bet_this_round=20, stack=110)
        request["your_stack"] = 110
        request["players"][1].update(chip_delta=0, bet_this_round=31, stack=99)

        self.assertEqual(decide_move(request), {"action": "fold"})


class PostRevealTests(unittest.TestCase):
    def test_any_exact_pair_calls_an_all_in_at_any_deficit(self):
        for number in (1, 7, 13):
            request = sample_request()
            request.update(
                your_number=number,
                community_number=number,
                to_call=92,
                min_raise_to=None,
                max_raise_to=None,
                legal_actions=["fold", "call"],
            )
            set_frozen_state(request, delta=-101, opponent_commitment=150)
            request["your_stack"] = request["players"][0]["stack"]
            with self.subTest(number=number):
                self.assertEqual(decide_move(request), {"action": "call"})

    def test_pair_can_use_the_only_legal_all_in_raise(self):
        request = sample_request()
        request.update(
            your_number=5,
            community_number=5,
            min_raise_to=185,
            max_raise_to=185,
        )

        self.assertEqual(
            decide_move(request),
            {"action": "raise", "amount": 185},
        )

    def test_non_pair_twelve_calls_but_does_not_raise_ordinary_bet(self):
        request = sample_request()
        request.update(your_number=12, community_number=13, to_call=12)
        set_frozen_state(request, delta=-20)

        self.assertEqual(decide_move(request), {"action": "call"})

    def test_critical_non_pair_twelve_folds_meaningful_bet(self):
        request = sample_request()
        request.update(your_number=12, community_number=13, to_call=12)
        set_frozen_state(request, delta=-101)

        self.assertEqual(decide_move(request), {"action": "fold"})

    def test_normal_eleven_calls_small_first_bet_but_folds_when_down(self):
        normal = sample_request()
        normal.update(your_number=11, community_number=13, to_call=8)
        set_frozen_state(normal, delta=-20)
        self.assertEqual(decide_move(normal), {"action": "call"})

        down = copy.deepcopy(normal)
        set_frozen_state(down, delta=-21)
        self.assertEqual(decide_move(down), {"action": "fold"})

    def test_free_weak_hand_checks_and_never_bluffs(self):
        request = sample_request()
        request.update(
            your_number=6,
            community_number=10,
            pot=14,
            to_call=0,
            legal_actions=["check", "bet"],
            min_raise_to=2,
            max_raise_to=185,
        )

        self.assertEqual(decide_move(request), {"action": "check"})

    def test_premium_free_value_bet_is_capped_below_all_in(self):
        request = sample_request()
        request.update(
            your_number=13,
            community_number=12,
            pot=100,
            to_call=0,
            legal_actions=["check", "bet"],
            min_raise_to=2,
            max_raise_to=185,
        )

        self.assertEqual(decide_move(request), {"action": "bet", "amount": 30})


class ProtocolSafetyTests(unittest.TestCase):
    def test_call_check_and_fold_never_include_amount(self):
        requests = [
            sample_request(),
            pre_request(11, to_call=4),
            pre_request(6, contribution=2, opponent_total=2, to_call=0),
        ]
        requests[2].update(legal_actions=["check"], min_raise_to=None, max_raise_to=None)

        for request in requests:
            with self.subTest(action=decide_move(request)["action"]):
                self.assertNotIn("amount", decide_move(request))

    def test_player_list_order_does_not_change_decision(self):
        request = pre_request(8, delta=-98)
        expected = decide_move(request)
        request["players"].reverse()

        self.assertEqual(decide_move(request), expected)

    def test_top_level_delta_is_used_when_player_delta_is_missing(self):
        request = pre_request(11, delta=0, to_call=4)
        del request["players"][0]["chip_delta"]
        request["chip_delta"] = -21

        self.assertEqual(decide_move(request), {"action": "fold"})

    def test_live_stack_does_not_replace_frozen_chip_delta(self):
        request = pre_request(11, delta=0, to_call=4)
        request["your_stack"] = 80
        request["players"][0]["stack"] = 80

        self.assertEqual(decide_move(request), {"action": "call"})

    def test_unknown_and_malformed_history_fields_are_ignored(self):
        request = sample_request()
        expected = decide_move(request)
        request["future_server_field"] = {"anything": [1, 2, 3]}
        request["recent_hands"] = [None, {"actions": "not-a-list"}]

        self.assertEqual(decide_move(request), expected)

    def test_repeated_request_is_deterministic(self):
        request = pre_request(13)

        self.assertEqual(decide_move(request), decide_move(copy.deepcopy(request)))

    def test_policy_matrix_never_emits_illegal_or_badly_sized_move(self):
        legal_shapes = [
            (["check"], 0, None, None),
            (["check", "bet"], 0, 2, 190),
            (["check", "raise"], 0, 4, 190),
            (["fold", "call"], 5, None, None),
            (["fold", "call", "raise"], 5, 10, 190),
            (["call"], 5, None, None),
        ]

        for number in range(1, 14):
            for community in (None, 1, 7, 12, 13):
                for delta in (0, -20, -21, -100, -101):
                    for legal, to_call, minimum, maximum in legal_shapes:
                        request = sample_request()
                        request.update(
                            your_number=number,
                            community_number=community,
                            round="pre_reveal" if community is None else "post_reveal",
                            legal_actions=legal,
                            to_call=to_call,
                            min_raise_to=minimum,
                            max_raise_to=maximum,
                        )
                        set_frozen_state(request, delta=delta)

                        result = decide_move(request)

                        self.assertIn(result["action"], legal)
                        if result["action"] in {"bet", "raise"}:
                            self.assertIn("amount", result)
                            self.assertIs(type(result["amount"]), int)
                            self.assertGreaterEqual(result["amount"], minimum)
                            self.assertLessEqual(result["amount"], maximum)
                        else:
                            self.assertNotIn("amount", result)


if __name__ == "__main__":
    unittest.main()
