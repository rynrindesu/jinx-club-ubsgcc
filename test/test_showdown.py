import copy
import unittest

from app.phase1.showdown import decide_move, showdown_equity


def sample_request():
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


class EquityTests(unittest.TestCase):
    def test_pre_reveal_anchor_points(self):
        self.assertAlmostEqual(showdown_equity(1, None), 18.5 / 169)
        self.assertAlmostEqual(showdown_equity(7, None), 0.5)
        self.assertAlmostEqual(showdown_equity(13, None), 150.5 / 169)

    def test_pair_and_non_pair_equity(self):
        self.assertAlmostEqual(showdown_equity(5, 5), 12.5 / 13)
        self.assertAlmostEqual(showdown_equity(3, 5), 2.5 / 13)
        self.assertAlmostEqual(showdown_equity(13, 5), 11.5 / 13)


class DecisionTests(unittest.TestCase):
    def test_challenge_example_folds_weak_hand(self):
        self.assertEqual(decide_move(sample_request()), {"action": "fold"})

    def test_pair_never_folds_when_facing_a_bet(self):
        request = sample_request()
        request["your_number"] = request["community_number"]

        result = decide_move(request)

        self.assertIn(result["action"], {"call", "raise"})
        if result["action"] == "raise":
            self.assertGreaterEqual(result["amount"], request["min_raise_to"])
            self.assertLessEqual(result["amount"], request["max_raise_to"])

    def test_call_omits_amount(self):
        request = sample_request()
        request.update(
            your_number=13,
            pot=100,
            to_call=5,
            legal_actions=["fold", "call"],
            min_raise_to=None,
            max_raise_to=None,
        )

        self.assertEqual(decide_move(request), {"action": "call"})

    def test_high_non_pair_calls_instead_of_value_raising_post_reveal(self):
        request = sample_request()
        request.update(
            your_number=13,
            community_number=11,
            pot=45,
            to_call=25,
            min_raise_to=50,
            max_raise_to=185,
        )

        self.assertEqual(decide_move(request), {"action": "call"})

    def test_final_hand_non_pair_folds_to_all_in_reraise(self):
        request = sample_request()
        request.update(
            match_id="phase1-final-hand",
            hand_number=45,
            total_hands=45,
            your_number=13,
            community_number=11,
            your_stack=131,
            pot=269,
            to_call=131,
            min_raise_to=None,
            max_raise_to=None,
            legal_actions=["fold", "call"],
            current_hand_actions=[
                {"round": "pre_reveal", "seat": 0, "action": "call", "amount": 2},
                {"round": "pre_reveal", "seat": 1, "action": "raise", "amount": 5},
                {"round": "pre_reveal", "seat": 0, "action": "raise", "amount": 10},
                {"round": "pre_reveal", "seat": 1, "action": "call", "amount": 10},
                {"round": "post_reveal", "seat": 1, "action": "bet", "amount": 25},
                {"round": "post_reveal", "seat": 0, "action": "raise", "amount": 54},
                {"round": "post_reveal", "seat": 1, "action": "raise", "amount": 195},
            ],
        )
        request["players"][0].update(bet_this_round=54, stack=131)
        request["players"][1].update(bet_this_round=195, stack=0, all_in=True)

        self.assertEqual(decide_move(request), {"action": "fold"})

    def test_free_weak_hand_checks(self):
        request = sample_request()
        request.update(
            match_id="force-a-stable-check",
            your_number=6,
            community_number=10,
            pot=14,
            to_call=0,
            legal_actions=["check", "bet"],
            min_raise_to=2,
            max_raise_to=185,
        )

        self.assertEqual(decide_move(request), {"action": "check"})

    def test_only_all_in_raise_is_sized_exactly(self):
        request = sample_request()
        request.update(
            match_id="all-in-pair",
            your_number=5,
            community_number=5,
            min_raise_to=185,
            max_raise_to=185,
        )

        result = decide_move(request)

        self.assertIn(result["action"], {"call", "raise"})
        if result["action"] == "raise":
            self.assertEqual(result["amount"], 185)

    def test_unknown_future_fields_are_ignored(self):
        request = sample_request()
        expected = decide_move(request)
        request["future_server_field"] = {"anything": [1, 2, 3]}

        self.assertEqual(expected, decide_move(request))

    def test_decision_is_deterministic_for_repeated_request(self):
        request = sample_request()
        first = decide_move(request)

        self.assertEqual(first, decide_move(copy.deepcopy(request)))

    def test_matrix_never_emits_an_illegal_action_or_amount(self):
        base = sample_request()
        for your_number in range(1, 14):
            for community in [None, 1, 7, 13]:
                request = copy.deepcopy(base)
                request["your_number"] = your_number
                request["community_number"] = community
                request["round"] = "pre_reveal" if community is None else "post_reveal"

                result = decide_move(request)

                self.assertIn(result["action"], request["legal_actions"])
                if result["action"] in {"bet", "raise"}:
                    self.assertIn("amount", result)
                    self.assertGreaterEqual(result["amount"], request["min_raise_to"])
                    self.assertLessEqual(result["amount"], request["max_raise_to"])
                else:
                    self.assertNotIn("amount", result)


if __name__ == "__main__":
    unittest.main()
