import copy
import unittest
from dataclasses import replace

from app.phase2.showdown.bot import (
    Phase2Engine,
    _post_reveal_move,
    _pre_reveal_move,
    _risk_context,
)
from app.phase2.showdown.rules import EquityEstimate
from app.phase2.showdown.state import OpponentProfile, Phase2State
from test.test_showdown_phase2 import phase2_request


def learned_equity(mean: float) -> EquityEstimate:
    return EquityEstimate(
        mean=mean,
        lower=mean,
        upper=mean,
        disagreement=0.0,
        coverage=1.0,
        candidate_count=1,
        observation_count=12,
        confidence="learned",
    )


def neutral_profile() -> OpponentProfile:
    return OpponentProfile(
        fold_to_open_rate=0.35,
        reraise_rate=0.18,
        post_fold_rate=0.35,
        post_reraise_rate=0.15,
        bet_after_check_rate=0.45,
        aggression_rate=0.45,
        open_responses=0,
        post_responses=0,
        checked_to=0,
        decisions=0,
    )


def facing_raise_five(*, hand_number: int = 10, delta: int = 0):
    request = phase2_request(
        7,
        table_rule="policy-regression",
        hand_number=hand_number,
        delta=delta,
    )
    request.update(
        button_seat=1,
        your_stack=198 + delta,
        pot=7,
        to_call=3,
        min_raise_to=8,
        max_raise_to=198 + delta,
        legal_actions=["fold", "call", "raise"],
        current_hand_actions=[
            {
                "round": "pre_reveal",
                "seat": 1,
                "action": "raise",
                "amount": 5,
            }
        ],
    )
    request["players"][0].update(
        chip_delta=delta,
        stack=198 + delta,
        bet_this_round=2,
    )
    request["players"][1].update(
        chip_delta=-delta,
        bet_this_round=5,
    )
    return request


class Phase2PolicyRegressionTests(unittest.TestCase):
    def test_exact_scouted_leg_two_card_twelve_stays_active(self):
        request = phase2_request(
            12,
            table_rule="scouted-guarded-open",
            match_id="scouted-guarded-open-match",
            leg_number=2,
            hand_number=37,
            delta=26,
        )

        self.assertEqual(
            Phase2Engine(Phase2State()).decide(request),
            {"action": "raise", "amount": 4},
        )

    def test_future_blinds_define_the_exact_coast_boundary(self):
        unsecured = phase2_request(12, hand_number=37, delta=30)
        secured = phase2_request(12, hand_number=37, delta=31)

        self.assertEqual(_risk_context(unsecured).coast_floor, 24)
        self.assertFalse(_risk_context(unsecured).secure_current_hand)
        self.assertEqual(_risk_context(secured).coast_floor, 25)
        self.assertTrue(_risk_context(secured).secure_current_hand)
        self.assertEqual(
            Phase2Engine(Phase2State(use_scouted_priors=False)).decide(secured),
            {"action": "fold"},
        )

    def test_aggression_increases_call_caution_without_bluff_evidence(self):
        request = phase2_request(7, hand_number=10)
        request.update(
            round="post_reveal",
            community_number=5,
            your_stack=198,
            pot=8,
            to_call=3,
            min_raise_to=None,
            max_raise_to=None,
            legal_actions=["fold", "call"],
            current_hand_actions=[
                {
                    "round": "post_reveal",
                    "seat": 1,
                    "action": "bet",
                    "amount": 3,
                }
            ],
        )
        request["players"][0].update(stack=198, bet_this_round=0)
        request["players"][1].update(bet_this_round=3)
        risk = _risk_context(request)
        neutral = neutral_profile()
        aggressive = replace(
            neutral,
            aggression_rate=0.80,
            decisions=8,
        )

        self.assertEqual(
            _post_reveal_move(request, learned_equity(0.32), neutral, risk),
            {"action": "call"},
        )
        self.assertEqual(
            _post_reveal_move(request, learned_equity(0.32), aggressive, risk),
            {"action": "fold"},
        )

    def test_raise_five_bet_seven_line_is_priced_before_entering(self):
        request = facing_raise_five()
        risk = _risk_context(request)
        profile = neutral_profile()

        self.assertEqual(
            _pre_reveal_move(request, learned_equity(0.58), profile, risk),
            {"action": "fold"},
        )
        self.assertEqual(
            _pre_reveal_move(request, learned_equity(0.70), profile, risk),
            {"action": "call"},
        )

    def test_unsecured_late_lead_raises_only_response_ready_hands(self):
        request = phase2_request(
            12,
            table_rule="guarded-open",
            hand_number=37,
            delta=26,
        )
        risk = _risk_context(request)
        profile = neutral_profile()

        self.assertEqual(risk.tier, "guarded")
        self.assertEqual(risk.coast_floor, 20)
        self.assertFalse(risk.secure_current_hand)
        self.assertEqual(
            _pre_reveal_move(request, learned_equity(0.885), profile, risk),
            {"action": "raise", "amount": 4},
        )
        self.assertEqual(
            _pre_reveal_move(request, learned_equity(0.80), profile, risk),
            {"action": "call"},
        )

        reraised = copy.deepcopy(request)
        reraised.update(
            your_stack=222,
            pot=14,
            to_call=6,
            min_raise_to=16,
            max_raise_to=222,
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
        reraised["players"][0].update(stack=222, bet_this_round=4)
        reraised["players"][1].update(bet_this_round=10)
        response_risk = _risk_context(reraised)

        self.assertEqual(
            _pre_reveal_move(
                reraised,
                learned_equity(0.885),
                profile,
                response_risk,
            ),
            {"action": "call"},
        )
        self.assertEqual(
            _pre_reveal_move(
                reraised,
                learned_equity(0.80),
                profile,
                response_risk,
            ),
            {"action": "fold"},
        )

    def test_guarded_open_can_continue_against_expected_post_bet(self):
        request = phase2_request(
            12,
            table_rule="guarded-post",
            hand_number=37,
            delta=26,
        )
        request.update(
            round="post_reveal",
            community_number=5,
            your_stack=222,
            pot=15,
            to_call=7,
            min_raise_to=21,
            max_raise_to=222,
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
                    "action": "call",
                    "amount": 4,
                },
                {
                    "round": "post_reveal",
                    "seat": 1,
                    "action": "bet",
                    "amount": 7,
                },
            ],
        )
        request["players"][0].update(stack=222, bet_this_round=0)
        request["players"][1].update(bet_this_round=7)

        self.assertEqual(
            _post_reveal_move(
                request,
                learned_equity(0.885),
                neutral_profile(),
                _risk_context(request),
            ),
            {"action": "call"},
        )


if __name__ == "__main__":
    unittest.main()
