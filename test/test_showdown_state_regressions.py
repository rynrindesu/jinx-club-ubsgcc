import unittest

from app.phase2.showdown.rules import (
    RuleKnowledge,
    ShowdownObservation,
    build_candidate_rules,
)
from app.phase2.showdown.state import (
    OpponentProfile,
    RangeEvidence,
    _live_range_context,
    _range_action_contexts,
)


def learned_higher_rule() -> RuleKnowledge:
    knowledge = RuleKnowledge(build_candidate_rules())
    higher = next(
        index
        for index, candidate in enumerate(knowledge.candidates)
        if candidate.name == "higher"
    )
    knowledge.active_candidates = {higher}
    knowledge.ingest(
        ShowdownObservation(
            key=("training", 1, "0", "1"),
            community=7,
            first_number=13,
            second_number=1,
            outcome=1,
        )
    )
    return knowledge


def empty_profile(**overrides: object) -> OpponentProfile:
    values: dict[str, object] = {
        "fold_to_open_rate": 0.35,
        "reraise_rate": 0.18,
        "post_fold_rate": 0.35,
        "post_reraise_rate": 0.15,
        "bet_after_check_rate": 0.45,
        "aggression_rate": 0.45,
        "open_responses": 0,
        "post_responses": 0,
        "checked_to": 0,
        "decisions": 0,
        "post_reraises": 0,
    }
    values.update(overrides)
    return OpponentProfile(**values)  # type: ignore[arg-type]


def live_payload(actions: list[dict[str, object]], pot: int) -> dict[str, object]:
    return {
        "your_seat": 0,
        "button_seat": 1,
        "small_blind": 1,
        "big_blind": 2,
        "round": actions[-1]["round"],
        "community_number": None,
        "pot": pot,
        "players": [{"seat": 0}, {"seat": 1}],
        "current_hand_actions": actions,
    }


class OpponentRangeRegressionTests(unittest.TestCase):
    def test_first_raise_and_reraise_have_soft_strong_priors(self):
        knowledge = learned_higher_rule()
        profile = empty_profile()
        first_raise = live_payload(
            [
                {
                    "round": "pre_reveal",
                    "seat": 1,
                    "action": "raise",
                    "amount": 5,
                }
            ],
            pot=7,
        )
        reraise = live_payload(
            [
                {
                    "round": "pre_reveal",
                    "seat": 0,
                    "action": "raise",
                    "amount": 5,
                },
                {
                    "round": "pre_reveal",
                    "seat": 1,
                    "action": "raise",
                    "amount": 10,
                },
            ],
            pot=15,
        )

        raise_range = profile.range_for(
            payload=first_raise, rule_knowledge=knowledge
        )
        reraise_range = profile.range_for(payload=reraise, rule_knowledge=knowledge)

        self.assertAlmostEqual(sum(raise_range), 1.0)
        self.assertGreater(min(raise_range), 0.0)
        self.assertGreater(raise_range[-1], raise_range[0])
        self.assertGreater(reraise_range[-1], raise_range[-1])
        self.assertGreater(
            sum((number + 1) * weight for number, weight in enumerate(reraise_range)),
            sum((number + 1) * weight for number, weight in enumerate(raise_range)),
        )

    def test_one_reveal_blends_with_prior_and_more_reveals_gain_weight(self):
        knowledge = learned_higher_rule()
        payload = live_payload(
            [
                {
                    "round": "pre_reveal",
                    "seat": 1,
                    "action": "raise",
                    "amount": 5,
                }
            ],
            pot=7,
        )
        weak_strength = knowledge.estimate(1, None).mean
        weak_raise = RangeEvidence(
            strength=weak_strength,
            action="raise",
            size_bucket="large",
            position="button",
            round_name="pre_reveal",
            reliability=1.0,
        )
        prior = empty_profile().range_for(payload=payload, rule_knowledge=knowledge)
        one = empty_profile(
            range_samples=(weak_raise,), range_shown_hands=1
        ).range_for(payload=payload, rule_knowledge=knowledge)
        many = empty_profile(
            range_samples=(weak_raise,) * 8, range_shown_hands=8
        ).range_for(payload=payload, rule_knowledge=knowledge)

        self.assertGreater(one[0], prior[0])
        self.assertGreater(many[0], one[0])
        self.assertLess(many[-1], one[-1])

    def test_open_reraise_count_is_sticky_after_rate_dilution(self):
        profile = empty_profile(
            open_responses=8,
            reraise_rate=(1 + 0.18 * 4) / (8 + 4),
            open_reraises=1,
        )

        self.assertLess(profile.reraise_rate, 0.30)
        self.assertTrue(profile.punishes_opens)

    def test_post_reraise_count_is_sticky_after_rate_dilution(self):
        profile = empty_profile(
            post_responses=10,
            post_reraise_rate=(1 + 0.15 * 4) / (10 + 4),
            post_reraises=1,
        )

        self.assertLess(profile.post_reraise_rate, 0.29)
        self.assertTrue(profile.punishes_post_bets)
        self.assertFalse(
            empty_profile(
                post_responses=10,
                post_reraise_rate=profile.post_reraise_rate,
            ).punishes_post_bets
        )
        self.assertTrue(
            empty_profile(
                post_responses=1,
                post_reraise_rate=0.30,
            ).punishes_post_bets
        )


class ActionSizingRegressionTests(unittest.TestCase):
    def test_live_and_history_use_increment_over_pot_before_action(self):
        completed_actions = [
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
            {"round": "post_reveal", "seat": 0, "action": "check"},
            {
                "round": "post_reveal",
                "seat": 1,
                "action": "bet",
                "amount": 7,
            },
            {
                "round": "post_reveal",
                "seat": 0,
                "action": "call",
                "amount": 7,
            },
        ]
        historical = _range_action_contexts(
            completed_actions,
            final_pot=24,
            button_seat=1,
            small_blind=1,
            big_blind=2,
            seats=(0, 1),
        )
        live = live_payload(completed_actions[:-1], pot=17)

        self.assertEqual(historical[0], ("raise", "large"))
        self.assertEqual(historical[3], ("bet", "medium"))
        self.assertEqual(_live_range_context(live)[:2], historical[3])

    def test_second_raise_is_classified_as_reraise(self):
        actions = [
            {
                "round": "pre_reveal",
                "seat": 0,
                "action": "raise",
                "amount": 5,
            },
            {
                "round": "pre_reveal",
                "seat": 1,
                "action": "raise",
                "amount": 10,
            },
        ]

        contexts = _range_action_contexts(
            actions,
            final_pot=15,
            button_seat=0,
            small_blind=1,
            big_blind=2,
            seats=(0, 1),
        )

        self.assertEqual(contexts[0][0], "raise")
        self.assertEqual(contexts[1][0], "reraise")


if __name__ == "__main__":
    unittest.main()
