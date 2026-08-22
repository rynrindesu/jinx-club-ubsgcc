import unittest

from app.phase2.showdown.rules import (
    RuleCandidate,
    RuleKnowledge,
    ShowdownObservation,
    build_candidate_rules,
    extract_observations,
)
from app.phase2.showdown.scouting import observations_for_leg


def replay_hand(**updates):
    hand = {
        "hand_number": 12,
        "community_number": 4,
        "shown_numbers": {"0": 4, "1": 6},
        "winners": [0, 1],
        "actions": [],
    }
    hand.update(updates)
    return hand


class CandidateRecoveryTests(unittest.TestCase):
    def test_live_showdown_supersedes_a_contradictory_baseline(self):
        candidates = (
            RuleCandidate("higher", (("number", 1),)),
            RuleCandidate("lower", (("number", -1),)),
        )
        knowledge = RuleKnowledge(candidates)
        knowledge.ingest(
            ShowdownObservation(
                ("scout", 1, "hero", "opponent"),
                5,
                3,
                2,
                1,
                is_baseline=True,
            )
        )

        knowledge.ingest(
            ShowdownObservation(
                ("live", 1, "0", "1"),
                5,
                1,
                2,
                1,
            )
        )

        self.assertEqual(knowledge.observation_count, 2)
        self.assertEqual(knowledge.active_candidates, {1})
        estimate = knowledge.estimate(1, 5)
        self.assertEqual(estimate.confidence, "partial")
        self.assertEqual(estimate.observation_count, 1)

    def test_decisive_scouting_resolves_the_inferred_legs(self):
        candidates = build_candidate_rules()
        expected = {
            1: {"pair_then_higher"},
            3: {"pair_then_seven_then_higher"},
            4: {"pair_loses_then_lower"},
        }

        for leg, expected_names in expected.items():
            with self.subTest(leg=leg):
                knowledge = RuleKnowledge(candidates)
                for observation in observations_for_leg(leg):
                    self.assertTrue(observation.is_baseline)
                    knowledge.ingest(observation)
                names = {
                    candidates[index].name
                    for index in knowledge.active_candidates
                }
                self.assertEqual(names, expected_names)

    def test_leg_three_candidate_prioritizes_pair_then_seven_then_high(self):
        candidate = next(
            candidate
            for candidate in build_candidate_rules()
            if candidate.name == "pair_then_seven_then_higher"
        )

        self.assertEqual(candidate.compare(4, 6, 4), 1)
        self.assertEqual(candidate.compare(10, 7, 2), -1)
        self.assertEqual(candidate.compare(7, 4, 4), -1)
        self.assertEqual(candidate.compare(7, 13, 2), 1)
        self.assertEqual(candidate.compare(10, 8, 2), 1)


class RefundExtractionTests(unittest.TestCase):
    def extract(self, hand):
        return extract_observations(
            table_rule="leg-three",
            match_id="replay",
            your_seat=0,
            hands=[hand],
        )

    def test_deltas_identify_the_showdown_winner_despite_refund_winners(self):
        observations = self.extract(
            replay_hand(deltas={"0": 141, "1": -141})
        )

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].outcome, 1)

    def test_player_delta_records_are_supported(self):
        observations = self.extract(
            replay_hand(
                players=[
                    {"seat": 0, "delta": -141},
                    {"seat": 1, "delta": 141},
                ]
            )
        )

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].outcome, -1)

    def test_ambiguous_all_in_shared_winners_are_skipped(self):
        hand = replay_hand(
            actions=[{"seat": 1, "action": "all_in", "amount": 156}]
        )

        self.assertEqual(self.extract(hand), [])

    def test_unequal_contributions_without_deltas_are_skipped(self):
        hand = replay_hand(contributions={"0": 141, "1": 156})

        self.assertEqual(self.extract(hand), [])

    def test_unequal_final_action_totals_without_deltas_are_skipped(self):
        hand = replay_hand(
            actions=[
                {
                    "round": "post_reveal",
                    "seat": 0,
                    "action": "raise",
                    "amount": 141,
                },
                {
                    "round": "post_reveal",
                    "seat": 1,
                    "action": "raise",
                    "amount": 156,
                },
            ]
        )

        self.assertEqual(self.extract(hand), [])

    def test_ordinary_shared_winners_remain_a_tie(self):
        observations = self.extract(replay_hand())

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].outcome, 0)


if __name__ == "__main__":
    unittest.main()
