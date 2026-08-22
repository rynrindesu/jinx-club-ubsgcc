import unittest

from app.phase2.toolbox.routing import next_hop
from app.phase2.toolbox.study import StudyDocument, select_passages


class CharacterEncoder:
    """Small deterministic encoder used to test token-cap plumbing."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))

    def decode(self, tokens: list[int]) -> str:
        return "x" * len(tokens)


class RouteTests(unittest.TestCase):
    def test_adds_entry_tolls_when_choosing_a_route(self):
        graph = {
            "adjacency": {"A": {"B": 4, "C": 2}, "B": {"D": 3}, "C": {"D": 2}, "D": {}},
            "tolls": {"A": 5, "B": 1, "C": 9, "D": 2},
        }

        self.assertEqual(next_hop(graph, "A", "D"), "B")

    def test_hop_allowance_can_require_a_more_expensive_route(self):
        graph = {
            "adjacency": {"A": {"B": 1, "D": 10}, "B": {"C": 1}, "C": {"D": 1}, "D": {}},
            "tolls": {"A": 0, "B": 0, "C": 0, "D": 0},
        }

        self.assertEqual(next_hop(graph, "A", "D"), "B")
        self.assertEqual(next_hop(graph, "A", "D", hops_remaining=2), "D")

    def test_never_uses_a_reported_visited_node(self):
        graph = {
            "adjacency": {"A": {"B": 1, "C": 3}, "B": {"D": 1}, "C": {"D": 1}, "D": {}},
            "tolls": {"A": 0, "B": 0, "C": 0, "D": 0},
        }

        self.assertEqual(next_hop(graph, "A", "D", visited_nodes=["B"]), "C")


class StudySelectionTests(unittest.TestCase):
    def test_returns_the_relevant_bounded_passage(self):
        documents = (
            StudyDocument("History", "https://example.test/history", "The orchard opened in May."),
            StudyDocument(
                "Engineering",
                "https://example.test/engineering",
                "The sensor grid was last brought back into alignment on 14 March.",
            ),
        )

        passages = select_passages(
            "When was the sensor grid last brought back into alignment?",
            documents,
            CharacterEncoder(),
        )

        self.assertEqual(len(passages), 1)
        self.assertIn("14 March", passages[0])

    def test_never_exceeds_the_token_budget(self):
        documents = (
            StudyDocument(
                "Engineering",
                "https://example.test/engineering",
                "Sensor calibration " + "details " * 2_000,
            ),
        )

        encoder = CharacterEncoder()
        passages = select_passages("sensor calibration", documents, encoder)

        self.assertLessEqual(sum(len(encoder.encode(passage)) for passage in passages), 900)
