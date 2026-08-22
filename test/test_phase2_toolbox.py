import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.phase2.toolbox import server as toolbox_server
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

    def test_hop_planner_prefers_fewer_hops_when_costs_tie(self):
        graph = {
            "adjacency": {"A": {"B": 0, "D": 10}, "B": {"D": 10}, "D": {}},
            "tolls": {"A": 0, "B": 0, "D": 0},
        }

        self.assertEqual(next_hop(graph, "A", "D", hops_remaining=2), "D")


class GraphFetchTests(unittest.TestCase):
    def test_fetches_each_map_id_once(self):
        response = Mock()
        response.json.return_value = {"adjacency": {"A": {}}, "tolls": {"A": 0}}
        toolbox_server._cached_graph.cache_clear()

        with patch("app.phase2.toolbox.server.httpx.get", return_value=response) as get:
            toolbox_server._fetch_graph("map-123")
            toolbox_server._fetch_graph("map-123")

        self.assertEqual(get.call_count, 1)


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

    def test_returns_a_compact_synonym_matched_sentence_window(self):
        documents = (
            StudyDocument(
                "Growers Cooperative",
                "https://example.test/growers",
                "## Cold Store\n"
                "Routine inventory checks took place throughout the spring. "
                "A refrigeration compressor failure on 6 April threatened the stored fruit. "
                "The backup unit restored the required temperature before noon. "
                "The cooperative later updated its maintenance schedule.",
            ),
        )

        passages = select_passages(
            "On what date did a mechanical fault in the cold-store threaten the stored fruit?",
            documents,
            CharacterEncoder(),
        )

        self.assertEqual(len(passages), 1)
        self.assertIn("6 April", passages[0])
        self.assertNotIn("Routine inventory checks", passages[0])
        self.assertLessEqual(len(passages[0]), 260)

    def test_returns_adjacent_context_when_it_contains_the_answer(self):
        documents = (
            StudyDocument(
                "Growers Cooperative",
                "https://example.test/growers",
                "## Incident Reports\n"
                "A refrigeration compressor failure on 6 April threatened the stored fruit. "
                "Investigators traced the failure to a worn drive belt. "
                "The replacement belt was fitted the next morning.",
            ),
        )

        passages = select_passages(
            "What caused the compressor failure?",
            documents,
            CharacterEncoder(),
        )

        self.assertEqual(len(passages), 1)
        self.assertIn("worn drive belt", passages[0])
        self.assertLessEqual(len(passages[0]), 220)


class McpDiscoveryTests(unittest.TestCase):
    def test_advertises_the_phase2_tools(self):
        initialize = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        }
        accept = {"Accept": "application/json, text/event-stream"}

        with TestClient(app) as client:
            initialized = client.post("/mcp", json=initialize, headers=accept)
            session_headers = {**accept, "mcp-session-id": initialized.headers["mcp-session-id"]}
            client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
                headers=session_headers,
            )
            response = client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
                headers=session_headers,
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn('"name":"retrieve"', response.text)
        self.assertIn('"name":"next_route_node"', response.text)
