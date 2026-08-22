import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.phase3.toolbox import server as toolbox_server
from app.phase3.toolbox.meetings import find_meeting_window, own_calendar_blocks
from app.phase3.toolbox.venues import open_venue_names, validate_hour


class VenueAvailabilityTests(unittest.TestCase):
    payload = {
        "day": "Thursday",
        "venues": [
            {"name": "Amber Hall", "x": 6, "y": 3, "available": [["08:00", "11:00"]]},
            {"name": "Nine Quarters", "x": 7, "y": 3, "available": [["11:00", "16:00"]]},
            {"name": "Late Cafe", "x": 1, "y": 1, "available": [["08:00", "11:00"]]},
        ],
    }

    def test_includes_opening_hour_and_excludes_closing_hour(self):
        self.assertEqual(open_venue_names(self.payload, "08:00"), "Amber Hall, Late Cafe")
        self.assertEqual(open_venue_names(self.payload, "11:00"), "Nine Quarters")

    def test_requires_zero_padded_on_the_hour_time(self):
        self.assertEqual(validate_hour("23:00"), "23:00")
        for invalid in ("9:00", "09:30", "07:00", "24:00"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_hour(invalid)


class VenueFetchTests(unittest.TestCase):
    def test_fetches_each_day_once(self):
        response = Mock()
        response.json.return_value = {"venues": []}
        toolbox_server._cached_venues.cache_clear()

        with patch("app.phase3.toolbox.server.httpx.get", return_value=response) as get:
            toolbox_server._fetch_venues("Thursday")
            toolbox_server._fetch_venues("Thursday")

        self.assertEqual(get.call_count, 1)
        self.assertTrue(get.call_args.args[0].endswith("/venues/Thursday"))


class InboxEndpointTests(unittest.TestCase):
    def test_uses_the_emails_endpoint_by_default(self):
        self.assertTrue(toolbox_server._inbox_url().endswith("/emails"))


class MeetingTimeTests(unittest.TestCase):
    inbox = """From: Marek Sould <m.sould@kesterline.example>
Response: ACCEPTED
When: Tuesday 10:00-11:00

We had this down for 12 pm originally, but it is no longer current.

From: Ada <ada@example.test>
Response: TENTATIVE
When: Tuesday 13:00-14:00
"""

    schedules = {
        "ada": {"busy": [["08:00", "10:00"]]},
        "bram": {"busy": [["12:00", "13:00"]]},
    }

    def test_prefers_a_later_clean_window_over_an_earlier_tentative_one(self):
        inbox = "Response: TENTATIVE\nWhen: Tuesday 10:00-11:00\n"
        schedules = {"ada": {"busy": [["11:00", "12:00"]]}}
        self.assertEqual(
            find_meeting_window(
                "Tuesday", ["ada"], "10:00", "13:00", 60, inbox, schedules
            ),
            "12:00, 13:00",
        )

    def test_uses_earliest_tentative_window_only_when_no_clean_window_exists(self):
        inbox = "Response: TENTATIVE\nWhen: Tuesday 12:00-13:00\n"
        schedules = {"ada": {"busy": [["13:00", "14:00"]]}}
        self.assertEqual(
            find_meeting_window("Tuesday", ["ada"], "12:00", "14:00", 60, inbox, schedules),
            "12:00, 13:00",
        )

    def test_events_touching_a_window_boundary_do_not_overlap(self):
        inbox = "Response: ACCEPTED\nWhen: Tuesday 08:00-10:00\n"
        self.assertEqual(
            find_meeting_window("Tuesday", ["ada"], "10:00", "11:00", 60, inbox, {"ada": {"busy": []}}),
            "10:00, 11:00",
        )

    def test_ignores_unstructured_times_in_message_prose(self):
        hard, soft = own_calendar_blocks(self.inbox, "Tuesday")
        self.assertEqual([(block.start, block.end) for block in hard], [(600, 660)])
        self.assertEqual([(block.start, block.end) for block in soft], [(780, 840)])


class McpDiscoveryTests(unittest.TestCase):
    def test_advertises_the_phase3_tool(self):
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
        self.assertIn('"name":"find_open_venues"', response.text)
        self.assertIn('"name":"find_meeting_time"', response.text)
