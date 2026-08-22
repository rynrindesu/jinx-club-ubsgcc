"""Validated Phase 2 scouting evidence from completed attempts.

The event fixes the rule order across retries.  These are genuine showdown
comparisons from match 3d3a46e3-afc5-4fdf-bf58-c3176419000e, not guessed rule
definitions.  Replaying the comparisons through the normal learner preserves
uncertainty where the scouting attempt did not distinguish two rules.
"""

from __future__ import annotations

from .rules import ShowdownObservation


# Each tuple is (hand_number, community, hero_number, opponent_number, outcome).
# Outcome is from the hero's perspective: 1 win, 0 tie, -1 loss.
_SCOUTED_COMPARISONS: dict[
    int, tuple[tuple[int, int, int, int, int], ...]
] = {
    1: (
        (1, 11, 12, 4, 1),
        (3, 9, 8, 2, 1),
        (4, 13, 4, 6, -1),
        (5, 11, 13, 3, 1),
        (8, 4, 11, 7, 1),
        (13, 5, 5, 4, 1),
        (14, 2, 1, 6, -1),
        (16, 13, 11, 7, 1),
        (18, 6, 3, 8, -1),
        (26, 4, 2, 7, -1),
        (28, 3, 11, 7, 1),
    ),
    2: (
        (2, 2, 12, 9, 1),
        (7, 9, 4, 7, -1),
        (9, 8, 9, 2, 1),
        (10, 6, 7, 8, -1),
        (17, 7, 6, 1, 1),
        (21, 2, 13, 6, 1),
        (26, 3, 7, 9, -1),
        (34, 1, 7, 8, -1),
        (40, 7, 11, 8, 1),
    ),
    3: (
        (2, 12, 8, 6, 1),
        (8, 8, 7, 6, 1),
        (9, 8, 7, 5, 1),
        (11, 11, 4, 2, 1),
        (15, 11, 6, 4, 1),
        (16, 7, 8, 6, 1),
        (19, 5, 10, 9, 1),
        (20, 11, 3, 6, -1),
        (21, 5, 9, 2, 1),
        (25, 5, 12, 3, 1),
        (27, 12, 6, 3, 1),
        (28, 8, 1, 9, -1),
        (34, 5, 1, 9, -1),
        (36, 3, 2, 9, -1),
    ),
    4: (
        (7, 11, 10, 7, -1),
        (14, 4, 5, 8, 1),
        (15, 11, 3, 10, 1),
        (17, 5, 6, 11, 1),
        (37, 9, 3, 8, 1),
        (38, 7, 2, 8, 1),
        (40, 7, 12, 8, -1),
    ),
}


def observations_for_leg(leg_number: int | None) -> tuple[ShowdownObservation, ...]:
    """Return repeat-safe observations for this fixed Phase 2 leg."""

    comparisons = _SCOUTED_COMPARISONS.get(leg_number or 0, ())
    return tuple(
        ShowdownObservation(
            key=(f"scout-3d3a46e3-leg-{leg_number}", hand, "hero", "opponent"),
            community=community,
            first_number=hero_number,
            second_number=opponent_number,
            outcome=outcome,
        )
        for hand, community, hero_number, opponent_number, outcome in comparisons
    )
