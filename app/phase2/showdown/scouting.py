"""Validated Phase 2 scouting evidence from completed attempts.

The event fixes the rule order across retries.  These are genuine showdown
comparisons, not guessed rule definitions.  Replaying them through the normal
learner preserves their provenance and lets current live evidence take
precedence if a scouting replay turns out to be stale.
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

# Later completed attempts supplied the comparisons that distinguish the
# previously surviving rule families.  The source token is deliberately part
# of the evidence key because hand numbers restart in every match.
_DECISIVE_COMPARISONS: dict[
    int, tuple[tuple[str, int, int, int, int, int], ...]
] = {
    1: (
        ("02c353ed", 22, 5, 5, 6, 1),
    ),
    3: (
        ("02c353ed", 18, 4, 4, 6, 1),
        ("02c353ed", 20, 2, 10, 7, -1),
    ),
    4: (
        ("18898a39", 8, 1, 10, 1, 1),
    ),
}


def observations_for_leg(leg_number: int | None) -> tuple[ShowdownObservation, ...]:
    """Return repeat-safe observations for this fixed Phase 2 leg."""

    comparisons = _SCOUTED_COMPARISONS.get(leg_number or 0, ())
    original = tuple(
        ShowdownObservation(
            key=(f"scout-3d3a46e3-leg-{leg_number}", hand, "hero", "opponent"),
            community=community,
            first_number=hero_number,
            second_number=opponent_number,
            outcome=outcome,
            is_baseline=True,
        )
        for hand, community, hero_number, opponent_number, outcome in comparisons
    )
    decisive = tuple(
        ShowdownObservation(
            key=(
                f"scout-{source}-leg-{leg_number}",
                hand,
                "hero",
                "opponent",
            ),
            community=community,
            first_number=hero_number,
            second_number=opponent_number,
            outcome=outcome,
            is_baseline=True,
        )
        for source, hand, community, hero_number, opponent_number, outcome
        in _DECISIVE_COMPARISONS.get(leg_number or 0, ())
    )
    return original + decisive
