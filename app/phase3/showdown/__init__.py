"""Independent SHOWDOWN Phase 3 bot.

This package deliberately has no dependency on earlier SHOWDOWN phases.
"""

from typing import Any


def decide_move(payload: dict[str, Any]) -> dict[str, str | int]:
    """Load the runtime lazily so analysis helpers can import independently."""

    from .engine import decide_move as _decide_move

    return _decide_move(payload)


__all__ = ["decide_move"]
