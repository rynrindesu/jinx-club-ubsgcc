"""Phase-aware SHOWDOWN request dispatcher."""

from typing import Any, Mapping

from ..phase1.showdown import decide_move as decide_phase1_move
from ..phase2.showdown import decide_move as decide_phase2_move
from ..phase3.showdown.engine import decide_move as decide_phase3_move


def decide_move(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Route each SHOWDOWN phase to its isolated decision engine."""

    phase = _phase(payload.get("phase"))
    if phase == 3:
        return decide_phase3_move(payload)
    if phase == 2:
        return decide_phase2_move(payload)
    return decide_phase1_move(payload)


def _phase(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


__all__ = ["decide_move"]
