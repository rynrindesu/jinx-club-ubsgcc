"""Phase-aware SHOWDOWN request dispatcher."""

from typing import Any, Mapping

from ..phase1.showdown import decide_move as decide_phase1_move
from ..phase2.showdown import decide_move as decide_phase2_move


def decide_move(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Route Phase 2 to its learner and retain Phase 1 as the safe default."""

    if _phase(payload.get("phase")) == 2:
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

