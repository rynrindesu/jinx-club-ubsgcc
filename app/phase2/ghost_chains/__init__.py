"""Ghost Chains Phase 2 structural and identity-risk engine."""

from .identity import (
    IdentityConfig,
    IdentityDimensionScore,
    IdentityEvent,
    IdentityScore,
    IdentityScorer,
)
from .models import (
    Transaction,
    TransactionConflictError,
    TransactionValidationError,
)
from .scoring import DiscountedWalkScorer, ScoreConfig, StructuralScore, TemporalEdge
from .solution import (
    EngineSnapshot,
    GhostChainsEngine,
    reset,
    score_batch,
    score_transaction,
    solve,
)

__all__ = [
    "DiscountedWalkScorer",
    "EngineSnapshot",
    "GhostChainsEngine",
    "IdentityConfig",
    "IdentityDimensionScore",
    "IdentityEvent",
    "IdentityScore",
    "IdentityScorer",
    "ScoreConfig",
    "StructuralScore",
    "TemporalEdge",
    "Transaction",
    "TransactionConflictError",
    "TransactionValidationError",
    "reset",
    "score_batch",
    "score_transaction",
    "solve",
]
