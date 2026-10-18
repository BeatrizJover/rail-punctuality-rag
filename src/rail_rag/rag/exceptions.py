"""Exceptions raised by the RAG layer.

They live here rather than in :mod:`rail_rag.core.exceptions` because several
sibling modules share them, unlike ``IngestionError`` which has a single home.
"""

from __future__ import annotations

from rail_rag.core.exceptions import RailRagError


class RagError(RailRagError):
    """Base class for every failure originating in the RAG layer."""


class ProviderError(RagError):
    """Raised when a model provider is misconfigured, absent or unreachable."""


class UnsafeQueryError(RagError):
    """Raised when generated SQL violates the execution policy.

    Distinct from :class:`QueryExecutionError`: nothing was sent to the database.
    """


class QueryExecutionError(RagError):
    """Raised when a validated query fails, times out, or the database is down."""

    def __init__(self, message: str, *, timed_out: bool = False) -> None:
        super().__init__(message)
        self.timed_out = timed_out


class AnswerError(RagError):
    """Raised when the pipeline cannot produce an answer it is willing to stand behind."""


class QueryTooExpensiveError(AnswerError):
    """Raised when the planner's estimate exceeds the cost ceiling; nothing was executed."""

    def __init__(self, estimated_cost: float, ceiling: float) -> None:
        super().__init__(
            "This question is too heavy to answer at this level of detail"
            f" (estimated cost {estimated_cost:,.0f}, limit {ceiling:,.0f})."
            " Try a shorter period or a coarser breakdown."
        )
        self.estimated_cost = estimated_cost
        self.ceiling = ceiling
