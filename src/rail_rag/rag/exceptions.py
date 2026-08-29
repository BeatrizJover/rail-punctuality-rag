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
