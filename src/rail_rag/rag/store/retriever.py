"""Retrieve the passages closest to a question.

The compatibility checks run once, in ``__init__``, rather than on every call:
a dimension mismatch is a startup condition, and re-reading the catalogue for
each question would pay for it on the hot path.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from sqlalchemy import Engine

from rail_rag.rag.exceptions import RagError
from rail_rag.rag.providers.base import Embedder
from rail_rag.rag.store.models import KbSchema
from rail_rag.rag.store.repository import RetrievedChunk, search
from rail_rag.rag.store.schema import verify_dimension

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 4


@dataclass(frozen=True)
class ScopedPassages:
    """The same question ranked over every source and over internal documents only.

    ``internal_only`` is a subset of ``all_sources``, not its complement.
    """

    all_sources: Sequence[RetrievedChunk]
    internal_only: Sequence[RetrievedChunk]


class Retriever:
    """Embeds a question and returns the nearest stored passages."""

    def __init__(
        self,
        engine: Engine,
        kb: KbSchema,
        embedder: Embedder,
        *,
        top_k: int = DEFAULT_TOP_K,
        external_ids: Collection[str] = (),
    ) -> None:
        """Verify that the stored vectors and the configured model agree.

        Raises:
            RagError: if the embedder does not match the schema, or top_k is not positive.
            DatabaseError: if the knowledge base is missing or its width differs.
        """
        if top_k <= 0:
            raise RagError("top_k must be positive")
        verify_dimension(engine, kb)
        if embedder.dimension != kb.dimension:
            raise RagError(
                f"Embedder {embedder.model_name!r} produces {embedder.dimension} dimensions"
                f" but the knowledge base stores {kb.dimension}."
            )
        self._engine = engine
        self._kb = kb
        self._embedder = embedder
        self._top_k = top_k
        self._external_ids = tuple(external_ids)

    @property
    def top_k(self) -> int:
        return self._top_k

    def retrieve(self, question: str, *, top_k: int | None = None) -> Sequence[RetrievedChunk]:
        """Return the passages most similar to ``question``, closest first.

        An empty question short-circuits: embedding whitespace costs quota and
        returns an arbitrary ranking dressed up as a result.

        Raises:
            ProviderError: if the embedding provider fails.
            DatabaseError: if the search query fails.
        """
        if not question.strip():
            return []
        vector = self._embedder.embed_query(question)
        results = search(self._engine, self._kb, vector, top_k=top_k or self._top_k)
        logger.debug("Retrieved %d passages", len(results))
        return results

    def retrieve_scoped(self, question: str, *, top_k: int | None = None) -> ScopedPassages:
        """Rank one question twice, over both scopes, for the price of one embedding.

        With no declared external ids the two scopes are the same object and the
        second scan never runs, so the default costs exactly what ``retrieve`` does.

        Raises:
            ProviderError: if the embedding provider fails.
            DatabaseError: if a search query fails.
        """
        if not question.strip():
            return ScopedPassages((), ())
        limit = top_k or self._top_k
        vector = self._embedder.embed_query(question)
        every = search(self._engine, self._kb, vector, top_k=limit)
        internal = every
        if self._external_ids:
            internal = search(
                self._engine, self._kb, vector, top_k=limit, exclude_docs=self._external_ids
            )
        logger.debug("Retrieved %d passages, %d of them internal", len(every), len(internal))
        return ScopedPassages(all_sources=every, internal_only=internal)
