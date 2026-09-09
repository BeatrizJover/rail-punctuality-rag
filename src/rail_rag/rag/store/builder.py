"""Turn the documentation corpus into stored vectors.

This is the only module in the store that spends provider quota, which is why
the persistence in :mod:`rail_rag.rag.store.repository` is kept free of it: an
upsert must be testable without an API key.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import Engine

from rail_rag.rag.exceptions import RagError
from rail_rag.rag.providers.base import Embedder
from rail_rag.rag.store.chunking import DEFAULT_MAX_CHARS, DEFAULT_MIN_CHARS, load_corpus
from rail_rag.rag.store.models import KbSchema
from rail_rag.rag.store.repository import (
    SyncReport,
    pending_chunks,
    save_embeddings,
    sync_chunks,
)
from rail_rag.rag.store.schema import verify_dimension

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildReport:
    """What one build did, in terms a CLI can print without interpretation."""

    sync: SyncReport
    embedded: int
    batches: int


def build_knowledge_base(
    engine: Engine,
    kb: KbSchema,
    embedder: Embedder,
    corpus_dir: Path,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
    batch_size: int = 32,
    force: bool = False,
) -> BuildReport:
    """Chunk the corpus, upsert it, and embed whatever still needs a vector.

    Raises:
        RagError: if the corpus is unusable or the embedder does not match the schema.
        DatabaseError: if the knowledge base is absent or cannot be written.
        ProviderError: if the embedding provider fails.
    """
    if batch_size <= 0:
        raise RagError("batch_size must be positive")
    # Checked before any quota is spent, not after.
    verify_dimension(engine, kb)
    if embedder.dimension != kb.dimension:
        raise RagError(
            f"Embedder {embedder.model_name!r} produces {embedder.dimension} dimensions"
            f" but the knowledge base stores {kb.dimension}. Rebuild it with"
            " 'kb-drop' then 'kb-init', or select a profile that matches."
        )

    chunks = load_corpus(corpus_dir, max_chars=max_chars, min_chars=min_chars)
    report = sync_chunks(engine, kb, chunks)

    pending = pending_chunks(engine, kb, model_name=embedder.model_name, force=force)
    if not pending:
        logger.info("Every chunk already carries a %s vector", embedder.model_name)
        return BuildReport(sync=report, embedded=0, batches=0)

    embedded = 0
    batches = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        vectors = embedder.embed_documents([stored.chunk.embedding_text for stored in batch])
        if len(vectors) != len(batch):
            raise RagError(f"Embedder returned {len(vectors)} vectors for {len(batch)} chunks")
        embedded += save_embeddings(
            engine,
            kb,
            list(zip([stored.id for stored in batch], vectors, strict=True)),
            model_name=embedder.model_name,
        )
        batches += 1
        logger.info("Embedded batch %d (%d chunks)", batches, len(batch))

    return BuildReport(sync=report, embedded=embedded, batches=batches)
