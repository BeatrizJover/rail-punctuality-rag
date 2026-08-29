"""Storage for the knowledge base, in its own PostgreSQL namespace.

Three separate ``MetaData`` instances now exist, and the split is deliberate:
``gold`` holds mirrored analytics, ``ops`` holds the load history, and ``rag``
holds the corpus. Rebuilding one must never take down another. Embeddings in
particular cost provider quota to regenerate, so ``db-drop`` has no business
touching them.

The vector column carries its dimension in the DDL, which means the schema
cannot be defined until the embedding model is known. Hence a factory rather
than a module-level table: a hard-coded dimension would be a claim the
configuration is free to contradict.
"""

from __future__ import annotations

from dataclasses import dataclass

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Identity,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
)

RAG_SCHEMA = "rag"

#: SHA-256 hex digest length, used to detect chunks whose text has not changed.
_HASH_LEN = 64


@dataclass(frozen=True)
class KbSchema:
    """The knowledge-base tables, bound to one embedding dimension."""

    metadata: MetaData
    kb_chunk: Table
    dimension: int


def build_kb_schema(dimension: int) -> KbSchema:
    """Define the knowledge-base tables for a given embedding dimension.

    Raises:
        ValueError: if the dimension is not positive.
    """
    if dimension <= 0:
        raise ValueError("embedding dimension must be positive")

    metadata = MetaData(schema=RAG_SCHEMA)
    kb_chunk = Table(
        "kb_chunk",
        metadata,
        Column("id", BigInteger, Identity(), primary_key=True),
        # Source document stem, so an answer can cite where a claim came from.
        Column("doc_id", String(128), nullable=False),
        Column("chunk_index", Integer, nullable=False),
        Column("heading", String(256), nullable=True),
        Column("content", Text, nullable=False),
        # Lets a rebuild skip chunks whose text is unchanged, saving provider quota.
        Column("content_hash", String(_HASH_LEN), nullable=False),
        # Null until the chunk is embedded, so ingestion and embedding stay separable.
        Column("embedding", Vector(dimension), nullable=True),
        # Stored per row: a model swap with a different dimension must be detectable.
        Column("embedding_model", String(128), nullable=True),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        UniqueConstraint("doc_id", "chunk_index", name="uq_kb_chunk_position"),
        Index("ix_kb_chunk_doc", "doc_id"),
        comment=(
            "One retrievable passage of the project's own documentation. "
            "No ANN index by design: the corpus is small enough that an exact "
            "scan is both faster and more accurate than an approximate one."
        ),
    )
    return KbSchema(metadata=metadata, kb_chunk=kb_chunk, dimension=dimension)
