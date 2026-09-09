"""Persistence for the knowledge base: upsert, embedding writes and vector search.

Nothing here knows about model providers. The module that spends API quota is
:mod:`rail_rag.rag.store.builder`, and keeping the two apart is what lets the
whole upsert and search surface be tested with no network and no API key.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from sqlalchemy import (
    CursorResult,
    Engine,
    bindparam,
    delete,
    func,
    literal_column,
    null,
    select,
    tuple_,
    update,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError

from rail_rag.core.exceptions import DatabaseError
from rail_rag.rag.exceptions import RagError
from rail_rag.rag.store.chunking import Chunk
from rail_rag.rag.store.models import KbSchema

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyncReport:
    """What one corpus synchronisation did to the table."""

    inserted: int
    updated: int
    unchanged: int
    deleted: int

    @property
    def needs_embedding(self) -> int:
        """Rows whose vector was just discarded or never existed."""
        return self.inserted + self.updated


@dataclass(frozen=True)
class StoredChunk:
    """A chunk as it lives in the table, carrying the key needed to write back."""

    id: int
    chunk: Chunk


@dataclass(frozen=True)
class RetrievedChunk:
    """A passage returned by similarity search, with enough provenance to cite it."""

    doc_id: str
    heading: str | None
    content: str
    similarity: float


@dataclass(frozen=True)
class KbStats:
    """Row counts, for the CLI to report what state the knowledge base is in."""

    total: int
    embedded: int

    @property
    def pending(self) -> int:
        return self.total - self.embedded


def sync_chunks(engine: Engine, kb: KbSchema, chunks: Sequence[Chunk]) -> SyncReport:
    """Make the table match ``chunks`` exactly, preserving vectors of unchanged text.

    The corpus is the whole truth: positions no longer present are deleted, which
    covers both a removed document and one that lost a section. A chunk whose text
    changed has its vector discarded, because a stale vector pointing at new text
    is worse than a missing one - the missing one is skipped by search, the stale
    one silently poisons it.

    Raises:
        RagError: if two chunks share a ``(doc_id, chunk_index)`` position.
        DatabaseError: if the statements cannot be applied.
    """
    keys = [(chunk.doc_id, chunk.chunk_index) for chunk in chunks]
    if len(set(keys)) != len(keys):
        raise RagError("Duplicate (doc_id, chunk_index) in the corpus; positions must be unique")

    table = kb.kb_chunk
    rows = [
        {
            "doc_id": chunk.doc_id,
            "chunk_index": chunk.chunk_index,
            "heading": chunk.heading,
            "content": chunk.content,
            "content_hash": chunk.content_hash,
        }
        for chunk in chunks
    ]

    try:
        with engine.begin() as conn:
            removal = delete(table).where(tuple_(table.c.doc_id, table.c.chunk_index).not_in(keys))
            deleted = conn.execute(removal).rowcount
            if not rows:
                return SyncReport(inserted=0, updated=0, unchanged=0, deleted=deleted)

            statement = insert(table).values(rows)
            upsert = statement.on_conflict_do_update(
                index_elements=["doc_id", "chunk_index"],
                set_={
                    "heading": statement.excluded.heading,
                    "content": statement.excluded.content,
                    "content_hash": statement.excluded.content_hash,
                    "embedding": null(),
                    "embedding_model": null(),
                    "updated_at": func.now(),
                },
                # An unchanged hash leaves the row, and its vector, untouched.
                where=table.c.content_hash.is_distinct_from(statement.excluded.content_hash),
            )
            # ``xmax = 0`` on a returned row means it was inserted, not updated.
            # ``returning`` on a literal column leaves SQLAlchemy's TypeVar unbound.
            returned = cast(
                "CursorResult[tuple[bool]]",
                conn.execute(upsert.returning(literal_column("xmax = 0").label("inserted"))),
            )
            touched = [bool(row[0]) for row in returned]
    except SQLAlchemyError as exc:
        raise DatabaseError(
            f"Could not synchronise the knowledge base: {type(exc).__name__}"
        ) from exc

    inserted = sum(touched)
    report = SyncReport(
        inserted=inserted,
        updated=len(touched) - inserted,
        unchanged=len(rows) - len(touched),
        deleted=deleted,
    )
    logger.info(
        "Corpus synced: %d inserted, %d updated, %d unchanged, %d deleted",
        report.inserted,
        report.updated,
        report.unchanged,
        report.deleted,
    )
    return report


def pending_chunks(
    engine: Engine, kb: KbSchema, *, model_name: str, force: bool = False
) -> list[StoredChunk]:
    """Return the chunks that still need a vector from ``model_name``.

    A chunk qualifies when it has no vector or when its vector came from a
    different model, so swapping models inside one dimension self-heals on the
    next build without anyone remembering to pass ``--force``.

    Raises:
        DatabaseError: if the table cannot be read.
    """
    table = kb.kb_chunk
    statement = select(
        table.c.id, table.c.doc_id, table.c.chunk_index, table.c.heading, table.c.content
    ).order_by(table.c.doc_id, table.c.chunk_index)
    if not force:
        statement = statement.where(
            table.c.embedding.is_(None) | table.c.embedding_model.is_distinct_from(model_name)
        )
    try:
        with engine.connect() as conn:
            rows = conn.execute(statement).all()
    except SQLAlchemyError as exc:
        raise DatabaseError(f"Could not read the knowledge base: {type(exc).__name__}") from exc

    return [
        StoredChunk(
            id=row.id,
            chunk=Chunk(
                doc_id=row.doc_id,
                chunk_index=row.chunk_index,
                heading=row.heading,
                content=row.content,
            ),
        )
        for row in rows
    ]


def save_embeddings(
    engine: Engine,
    kb: KbSchema,
    vectors: Sequence[tuple[int, Sequence[float]]],
    *,
    model_name: str,
) -> int:
    """Write one batch of vectors and commit it on its own.

    Committing per batch is deliberate: these vectors already cost provider quota,
    so a failure on a later batch must not roll back the ones already paid for.

    Raises:
        RagError: if a vector does not match the schema's dimension.
        DatabaseError: if the rows cannot be written.
    """
    if not vectors:
        return 0
    for row_id, vector in vectors:
        if len(vector) != kb.dimension:
            raise RagError(
                f"Chunk {row_id} got a vector of length {len(vector)};"
                f" the schema expects {kb.dimension}"
            )

    table = kb.kb_chunk
    statement = (
        update(table)
        .where(table.c.id == bindparam("row_id"))
        .values(
            embedding=bindparam("vector", type_=table.c.embedding.type),
            embedding_model=model_name,
            updated_at=func.now(),
        )
    )
    parameters = [{"row_id": row_id, "vector": list(vector)} for row_id, vector in vectors]
    try:
        with engine.begin() as conn:
            conn.execute(statement, parameters)
    except SQLAlchemyError as exc:
        raise DatabaseError(f"Could not store embeddings: {type(exc).__name__}") from exc
    return len(parameters)


def search(
    engine: Engine, kb: KbSchema, vector: Sequence[float], *, top_k: int
) -> list[RetrievedChunk]:
    """Return the ``top_k`` passages closest to ``vector`` by cosine distance.

    Unembedded rows are excluded rather than ranked last: a NULL vector has no
    distance, and treating it as one would put unrelated text at the top.

    Raises:
        RagError: if the vector does not match the schema's dimension.
        DatabaseError: if the query fails.
    """
    if len(vector) != kb.dimension:
        raise RagError(f"Query vector has length {len(vector)}; the schema expects {kb.dimension}")
    if top_k <= 0:
        return []

    table = kb.kb_chunk
    distance = table.c.embedding.cosine_distance(list(vector)).label("distance")
    statement = (
        select(table.c.doc_id, table.c.heading, table.c.content, distance)
        .where(table.c.embedding.is_not(None))
        # ``id`` breaks ties, so equal distances come back in a stable order.
        .order_by(distance, table.c.id)
        .limit(top_k)
    )
    try:
        with engine.connect() as conn:
            rows = conn.execute(statement).all()
    except SQLAlchemyError as exc:
        raise DatabaseError(f"Knowledge-base search failed: {type(exc).__name__}") from exc

    return [
        RetrievedChunk(
            doc_id=row.doc_id,
            heading=row.heading,
            content=row.content,
            similarity=1.0 - float(row.distance),
        )
        for row in rows
    ]


def kb_stats(engine: Engine, kb: KbSchema) -> KbStats:
    """Count the stored chunks and how many of them carry a vector.

    Raises:
        DatabaseError: if the table cannot be read.
    """
    table = kb.kb_chunk
    statement = select(
        func.count().label("total"),
        func.count(table.c.embedding).label("embedded"),
    )
    try:
        with engine.connect() as conn:
            row = conn.execute(statement).one()
    except SQLAlchemyError as exc:
        raise DatabaseError(f"Could not count knowledge-base rows: {type(exc).__name__}") from exc
    return KbStats(total=int(row.total), embedded=int(row.embedded))
