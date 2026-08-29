"""Create, drop and inspect the knowledge-base schema.

Mirrors the shape of :mod:`rail_rag.db.schema` — explicit engine, idempotent
DDL, errors wrapped so the DSN never reaches a log — but keeps its own
functions rather than extending the Gold ones. The Gold schema knows nothing
about the RAG layer, and a one-directional dependency is worth more than the
handful of lines it costs.
"""

from __future__ import annotations

import logging

from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from rail_rag.core.exceptions import DatabaseError
from rail_rag.rag.store.models import RAG_SCHEMA, KbSchema

logger = logging.getLogger(__name__)


def create_kb_schema(engine: Engine, kb: KbSchema) -> None:
    """Create the ``vector`` extension, the namespace and the tables, if absent.

    Idempotent. The extension is per-database, so a test database needs it just
    as much as the development one.

    Raises:
        DatabaseError: if the DDL cannot be applied.
    """
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{RAG_SCHEMA}"'))
            kb.metadata.create_all(bind=conn, checkfirst=True)
    except SQLAlchemyError as exc:
        raise DatabaseError(
            f"Could not create the knowledge-base schema: {type(exc).__name__}."
            " The vector extension requires a superuser and the pgvector image."
        ) from exc
    logger.info("Knowledge-base schema ready (dimension %d)", kb.dimension)


def drop_kb_schema(engine: Engine) -> None:
    """Drop the knowledge-base namespace.

    Never called by ``db-drop``: rebuilding Gold must not discard embeddings
    that cost provider quota to produce.

    Raises:
        DatabaseError: if the DDL cannot be applied.
    """
    try:
        with engine.begin() as conn:
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{RAG_SCHEMA}" CASCADE'))
    except SQLAlchemyError as exc:
        raise DatabaseError(
            f"Could not drop the knowledge-base schema: {type(exc).__name__}"
        ) from exc
    logger.warning("Knowledge-base schema dropped")


def kb_schema_exists(engine: Engine) -> bool:
    """Return whether the knowledge-base table is present."""
    inspector = inspect(engine)
    if RAG_SCHEMA not in inspector.get_schema_names():
        return False
    return "kb_chunk" in inspector.get_table_names(schema=RAG_SCHEMA)


def stored_dimension(engine: Engine) -> int | None:
    """Return the vector width the table was created with, or None if absent.

    Read from the column type rather than from a row, so it answers even for an
    empty corpus.

    Raises:
        DatabaseError: if the catalogue cannot be queried.
    """
    statement = text(
        "SELECT format_type(a.atttypid, a.atttypmod)"
        " FROM pg_attribute a"
        " JOIN pg_class c ON c.oid = a.attrelid"
        " JOIN pg_namespace n ON n.oid = c.relnamespace"
        " WHERE n.nspname = :schema AND c.relname = 'kb_chunk' AND a.attname = 'embedding'"
    )
    try:
        with engine.connect() as conn:
            rendered = conn.execute(statement, {"schema": RAG_SCHEMA}).scalar()
    except SQLAlchemyError as exc:
        raise DatabaseError(f"Could not inspect the vector column: {type(exc).__name__}") from exc
    if rendered is None:
        return None
    # format_type renders the column as "vector(768)".
    _, _, tail = str(rendered).partition("(")
    width = tail.rstrip(")")
    return int(width) if width.isdigit() else None


def verify_dimension(engine: Engine, kb: KbSchema) -> None:
    """Fail fast when the configured model no longer matches the stored vectors.

    Cosine distance between vectors from different models is arithmetically
    valid and semantically meaningless, so a silent mismatch degrades answers
    without producing a single error.

    Raises:
        DatabaseError: if the widths differ or the table is missing.
    """
    stored = stored_dimension(engine)
    if stored is None:
        raise DatabaseError("Knowledge-base table is missing. Run 'kb-init'.")
    if stored != kb.dimension:
        raise DatabaseError(
            f"Embedding dimension mismatch: table has {stored}, configuration expects"
            f" {kb.dimension}. Changing the embedding model requires rebuilding the"
            " knowledge base with 'kb-drop' then 'kb-init'."
        )
