"""Create, drop and inspect the Gold schema in PostgreSQL.

Thin, idempotent wrappers around the ``MetaData`` in :mod:`rail_rag.db.models`.
Every function takes an ``Engine`` explicitly rather than reaching for global
settings, so tests can point them at a throwaway database.
"""

import logging

from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from rail_rag.core.exceptions import DatabaseError
from rail_rag.db.models import (
    GOLD_SCHEMA,
    OPS_SCHEMA,
    ORDERED_OPS_TABLES,
    ORDERED_TABLES,
    metadata,
    ops_metadata,
)

logger = logging.getLogger(__name__)


def ping(engine: Engine) -> str:
    """Check connectivity and return the server version.

    Raises:
        DatabaseError: if the database is unreachable or rejects the connection.
    """
    try:
        with engine.connect() as conn:
            version = conn.execute(text("SELECT version()")).scalar_one()
    except SQLAlchemyError as exc:
        raise DatabaseError(f"Could not reach the database: {type(exc).__name__}") from exc
    return str(version)


def create_schema(engine: Engine) -> None:
    """Create the Gold and ops namespaces and all tables, if absent.

    Idempotent: safe to run against an already-initialised database.

    Raises:
        DatabaseError: if the DDL cannot be applied.
    """
    try:
        with engine.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{GOLD_SCHEMA}"'))
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{OPS_SCHEMA}"'))
            metadata.create_all(bind=conn, checkfirst=True)
            ops_metadata.create_all(bind=conn, checkfirst=True)
    except SQLAlchemyError as exc:
        raise DatabaseError(f"Could not create the Gold schema: {type(exc).__name__}") from exc
    logger.info(
        "Schemas ready (%d Gold tables, %d ops tables)",
        len(ORDERED_TABLES),
        len(ORDERED_OPS_TABLES),
    )


def drop_schema(engine: Engine, *, include_ops: bool = False) -> None:
    """Drop every Gold table and the namespace itself.

    ``ops`` is preserved unless ``include_ops`` is set: the load history outlives
    the data it describes, and a reset must not erase the audit trail.

    Raises:
        DatabaseError: if the DDL cannot be applied.
    """
    try:
        with engine.begin() as conn:
            metadata.drop_all(bind=conn, checkfirst=True)
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{GOLD_SCHEMA}" CASCADE'))
            if include_ops:
                ops_metadata.drop_all(bind=conn, checkfirst=True)
                conn.execute(text(f'DROP SCHEMA IF EXISTS "{OPS_SCHEMA}" CASCADE'))
    except SQLAlchemyError as exc:
        raise DatabaseError(f"Could not drop the Gold schema: {type(exc).__name__}") from exc
    logger.warning("Gold schema dropped (ops included: %s)", include_ops)


def existing_tables(engine: Engine) -> set[str]:
    """Return the names of the Gold tables currently present."""
    return _tables_in(engine, GOLD_SCHEMA)


def existing_ops_tables(engine: Engine) -> set[str]:
    """Return the names of the ops tables currently present."""
    return _tables_in(engine, OPS_SCHEMA)


def _tables_in(engine: Engine, schema: str) -> set[str]:
    inspector = inspect(engine)
    if schema not in inspector.get_schema_names():
        return set()
    return set(inspector.get_table_names(schema=schema))


def missing_tables(engine: Engine) -> set[str]:
    """Return the Gold and ops tables that are expected but absent.

    An empty set means the database is ready for the loader; this is what the
    API's readiness probe will consume in Stage 3C.
    """
    missing_gold = {table.name for table in ORDERED_TABLES} - existing_tables(engine)
    missing_ops = {table.name for table in ORDERED_OPS_TABLES} - existing_ops_tables(engine)
    return missing_gold | missing_ops
