"""Create, drop and inspect the Gold schema in PostgreSQL.

Thin, idempotent wrappers around the ``MetaData`` in :mod:`rail_rag.db.models`.
Every function takes an ``Engine`` explicitly rather than reaching for global
settings, so tests can point them at a throwaway database.
"""

import logging

from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from rail_rag.core.exceptions import DatabaseError
from rail_rag.db.models import GOLD_SCHEMA, ORDERED_TABLES, metadata

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
    """Create the Gold namespace and all tables, if absent.

    Idempotent: safe to run against an already-initialised database.

    Raises:
        DatabaseError: if the DDL cannot be applied.
    """
    try:
        with engine.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{GOLD_SCHEMA}"'))
            metadata.create_all(bind=conn, checkfirst=True)
    except SQLAlchemyError as exc:
        raise DatabaseError(f"Could not create the Gold schema: {type(exc).__name__}") from exc
    logger.info("Gold schema ready (%d tables)", len(ORDERED_TABLES))


def drop_schema(engine: Engine) -> None:
    """Drop every Gold table and the namespace itself.

    Destructive; exposed for tests and for a clean local reset.

    Raises:
        DatabaseError: if the DDL cannot be applied.
    """
    try:
        with engine.begin() as conn:
            metadata.drop_all(bind=conn, checkfirst=True)
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{GOLD_SCHEMA}" CASCADE'))
    except SQLAlchemyError as exc:
        raise DatabaseError(f"Could not drop the Gold schema: {type(exc).__name__}") from exc
    logger.warning("Gold schema dropped")


def existing_tables(engine: Engine) -> set[str]:
    """Return the names of the Gold tables currently present."""
    inspector = inspect(engine)
    if GOLD_SCHEMA not in inspector.get_schema_names():
        return set()
    return set(inspector.get_table_names(schema=GOLD_SCHEMA))


def missing_tables(engine: Engine) -> set[str]:
    """Return the Gold tables that are expected but absent.

    An empty set means the database is ready for the loader; this is what the
    API's readiness probe will consume in Stage 3C.
    """
    return {table.name for table in ORDERED_TABLES} - existing_tables(engine)
