"""Database engine and session management (SQLAlchemy)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from rail_rag.core.config import Settings, get_settings
from rail_rag.core.exceptions import DatabaseError

logger = logging.getLogger(__name__)


def create_db_engine(settings: Settings | None = None) -> Engine:
    """Create a SQLAlchemy engine from settings."""
    settings = settings or get_settings()
    try:
        engine = create_engine(settings.database_url, pool_pre_ping=True)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise DatabaseError("failed to create the database engine") from exc

    logger.info(
        "database engine created (host=%s port=%s db=%s)",
        settings.db_host,
        settings.db_port,
        settings.db_name,
    )
    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a session factory bound to ``engine``."""
    return sessionmaker(bind=engine, expire_on_commit=False, class_=Session)


@contextmanager
def session_scope(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """
    Provide a transactional scope around a series of operations.
    Commits on success, rolls back on any exception, and always closes the session.
    """
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
