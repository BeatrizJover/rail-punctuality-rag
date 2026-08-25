"""Tests for engine and session-factory construction (no live connection)."""

from __future__ import annotations

from pydantic import SecretStr
from sqlalchemy import Engine

from rail_rag.core.config import Settings
from rail_rag.db.engine import create_db_engine, create_session_factory


def _settings() -> Settings:
    return Settings(db_password=SecretStr("pw"))


def test_create_db_engine_returns_postgres_engine() -> None:
    engine = create_db_engine(_settings())
    assert isinstance(engine, Engine)
    assert engine.url.get_backend_name() == "postgresql"
    assert engine.url.database == "rail_rag"
    # SQLAlchemy masks the password in the URL's string form.
    assert "pw" not in str(engine.url)


def test_session_factory_is_bound_to_engine() -> None:
    engine = create_db_engine(_settings())
    factory = create_session_factory(engine)
    session = factory()
    try:
        assert session.get_bind() is engine
    finally:
        session.close()
