"""Tests for application settings, including a credential-leak regression."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from rail_rag.core.config import Settings


def test_defaults() -> None:
    settings = Settings()
    assert settings.db_host == "localhost"
    assert settings.db_port == 5432
    assert settings.db_name == "rail_rag"


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAIL_RAG_DB_HOST", "db.internal")
    monkeypatch.setenv("RAIL_RAG_DB_PORT", "6543")
    settings = Settings()
    assert settings.db_host == "db.internal"
    assert settings.db_port == 6543


def test_database_url_is_well_formed() -> None:
    settings = Settings(
        db_user="alice",
        db_password=SecretStr("s3cr3t"),
        db_host="localhost",
        db_port=5432,
        db_name="rail_rag",
    )
    assert settings.database_url == "postgresql+psycopg://alice:s3cr3t@localhost:5432/rail_rag"


def test_password_is_not_leaked_by_repr() -> None:
    """Regression: the DB password must never appear in the settings repr.

    A previous version exposed the password through ``repr`` / ``model_dump``.
    """
    secret = "super-secret-value"
    settings = Settings(db_password=SecretStr(secret))

    assert secret not in repr(settings)
    assert secret not in str(settings)
    assert secret not in str(settings.model_dump())
    # The value is still retrievable when explicitly requested.
    assert settings.db_password.get_secret_value() == secret
