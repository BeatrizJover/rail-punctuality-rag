"""Shared test fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.exc import SQLAlchemyError

from rail_rag.db.schema import create_schema, drop_schema

_TEST_DSN_VAR = "TEST_DATABASE_URL"
_DEFAULT_TEST_DSN = "postgresql+psycopg://rail_rag:rail_rag@localhost:5432/rail_rag_test"


@pytest.fixture(autouse=True)
def isolated_settings_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run each test in a clean directory with no ``.env`` and no app vars."""
    for key in list(os.environ):
        if key.startswith("RAIL_RAG_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope="session")
def postgres_engine() -> Iterator[Engine]:
    """Yield an engine against a live PostgreSQL, or skip the test.

    Skipping rather than failing keeps ``pytest`` green on a clean checkout and
    in CI, where no database is provisioned. Run ``docker compose up -d`` to
    exercise these tests locally.
    """
    dsn = os.environ.get(_TEST_DSN_VAR, _DEFAULT_TEST_DSN)
    engine = create_engine(dsn, pool_pre_ping=True, future=True)
    try:
        with engine.connect():
            pass
    except SQLAlchemyError:
        engine.dispose()
        pytest.skip(f"No PostgreSQL reachable; set {_TEST_DSN_VAR} or start docker compose")
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def clean_schema(postgres_engine: Engine) -> Iterator[Engine]:
    """Provide an engine whose Gold schema has just been created from scratch."""
    drop_schema(postgres_engine)
    create_schema(postgres_engine)
    try:
        yield postgres_engine
    finally:
        drop_schema(postgres_engine)
