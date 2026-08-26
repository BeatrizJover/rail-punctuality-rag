"""Shared test fixtures."""
 
from __future__ import annotations 
import os
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote_plus
 
import pytest
from dotenv import dotenv_values
from sqlalchemy import Engine, create_engine
from sqlalchemy.exc import OperationalError
 
from rail_rag.core.config import get_settings
from rail_rag.db.schema import create_schema, drop_schema
 
_TEST_DSN_VAR = "TEST_DATABASE_URL"
_TEST_DB_SUFFIX = "_test"
_REPO_ROOT = Path(__file__).resolve().parent.parent 
_UNREACHABLE_SIGNALS = (
    "connection refused",
    "could not connect",
    "could not translate host",
    "failed to resolve host",
    "timeout expired",
)
  
def _default_test_dsn() -> str:
    """Derive the integration-test DSN from the repository's own ``.env``."""
    values = dotenv_values(_REPO_ROOT / ".env")
    user = values.get("RAIL_RAG_DB_USER") or "rail_rag"
    password = values.get("RAIL_RAG_DB_PASSWORD") or ""
    host = values.get("RAIL_RAG_DB_HOST") or "localhost"
    port = values.get("RAIL_RAG_DB_PORT") or "5432"
    name = (values.get("RAIL_RAG_DB_NAME") or "rail_rag") + _TEST_DB_SUFFIX
    return f"postgresql+psycopg://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{name}"
 
 
@pytest.fixture(autouse=True)
def isolated_settings_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run each test in a clean directory with no ``.env`` and no app vars."""
    for key in list(os.environ):
        if key.startswith("RAIL_RAG_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
 
 
@pytest.fixture(scope="session")
def default_test_dsn() -> str:
    """Expose the derived DSN so its guarantees can be asserted in a test."""
    return _default_test_dsn()
 
 
@pytest.fixture(scope="session")
def postgres_engine() -> Iterator[Engine]:
    """Yield an engine against a live PostgreSQL, or skip the test.
    
    Skipping rather than failing keeps ``pytest`` green on a clean checkout and
    in CI, where no database is provisioned. 
    """
    dsn = os.environ.get(_TEST_DSN_VAR) or _default_test_dsn()
    engine = create_engine(dsn, pool_pre_ping=True, future=True)
    try:
        with engine.connect():
            pass
    except OperationalError as exc:
        engine.dispose()
        if not any(signal in str(exc).lower() for signal in _UNREACHABLE_SIGNALS):
            raise
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

