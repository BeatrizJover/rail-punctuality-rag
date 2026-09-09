"""Integration tests for the knowledge-base schema.

The isolation tests matter most: three schemas now share one database, and the
whole point of separate MetaData instances is that resetting one leaves the
others intact.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, inspect

from rail_rag.core.exceptions import DatabaseError
from rail_rag.db.schema import create_schema, drop_schema
from rail_rag.rag.store.models import RAG_SCHEMA, build_kb_schema
from rail_rag.rag.store.schema import (
    create_kb_schema,
    drop_kb_schema,
    kb_schema_exists,
    stored_dimension,
    verify_dimension,
)

pytestmark = pytest.mark.integration

DIMENSION = 768


@pytest.fixture
def kb_engine(postgres_engine: Engine) -> Iterator[Engine]:
    """An engine whose rag schema has just been created from scratch."""
    drop_kb_schema(postgres_engine)
    create_kb_schema(postgres_engine, build_kb_schema(DIMENSION))
    try:
        yield postgres_engine
    finally:
        drop_kb_schema(postgres_engine)


def test_schema_is_created(kb_engine: Engine) -> None:
    assert kb_schema_exists(kb_engine)
    assert "kb_chunk" in inspect(kb_engine).get_table_names(schema=RAG_SCHEMA)


def test_creation_is_idempotent(kb_engine: Engine) -> None:
    create_kb_schema(kb_engine, build_kb_schema(DIMENSION))
    assert kb_schema_exists(kb_engine)


def test_drop_is_idempotent(postgres_engine: Engine) -> None:
    drop_kb_schema(postgres_engine)
    drop_kb_schema(postgres_engine)
    assert not kb_schema_exists(postgres_engine)


def test_stored_dimension_is_readable_from_an_empty_table(kb_engine: Engine) -> None:
    """Read from the column type, so it answers before any chunk is embedded."""
    assert stored_dimension(kb_engine) == DIMENSION


def test_stored_dimension_is_none_without_the_table(postgres_engine: Engine) -> None:
    drop_kb_schema(postgres_engine)
    assert stored_dimension(postgres_engine) is None


def test_verify_dimension_accepts_a_match(kb_engine: Engine) -> None:
    verify_dimension(kb_engine, build_kb_schema(DIMENSION))


def test_verify_dimension_rejects_a_mismatch(kb_engine: Engine) -> None:
    """Cosine distance across two embedding spaces is valid arithmetic and nonsense."""
    with pytest.raises(DatabaseError, match="dimension mismatch"):
        verify_dimension(kb_engine, build_kb_schema(1536))


def test_verify_dimension_reports_a_missing_table(postgres_engine: Engine) -> None:
    drop_kb_schema(postgres_engine)
    with pytest.raises(DatabaseError, match="kb-init"):
        verify_dimension(postgres_engine, build_kb_schema(DIMENSION))


def test_zero_dimension_is_rejected() -> None:
    with pytest.raises(ValueError, match="dimension"):
        build_kb_schema(0)


# --- isolation between the three schemas -----------------------------------


def test_dropping_gold_leaves_the_knowledge_base_intact(kb_engine: Engine) -> None:
    """Embeddings cost provider quota; a Gold reset must not discard them."""
    create_schema(kb_engine)
    drop_schema(kb_engine)
    assert kb_schema_exists(kb_engine)
    assert stored_dimension(kb_engine) == DIMENSION


def test_dropping_the_knowledge_base_leaves_gold_intact(kb_engine: Engine) -> None:
    create_schema(kb_engine)
    try:
        drop_kb_schema(kb_engine)
        assert "fact_stop_event" in inspect(kb_engine).get_table_names(schema="gold")
    finally:
        drop_schema(kb_engine)
