"""Structural tests for ``ops.load_runs``.

The provenance table lives in its own schema and its own ``MetaData`` so that
dropping Gold cannot take the load history with it.
"""

from __future__ import annotations

from sqlalchemy import CheckConstraint

from rail_rag.db.models import (
    GOLD_SCHEMA,
    OPS_SCHEMA,
    ORDERED_OPS_TABLES,
    ORDERED_TABLES,
    load_runs,
    metadata,
    ops_metadata,
)


def test_load_runs_lives_in_the_ops_schema() -> None:
    assert load_runs.schema == OPS_SCHEMA
    assert OPS_SCHEMA != GOLD_SCHEMA


def test_load_runs_is_not_part_of_the_gold_metadata() -> None:
    """A drop of the Gold MetaData must not reach the audit trail."""
    assert f"{OPS_SCHEMA}.load_runs" not in metadata.tables
    assert load_runs not in ORDERED_TABLES


def test_ops_metadata_holds_only_the_ops_tables() -> None:
    assert set(ops_metadata.tables.values()) == set(ORDERED_OPS_TABLES)


def test_counts_are_not_nullable_and_default_to_zero() -> None:
    """An unset tally would read as 'unknown' and defeat the point of the row."""
    for name in ("rows_read", "rows_rejected", "rows_inserted", "rows_updated"):
        column = load_runs.c[name]
        assert not column.nullable
        assert column.server_default is not None


def test_status_is_constrained_to_the_known_values() -> None:
    clauses = " ".join(
        str(c.sqltext) for c in load_runs.constraints if isinstance(c, CheckConstraint)
    )
    assert "running" in clauses
    assert "succeeded" in clauses
    assert "failed" in clauses


def test_run_id_is_indexed() -> None:
    """Grouping a whole invocation is the common query, so it must not scan."""
    indexed = {tuple(column.name for column in index.columns) for index in load_runs.indexes}
    assert ("run_id",) in indexed


def test_date_key_is_nullable() -> None:
    """Dimension loads have no service date; only the fact is incremental."""
    assert load_runs.c.date_key.nullable
