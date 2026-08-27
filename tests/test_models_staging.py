"""Structural tests for the staging table and its relationship to the fact.

Staging exists so that a malformed batch lands successfully and can be reported in
SQL. Any constraint added to it silently converts a data-quality report into a
driver exception, so its permissiveness is pinned here rather than left to a comment.
"""

from __future__ import annotations

from sqlalchemy import CheckConstraint

from rail_rag.db.models import fact_stop_event, stg_fact_stop_event

#: Columns the fact owns but the export does not supply.
_FACT_ONLY_COLUMNS = frozenset({"measured_arrivals", "loaded_at"})


def test_staging_columns_are_all_nullable() -> None:
    not_nullable = [column.name for column in stg_fact_stop_event.columns if not column.nullable]
    assert not_nullable == []


def test_staging_has_no_primary_key() -> None:
    assert list(stg_fact_stop_event.primary_key.columns) == []


def test_staging_has_no_foreign_keys() -> None:
    assert stg_fact_stop_event.foreign_keys == set()


def test_staging_has_no_check_constraints() -> None:
    checks = [c for c in stg_fact_stop_event.constraints if isinstance(c, CheckConstraint)]
    assert checks == []


def test_staging_has_no_indexes() -> None:
    assert stg_fact_stop_event.indexes == set()


def test_staging_mirrors_the_source_columns_of_the_fact() -> None:
    """A column added to one side and forgotten on the other is a silent data loss."""
    expected = [c.name for c in fact_stop_event.columns if c.name not in _FACT_ONLY_COLUMNS]
    assert [c.name for c in stg_fact_stop_event.columns] == expected


def test_measured_arrivals_never_reaches_staging() -> None:
    """It is derived in SQL on promotion, so it must not be transportable."""
    assert "measured_arrivals" not in stg_fact_stop_event.columns
    assert "measured_arrivals" in fact_stop_event.columns


def test_fact_keeps_the_constraints_staging_gives_up() -> None:
    """The strictness is not removed, only relocated to the destination table."""
    check_names = {c.name for c in fact_stop_event.constraints if isinstance(c, CheckConstraint)}
    assert "ck_fact_measured_arrivals_matches_punctual" in check_names
    assert fact_stop_event.foreign_keys != set()
    assert any(index.unique for index in fact_stop_event.indexes)


def test_delay_columns_carry_no_range_constraint() -> None:
    """Negative delays are early arrivals: valid data on both tables."""
    clauses = " ".join(
        str(c.sqltext) for c in fact_stop_event.constraints if isinstance(c, CheckConstraint)
    )
    assert "delay_arr_s" not in clauses
    assert "delay_dep_s" not in clauses
