"""Tests for the Gold star schema definition.

These assert the modelling decisions themselves, without a database: the grain,
the nullability the real export exhibits, and the invariants that protect the
punctuality ratio.
"""

from sqlalchemy import Table

from rail_rag.db.models import (
    GOLD_SCHEMA,
    ORDERED_TABLES,
    dim_relation,
    dim_station,
    fact_stop_event,
    metadata,
    stg_fact_stop_event,
)


def _check_names(table: Table) -> set[str]:
    """Return the names of a table's CHECK constraints."""
    return {
        c.name for c in table.constraints if isinstance(c.name, str) and c.name.startswith("ck_")
    }


def test_all_tables_live_in_the_gold_schema() -> None:
    """Gold tables must not leak into the public namespace."""
    assert all(table.schema == GOLD_SCHEMA for table in ORDERED_TABLES)
    assert set(metadata.tables) == {f"{GOLD_SCHEMA}.{t.name}" for t in ORDERED_TABLES}


def test_dimensions_are_created_before_the_fact() -> None:
    """Foreign keys only resolve if dimensions come first."""
    order = [table.name for table in ORDERED_TABLES]
    assert order.index("fact_stop_event") > order.index("dim_date")
    assert order.index("fact_stop_event") > order.index("dim_station")
    assert order.index("fact_stop_event") > order.index("dim_relation")


def test_fact_grain_is_unique_on_the_upstream_merge_key() -> None:
    """The grain must match the Databricks MERGE key, or a reload duplicates."""
    unique_indexes = [idx for idx in fact_stop_event.indexes if idx.unique]
    assert len(unique_indexes) == 1
    assert [c.name for c in unique_indexes[0].columns] == [
        "date_key",
        "station_key",
        "train_no",
    ]


def test_measures_that_the_export_never_leaves_null() -> None:
    """Keys and stop_events are complete in every sampled row."""
    for column in ("date_key", "station_key", "relation_key", "train_no", "stop_events"):
        assert not fact_stop_event.c[column].nullable, column


def test_measures_that_the_export_does_leave_null() -> None:
    """Observed as NULL in the real Gold sample; a NOT NULL here would fail the load."""
    for column in ("planned_hour", "delay_arr_s", "delay_dep_s", "dwell_delta_s"):
        assert fact_stop_event.c[column].nullable, column


def test_measured_arrivals_is_derived_and_never_null() -> None:
    """The denominator of the punctuality ratio must be a real additive measure."""
    assert fact_stop_event.c["punctual_arrivals"].nullable
    assert not fact_stop_event.c["measured_arrivals"].nullable
    assert "ck_fact_measured_arrivals_matches_punctual" in _check_names(fact_stop_event)


def test_delay_columns_have_no_range_check() -> None:
    """Negative delays are early arrivals: valid data, not errors."""
    conditions = " ".join(
        str(c.sqltext) for c in fact_stop_event.constraints if hasattr(c, "sqltext")
    )
    assert "delay_arr_s" not in conditions
    assert "delay_dep_s" not in conditions


def test_nullable_dimension_attributes_match_the_source() -> None:
    """ptcar_no and relation_direction are genuinely absent in production rows."""
    assert dim_station.c["ptcar_no"].nullable
    assert dim_relation.c["relation_direction"].nullable


def test_staging_table_is_unconstrained() -> None:
    """Staging must accept a bad export so the loader can diagnose it in SQL."""
    assert not stg_fact_stop_event.foreign_keys
    assert not _check_names(stg_fact_stop_event)
    assert stg_fact_stop_event.primary_key.columns.keys() == []


def test_staging_mirrors_the_fact_source_columns() -> None:
    """The loader can promote staging rows into the fact table without column mapping."""    
    fact_only = {"loaded_at", "measured_arrivals"}
    assert set(stg_fact_stop_event.c.keys()) == set(fact_stop_event.c.keys()) - fact_only