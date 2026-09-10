"""Tests for the fact table's monthly range partitions.

The unit half pins the bounds arithmetic. The integration half pins what
PostgreSQL actually does with them: the constraints the parent keeps, the
rows it refuses, and the pruning that is the whole point of partitioning.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError

from rail_rag.db.models import GOLD_SCHEMA, fact_stop_event
from rail_rag.db.partitions import (
    BASELINE_END,
    BASELINE_START,
    ensure_partitions,
    existing_partitions,
    monthly_bounds,
    partition_name,
)

_DIM_ROWS = f"""
INSERT INTO {GOLD_SCHEMA}.dim_date
  (date_key, year, quarter, month, month_name, week_of_year,
   day_of_week, day_name, is_weekend)
VALUES ('2026-08-23', 2026, 3, 8, 'August', 34, 7, 'Sunday', true);
INSERT INTO {GOLD_SCHEMA}.dim_station (station_key) VALUES (repeat('a', 32));
INSERT INTO {GOLD_SCHEMA}.dim_relation (relation_key) VALUES (repeat('b', 32));
"""

_INSERT_FACT = f"""
INSERT INTO {GOLD_SCHEMA}.fact_stop_event
  (date_key, station_key, relation_key, train_no, stop_events, measured_arrivals)
VALUES (:date_key, :station_key, repeat('b', 32), :train_no, 1, 0)
"""


# --- bounds arithmetic -----------------------------------------------------


def test_a_single_month_yields_one_partition() -> None:
    bounds = monthly_bounds(dt.date(2026, 8, 1), dt.date(2026, 9, 1))
    assert bounds == [("fact_stop_event_2026_08", dt.date(2026, 8, 1), dt.date(2026, 9, 1))]


def test_bounds_are_contiguous_and_half_open() -> None:
    """A gap or an overlap between children is a load that silently loses rows."""
    bounds = monthly_bounds(dt.date(2025, 11, 1), dt.date(2026, 3, 1))
    for (_, _, upper), (_, next_lower, _) in zip(bounds, bounds[1:], strict=False):
        assert upper == next_lower


def test_a_partial_month_still_gets_its_whole_partition() -> None:
    """The export stops mid-month; that month must be fully covered anyway."""
    bounds = monthly_bounds(dt.date(2026, 9, 7), dt.date(2026, 9, 8))
    assert bounds == [("fact_stop_event_2026_09", dt.date(2026, 9, 1), dt.date(2026, 10, 1))]


def test_the_year_boundary_rolls_over() -> None:
    bounds = monthly_bounds(dt.date(2026, 12, 1), dt.date(2027, 2, 1))
    assert [name for name, _, _ in bounds] == [
        "fact_stop_event_2026_12",
        "fact_stop_event_2027_01",
    ]


def test_an_empty_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be after"):
        monthly_bounds(dt.date(2026, 1, 1), dt.date(2026, 1, 1))


def test_the_baseline_window_covers_the_real_export() -> None:
    """Gold spans 2024-01-01 to 2026-09-07; db-init must not leave it homeless."""
    names = {name for name, _, _ in monthly_bounds(BASELINE_START, BASELINE_END)}
    assert partition_name(dt.date(2024, 1, 1)) in names
    assert partition_name(dt.date(2026, 9, 7)) in names
    assert len(names) == 36


def test_the_fact_declares_its_partitioning() -> None:
    """Without this the parent is an ordinary table and every child DDL fails."""
    options = fact_stop_event.dialect_options["postgresql"]
    assert options["partition_by"] == "RANGE (date_key)"


def test_the_unique_index_leads_with_the_partition_key() -> None:
    """PostgreSQL rejects a unique index on a partitioned table without it."""
    unique = next(index for index in fact_stop_event.indexes if index.unique)
    assert [column.name for column in unique.columns][0] == "date_key"


# --- live PostgreSQL -------------------------------------------------------


@pytest.mark.integration
def test_db_init_creates_the_partitioned_parent(clean_schema: Engine) -> None:
    with clean_schema.connect() as conn:
        relkind = conn.execute(
            text("SELECT relkind FROM pg_class WHERE relname = 'fact_stop_event'")
        ).scalar_one()
    assert relkind == "p"


@pytest.mark.integration
def test_db_init_creates_the_baseline_children(clean_schema: Engine) -> None:
    assert len(existing_partitions(clean_schema)) == 36


@pytest.mark.integration
def test_creating_partitions_again_is_a_no_op(clean_schema: Engine) -> None:
    """db-init runs on an initialised database as often as anyone types it."""
    with clean_schema.begin() as conn:
        ensure_partitions(conn, BASELINE_START, BASELINE_END)
    assert len(existing_partitions(clean_schema)) == 36


@pytest.mark.integration
def test_the_unique_index_reaches_every_child(clean_schema: Engine) -> None:
    """Grain uniqueness lives in the children; the parent alone enforces nothing."""
    with clean_schema.connect() as conn:
        count = conn.execute(
            text(
                "SELECT count(*) FROM pg_index"
                " JOIN pg_class ON pg_class.oid = pg_index.indrelid"
                " WHERE pg_class.relname LIKE 'fact_stop_event\\_%' AND indisunique"
            )
        ).scalar_one()
    assert count == 36


@pytest.mark.integration
def test_a_date_outside_every_partition_is_rejected(clean_schema: Engine) -> None:
    """No DEFAULT partition on purpose: an unplanned date must fail loudly."""
    with clean_schema.begin() as conn:
        conn.execute(text(_DIM_ROWS))
    with pytest.raises(IntegrityError, match="no partition"), clean_schema.begin() as conn:
        conn.execute(
            text(_INSERT_FACT),
            {"date_key": dt.date(2030, 1, 1), "station_key": "a" * 32, "train_no": 1},
        )


@pytest.mark.integration
def test_foreign_keys_still_bite_from_inside_a_child(clean_schema: Engine) -> None:
    """Referential integrity survives the move to a partitioned parent."""
    with clean_schema.begin() as conn:
        conn.execute(text(_DIM_ROWS))
    with pytest.raises(IntegrityError, match="fk_fact_station"), clean_schema.begin() as conn:
        conn.execute(
            text(_INSERT_FACT),
            {"date_key": dt.date(2026, 8, 23), "station_key": "c" * 32, "train_no": 1},
        )


@pytest.mark.integration
def test_a_row_lands_in_the_child_for_its_month(clean_schema: Engine) -> None:
    with clean_schema.begin() as conn:
        conn.execute(text(_DIM_ROWS))
        conn.execute(
            text(_INSERT_FACT),
            {"date_key": dt.date(2026, 8, 23), "station_key": "a" * 32, "train_no": 1},
        )
    with clean_schema.connect() as conn:
        rows = conn.execute(
            text(f"SELECT count(*) FROM {GOLD_SCHEMA}.fact_stop_event_2026_08")
        ).scalar_one()
    assert rows == 1


@pytest.mark.integration
def test_a_date_predicate_prunes_to_one_child(clean_schema: Engine) -> None:
    """The reason to partition at all: a scoped question reads one month, not three years."""
    with clean_schema.connect() as conn:
        plan = "\n".join(
            conn.execute(
                text(
                    f"EXPLAIN SELECT count(*) FROM {GOLD_SCHEMA}.fact_stop_event"
                    " WHERE date_key >= DATE '2026-08-01' AND date_key < DATE '2026-09-01'"
                )
            ).scalars()
        )
    assert "fact_stop_event_2026_08" in plan
    assert "fact_stop_event_2026_07" not in plan
    assert "fact_stop_event_2025_08" not in plan
