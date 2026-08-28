"""Integration tests for ``ops.load_runs`` against a live PostgreSQL.

Marked ``integration`` and skipped automatically when no database is reachable.
Run them with ``docker compose up -d`` and ``pytest -m integration``.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

import pytest
from sqlalchemy import Engine, Row, func, select
from sqlalchemy.exc import IntegrityError

from rail_rag.core.exceptions import DatabaseError
from rail_rag.db.models import dim_date, load_runs
from rail_rag.db.run_log import (
    LoadCounts,
    fail_run,
    finish_run,
    new_run_id,
    start_run,
)
from rail_rag.db.schema import drop_schema, existing_ops_tables, missing_tables

pytestmark = pytest.mark.integration

_DIM_DATE_ROW = {
    "date_key": dt.date(2026, 8, 23),
    "year": 2026,
    "quarter": 3,
    "month": 8,
    "month_name": "August",
    "week_of_year": 34,
    "day_of_week": 1,
    "day_name": "Sunday",
    "is_weekend": True,
}


def _entry(engine: Engine, entry_id: int) -> Row[Any]:
    with engine.connect() as conn:
        return conn.execute(select(load_runs).where(load_runs.c.id == entry_id)).one()


def _dim_date_count(engine: Engine) -> int:
    with engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(dim_date)).scalar_one())


def test_create_schema_provisions_the_ops_table(clean_schema: Engine) -> None:
    assert "load_runs" in existing_ops_tables(clean_schema)
    assert missing_tables(clean_schema) == set()


def test_start_run_opens_a_running_entry(clean_schema: Engine) -> None:
    run_id = new_run_id()
    entry_id = start_run(
        clean_schema,
        run_id=run_id,
        table_name="dim_date",
        source_file="dim_date.parquet",
    )
    row = _entry(clean_schema, entry_id)
    assert row.run_id == run_id
    assert row.status == "running"
    assert row.finished_at is None
    assert row.rows_read == 0


def test_finish_run_records_the_tallies(clean_schema: Engine) -> None:
    entry_id = start_run(
        clean_schema,
        run_id=new_run_id(),
        table_name="fact_stop_event",
        source_file="fact_stop_event.parquet",
        date_key=dt.date(2026, 8, 23),
    )
    finish_run(
        clean_schema,
        entry_id,
        LoadCounts(rows_read=100, rows_rejected=2, rows_inserted=90, rows_updated=8),
    )
    row = _entry(clean_schema, entry_id)
    assert row.status == "succeeded"
    assert (row.rows_inserted, row.rows_updated, row.rows_rejected) == (90, 8, 2)
    assert row.date_key == dt.date(2026, 8, 23)
    assert row.finished_at is not None


def test_one_run_id_groups_several_tables(clean_schema: Engine) -> None:
    """load-dims touches three tables; a single aggregate row would hide which failed."""
    run_id = new_run_id()
    for name in ("dim_date", "dim_station", "dim_relation"):
        start_run(clean_schema, run_id=run_id, table_name=name, source_file=f"{name}.parquet")
    with clean_schema.connect() as conn:
        rows = conn.execute(select(load_runs).where(load_runs.c.run_id == run_id)).all()
    assert {row.table_name for row in rows} == {"dim_date", "dim_station", "dim_relation"}


def test_failed_run_survives_the_rollback_of_the_data_transaction(clean_schema: Engine) -> None:
    """The record of a failed load is exactly the row a shared transaction would erase."""
    entry_id = start_run(
        clean_schema,
        run_id=new_run_id(),
        table_name="dim_date",
        source_file="dim_date.parquet",
    )
    with pytest.raises(RuntimeError), clean_schema.begin() as conn:
        conn.execute(dim_date.insert().values(**_DIM_DATE_ROW))
        try:
            raise RuntimeError("promotion failed")
        except RuntimeError as exc:
            fail_run(clean_schema, entry_id, str(exc), LoadCounts(rows_read=1))
            raise

    assert _dim_date_count(clean_schema) == 0
    row = _entry(clean_schema, entry_id)
    assert row.status == "failed"
    assert row.error == "promotion failed"
    assert row.rows_read == 1


def test_long_error_is_truncated_not_rejected(clean_schema: Engine) -> None:
    entry_id = start_run(
        clean_schema,
        run_id=new_run_id(),
        table_name="dim_date",
        source_file="dim_date.parquet",
    )
    fail_run(clean_schema, entry_id, "x" * 5000)
    assert len(_entry(clean_schema, entry_id).error) == 2000


def test_closing_an_unknown_entry_raises(clean_schema: Engine) -> None:
    with pytest.raises(DatabaseError, match="not found"):
        finish_run(clean_schema, -1, LoadCounts())


def test_running_entry_must_not_carry_a_finish_time(clean_schema: Engine) -> None:
    """Guards the status/finished_at pair against drifting apart."""
    with pytest.raises(IntegrityError), clean_schema.begin() as conn:
        conn.execute(
            load_runs.insert().values(
                run_id=uuid.uuid4(),
                table_name="dim_date",
                source_file="dim_date.parquet",
                status="running",
                finished_at=dt.datetime.now(dt.UTC),
            )
        )


def test_dropping_gold_preserves_the_load_history(clean_schema: Engine) -> None:
    """A reset of the warehouse must not erase how it was built."""
    run_id = new_run_id()
    start_run(clean_schema, run_id=run_id, table_name="dim_date", source_file="dim_date.parquet")
    drop_schema(clean_schema)
    with clean_schema.connect() as conn:
        surviving = conn.execute(select(load_runs).where(load_runs.c.run_id == run_id)).all()
    assert len(surviving) == 1


def test_include_ops_drops_the_history_explicitly(clean_schema: Engine) -> None:
    drop_schema(clean_schema, include_ops=True)
    assert existing_ops_tables(clean_schema) == set()
