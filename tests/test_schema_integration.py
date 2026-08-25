"""Integration tests for schema management against a live PostgreSQL.

Marked ``integration`` and skipped automatically when no database is reachable.
Run them with ``docker compose up -d`` and ``pytest -m integration``.
"""

import datetime as dt

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError

from rail_rag.db.models import GOLD_SCHEMA
from rail_rag.db.schema import create_schema, existing_tables, missing_tables, ping

pytestmark = pytest.mark.integration

_DIM_ROWS = (
    f"INSERT INTO {GOLD_SCHEMA}.dim_date"
    " (date_key, year, quarter, month, month_name, week_of_year, day_of_week,"
    "  day_name, is_weekend)"
    " VALUES ('2026-08-23', 2026, 3, 8, 'August', 34, 1, 'Sunday', true);"
    f"INSERT INTO {GOLD_SCHEMA}.dim_station (station_key, station_name)"
    " VALUES ('a' || repeat('0', 31), 'MOENSBERG');"
    f"INSERT INTO {GOLD_SCHEMA}.dim_relation (relation_key, relation)"
    " VALUES ('b' || repeat('0', 31), 'IC 01');"
)


def _insert_fact(conn: object, **overrides: object) -> None:
    """Insert one fact row, with sane defaults for everything not overridden."""
    row: dict[str, object] = {
        "date_key": dt.date(2026, 8, 23),
        "station_key": "a" + "0" * 31,
        "relation_key": "b" + "0" * 31,
        "train_no": 501,
        "planned_hour": 7,
        "delay_arr_s": -120,
        "delay_dep_s": 41,
        "dwell_delta_s": 0,
        "punctual_arrivals": 1,
        "measured_arrivals": 1,
        "stop_events": 1,
    }
    row.update(overrides)
    columns = ", ".join(row)
    values = ", ".join(f":{name}" for name in row)
    conn.execute(  # type: ignore[attr-defined]
        text(f"INSERT INTO {GOLD_SCHEMA}.fact_stop_event ({columns}) VALUES ({values})"),
        row,
    )


def test_ping_returns_the_server_version(postgres_engine: Engine) -> None:
    """A reachable database reports its version instead of raising."""
    assert "PostgreSQL" in ping(postgres_engine)


def test_create_schema_is_idempotent(clean_schema: Engine) -> None:
    """Re-running db-init on an initialised database is a no-op."""
    create_schema(clean_schema)
    assert missing_tables(clean_schema) == set()
    assert "fact_stop_event" in existing_tables(clean_schema)


def test_orphan_fact_row_is_rejected(clean_schema: Engine) -> None:
    """A station_key absent from the dimension must fail loudly, not load."""
    with clean_schema.begin() as conn:
        conn.execute(text(_DIM_ROWS))
    with pytest.raises(IntegrityError), clean_schema.begin() as conn:
        _insert_fact(conn, station_key="c" + "0" * 31)


def test_negative_delays_are_accepted(clean_schema: Engine) -> None:
    """Regression guard: early arrivals are valid data."""
    with clean_schema.begin() as conn:
        conn.execute(text(_DIM_ROWS))
        _insert_fact(conn, delay_arr_s=-1808, delay_dep_s=-1808)


def test_measured_arrivals_must_agree_with_punctual_arrivals(clean_schema: Engine) -> None:
    """Guards the SUM/SUM ratio against a desynchronised denominator."""
    with clean_schema.begin() as conn:
        conn.execute(text(_DIM_ROWS))
    with pytest.raises(IntegrityError), clean_schema.begin() as conn:
        _insert_fact(conn, punctual_arrivals=None, measured_arrivals=1)


def test_unmeasured_arrival_is_accepted(clean_schema: Engine) -> None:
    """2% of real rows have no measured arrival; they must still load."""
    with clean_schema.begin() as conn:
        conn.execute(text(_DIM_ROWS))
        _insert_fact(conn, punctual_arrivals=None, measured_arrivals=0, delay_arr_s=None)


def test_duplicate_grain_is_rejected(clean_schema: Engine) -> None:
    """The unique index mirrors the upstream MERGE key."""
    with clean_schema.begin() as conn:
        conn.execute(text(_DIM_ROWS))
        _insert_fact(conn)
    with pytest.raises(IntegrityError), clean_schema.begin() as conn:
        _insert_fact(conn, delay_arr_s=999)
