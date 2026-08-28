"""Integration tests for the loader against a live PostgreSQL.

Happy path uses the synthetic export, which is referentially coherent by construction.
Orphan detection uses the real converted samples, which are three independent cuts and
therefore contain fact keys with no matching dimension row.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, FromClause, func, select

from rail_rag.db.models import (
    dim_date,
    dim_relation,
    dim_station,
    fact_stop_event,
    load_runs,
    stg_fact_stop_event,
)
from rail_rag.db.run_log import new_run_id
from rail_rag.ingestion.loader import load_dimensions, load_fact
from rail_rag.ingestion.validation import DataQualityError

pytestmark = pytest.mark.integration


def _count(engine: Engine, table: FromClause) -> int:
    with engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(table)).scalar_one())


def _fact_rows(engine: Engine) -> list[dict[str, Any]]:
    with engine.connect() as conn:
        return [dict(row) for row in conn.execute(select(fact_stop_event)).mappings()]


def test_load_dimensions_inserts_then_updates(
    clean_schema: Engine, synthetic_gold_dir: Path
) -> None:
    """The second run must update in place, not duplicate or fail on the primary key."""
    first = load_dimensions(clean_schema, synthetic_gold_dir)
    assert first["dim_station"].rows_inserted == 20
    assert first["dim_station"].rows_updated == 0

    second = load_dimensions(clean_schema, synthetic_gold_dir)
    assert second["dim_station"].rows_inserted == 0
    assert second["dim_station"].rows_updated == 20
    assert _count(clean_schema, dim_station) == 20


def test_load_dimensions_covers_all_three(clean_schema: Engine, synthetic_gold_dir: Path) -> None:
    load_dimensions(clean_schema, synthetic_gold_dir)
    assert _count(clean_schema, dim_date) == 3
    assert _count(clean_schema, dim_station) == 20
    assert _count(clean_schema, dim_relation) == 7


def test_happy_path_promotes_every_row(clean_schema: Engine, synthetic_gold_dir: Path) -> None:
    load_dimensions(clean_schema, synthetic_gold_dir)
    counts = load_fact(clean_schema, synthetic_gold_dir)
    assert counts.rows_rejected == 0
    assert counts.rows_inserted == 60
    assert _count(clean_schema, fact_stop_event) == 60


def test_measured_arrivals_is_derived_in_sql(
    clean_schema: Engine, synthetic_gold_dir: Path
) -> None:
    """The generator plants unmeasured arrivals; the CHECK would reject a wrong derivation."""
    load_dimensions(clean_schema, synthetic_gold_dir)
    load_fact(clean_schema, synthetic_gold_dir)
    rows = _fact_rows(clean_schema)
    assert any(row["punctual_arrivals"] is None for row in rows)
    for row in rows:
        expected = 0 if row["punctual_arrivals"] is None else 1
        assert row["measured_arrivals"] == expected


def test_reloading_a_day_is_idempotent(clean_schema: Engine, synthetic_gold_dir: Path) -> None:
    """Same semantics as the upstream MERGE: reprocessing must not double the grain."""
    load_dimensions(clean_schema, synthetic_gold_dir)
    load_fact(clean_schema, synthetic_gold_dir)
    second = load_fact(clean_schema, synthetic_gold_dir)
    assert second.rows_inserted == 0
    assert second.rows_updated == 60
    assert _count(clean_schema, fact_stop_event) == 60


def test_reload_refreshes_loaded_at(clean_schema: Engine, synthetic_gold_dir: Path) -> None:
    load_dimensions(clean_schema, synthetic_gold_dir)
    load_fact(clean_schema, synthetic_gold_dir)
    before = min(row["loaded_at"] for row in _fact_rows(clean_schema))
    load_fact(clean_schema, synthetic_gold_dir)
    after = min(row["loaded_at"] for row in _fact_rows(clean_schema))
    assert after > before


def test_date_filter_loads_only_that_day(clean_schema: Engine, synthetic_gold_dir: Path) -> None:
    load_dimensions(clean_schema, synthetic_gold_dir)
    with clean_schema.connect() as conn:
        one_day = conn.execute(select(func.min(dim_date.c.date_key))).scalar_one()
    counts = load_fact(clean_schema, synthetic_gold_dir, service_date=one_day)
    assert counts.rows_inserted == 20
    assert counts.rows_rejected == 0
    assert _count(clean_schema, fact_stop_event) == 20


def test_orphans_fail_the_load_by_default(clean_schema: Engine, gold_parquet_dir: Path) -> None:
    """The real samples are independent cuts, so many fact keys do not resolve."""
    load_dimensions(clean_schema, gold_parquet_dir)
    with pytest.raises(DataQualityError, match="orphan_station"):
        load_fact(clean_schema, gold_parquet_dir)
    assert _count(clean_schema, fact_stop_event) == 0


def test_failed_load_leaves_staging_intact_for_diagnosis(
    clean_schema: Engine, gold_parquet_dir: Path
) -> None:
    """Staging is the evidence; a rollback that wipes it would defeat the design."""
    load_dimensions(clean_schema, gold_parquet_dir)
    with pytest.raises(DataQualityError):
        load_fact(clean_schema, gold_parquet_dir)
    assert _count(clean_schema, stg_fact_stop_event) == 0


def test_skip_policy_loads_the_good_rows(clean_schema: Engine, gold_parquet_dir: Path) -> None:
    load_dimensions(clean_schema, gold_parquet_dir)
    counts = load_fact(clean_schema, gold_parquet_dir, on_violation="skip")
    assert counts.rows_read == 100
    assert counts.rows_rejected > 0
    assert counts.rows_inserted == counts.rows_read - counts.rows_rejected
    assert _count(clean_schema, fact_stop_event) == counts.rows_inserted


def test_a_failed_load_is_recorded_as_failed(clean_schema: Engine, gold_parquet_dir: Path) -> None:
    load_dimensions(clean_schema, gold_parquet_dir)
    with pytest.raises(DataQualityError):
        load_fact(clean_schema, gold_parquet_dir)
    with clean_schema.connect() as conn:
        row = conn.execute(
            select(load_runs)
            .where(load_runs.c.table_name == "fact_stop_event")
            .order_by(load_runs.c.id.desc())
            .limit(1)
        ).one()
    assert row.status == "failed"
    assert "orphan" in row.error


def test_a_successful_load_records_its_tallies(
    clean_schema: Engine, synthetic_gold_dir: Path
) -> None:
    load_dimensions(clean_schema, synthetic_gold_dir)
    load_fact(clean_schema, synthetic_gold_dir, service_date=dt.date(2026, 8, 23))
    with clean_schema.connect() as conn:
        row = conn.execute(
            select(load_runs)
            .where(load_runs.c.table_name == "fact_stop_event")
            .order_by(load_runs.c.id.desc())
            .limit(1)
        ).one()
    assert row.status == "succeeded"
    assert row.date_key == dt.date(2026, 8, 23)
    assert row.rows_inserted == 20


def test_dimension_loads_share_one_run_id(clean_schema: Engine, synthetic_gold_dir: Path) -> None:
    """One invocation, three tables: the run_id is what ties them together."""
    run_id = new_run_id()
    load_dimensions(clean_schema, synthetic_gold_dir, run_id=run_id)
    with clean_schema.connect() as conn:
        names = conn.execute(
            select(load_runs.c.table_name).where(load_runs.c.run_id == run_id)
        ).scalars()
    assert set(names) == {"dim_date", "dim_station", "dim_relation"}
