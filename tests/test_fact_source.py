"""Tests for the partitioned DuckDB fact reader and the COPY load path.

The synthetic Gold generator writes a single-file fact; the real export is a
Hive-partitioned directory. ``_partition`` rewrites the former into the latter -
one ``date_key=YYYY-MM-DD/`` directory per day, with ``date_key`` dropped from
the Parquet exactly as Spark leaves it - so the DuckDB path is exercised on the
layout it will meet in production, not on the single file the other tests use.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from sqlalchemy import Engine, FromClause, func, select, text

from rail_rag.db.models import GOLD_SCHEMA, fact_stop_event, stg_fact_stop_event
from rail_rag.ingestion.fact_source import (
    count_fact_rows,
    is_partitioned,
    read_fact_batches,
)
from rail_rag.ingestion.gold_source import IngestionError
from rail_rag.ingestion.loader import load_dimensions, load_fact

pytestmark = pytest.mark.integration

_STAGING_COLUMNS = [column.name for column in stg_fact_stop_event.columns]


def _partition(single_file_dir: Path, target: Path) -> Path:
    """Rewrite a single-file synthetic export into the Hive-partitioned layout."""
    table = pq.read_table(single_file_dir / "fact_stop_event.parquet")
    fact_root = target / "fact_stop_event"
    for day in sorted({d for d in table.column("date_key").to_pylist()}):
        mask = [d == day for d in table.column("date_key").to_pylist()]
        one_day = table.filter(mask).drop_columns(["date_key"])
        partition_dir = fact_root / f"date_key={day.isoformat()}"
        partition_dir.mkdir(parents=True)
        pq.write_table(one_day, partition_dir / "part-00000.parquet")
        (partition_dir / "_SUCCESS").touch()
    for name in ("dim_date", "dim_station", "dim_relation"):
        (target / name).mkdir()
        pq.write_table(
            pq.read_table(single_file_dir / f"{name}.parquet"),
            target / name / "part-00000.parquet",
        )
    return target


@pytest.fixture
def partitioned_gold_dir(synthetic_gold_dir: Path, tmp_path: Path) -> Path:
    """The synthetic export, re-laid-out as a partitioned directory."""
    return _partition(synthetic_gold_dir, tmp_path / "partitioned")


def _count(engine: Engine, table: FromClause) -> int:
    with engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(table)).scalar_one())


# --- the reader ------------------------------------------------------------


def test_a_partitioned_directory_is_detected(partitioned_gold_dir: Path) -> None:
    assert is_partitioned(partitioned_gold_dir)


def test_a_single_file_export_is_not_partitioned(synthetic_gold_dir: Path) -> None:
    assert not is_partitioned(synthetic_gold_dir)


def test_the_reader_recovers_date_key_from_the_path(partitioned_gold_dir: Path) -> None:
    """date_key lives in the directory name, not the Parquet; the reader must restore it."""
    batches = list(read_fact_batches(partitioned_gold_dir, _STAGING_COLUMNS))
    assert batches
    assert batches[0].schema.names == _STAGING_COLUMNS
    dates = {d for batch in batches for d in batch.column("date_key").to_pylist()}
    assert all(isinstance(d, dt.date) for d in dates)


def test_a_range_prunes_to_the_days_it_covers(partitioned_gold_dir: Path) -> None:
    total = count_fact_rows(partitioned_gold_dir, None, None)
    with pq.ParquetFile(
        next((partitioned_gold_dir / "fact_stop_event").glob("date_key=*/*.parquet"))
    ) as handle:
        one_day_rows = handle.metadata.num_rows
    days = sorted(
        p.name.removeprefix("date_key=")
        for p in (partitioned_gold_dir / "fact_stop_event").glob("date_key=*")
    )
    first = dt.date.fromisoformat(days[0])
    scoped = count_fact_rows(partitioned_gold_dir, first, first + dt.timedelta(days=1))
    assert scoped == one_day_rows
    assert scoped < total


def test_schema_drift_fails_before_transport(partitioned_gold_dir: Path) -> None:
    """An unannounced column in the export schema is a contract breach, caught at read time."""
    for partition in (partitioned_gold_dir / "fact_stop_event").glob("date_key=*"):
        table = pq.read_table(partition / "part-00000.parquet")
        pq.write_table(
            table.add_column(0, "surprise", [[1] * table.num_rows]),
            partition / "part-00000.parquet",
        )
    with pytest.raises(IngestionError, match="do not match the contract"):
        count_fact_rows(partitioned_gold_dir, None, None)


def test_a_missing_export_raises_ingestion_error(tmp_path: Path) -> None:
    """A path with no Parquet part files fails as IngestionError, not a raw DuckDB error."""
    with pytest.raises(IngestionError, match="No Parquet part files"):
        count_fact_rows(tmp_path / "absent", None, None)


# --- the COPY load path ----------------------------------------------------


def test_the_partitioned_path_loads_every_row(
    clean_schema: Engine, partitioned_gold_dir: Path
) -> None:
    load_dimensions(clean_schema, partitioned_gold_dir)
    counts = load_fact(clean_schema, partitioned_gold_dir)
    on_disk = count_fact_rows(partitioned_gold_dir, None, None)
    assert counts.rows_read == on_disk
    assert counts.rows_inserted == on_disk
    assert _count(clean_schema, fact_stop_event) == on_disk


def test_date_key_lands_as_a_date(clean_schema: Engine, partitioned_gold_dir: Path) -> None:
    load_dimensions(clean_schema, partitioned_gold_dir)
    load_fact(clean_schema, partitioned_gold_dir)
    with clean_schema.connect() as conn:
        typename = conn.execute(
            text(f"SELECT pg_typeof(date_key) FROM {GOLD_SCHEMA}.fact_stop_event LIMIT 1")
        ).scalar_one()
    assert typename == "date"


def test_disjoint_nulls_survive_the_copy(clean_schema: Engine, partitioned_gold_dir: Path) -> None:
    """The COPY must not coerce an unmeasured arrival to zero."""
    load_dimensions(clean_schema, partitioned_gold_dir)
    load_fact(clean_schema, partitioned_gold_dir)
    with clean_schema.connect() as conn:
        unmeasured = conn.execute(
            text(
                f"SELECT count(*) FROM {GOLD_SCHEMA}.fact_stop_event"
                " WHERE punctual_arrivals IS NULL AND measured_arrivals = 0"
            )
        ).scalar_one()
    assert unmeasured > 0


def test_reloading_a_partitioned_range_is_idempotent(
    clean_schema: Engine, partitioned_gold_dir: Path
) -> None:
    load_dimensions(clean_schema, partitioned_gold_dir)
    load_fact(clean_schema, partitioned_gold_dir)
    on_disk = count_fact_rows(partitioned_gold_dir, None, None)
    second = load_fact(clean_schema, partitioned_gold_dir)
    assert second.rows_inserted == 0
    assert second.rows_updated == on_disk
    assert _count(clean_schema, fact_stop_event) == on_disk


def test_date_scoped_load_stages_only_that_day(
    clean_schema: Engine, partitioned_gold_dir: Path
) -> None:
    """--date pushes down to the reader: staging holds one day, not the whole export."""
    load_dimensions(clean_schema, partitioned_gold_dir)
    with clean_schema.connect() as conn:
        first = conn.execute(text(f"SELECT min(date_key) FROM {GOLD_SCHEMA}.dim_date")).scalar_one()
    one_day = count_fact_rows(partitioned_gold_dir, first, first + dt.timedelta(days=1))
    counts = load_fact(clean_schema, partitioned_gold_dir, service_date=first)
    assert counts.rows_read == one_day
    assert _count(clean_schema, fact_stop_event) == one_day


def test_a_failed_partitioned_load_rolls_back_staging(
    clean_schema: Engine, partitioned_gold_dir: Path
) -> None:
    """The COPY shares the load's transaction, so a rejected batch leaves no trace."""
    counts_before = _count(clean_schema, stg_fact_stop_event)
    with pytest.raises(Exception):  # noqa: B017 - dims absent, so every fact key is an orphan
        load_fact(clean_schema, partitioned_gold_dir)
    assert _count(clean_schema, stg_fact_stop_event) == counts_before
