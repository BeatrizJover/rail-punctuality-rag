"""Read the Hive-partitioned ``fact_stop_event`` export with DuckDB.

The dimensions stream through :mod:`rail_rag.ingestion.gold_source` row by row;
the fact is far too large for that. Here DuckDB reads the partitioned dataset,
prunes by date range against the ``date_key=`` directory names, and hands back
Arrow batches the loader COPYs straight into staging. ``date_key`` lives in the
partition path, not the Parquet files, so ``hive_types`` pins it to ``DATE``.

The contract check is a column-set assertion once per load, not Pydantic per
row: at 60M rows the schema is the boundary worth guarding, and any bad row is
caught later by the staging data-quality rules.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pyarrow as pa

from rail_rag.ingestion.gold_source import IngestionError

#: Fact directory inside a Gold export root.
FACT_DIRNAME = "fact_stop_event"

#: Rows per Arrow batch handed to the loader. Bounds the CSV buffer, not the scan.
_BATCH_SIZE = 100_000


@dataclass(frozen=True)
class DateRange:
    """Half-open ``[start, end)`` load scope. ``None`` on a bound means unbounded."""

    start: dt.date | None = None
    end: dt.date | None = None

    @classmethod
    def for_day(cls, day: dt.date) -> DateRange:
        """The single-day scope ``[day, day + 1)`` that ``--date`` maps to."""
        return cls(day, day + dt.timedelta(days=1))

    @property
    def is_full(self) -> bool:
        """True when neither bound is set: the whole export is in scope."""
        return self.start is None and self.end is None

    def describe(self) -> str:
        """Human-readable scope for logs and error messages."""
        if self.is_full:
            return "full export"
        return f"[{self.start or '-inf'}, {self.end or '+inf'})"


#: The columns the export must carry, ``date_key`` included (it comes from the path).
_EXPECTED_COLUMNS: tuple[str, ...] = (
    "date_key",
    "station_key",
    "relation_key",
    "train_no",
    "planned_hour",
    "delay_arr_s",
    "delay_dep_s",
    "dwell_delta_s",
    "punctual_arrivals",
    "stop_events",
)


def fact_dir(source_dir: Path) -> Path:
    """The partitioned fact directory under a Gold export root."""
    return source_dir / FACT_DIRNAME


def is_partitioned(source_dir: Path) -> bool:
    """True when the fact is a partitioned directory rather than a single file."""
    return fact_dir(source_dir).is_dir()


def _dataset_expr(source_dir: Path) -> str:
    glob = fact_dir(source_dir) / "**" / "*.parquet"
    return f"read_parquet('{glob}', hive_partitioning=true, hive_types={{'date_key': DATE}})"


def _assert_columns(con: duckdb.DuckDBPyConnection, dataset: str, source_dir: Path) -> None:
    """Fail loudly on schema drift before a single row is transported."""
    found = {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {dataset}").fetchall()}
    expected = set(_EXPECTED_COLUMNS)
    if found != expected:
        missing = expected - found
        extra = found - expected
        raise IngestionError(
            f"Fact export columns do not match the contract in {fact_dir(source_dir)}: "
            f"missing={sorted(missing)}, unexpected={sorted(extra)}"
        )


def _range_predicate(start: dt.date | None, end: dt.date | None) -> str:
    clauses: list[str] = []
    if start is not None:
        clauses.append(f"date_key >= DATE '{start.isoformat()}'")
    if end is not None:
        clauses.append(f"date_key < DATE '{end.isoformat()}'")
    return " WHERE " + " AND ".join(clauses) if clauses else ""


def count_fact_rows(source_dir: Path, start: dt.date | None, end: dt.date | None) -> int:
    """Rows on disk within ``[start, end)``; the loaded-count gate compares against this.

    Raises:
        IngestionError: if the export holds no Parquet part files or its columns drift.
    """
    con = duckdb.connect()
    try:
        dataset = _dataset_expr(source_dir)
        _assert_columns(con, dataset, source_dir)
        query = f"SELECT count(*) FROM {dataset}{_range_predicate(start, end)}"
        row = con.execute(query).fetchone()
        return int(row[0]) if row is not None else 0
    except duckdb.IOException as exc:
        raise IngestionError(
            f"No Parquet part files under fact export: {fact_dir(source_dir)}"
        ) from exc
    finally:
        con.close()


def read_fact_batches(
    source_dir: Path,
    columns: list[str],
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> Iterator[pa.RecordBatch]:
    """Stream the fact within ``[start, end)`` as Arrow batches in ``columns`` order.

    DuckDB prunes to the partitions the range touches, so an unbounded read and a
    one-day read cost proportionally to what they return, not to the whole export.

    Raises:
        IngestionError: if the export holds no Parquet part files or its columns drift.
    """
    con = duckdb.connect()
    try:
        dataset = _dataset_expr(source_dir)
        _assert_columns(con, dataset, source_dir)
        projection = ", ".join(columns)
        query = f"SELECT {projection} FROM {dataset}{_range_predicate(start, end)}"
        reader = con.execute(query).fetch_record_batch(_BATCH_SIZE)
        yield from reader
    except duckdb.IOException as exc:
        raise IngestionError(
            f"No Parquet part files under fact export: {fact_dir(source_dir)}"
        ) from exc
    finally:
        con.close()
