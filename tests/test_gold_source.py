"""Tests for the Parquet Gold reader.

Exercised against the real converted samples (faithful nulls and types) and against
freshly generated synthetic data, plus the failure paths: missing file, unreadable
file, and a row that violates its contract.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rail_rag.ingestion.gold_source import (
    IngestionError,
    read_dim_date,
    read_dim_relation,
    read_dim_station,
    read_fact_stop_event,
)


def test_reads_all_real_dim_date_rows(gold_parquet_dir: Path) -> None:
    rows = list(read_dim_date(gold_parquet_dir / "dim_date.parquet"))
    assert len(rows) == 100
    assert all(isinstance(r.date_key, dt.date) for r in rows)


def test_reads_real_fact_with_null_measures(gold_parquet_dir: Path) -> None:
    """The real sample contains unmeasured arrivals; they must survive as None."""
    rows = list(read_fact_stop_event(gold_parquet_dir / "fact_stop_event.parquet"))
    assert len(rows) == 100
    assert any(r.punctual_arrivals is None for r in rows)
    assert any(r.delay_arr_s is not None and r.delay_arr_s < 0 for r in rows)


def test_reads_real_relation_with_null_direction(gold_parquet_dir: Path) -> None:
    rows = list(read_dim_relation(gold_parquet_dir / "dim_relation.parquet"))
    assert any(r.relation_direction is None for r in rows)


def test_reads_real_station(gold_parquet_dir: Path) -> None:
    rows = list(read_dim_station(gold_parquet_dir / "dim_station.parquet"))
    assert len(rows) == 100
    assert all(len(r.station_key) == 32 for r in rows)


def test_returns_a_lazy_iterator(gold_parquet_dir: Path) -> None:
    """The reader must stream, not materialise: the loader relies on this for 26M rows."""
    from collections.abc import Iterator

    result = read_fact_stop_event(gold_parquet_dir / "fact_stop_event.parquet")
    assert isinstance(result, Iterator)


def test_missing_file_raises_ingestion_error(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="not found"):
        list(read_dim_date(tmp_path / "does_not_exist.parquet"))


def test_unreadable_file_raises_ingestion_error(tmp_path: Path) -> None:
    bad = tmp_path / "not_parquet.parquet"
    bad.write_bytes(b"this is not parquet")
    with pytest.raises(IngestionError, match="Could not open"):
        list(read_dim_date(bad))


def test_contract_violation_names_the_row(tmp_path: Path) -> None:
    """A bad row must fail with its position and field, not a bare pydantic trace."""
    table = pa.table(
        {
            "station_key": ["0" * 32, "tooshort"],
            "station_name": ["A", "B"],
            "ptcar_no": [1, 2],
            "observed_stop_events": [10, 20],
            "first_seen": [dt.date(2026, 8, 1), dt.date(2026, 8, 1)],
            "last_seen": [dt.date(2026, 8, 2), dt.date(2026, 8, 2)],
        }
    )
    path = tmp_path / "dim_station.parquet"
    pq.write_table(table, path)
    with pytest.raises(IngestionError, match="row 1"):
        list(read_dim_station(path))
        

def _write_parts(directory: Path, tables: list[pa.Table]) -> None:
    """Write one Parquet part file per table into a directory, Spark-style."""
    directory.mkdir(parents=True, exist_ok=True)
    for i, table in enumerate(tables):
        pq.write_table(table, directory / f"part-{i:05d}.parquet")


def _dim_relation_table(n: int) -> pa.Table:
    """Minimal dim_relation table with n rows, matching the export schema."""
    rows = [
        {
            "relation_key": f"{i:032d}",
            "relation": f"REL-{i}",
            "relation_direction": None if i == 0 else "A -> B",
            "operator": "SNCB/NMBS",
        }
        for i in range(n)
    ]
    return pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("relation_key", pa.string()),
                ("relation", pa.string()),
                ("relation_direction", pa.string()),
                ("operator", pa.string()),
            ]
        ),
    )


def test_a_dimension_directory_is_read_across_all_part_files(tmp_path: Path) -> None:
    table = _dim_relation_table(6)
    half = table.num_rows // 2
    _write_parts(tmp_path / "dim_relation", [table.slice(0, half), table.slice(half)])

    rows = list(read_dim_relation(tmp_path / "dim_relation"))

    assert len(rows) == table.num_rows


def test_a_single_parquet_file_is_still_read(tmp_path: Path) -> None:
    table = _dim_relation_table(6)
    path = tmp_path / "dim_relation.parquet"
    pq.write_table(table, path)

    rows = list(read_dim_relation(path))

    assert len(rows) == table.num_rows


def test_an_empty_directory_is_a_loud_error(tmp_path: Path) -> None:
    (tmp_path / "dim_relation").mkdir()
    with pytest.raises(IngestionError, match="No Parquet part files"):
        list(read_dim_relation(tmp_path / "dim_relation"))


def test_a_missing_source_is_a_loud_error(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="not found"):
        list(read_dim_relation(tmp_path / "does_not_exist"))
