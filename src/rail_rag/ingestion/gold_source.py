"""Read the Gold export from Parquet and validate it against the contracts.

One reader function per table. Each opens a Parquet file, walks it in batches, and
yields validated Pydantic rows one at a time. Streaming - rather than loading the
whole table into memory - is deliberate: the real fact table is ~26M rows a year,
far past what fits in a process, and the loader consumes this as an iterator.

A missing file, an unreadable file or a row that violates its contract all surface
as :class:`IngestionError` with enough context to locate the problem, never as a
bare ``pyarrow`` or ``pydantic`` traceback.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar

import pyarrow.parquet as pq
from pydantic import ValidationError

from rail_rag.core.exceptions import RailRagError
from rail_rag.ingestion.data_contracts import (
    DimDateRow,
    DimRelationRow,
    DimStationRow,
    FactStopEventRow,
    _GoldRow,
)

#: Rows read from Parquet per batch. Bounds peak memory independently of file size.
_BATCH_SIZE = 10_000

RowT = TypeVar("RowT", bound=_GoldRow)


class IngestionError(RailRagError):
    """Raised when a Gold source file is missing, unreadable, or violates a contract."""


def _read_table(
    path: Path,
    model: type[RowT],
) -> Iterator[RowT]:
    """Stream one Parquet file, yielding validated ``model`` rows.

    Raises:
        IngestionError: if the file is missing or unreadable, or if any row fails
            validation. The message carries the file, the model and - for a bad
            row - its zero-based position and the underlying field errors.
    """
    if not path.is_file():
        raise IngestionError(f"Gold source file not found: {path}")

    try:
        parquet_file = pq.ParquetFile(path)
    except Exception as exc:  # pyarrow raises a broad OSError/ArrowInvalid family
        raise IngestionError(f"Could not open Parquet file {path}: {type(exc).__name__}") from exc

    row_index = 0
    for batch in parquet_file.iter_batches(batch_size=_BATCH_SIZE):
        for record in batch.to_pylist():
            try:
                yield model.model_validate(record)
            except ValidationError as exc:
                raise IngestionError(
                    f"{model.__name__} row {row_index} in {path.name} "
                    f"violates its contract: {exc.error_count()} error(s); {exc}"
                ) from exc
            row_index += 1


def read_dim_date(path: Path) -> Iterator[DimDateRow]:
    """Stream validated ``dim_date`` rows from ``path``."""
    return _read_table(path, DimDateRow)


def read_dim_station(path: Path) -> Iterator[DimStationRow]:
    """Stream validated ``dim_station`` rows from ``path``."""
    return _read_table(path, DimStationRow)


def read_dim_relation(path: Path) -> Iterator[DimRelationRow]:
    """Stream validated ``dim_relation`` rows from ``path``."""
    return _read_table(path, DimRelationRow)


def read_fact_stop_event(path: Path) -> Iterator[FactStopEventRow]:
    """Stream validated ``fact_stop_event`` rows from ``path``."""
    return _read_table(path, FactStopEventRow)
