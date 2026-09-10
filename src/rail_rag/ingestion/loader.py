"""Load the Gold export into PostgreSQL.

Dimensions are upserted directly: they are small, and a ``TRUNCATE`` is impossible
anyway while the fact's foreign keys reference them. The fact goes through the
unconstrained staging table first, so a bad batch is diagnosed in SQL as a set
rather than aborting on whichever row hit a constraint first.

``measured_arrivals`` is derived in the promotion ``INSERT ... SELECT`` and never
travels through Python, which makes it impossible to desynchronise from the CHECK
that validates it.
"""

from __future__ import annotations

import datetime as dt
import io
import logging
import uuid
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import Any, Literal

import pyarrow.csv as pacsv
from sqlalchemy import (
    Boolean,
    ColumnClause,
    ColumnElement,
    Connection,
    Engine,
    Table,
    and_,
    case,
    func,
    literal_column,
    select,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError

from rail_rag.core.exceptions import DatabaseError
from rail_rag.db.models import (
    dim_date,
    dim_relation,
    dim_station,
    fact_stop_event,
    stg_fact_stop_event,
)
from rail_rag.db.run_log import LoadCounts, fail_run, finish_run, new_run_id, start_run
from rail_rag.ingestion.data_contracts import _GoldRow
from rail_rag.ingestion.fact_source import (
    DateRange,
    fact_dir,
    is_partitioned,
    read_fact_batches,
)
from rail_rag.ingestion.gold_source import (
    read_dim_date,
    read_dim_relation,
    read_dim_station,
    read_fact_stop_event,
)
from rail_rag.ingestion.validation import (
    GRAIN_COLUMNS,
    DataQualityError,
    Violation,
    valid_rows_clause,
    validate_staged_fact,
)

logger = logging.getLogger(__name__)

#: Rows per INSERT. Bounds memory and statement size independently of file size.
BATCH_SIZE = 10_000

#: What to do when staged rows fail a rule. ``fail`` is the default on purpose.
OnViolation = Literal["fail", "skip"]

_DIMENSIONS: tuple[tuple[str, Table, Any], ...] = (
    ("dim_date", dim_date, read_dim_date),
    ("dim_station", dim_station, read_dim_station),
    ("dim_relation", dim_relation, read_dim_relation),
)

#: ``xmax = 0`` on a returned row means it was inserted, not updated by ON CONFLICT.
#: Dimensions only: PostgreSQL refuses to return a system column from a partitioned
#: table, so the fact counts its split from the row count instead.
_WAS_INSERTED: ColumnClause[bool] = literal_column("(xmax = 0)", Boolean)


def _batched(rows: Iterable[_GoldRow], size: int) -> Iterator[list[dict[str, Any]]]:
    iterator = iter(rows)
    while batch := list(islice(iterator, size)):
        yield [row.model_dump() for row in batch]


def _upsert_dimension(conn: Connection, table: Table, rows: list[dict[str, Any]]) -> LoadCounts:
    """Insert or update one dimension batch, counting each outcome separately."""
    primary_key = [column.name for column in table.primary_key.columns]
    updatable = [
        column.name for column in table.columns if column.name not in primary_key + ["loaded_at"]
    ]
    insert_stmt = pg_insert(table).values(rows)
    upsert = insert_stmt.on_conflict_do_update(
        index_elements=primary_key,
        set_={name: insert_stmt.excluded[name] for name in updatable} | {"loaded_at": func.now()},
    ).returning(_WAS_INSERTED)
    outcomes = [bool(row[0]) for row in conn.execute(upsert)]
    inserted = sum(outcomes)
    return LoadCounts(
        rows_read=len(rows),
        rows_inserted=inserted,
        rows_updated=len(outcomes) - inserted,
    )


def load_dimensions(
    engine: Engine,
    source_dir: Path,
    *,
    run_id: uuid.UUID | None = None,
) -> dict[str, LoadCounts]:
    """Upsert the three dimensions from ``source_dir``.

    Rows absent from the export are left in place: the fact may still reference them.

    Raises:
        IngestionError: if a source file is missing or violates its contract.
        DatabaseError: if the upsert fails.
    """
    run_id = run_id or new_run_id()
    results: dict[str, LoadCounts] = {}
    for name, table, reader in _DIMENSIONS:
        path = _dimension_source(source_dir, name)
        entry_id = start_run(engine, run_id=run_id, table_name=name, source_file=str(path))
        try:
            counts = LoadCounts()
            with engine.begin() as conn:
                for batch in _batched(reader(path), BATCH_SIZE):
                    batch_counts = _upsert_dimension(conn, table, batch)
                    counts = LoadCounts(
                        rows_read=counts.rows_read + batch_counts.rows_read,
                        rows_inserted=counts.rows_inserted + batch_counts.rows_inserted,
                        rows_updated=counts.rows_updated + batch_counts.rows_updated,
                    )
        except SQLAlchemyError as exc:
            fail_run(engine, entry_id, f"{type(exc).__name__}")
            raise DatabaseError(f"Could not load {name}: {type(exc).__name__}") from exc
        except Exception as exc:
            fail_run(engine, entry_id, str(exc))
            raise
        finish_run(engine, entry_id, counts)
        results[name] = counts
        logger.info("%s: %s", name, counts)
    return results


def _scope_column(column: ColumnElement[Any], scope: DateRange) -> ColumnElement[bool] | None:
    """A ``[start, end)`` predicate on ``column``, or ``None`` for a full scope."""
    clauses: list[ColumnElement[bool]] = []
    if scope.start is not None:
        clauses.append(column >= scope.start)
    if scope.end is not None:
        clauses.append(column < scope.end)
    if not clauses:
        return None
    return and_(*clauses)


def _stage_fact(conn: Connection, source_dir: Path, scope: DateRange) -> None:
    """Empty staging and stream the fact into it, choosing the path by layout."""
    conn.execute(stg_fact_stop_event.delete())
    if is_partitioned(source_dir):
        _copy_partitioned_fact(conn, source_dir, scope)
    else:
        single_file = source_dir / "fact_stop_event.parquet"
        for batch in _batched(read_fact_stop_event(single_file), BATCH_SIZE):
            conn.execute(stg_fact_stop_event.insert(), batch)


def _copy_partitioned_fact(conn: Connection, source_dir: Path, scope: DateRange) -> None:
    """DuckDB reads the partitioned dataset; psycopg COPYs it into staging.

    The COPY runs on this connection, so it shares the load's transaction: the
    validation that follows sees the staged rows, and a rollback discards them.
    """
    columns = [column.name for column in stg_fact_stop_event.columns]
    column_list = ", ".join(f'"{name}"' for name in columns)
    copy_sql = (
        f'COPY "{stg_fact_stop_event.schema}"."{stg_fact_stop_event.name}" ({column_list})'
        " FROM STDIN WITH (FORMAT csv, HEADER false)"
    )
    write_options = pacsv.WriteOptions(include_header=False)
    driver_connection = conn.connection.driver_connection
    if driver_connection is None:  # pragma: no cover - defensive, psycopg always sets it
        raise DatabaseError("No DBAPI connection available for COPY")
    with driver_connection.cursor().copy(copy_sql) as copy:
        for batch in read_fact_batches(source_dir, columns, scope.start, scope.end):
            buffer = io.BytesIO()
            pacsv.write_csv(batch, buffer, write_options=write_options)
            copy.write(buffer.getvalue())


def _staged_in_scope(conn: Connection, scope: DateRange) -> int:
    """Rows the run is responsible for; the range narrows what counts as rejected."""
    statement = select(func.count()).select_from(stg_fact_stop_event)
    predicate = _scope_column(stg_fact_stop_event.c.date_key, scope)
    if predicate is not None:
        statement = statement.where(predicate)
    return int(conn.execute(statement).scalar_one())


def _delete_fact_scope(conn: Connection, scope: DateRange) -> None:
    """Clear the fact within the range so a reload replaces rather than merges.

    The predicate prunes to the range's partitions, and dropping obsolete rows -
    ones the new export no longer carries - is what makes a reload reproducible
    rather than an accreting upsert.
    """
    predicate = _scope_column(fact_stop_event.c.date_key, scope)
    statement = fact_stop_event.delete()
    if predicate is not None:
        statement = statement.where(predicate)
    conn.execute(statement)


def _promote(conn: Connection, scope: DateRange, *, only_valid: bool) -> int:
    """Move staged rows into the fact, deriving ``measured_arrivals`` in SQL.

    The scope is already cleared, so every promoted row is an insert; the count
    of promoted rows comes straight from the source projection.
    """
    source_names = [column.name for column in stg_fact_stop_event.columns]
    selected: list[ColumnElement[Any]] = [stg_fact_stop_event.c[name] for name in source_names]
    selected.append(
        case((stg_fact_stop_event.c.punctual_arrivals.is_(None), 0), else_=1).label(
            "measured_arrivals"
        )
    )
    source = select(*selected)
    if only_valid:
        source = source.where(valid_rows_clause())
    else:
        predicate = _scope_column(stg_fact_stop_event.c.date_key, scope)
        if predicate is not None:
            source = source.where(predicate)

    target_names = [*source_names, "measured_arrivals"]
    insert_stmt = pg_insert(fact_stop_event).from_select(target_names, source)
    updatable = [name for name in target_names if name not in GRAIN_COLUMNS]
    upsert = insert_stmt.on_conflict_do_update(
        index_elements=list(GRAIN_COLUMNS),
        set_={name: insert_stmt.excluded[name] for name in updatable} | {"loaded_at": func.now()},
    )
    promoted = int(conn.execute(select(func.count()).select_from(source.subquery())).scalar_one())
    conn.execute(upsert)
    return promoted


def load_fact(
    engine: Engine,
    source_dir: Path,
    *,
    service_date: dt.date | None = None,
    date_range: DateRange | None = None,
    on_violation: OnViolation = "fail",
    run_id: uuid.UUID | None = None,
) -> LoadCounts:
    """Stage, validate and promote the fact export over a date scope.

    ``service_date`` loads one day; ``date_range`` loads ``[start, end)``; neither
    loads the whole export. They are mutually exclusive. The scope is cleared before
    promotion, so a reload replaces its window rather than accreting onto it.

    Under ``fail`` a single violation aborts the load and leaves staging intact for
    diagnosis. Under ``skip`` every offending row is excluded and counted as rejected.

    Raises:
        ValueError: if both ``service_date`` and ``date_range`` are given.
        DataQualityError: if rules failed and ``on_violation`` is ``fail``.
        IngestionError: if the source file is missing or violates its contract.
        DatabaseError: if staging or promotion fails.
    """
    if service_date is not None and date_range is not None:
        raise ValueError("Pass service_date or date_range, not both")
    scope = (
        DateRange.for_day(service_date) if service_date is not None else date_range or DateRange()
    )

    path = (
        fact_dir(source_dir)
        if is_partitioned(source_dir)
        else source_dir / "fact_stop_event.parquet"
    )
    run_id = run_id or new_run_id()
    entry_id = start_run(
        engine,
        run_id=run_id,
        table_name="fact_stop_event",
        source_file=str(path),
        date_key=scope.start,
    )
    try:
        with engine.begin() as conn:
            _stage_fact(conn, source_dir, scope)
            staged = _staged_in_scope(conn, scope)
            violations = validate_staged_fact(conn)
            if violations:
                _report(violations, on_violation)
            if violations and on_violation == "fail":
                raise DataQualityError(
                    f"{len(violations)} rule(s) failed on {staged} staged row(s): "
                    + " | ".join(v.describe() for v in violations)
                )
            _delete_fact_scope(conn, scope)
            promoted = _promote(conn, scope, only_valid=bool(violations))
            counts = LoadCounts(
                rows_read=staged,
                rows_rejected=staged - promoted,
                rows_inserted=promoted,
                rows_updated=0,
            )
    except SQLAlchemyError as exc:
        fail_run(engine, entry_id, f"{type(exc).__name__}")
        raise DatabaseError(f"Could not load the fact: {type(exc).__name__}") from exc
    except Exception as exc:
        fail_run(engine, entry_id, str(exc))
        raise
    finish_run(engine, entry_id, counts)
    logger.info("fact_stop_event: %s", counts)
    return counts


def _report(violations: list[Violation], on_violation: OnViolation) -> None:
    for violation in violations:
        logger.warning("%s (policy=%s)", violation.describe(), on_violation)


def _dimension_source(source_dir: Path, name: str) -> Path:
    """Resolve a dimension's source: a ``{name}/`` export directory if present,
    else the ``{name}.parquet`` single-file sample. Lets the same loader read
    both the real Databricks export and the synthetic fixture."""
    directory = source_dir / name
    if directory.is_dir():
        return directory
    return source_dir / f"{name}.parquet"
