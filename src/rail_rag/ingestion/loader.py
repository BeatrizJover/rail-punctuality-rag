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
import logging
import uuid
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import (
    Boolean,
    ColumnClause,
    ColumnElement,
    Connection,
    Engine,
    Table,
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
        path = source_dir / f"{name}.parquet"
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


def _stage_fact(conn: Connection, path: Path) -> None:
    """Empty staging and stream the export into it."""
    conn.execute(stg_fact_stop_event.delete())
    for batch in _batched(read_fact_stop_event(path), BATCH_SIZE):
        conn.execute(stg_fact_stop_event.insert(), batch)


def _staged_in_scope(conn: Connection, service_date: dt.date | None) -> int:
    """Rows the run is responsible for; ``--date`` narrows what counts as rejected."""
    statement = select(func.count()).select_from(stg_fact_stop_event)
    if service_date is not None:
        statement = statement.where(stg_fact_stop_event.c.date_key == service_date)
    return int(conn.execute(statement).scalar_one())


def _promote(conn: Connection, service_date: dt.date | None, *, only_valid: bool) -> LoadCounts:
    """Move staged rows into the fact, deriving ``measured_arrivals`` in SQL."""
    source_names = [column.name for column in stg_fact_stop_event.columns]
    selected: list[ColumnElement[Any]] = [stg_fact_stop_event.c[name] for name in source_names]
    selected.append(
        case((stg_fact_stop_event.c.punctual_arrivals.is_(None), 0), else_=1).label(
            "measured_arrivals"
        )
    )
    source = select(*selected)
    if only_valid:
        source = source.where(valid_rows_clause(service_date))
    elif service_date is not None:
        source = source.where(stg_fact_stop_event.c.date_key == service_date)

    target_names = [*source_names, "measured_arrivals"]
    insert_stmt = pg_insert(fact_stop_event).from_select(target_names, source)
    updatable = [name for name in target_names if name not in GRAIN_COLUMNS]
    upsert = insert_stmt.on_conflict_do_update(
        index_elements=list(GRAIN_COLUMNS),
        set_={name: insert_stmt.excluded[name] for name in updatable} | {"loaded_at": func.now()},
    ).returning(_WAS_INSERTED)
    outcomes = [bool(row[0]) for row in conn.execute(upsert)]
    inserted = sum(outcomes)
    return LoadCounts(rows_inserted=inserted, rows_updated=len(outcomes) - inserted)


def load_fact(
    engine: Engine,
    source_dir: Path,
    *,
    service_date: dt.date | None = None,
    on_violation: OnViolation = "fail",
    run_id: uuid.UUID | None = None,
) -> LoadCounts:
    """Stage, validate and promote the fact export.

    Under ``fail`` a single violation aborts the load and leaves staging intact for
    diagnosis. Under ``skip`` every offending row is excluded and counted as rejected.

    Raises:
        DataQualityError: if rules failed and ``on_violation`` is ``fail``.
        IngestionError: if the source file is missing or violates its contract.
        DatabaseError: if staging or promotion fails.
    """
    path = source_dir / "fact_stop_event.parquet"
    run_id = run_id or new_run_id()
    entry_id = start_run(
        engine,
        run_id=run_id,
        table_name="fact_stop_event",
        source_file=str(path),
        date_key=service_date,
    )
    try:
        with engine.begin() as conn:
            _stage_fact(conn, path)
            staged = _staged_in_scope(conn, service_date)
            violations = validate_staged_fact(conn, service_date)
            if violations:
                _report(violations, on_violation)
            if violations and on_violation == "fail":
                raise DataQualityError(
                    f"{len(violations)} rule(s) failed on {staged} staged row(s): "
                    + " | ".join(v.describe() for v in violations)
                )
            counts = _promote(conn, service_date, only_valid=bool(violations))
            promoted = counts.rows_inserted + counts.rows_updated
            counts = LoadCounts(
                rows_read=staged,
                rows_rejected=staged - promoted,
                rows_inserted=counts.rows_inserted,
                rows_updated=counts.rows_updated,
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
