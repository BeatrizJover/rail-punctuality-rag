"""Write loader provenance to ``ops.load_runs``.

Every function commits on its own connection, deliberately outside whatever
transaction the loader has open: the record of a failed load must survive the
rollback that made it a failure.
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import Engine, func, insert, update
from sqlalchemy.exc import SQLAlchemyError

from rail_rag.core.exceptions import DatabaseError
from rail_rag.db.models import (
    ERROR_MAX_LEN,
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
    RUN_STATUS_SUCCEEDED,
    load_runs,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadCounts:
    """Row tallies for one table in one load run."""

    rows_read: int = 0
    rows_rejected: int = 0
    rows_inserted: int = 0
    rows_updated: int = 0


def new_run_id() -> uuid.UUID:
    """Return an identifier grouping every table touched by one invocation."""
    return uuid.uuid4()


def start_run(
    engine: Engine,
    *,
    run_id: uuid.UUID,
    table_name: str,
    source_file: str,
    date_key: dt.date | None = None,
) -> int:
    """Open a ``running`` entry and return its id.

    Raises:
        DatabaseError: if the entry cannot be written.
    """
    statement = (
        insert(load_runs)
        .values(
            run_id=run_id,
            table_name=table_name,
            source_file=source_file,
            date_key=date_key,
            status=RUN_STATUS_RUNNING,
        )
        .returning(load_runs.c.id)
    )
    try:
        # A fresh connection from the pool, so the caller's open transaction is untouched.
        with engine.begin() as conn:
            entry_id = conn.execute(statement).scalar_one()
    except SQLAlchemyError as exc:
        raise DatabaseError(f"Could not open a load run entry: {type(exc).__name__}") from exc
    logger.info("load run %s started for %s (entry %d)", run_id, table_name, entry_id)
    return int(entry_id)


def finish_run(engine: Engine, entry_id: int, counts: LoadCounts) -> None:
    """Close an entry as ``succeeded`` with its final tallies."""
    _close(engine, entry_id, status=RUN_STATUS_SUCCEEDED, counts=counts, error=None)


def fail_run(
    engine: Engine,
    entry_id: int,
    error: str,
    counts: LoadCounts | None = None,
) -> None:
    """Close an entry as ``failed``, keeping whatever tallies were reached."""
    _close(
        engine,
        entry_id,
        status=RUN_STATUS_FAILED,
        counts=counts or LoadCounts(),
        error=error[:ERROR_MAX_LEN],
    )


def _close(
    engine: Engine,
    entry_id: int,
    *,
    status: str,
    counts: LoadCounts,
    error: str | None,
) -> None:
    statement = (
        update(load_runs)
        .where(load_runs.c.id == entry_id)
        .values(
            status=status,
            error=error,
            rows_read=counts.rows_read,
            rows_rejected=counts.rows_rejected,
            rows_inserted=counts.rows_inserted,
            rows_updated=counts.rows_updated,
            finished_at=func.now(),
        )
    )
    try:
        with engine.begin() as conn:
            result = conn.execute(statement)
    except SQLAlchemyError as exc:
        raise DatabaseError(f"Could not close load run entry: {type(exc).__name__}") from exc
    if result.rowcount != 1:
        raise DatabaseError(f"Load run entry {entry_id} not found")
    logger.info("load run entry %d closed as %s (%s)", entry_id, status, counts)
