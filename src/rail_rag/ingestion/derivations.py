"""Post-load steps that close the ingestion: derive station activity from the
fact, then assert referential integrity across the whole load.

These run after the fact is in place. The integrity assertion is the counterpart
to dropping enforced foreign keys on the partitioned fact: correctness is checked
once over the finished load, not row by row on every insert.
"""

from __future__ import annotations

import logging

from sqlalchemy import Engine, exists, func, select, text
from sqlalchemy.exc import SQLAlchemyError

from rail_rag.core.exceptions import DatabaseError, RailRagError
from rail_rag.db.models import (
    GOLD_SCHEMA,
    dim_date,
    dim_relation,
    dim_station,
    fact_stop_event,
)

logger = logging.getLogger(__name__)


class ReferentialIntegrityError(RailRagError):
    """Raised when fact rows reference a dimension key that does not exist."""


#: SUM over the additive measure, not COUNT(*): stop_events is the grain's weight,
#: and a future non-unit weight must still total correctly.
_DERIVE_STATION_ACTIVITY = text(
    f"""
    UPDATE "{GOLD_SCHEMA}".dim_station AS s
    SET observed_stop_events = COALESCE(f.events, 0),
        first_seen = f.first_seen,
        last_seen = f.last_seen
    FROM (
        SELECT ds.station_key,
               SUM(fs.stop_events) AS events,
               MIN(fs.date_key) AS first_seen,
               MAX(fs.date_key) AS last_seen
        FROM "{GOLD_SCHEMA}".dim_station AS ds
        LEFT JOIN "{GOLD_SCHEMA}".fact_stop_event AS fs
            ON fs.station_key = ds.station_key
        GROUP BY ds.station_key
    ) AS f
    WHERE f.station_key = s.station_key
    """
)


def derive_station_activity(engine: Engine) -> int:
    """Fill dim_station's activity columns from the loaded fact.

    ``observed_stop_events`` becomes the station's SUM of stop_events (0 when the
    station never appears in the fact); ``first_seen``/``last_seen`` its date span
    (NULL when absent). Returns the number of dimension rows updated.

    Raises:
        DatabaseError: if the update fails.
    """
    try:
        with engine.begin() as conn:
            updated = conn.execute(_DERIVE_STATION_ACTIVITY).rowcount
    except SQLAlchemyError as exc:
        raise DatabaseError(f"Could not derive station activity: {type(exc).__name__}") from exc
    logger.info("derived station activity for %d station(s)", updated)
    return updated


def _orphan_count(engine: Engine) -> int:
    """Fact rows whose date, station or relation key is missing from its dimension."""
    missing_date = ~exists(select(1).where(dim_date.c.date_key == fact_stop_event.c.date_key))
    missing_station = ~exists(
        select(1).where(dim_station.c.station_key == fact_stop_event.c.station_key)
    )
    missing_relation = ~exists(
        select(1).where(dim_relation.c.relation_key == fact_stop_event.c.relation_key)
    )
    statement = (
        select(func.count())
        .select_from(fact_stop_event)
        .where(missing_date | missing_station | missing_relation)
    )
    with engine.connect() as conn:
        return int(conn.execute(statement).scalar_one())


_OPTIMIZE_STATEMENTS: tuple[str, ...] = (
    f'ALTER TABLE "{GOLD_SCHEMA}".fact_stop_event ALTER COLUMN station_key SET STATISTICS 500',
    f'ALTER TABLE "{GOLD_SCHEMA}".fact_stop_event ALTER COLUMN relation_key SET STATISTICS 500',
    "CREATE INDEX IF NOT EXISTS ix_fact_stop_event_station_covering"
    f' ON "{GOLD_SCHEMA}".fact_stop_event (station_key)'
    " INCLUDE (stop_events, punctual_arrivals, measured_arrivals)",
    "CREATE INDEX IF NOT EXISTS ix_fact_stop_event_relation_covering"
    f' ON "{GOLD_SCHEMA}".fact_stop_event (relation_key)'
    " INCLUDE (stop_events, punctual_arrivals, measured_arrivals)",
    f'DROP INDEX IF EXISTS "{GOLD_SCHEMA}".ix_fact_stop_event_station_key',
    f'DROP INDEX IF EXISTS "{GOLD_SCHEMA}".ix_fact_stop_event_relation_key',
    f'VACUUM ANALYZE "{GOLD_SCHEMA}".fact_stop_event',
)


def optimize_fact_for_reads(engine: Engine) -> None:
    """Build the dimension-filter indexes and refresh planner statistics.

    Idempotent, and runs in AUTOCOMMIT: index builds on a partitioned parent do
    not belong in a shared transaction.

    Raises:
        DatabaseError: if any statement fails.
    """
    try:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            for statement in _OPTIMIZE_STATEMENTS:
                conn.execute(text(statement))
    except SQLAlchemyError as exc:
        raise DatabaseError(f"Could not optimize fact for reads: {type(exc).__name__}") from exc
    logger.info("optimized fact_stop_event for reads (indexes + analyze)")


def assert_referential_integrity(engine: Engine) -> None:
    """Assert every fact row resolves to a row in each dimension.

    The fact carries no enforced foreign keys, so this is where the guarantee is
    made: a single count over the finished load, not a constraint on every insert.

    Raises:
        ReferentialIntegrityError: if any fact row references a missing key.
        DatabaseError: if the check cannot run.
    """
    try:
        orphans = _orphan_count(engine)
    except SQLAlchemyError as exc:
        raise DatabaseError(f"Could not check referential integrity: {type(exc).__name__}") from exc
    if orphans:
        raise ReferentialIntegrityError(
            f"{orphans} fact row(s) reference a dimension key that does not exist"
        )
    logger.info("referential integrity holds: no orphan fact rows")
