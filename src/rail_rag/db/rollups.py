"""Whether the daily rollups still cover everything the fact holds."""

from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy import Engine, func, select
from sqlalchemy.exc import SQLAlchemyError

from rail_rag.core.exceptions import DatabaseError
from rail_rag.db.models import agg_relation_day, agg_station_day, fact_stop_event
from rail_rag.observability import annotate

logger = logging.getLogger(__name__)

_REFRESH_HINT = "run 'python scripts/manage.py refresh-rollups'"


class StaleRollupsError(DatabaseError):
    """Raised when a rollup ends on a different date than the fact."""


def assert_rollups_fresh(engine: Engine) -> None:
    """Compare the last date of each rollup with the fact's, and record both on the trace.

    Only the latest date is compared, so a reload of an earlier day goes unnoticed.
    An empty fact is always fresh.

    Raises:
        StaleRollupsError: if a rollup ends on another date than the fact.
        DatabaseError: if the dates cannot be read, for instance a rollup is missing.
    """
    try:
        with engine.connect() as conn:
            fact = conn.execute(select(func.max(fact_stop_event.c.date_key))).scalar_one()
            station = conn.execute(select(func.max(agg_station_day.c.date_key))).scalar_one()
            relation = conn.execute(select(func.max(agg_relation_day.c.date_key))).scalar_one()
    except SQLAlchemyError as exc:
        raise DatabaseError(
            f"Could not read the rollups ({type(exc).__name__}); "
            f"run 'db-init', then {_REFRESH_HINT}"
        ) from exc

    stale = fact is not None and (station != fact or relation != fact)
    annotate(
        fact_max=_iso(fact),
        station_rollup_max=_iso(station),
        relation_rollup_max=_iso(relation),
        fresh=not stale,
    )
    if stale:
        raise StaleRollupsError(
            f"The daily rollups are out of date (fact ends {_iso(fact)}, agg_station_day "
            f"{_iso(station)}, agg_relation_day {_iso(relation)}); {_REFRESH_HINT}"
        )
    logger.info("rollups are fresh (fact ends %s)", _iso(fact))


def _iso(day: dt.date | None) -> str | None:
    return None if day is None else day.isoformat()
