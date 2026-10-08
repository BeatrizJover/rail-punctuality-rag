"""Monthly range partitions for ``gold.fact_stop_event``.

SQLAlchemy Core declares the partitioned parent but has no construct for the
children, so their DDL is emitted here. There is deliberately no ``DEFAULT``
partition: a row outside the declared window must fail the load rather than
land in a catch-all that later blocks attaching the partition it belongs to.
"""

from __future__ import annotations

import datetime as dt
import logging
import re

from sqlalchemy import Connection, Engine, MetaData, text

from rail_rag.db.models import GOLD_SCHEMA, fact_stop_event, metadata

logger = logging.getLogger(__name__)

#: Window created by ``db-init``, wide enough for the three-year export plus headroom.
BASELINE_START = dt.date(2024, 1, 1)
BASELINE_END = dt.date(2027, 1, 1)

_PARENT = fact_stop_event.name

_PARTITION_BY = re.compile(r"\s*(?:RANGE|LIST|HASH)\s*\(\s*(\w+)\s*\)\s*", re.IGNORECASE)


def _first_of_next_month(day: dt.date) -> dt.date:
    return dt.date(day.year + day.month // 12, day.month % 12 + 1, 1)


def partition_name(month: dt.date) -> str:
    """Child table name for the month containing ``month``."""
    return f"{_PARENT}_{month:%Y_%m}"


def monthly_bounds(start: dt.date, end: dt.date) -> list[tuple[str, dt.date, dt.date]]:
    """Name and half-open bounds of every month overlapping ``[start, end)``."""
    if end <= start:
        raise ValueError(f"end ({end}) must be after start ({start})")
    bounds: list[tuple[str, dt.date, dt.date]] = []
    lower = start.replace(day=1)
    while lower < end:
        upper = _first_of_next_month(lower)
        bounds.append((partition_name(lower), lower, upper))
        lower = upper
    return bounds


def ensure_partitions(conn: Connection, start: dt.date, end: dt.date) -> list[str]:
    """Create the monthly children covering ``[start, end)``, returning their names."""
    names: list[str] = []
    for name, lower, upper in monthly_bounds(start, end):
        conn.execute(
            text(
                f'CREATE TABLE IF NOT EXISTS "{GOLD_SCHEMA}"."{name}"'
                f' PARTITION OF "{GOLD_SCHEMA}"."{_PARENT}"'
                f" FOR VALUES FROM ('{lower.isoformat()}') TO ('{upper.isoformat()}')"
            )
        )
        names.append(name)
    return names


def existing_partitions(engine: Engine) -> set[str]:
    """Return the fact's current child partitions."""
    statement = text(
        "SELECT child.relname FROM pg_inherits"
        " JOIN pg_class AS child ON child.oid = inhrelid"
        " JOIN pg_class AS parent ON parent.oid = inhparent"
        " JOIN pg_namespace AS ns ON ns.oid = parent.relnamespace"
        " WHERE parent.relname = :parent AND ns.nspname = :schema"
    )
    with engine.connect() as conn:
        rows = conn.execute(statement, {"parent": _PARENT, "schema": GOLD_SCHEMA}).scalars()
        return {str(name) for name in rows}


def partition_keys(models: MetaData = metadata) -> dict[str, str]:
    """Map each partitioned table, qualified and lower-case, to its single partition key column."""
    keys: dict[str, str] = {}
    for name, table in models.tables.items():
        clause = table.dialect_options["postgresql"]["partition_by"]
        if clause is None:
            continue
        match = _PARTITION_BY.fullmatch(str(clause))
        if match is None:
            raise ValueError(f"Unsupported partition_by on {name}: {clause!r}")
        keys[name.lower()] = match.group(1).lower()
    return keys
