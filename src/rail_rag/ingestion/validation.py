"""Data-quality rules evaluated in SQL against the staging table.

Rules are expressed as failure conditions, mirroring the Databricks ``ops.dq_results``
convention: ``rows_failed == 0`` means PASS. They run over staging rather than over the
incoming stream so that a bad batch is diagnosed as a set, with counts and samples,
instead of as whichever row happened to hit a constraint first.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import ColumnElement, Connection, and_, exists, func, literal, or_, select

from rail_rag.core.exceptions import RailRagError
from rail_rag.db.models import (
    dim_date,
    dim_relation,
    dim_station,
    stg_fact_stop_event,
)

#: Offending rows reported per rule; enough to diagnose, short enough to log.
SAMPLE_SIZE = 5

#: The natural key of the fact, and the conflict target of the promotion upsert.
GRAIN_COLUMNS = ("date_key", "station_key", "train_no")

_KEY_COLUMNS = ("date_key", "station_key", "relation_key", "train_no")

RULE_NULL_KEYS = "null_keys"
RULE_DUPLICATE_GRAIN = "duplicate_grain"
RULE_ORPHAN_DATE = "orphan_date"
RULE_ORPHAN_STATION = "orphan_station"
RULE_ORPHAN_RELATION = "orphan_relation"


class DataQualityError(RailRagError):
    """Raised when staged rows violate a rule and the policy is to fail."""


@dataclass(frozen=True)
class Violation:
    """One failing rule, with its count and a sample of offending rows."""

    rule: str
    rows_failed: int
    sample: tuple[dict[str, Any], ...]

    def describe(self) -> str:
        """One-line summary for logs and error messages."""
        return f"{self.rule}: {self.rows_failed} row(s); sample={list(self.sample)}"


def _has_null_key() -> ColumnElement[bool]:
    return or_(*(stg_fact_stop_event.c[name].is_(None) for name in _KEY_COLUMNS))


def _is_orphan(
    dimension_key: ColumnElement[Any], staged_key: ColumnElement[Any]
) -> ColumnElement[bool]:
    """True when a non-null staged key has no match in its dimension."""
    return and_(
        staged_key.is_not(None),
        ~exists(select(literal(1)).where(dimension_key == staged_key)),
    )


def _duplicate_grain_groups() -> Any:
    """Subquery of grain tuples appearing more than once in staging."""
    grain = [stg_fact_stop_event.c[name] for name in GRAIN_COLUMNS]
    return (
        select(*grain).where(~_has_null_key()).group_by(*grain).having(func.count() > 1).subquery()
    )


def _is_duplicated() -> ColumnElement[bool]:
    duplicates = _duplicate_grain_groups()
    return exists(
        select(literal(1)).where(
            and_(*(stg_fact_stop_event.c[name] == duplicates.c[name] for name in GRAIN_COLUMNS))
        )
    )


def _date_scope(service_date: dt.date | None) -> ColumnElement[bool]:
    if service_date is None:
        return literal(True)
    return stg_fact_stop_event.c.date_key == service_date


def valid_rows_clause(service_date: dt.date | None = None) -> ColumnElement[bool]:
    """Rows that pass every rule; the promotion filter under the ``skip`` policy."""
    return and_(
        _date_scope(service_date),
        ~_has_null_key(),
        ~_is_duplicated(),
        ~_is_orphan(dim_date.c.date_key, stg_fact_stop_event.c.date_key),
        ~_is_orphan(dim_station.c.station_key, stg_fact_stop_event.c.station_key),
        ~_is_orphan(dim_relation.c.relation_key, stg_fact_stop_event.c.relation_key),
    )


def _evaluate(
    conn: Connection,
    rule: str,
    condition: ColumnElement[bool],
    service_date: dt.date | None,
) -> Violation | None:
    scoped = and_(_date_scope(service_date), condition)
    rows_failed = int(
        conn.execute(
            select(func.count()).select_from(stg_fact_stop_event).where(scoped)
        ).scalar_one()
    )
    if rows_failed == 0:
        return None
    sample_rows = conn.execute(
        select(*(stg_fact_stop_event.c[name] for name in _KEY_COLUMNS))
        .where(scoped)
        .limit(SAMPLE_SIZE)
    ).mappings()
    return Violation(rule=rule, rows_failed=rows_failed, sample=tuple(dict(r) for r in sample_rows))


def validate_staged_fact(
    conn: Connection,
    service_date: dt.date | None = None,
) -> list[Violation]:
    """Evaluate every rule over staging, returning only the ones that failed."""
    checks: Sequence[tuple[str, ColumnElement[bool]]] = (
        (RULE_NULL_KEYS, _has_null_key()),
        (RULE_DUPLICATE_GRAIN, _is_duplicated()),
        (
            RULE_ORPHAN_DATE,
            _is_orphan(dim_date.c.date_key, stg_fact_stop_event.c.date_key),
        ),
        (
            RULE_ORPHAN_STATION,
            _is_orphan(dim_station.c.station_key, stg_fact_stop_event.c.station_key),
        ),
        (
            RULE_ORPHAN_RELATION,
            _is_orphan(dim_relation.c.relation_key, stg_fact_stop_event.c.relation_key),
        ),
    )
    violations = [
        violation
        for rule, condition in checks
        if (violation := _evaluate(conn, rule, condition, service_date)) is not None
    ]
    return violations
