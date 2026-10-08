"""Execution of validated queries.

The guard decides what may run; this module decides how. The two are separate
because they fail differently: a rejected query is a modelling problem the user
should hear about, while a timeout is an operational one.

Three defences apply at execution time, none of which trusts the guard:

* the transaction is declared ``READ ONLY``, so the server refuses writes even
  if a statement slipped past static validation;
* ``statement_timeout`` is set on the server, so a runaway query is killed
  without the client having to stay alive to cancel it;
* the transaction is always rolled back, never committed.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError

from rail_rag.db.partitions import partition_keys
from rail_rag.rag.exceptions import QueryExecutionError
from rail_rag.rag.sql.guard import SafeQuery
from rail_rag.rag.sql.policy import SqlPolicy

logger = logging.getLogger(__name__)


class QueryResult:
    """Rows returned by a validated query, plus what it cost to get them."""

    __slots__ = ("columns", "rows", "truncated", "elapsed_ms")

    def __init__(
        self,
        columns: Sequence[str],
        rows: Sequence[tuple[Any, ...]],
        truncated: bool,
        elapsed_ms: float,
    ) -> None:
        self.columns = list(columns)
        self.rows = [tuple(row) for row in rows]
        self.truncated = truncated
        self.elapsed_ms = elapsed_ms

    @property
    def is_empty(self) -> bool:
        """True when the query ran fine and matched nothing.

        The caller must distinguish this from an error: an empty result is a
        fact about the data, and the answer has to say so rather than stay silent.
        """
        return not self.rows

    def __repr__(self) -> str:
        return f"QueryResult(rows={len(self.rows)}, truncated={self.truncated})"


def execute_safe_query(engine: Engine, query: SafeQuery, policy: SqlPolicy) -> QueryResult:
    """Run a validated query inside a read-only, time-bounded transaction.

    Raises:
        QueryExecutionError: if the query fails, times out, or the database is down.
    """
    started = time.perf_counter()
    try:
        with engine.connect() as conn:
            # SET LOCAL dies with the transaction, so it cannot leak into a pooled
            # connection reused by the loader or the health check.
            conn.execute(text("SET TRANSACTION READ ONLY"))
            conn.execute(text(f"SET LOCAL statement_timeout = {policy.statement_timeout_ms}"))
            cursor = conn.execute(text(query.sql))
            columns = list(cursor.keys())
            rows = cursor.fetchall()
            conn.rollback()
    except SQLAlchemyError as exc:
        # The DSN can appear in the driver's message; only the class name is safe to surface.
        raise QueryExecutionError(f"Query failed: {type(exc).__name__}") from exc

    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info("query returned %d rows in %.1f ms", len(rows), elapsed_ms)
    return QueryResult(
        columns=columns,
        rows=[tuple(row) for row in rows],
        # Hitting the cap exactly means the real answer is probably larger.
        truncated=len(rows) >= query.limit,
        elapsed_ms=elapsed_ms,
    )


def run_query(engine: Engine, sql: str, policy: SqlPolicy) -> QueryResult:
    """Validate then execute, which is the only supported path to the database.

    Raises:
        UnsafeQueryError: if validation rejects the query.
        QueryExecutionError: if execution fails.
    """
    from rail_rag.rag.sql.guard import validate_sql

    return execute_safe_query(engine, validate_sql(sql, policy), policy)


@dataclass(frozen=True)
class PlanEstimate:
    """What the planner expects a query to cost, without running it."""

    total_cost: float
    rows: float
    #: Distinct child partitions in the plan, keyed by the partitioned parent's qualified name.
    partitions_scanned: Mapping[str, int]


def explain_safe_query(engine: Engine, query: SafeQuery, policy: SqlPolicy) -> PlanEstimate:
    """Plan a validated query without executing it, under the same defences as execution.

    Raises:
        QueryExecutionError: if planning fails, times out, or the database is down.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SET TRANSACTION READ ONLY"))
            conn.execute(text(f"SET LOCAL statement_timeout = {policy.statement_timeout_ms}"))
            raw = conn.execute(text(f"EXPLAIN (FORMAT JSON) {query.sql}")).scalar_one()
            conn.rollback()
    except SQLAlchemyError as exc:
        raise QueryExecutionError(f"Plan failed: {type(exc).__name__}") from exc

    plan = (json.loads(raw) if isinstance(raw, str) else raw)[0]["Plan"]
    return PlanEstimate(
        total_cost=float(plan["Total Cost"]),
        rows=float(plan["Plan Rows"]),
        partitions_scanned=_partitions_in_plan(plan),
    )


def _partitions_in_plan(plan: dict[str, Any]) -> dict[str, int]:
    """Count distinct children of each partitioned table, by the ``<parent>_`` name prefix."""
    children: dict[str, set[str]] = {parent: set() for parent in partition_keys()}
    pending = [plan]
    while pending:
        node = pending.pop()
        pending.extend(node.get("Plans", []))
        relation = node.get("Relation Name")
        if relation is None:
            continue
        for parent, found in children.items():
            # The plan names no schema outside VERBOSE; the allow-list keeps a lookalike out.
            if relation.startswith(f"{parent.split('.', 1)[1]}_"):
                found.add(relation)
    return {parent: len(found) for parent, found in children.items()}
