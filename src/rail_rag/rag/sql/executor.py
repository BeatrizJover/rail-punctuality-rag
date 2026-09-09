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

import logging
import time
from collections.abc import Sequence
from typing import Any

from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError

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
