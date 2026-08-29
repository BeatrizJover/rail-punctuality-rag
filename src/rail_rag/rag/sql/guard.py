"""Static validation of model-generated SQL.

The guard runs before anything reaches a database connection, and it works on a
parsed syntax tree rather than on the query text. String matching is not an
option here: ``DROP`` hidden inside a comment, a second statement after a
semicolon, or a ``DELETE`` wrapped in a CTE all defeat a regular expression
while remaining perfectly visible in an AST.

Validation is an allow-list at every level that has one — statement kind, table
names — and a deny-list only where an allow-list is impractical, namely
functions.
"""

from __future__ import annotations

import logging
from typing import cast

import sqlglot
from sqlglot import exp

from rail_rag.rag.exceptions import UnsafeQueryError
from rail_rag.rag.sql.policy import DEFAULT_SCHEMA, SqlPolicy

logger = logging.getLogger(__name__)

_DIALECT = "postgres"

#: Node kinds that must not appear anywhere in the tree, at any depth.
#: ``Command`` is the important one: it is what sqlglot produces for statements it
#: does not model, so it catches VACUUM, COPY, SET and anything else exotic.
_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.TruncateTable,
    exp.Merge,
    exp.Grant,
    exp.Copy,
    exp.Command,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Use,
    exp.Attach,
    exp.Set,
    # ``SELECT ... INTO`` writes a new table; ``FOR UPDATE`` takes row locks.
    exp.Into,
    exp.Lock,
)


class SafeQuery:
    """A query that passed validation, rewritten with an enforced ``LIMIT``."""

    __slots__ = ("sql", "limit", "tables")

    def __init__(self, sql: str, limit: int, tables: frozenset[str]) -> None:
        self.sql = sql
        self.limit = limit
        self.tables = tables

    def __repr__(self) -> str:
        return f"SafeQuery(limit={self.limit}, tables={sorted(self.tables)})"


def validate_sql(sql: str, policy: SqlPolicy) -> SafeQuery:
    """Parse, check and normalise a generated query.

    Returns:
        The rewritten SQL, guaranteed to be a single read-only statement over
        allowed tables with a bounded row count.

    Raises:
        UnsafeQueryError: if the query is unparseable or violates the policy.
    """
    statements = _parse(sql)
    if len(statements) != 1:
        # A trailing semicolon parses to a single statement; two real ones do not.
        raise UnsafeQueryError(f"Expected exactly one statement, found {len(statements)}")

    root = statements[0]
    # Explicit isinstance rather than a tuple constant, so the type narrows to the
    # two shapes that actually carry a LIMIT clause.
    if not isinstance(root, exp.Select | exp.Union):
        raise UnsafeQueryError(f"Only SELECT queries are allowed, found {type(root).__name__}")

    _reject_forbidden_nodes(root)
    _reject_blocked_functions(root, policy)
    tables = _referenced_tables(root)
    _reject_unknown_tables(tables, policy)

    limited = _enforce_limit(root, policy.max_rows)
    return SafeQuery(
        sql=limited.sql(dialect=_DIALECT),
        limit=policy.max_rows,
        tables=tables,
    )


def _parse(sql: str) -> list[exp.Expression]:
    if not sql or not sql.strip():
        raise UnsafeQueryError("Query is empty")
    try:
        # sqlglot types parse() with an unbound TypeVar, so the cast states what
        # it actually returns rather than widening every caller to Any.
        parsed = cast(list[exp.Expression | None], sqlglot.parse(sql, dialect=_DIALECT))
    except sqlglot.ParseError as exc:
        raise UnsafeQueryError(f"Query could not be parsed: {exc}") from exc
    # sqlglot yields None for an empty fragment, e.g. the tail of "SELECT 1;;".
    return [item for item in parsed if item is not None]


def _reject_forbidden_nodes(root: exp.Expression) -> None:
    for node in root.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise UnsafeQueryError(f"{type(node).__name__.upper()} is not allowed in a query")


def _reject_blocked_functions(root: exp.Expression, policy: SqlPolicy) -> None:
    if not policy.blocked_functions:
        return
    blocked = {name.lower() for name in policy.blocked_functions}
    for func in root.find_all(exp.Func):
        # Anonymous covers anything sqlglot has no dedicated node for, which is
        # where the dangerous ones live: pg_read_file, pg_sleep, dblink.
        name = str(func.this) if isinstance(func, exp.Anonymous) else func.sql_name()
        if name.lower() in blocked:
            raise UnsafeQueryError(f"Function {name} is not allowed")


def _referenced_tables(root: exp.Expression) -> frozenset[str]:
    """Return qualified names of real tables, excluding CTE aliases."""
    cte_names = {cte.alias.lower() for cte in root.find_all(exp.CTE)}
    tables: set[str] = set()
    for table in root.find_all(exp.Table):
        name = table.name.lower()
        schema = (table.db or "").lower()
        if not schema and name in cte_names:
            continue
        tables.add(f"{schema or DEFAULT_SCHEMA}.{name}")
    return frozenset(tables)


def _reject_unknown_tables(tables: frozenset[str], policy: SqlPolicy) -> None:
    allowed = {name.lower() for name in policy.allowed_tables}
    unknown = sorted(tables - allowed)
    if unknown:
        raise UnsafeQueryError(f"Table(s) not allowed: {', '.join(unknown)}")


def _enforce_limit(root: exp.Select | exp.Union, max_rows: int) -> exp.Select | exp.Union:
    """Cap the outermost row count, tightening a larger limit and adding a missing one."""
    current = root.args.get("limit")
    if current is not None:
        try:
            written = int(current.expression.name)
        except (AttributeError, ValueError):
            written = max_rows + 1
        if written <= max_rows:
            return root
    return root.limit(max_rows, copy=True)
