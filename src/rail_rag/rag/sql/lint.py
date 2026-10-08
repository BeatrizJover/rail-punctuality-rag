"""Static checks that flag likely performance problems in validated SQL.

Nothing here rejects a query. The point is to make a known cause visible before
execution: a filter on the fact reached only through a dimension gives the
planner no predicate on the partition key, so every partition is scanned.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import Scope, traverse_scope

from rail_rag.db.models import metadata
from rail_rag.db.partitions import partition_keys
from rail_rag.rag.sql.policy import DEFAULT_SCHEMA

FACT_WITHOUT_PARTITION_FILTER = "fact_without_partition_filter"

_DIALECT = "postgres"
_COMPARISONS = (exp.EQ, exp.LT, exp.LTE, exp.GT, exp.GTE)

_IsKey = Callable[[exp.Expression], bool]


def lint_sql(sql: str, keys: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Return the distinct finding codes for ``sql``, in the order they were found."""
    partitioned = partition_keys() if keys is None else keys
    findings: list[str] = []
    for scope in traverse_scope(sqlglot.parse_one(sql, dialect=_DIALECT)):
        for alias, (_, source) in scope.selected_sources.items():
            if not isinstance(source, exp.Table):
                continue
            key = partitioned.get(_qualified(source))
            if key is not None and not _prunes_scope(scope, alias, key):
                findings.append(FACT_WITHOUT_PARTITION_FILTER)
    return tuple(dict.fromkeys(findings))


def _qualified(table: exp.Table) -> str:
    return f"{(table.db or DEFAULT_SCHEMA).lower()}.{table.name.lower()}"


def _prunes_scope(scope: Scope, alias: str, key: str) -> bool:
    """True when this scope's WHERE or JOIN conditions bound the table's partition key."""
    select = scope.expression
    conditions: list[exp.Expression] = []
    where = select.args.get("where")
    if where is not None:
        conditions.append(where.this)
    conditions.extend(
        join.args["on"] for join in select.args.get("joins") or [] if join.args.get("on")
    )

    def is_key(node: exp.Expression) -> bool:
        while isinstance(node, exp.Paren):
            node = node.this
        if not isinstance(node, exp.Column) or node.name.lower() != key:
            return False
        if node.table:
            return node.table.lower() == alias.lower()
        return _unambiguous(scope, alias, key)

    return any(_prunes(condition, is_key) for condition in conditions)


def _unambiguous(scope: Scope, alias: str, column: str) -> bool:
    """An unqualified column belongs to the table only if no other source could supply it."""
    for other_alias, (_, other) in scope.selected_sources.items():
        if other_alias == alias:
            continue
        if not isinstance(other, exp.Table) or column in _model_columns(other):
            return False
    return True


def _model_columns(table: exp.Table) -> set[str]:
    model = metadata.tables.get(_qualified(table))
    return set() if model is None else {column.name.lower() for column in model.c}


def _prunes(condition: exp.Expression, is_key: _IsKey) -> bool:
    """True when the condition cannot hold without bounding the key by constants."""
    if isinstance(condition, exp.Paren):
        return _prunes(condition.this, is_key)
    if isinstance(condition, exp.And):
        return _prunes(condition.this, is_key) or _prunes(condition.expression, is_key)
    if isinstance(condition, exp.Or):
        return _prunes(condition.this, is_key) and _prunes(condition.expression, is_key)
    if isinstance(condition, _COMPARISONS):
        left, right = condition.this, condition.expression
        return (is_key(left) and _is_constant(right)) or (is_key(right) and _is_constant(left))
    if isinstance(condition, exp.Between):
        return (
            is_key(condition.this)
            and _is_constant(condition.args["low"])
            and _is_constant(condition.args["high"])
        )
    if isinstance(condition, exp.In):
        values = condition.expressions
        return (
            is_key(condition.this)
            and bool(values)
            and not condition.args.get("query")
            and all(_is_constant(value) for value in values)
        )
    return False


def _is_constant(node: exp.Expression) -> bool:
    return node.find(exp.Column, exp.Query) is None
