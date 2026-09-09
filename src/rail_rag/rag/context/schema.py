"""The schema description handed to the SQL generator.

Rendered from :mod:`rail_rag.db.models`, never written by hand and never read
back from the live database. The metadata is already the single source of truth
for the tables themselves, so deriving the prompt from it means a new column
reaches the model the moment it reaches the database. A hand-maintained copy
would drift silently, and drift here produces confidently wrong SQL.

Only tables named in the policy are rendered: what the model cannot see, it does
not query. That is the same allow-list the guard enforces, used a second time to
prevent the mistake rather than to catch it.
"""

from __future__ import annotations

from sqlalchemy import Column, Table
from sqlalchemy.dialects.postgresql import dialect as postgresql_dialect
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.schema import CheckConstraint, UniqueConstraint

from rail_rag.db.models import ORDERED_TABLES
from rail_rag.rag.exceptions import RagError
from rail_rag.rag.sql.policy import SqlPolicy

#: SQLAlchemy ships no annotation for its dialect constructors; the type is still known.
_DIALECT: Dialect = postgresql_dialect()  # type: ignore[no-untyped-call]

#: Column width for the rendered name, chosen so the three dimensions align.
_NAME_WIDTH = 22
_TYPE_WIDTH = 14


def render_schema(policy: SqlPolicy) -> str:
    """Return a compact DDL description of every table the model may query.

    Raises:
        RagError: if the policy names a table the metadata does not define,
            which means configuration and schema have diverged.
    """
    tables = _selected_tables(policy)
    blocks = [_render_table(table) for table in tables]
    return "\n\n".join(blocks)


def _selected_tables(policy: SqlPolicy) -> list[Table]:
    by_name = {_qualified(table): table for table in ORDERED_TABLES}
    unknown = sorted({name.lower() for name in policy.allowed_tables} - set(by_name))
    if unknown:
        raise RagError(f"Policy allows table(s) absent from the schema: {', '.join(unknown)}")
    allowed = {name.lower() for name in policy.allowed_tables}
    return [table for name, table in by_name.items() if name in allowed]


def _qualified(table: Table) -> str:
    return f"{table.schema}.{table.name}".lower()


def _render_table(table: Table) -> str:
    lines = [f"TABLE {_qualified(table)}"]
    if table.comment:
        lines.append(f"  -- {table.comment}")
    lines.append("  COLUMNS:")
    lines.extend(f"    {_render_column(column)}" for column in table.columns)

    keys = [column.name for column in table.primary_key.columns]
    if keys:
        lines.append(f"  PRIMARY KEY: ({', '.join(keys)})")

    for constraint in table.constraints:
        if isinstance(constraint, UniqueConstraint):
            names = ", ".join(column.name for column in constraint.columns)
            lines.append(f"  UNIQUE: ({names})")
    for index in table.indexes:
        if index.unique:
            names = ", ".join(column.name for column in index.columns)
            lines.append(f"  UNIQUE: ({names})")

    checks = [
        str(constraint.sqltext)
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    ]
    if checks:
        lines.append("  CHECKS:")
        lines.extend(f"    {check}" for check in sorted(checks))
    return "\n".join(lines)


def _render_column(column: Column[object]) -> str:
    type_name = column.type.compile(dialect=_DIALECT)
    nullable = "NULL" if column.nullable else "NOT NULL"
    rendered = f"{column.name:<{_NAME_WIDTH}} {type_name:<{_TYPE_WIDTH}} {nullable}"
    targets = sorted(key.target_fullname for key in column.foreign_keys)
    if targets:
        rendered += f"  REFERENCES {', '.join(targets)}"
    return rendered
