"""Request and response bodies for the HTTP layer.

These are deliberately separate from the pipeline's own dataclasses. ``Answer``
describes what the pipeline computed; these describe what the service promises
over the wire. Collapsing the two would make an internal refactor a breaking
API change.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

#: Guards against a body large enough to cost embedding quota on nonsense.
MAX_QUESTION_CHARS = 500


class AskRequest(BaseModel):
    """One natural-language question."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class SourceOut(BaseModel):
    """A documentation passage the answer drew on."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str
    heading: str | None = None


class ResultOut(BaseModel):
    """The tabular result of a generated query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    columns: list[str]
    rows: list[list[Any]]
    #: True when the executor cut the result at the policy's row limit.
    truncated: bool


class AskResponse(BaseModel):
    """The answer, plus everything needed to audit how it was produced."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    route: str
    #: Always returned rather than gated behind a flag: the query is the audit trail.
    sql: str | None = None
    result: ResultOut | None = None
    sources: list[SourceOut] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Liveness: the process is up and serving."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    environment: str


class ReadyResponse(BaseModel):
    """Readiness: every dependency the pipeline needs is actually usable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ready: bool
    #: Gold or ops tables expected by the loader but absent.
    missing_tables: list[str]
    knowledge_base: bool
    embedding_dimension: int | None = None
    detail: str | None = None


def jsonable_row(row: Sequence[Any]) -> list[Any]:
    """Convert one result row into values the JSON encoder can carry.

    Postgres returns ``Decimal`` for numeric aggregates and ``date`` for date
    columns. Both round-trip badly or not at all as raw JSON, so they are
    normalised here rather than left for the encoder to guess at.
    """
    return [_jsonable_value(value) for value in row]


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    return value
