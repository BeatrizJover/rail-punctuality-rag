"""Pydantic v2 contracts for the Gold export.

One model per Gold table. They are the boundary between the untyped export on disk
and the typed load into PostgreSQL: a row that does not satisfy its contract never
reaches staging, and the failure names the table, the row and the offending field.

The contracts encode the nullability the *real* Gold sample exhibits, not a guess.
``ptcar_no`` and ``relation_direction`` are nullable because production rows leave
them empty; the fact's keys and ``stop_events`` are required because every sampled
row carries them. ``measured_arrivals`` is absent on purpose: it is derived in SQL
when the loader promotes staging into the fact, and never travels through Python.

Reading Parquet (not CSV) means nulls arrive as ``None`` natively, so there is no
``"null"`` string literal to intercept here - that was an artefact of the manual
CSV extraction and does not survive a real export.
"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict, Field


class _GoldRow(BaseModel):
    """Base for every Gold contract.

    ``extra="forbid"`` turns an unannounced upstream column into a loud failure
    instead of a silently dropped field - the ingestion-side mirror of the
    ``unexpected_source_columns`` drift check in the Databricks pipeline.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class DimDateRow(_GoldRow):
    """One calendar day. Generated upstream, 2014-2027, so no field is null."""

    date_key: dt.date
    year: int
    quarter: int = Field(ge=1, le=4)
    month: int = Field(ge=1, le=12)
    month_name: str
    week_of_year: int = Field(ge=1, le=53)
    # Spark's dayofweek(): 1 = Sunday .. 7 = Saturday. Kept faithful to the source.
    day_of_week: int = Field(ge=1, le=7)
    day_name: str
    is_weekend: bool


class DimStationRow(_GoldRow):
    """A measuring point, keyed by MD5 of the normalized name.

    ``ptcar_no`` is nullable: it only reaches Gold via the monthly export, so the
    daily-driven rows legitimately lack it.
    """

    station_key: str = Field(min_length=32, max_length=32)
    station_name: str | None = None
    ptcar_no: int | None = None
    observed_stop_events: int | None = Field(default=None, ge=0)
    first_seen: dt.date | None = None
    last_seen: dt.date | None = None


class DimRelationRow(_GoldRow):
    """A train relation.

    ``relation_direction`` is nullable, and this is where the upstream
    ``concat_ws`` behaviour surfaces: a NULL direction is skipped in the hash
    input, so ``(THAL, NULL, SNCB/NMBS)`` hashes as ``THAL|SNCB/NMBS``. The
    contract accepts the NULL as-is; reconciling the collision is upstream's job.
    """

    relation_key: str = Field(min_length=32, max_length=32)
    relation: str | None = None
    relation_direction: str | None = None
    operator: str | None = None


class FactStopEventRow(_GoldRow):
    """One train passing one measuring point on one service date.

    ``punctual_arrivals`` is preserved exactly as it arrives - ``None`` means the
    arrival was never measured, and it must never be coerced to 0, or the
    punctuality ratio's denominator would count a measurement that did not happen.
    ``measured_arrivals`` is intentionally not part of this contract.
    """

    date_key: dt.date
    station_key: str = Field(min_length=32, max_length=32)
    relation_key: str = Field(min_length=32, max_length=32)
    train_no: int
    planned_hour: int | None = Field(default=None, ge=0, le=23)
    # Negative delays are early arrivals: valid data, so no lower bound.
    delay_arr_s: int | None = None
    delay_dep_s: int | None = None
    dwell_delta_s: int | None = None
    punctual_arrivals: int | None = Field(default=None, ge=0, le=1)
    stop_events: int = Field(ge=0)
