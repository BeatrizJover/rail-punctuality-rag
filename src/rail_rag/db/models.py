"""
Gold star schema definition for PostgreSQL.

The upstream Gold layer (Databricks / Delta) is mirrored here as a SQLAlchemy
Core ``MetaData``: three conformed dimensions around one additive fact table,
plus an unconstrained staging table used by the loader.

"""

from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    SmallInteger,
    String,
    Table,
    func,
)

#: Dedicated PostgreSQL schema, so Gold tables never collide with future
#: RAG-side tables (embeddings, chat traces) living in ``public``.
GOLD_SCHEMA = "gold"

#: MD5 hex digest length, used by ``station_key`` and ``relation_key``.
_HASH_LEN = 32

metadata = MetaData(schema=GOLD_SCHEMA)


dim_date = Table(
    "dim_date",
    metadata,
    Column("date_key", Date, primary_key=True),
    Column("year", SmallInteger, nullable=False),
    Column("quarter", SmallInteger, nullable=False),
    Column("month", SmallInteger, nullable=False),
    Column("month_name", String(16), nullable=False),
    Column("week_of_year", SmallInteger, nullable=False),
    # Spark's dayofweek(): 1 = Sunday .. 7 = Saturday. Deliberately NOT the ISO
    # convention; kept as-is so the mirror stays faithful to the source.
    Column("day_of_week", SmallInteger, nullable=False),
    Column("day_name", String(16), nullable=False),
    Column("is_weekend", Boolean, nullable=False),
    Column("loaded_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    CheckConstraint("day_of_week BETWEEN 1 AND 7", name="ck_dim_date_day_of_week"),
    CheckConstraint("month BETWEEN 1 AND 12", name="ck_dim_date_month"),
    CheckConstraint("quarter BETWEEN 1 AND 4", name="ck_dim_date_quarter"),
    comment="Generated calendar, 2014-2027. Far wider than the fact window.",
)


dim_station = Table(
    "dim_station",
    metadata,
    Column("station_key", String(_HASH_LEN), primary_key=True),
    Column("station_name", String(128), nullable=True),
    # Nullable by design: ptcar_no only reaches Gold via the monthly export.
    Column("ptcar_no", Integer, nullable=True),
    Column("observed_stop_events", BigInteger, nullable=True),
    Column("first_seen", Date, nullable=True),
    Column("last_seen", Date, nullable=True),
    Column("loaded_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    comment="Measuring point identity, keyed by MD5 of the normalized name.",
)


dim_relation = Table(
    "dim_relation",
    metadata,
    Column("relation_key", String(_HASH_LEN), primary_key=True),
    Column("relation", String(32), nullable=True),
    Column("relation_direction", String(256), nullable=True),
    Column("operator", String(64), nullable=True),
    Column("loaded_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    comment="MD5 of relation | direction | operator, as produced upstream.",
)


def _fact_columns() -> list[Column[Any]]:
    """
    Return a fresh set of fact columns, shared by the fact and its staging twin.
    Columns cannot be attached to two Tables, so they are rebuilt per call.
    """
    return [
        Column("date_key", Date, nullable=False),
        Column("station_key", String(_HASH_LEN), nullable=False),
        Column("relation_key", String(_HASH_LEN), nullable=False),
        Column("train_no", Integer, nullable=False),
        Column("planned_hour", SmallInteger, nullable=True),
        # Negative delays are early arrivals: valid data.
        Column("delay_arr_s", Integer, nullable=True),
        Column("delay_dep_s", Integer, nullable=True),
        Column("dwell_delta_s", Integer, nullable=True),
        Column("punctual_arrivals", SmallInteger, nullable=True),
        Column("measured_arrivals", SmallInteger, nullable=False),
        Column("stop_events", SmallInteger, nullable=False),
    ]


fact_stop_event = Table(
    "fact_stop_event",
    metadata,
    *_fact_columns(),
    Column("loaded_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Index(None, "date_key", "station_key", "train_no", unique=True),
    ForeignKeyConstraint(["date_key"], [dim_date.c.date_key], name="fk_fact_date"),
    ForeignKeyConstraint(["station_key"], [dim_station.c.station_key], name="fk_fact_station"),
    ForeignKeyConstraint(["relation_key"], [dim_relation.c.relation_key], name="fk_fact_relation"),
    CheckConstraint("planned_hour BETWEEN 0 AND 23", name="ck_fact_planned_hour"),
    CheckConstraint("punctual_arrivals IN (0, 1)", name="ck_fact_punctual_arrivals"),
    CheckConstraint("stop_events >= 0", name="ck_fact_stop_events"),
    CheckConstraint(
        "(punctual_arrivals IS NULL AND measured_arrivals = 0)"
        " OR (punctual_arrivals IS NOT NULL AND measured_arrivals = 1)",
        name="ck_fact_measured_arrivals_matches_punctual",
    ),
    comment=(
        "Grain: one train passing one measuring point on one service date. "
        "Additive measures: stop_events, measured_arrivals, punctual_arrivals. "
        "Punctuality rate = SUM(punctual_arrivals) / SUM(measured_arrivals)."
    ),
)


stg_fact_stop_event = Table(
    "stg_fact_stop_event",
    metadata,
    *_fact_columns(),
    comment="Unconstrained landing zone for one load batch. Truncated per run.",
)


#: Creation order: dimensions first, so the fact's foreign keys resolve.
ORDERED_TABLES: tuple[Table, ...] = (
    dim_date,
    dim_station,
    dim_relation,
    fact_stop_event,
    stg_fact_stop_event,
)
