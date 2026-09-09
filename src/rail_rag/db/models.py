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
    Identity,
    Index,
    Integer,
    MetaData,
    SmallInteger,
    String,
    Table,
    Uuid,
    func,
    text,
)
from sqlalchemy.types import TypeEngine

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

#: The fact columns exactly as they arrive from the Gold export, in export order.
#: ``measured_arrivals`` is deliberately absent: it is a derived measure, computed
#: in SQL when the loader promotes staging rows into the fact table.
#: Negative delays are early arrivals, so no range constraint belongs on them.

_SOURCE_FACT_COLUMNS: tuple[tuple[str, type[TypeEngine[Any]] | TypeEngine[Any]], ...] = (
    ("date_key", Date),
    ("station_key", String(_HASH_LEN)),
    ("relation_key", String(_HASH_LEN)),
    ("train_no", Integer),
    ("planned_hour", SmallInteger),
    ("delay_arr_s", Integer),
    ("delay_dep_s", Integer),
    ("dwell_delta_s", Integer),
    ("punctual_arrivals", SmallInteger),
    ("stop_events", SmallInteger),
)

_REQUIRED_SOURCE_COLUMNS = frozenset(
    {"date_key", "station_key", "relation_key", "train_no", "stop_events"}
)


def _source_columns(*, permissive: bool) -> list[Column[Any]]:
    """
    Build a fresh set of export columns.
    """
    return [
        Column(name, type_, nullable=permissive or name not in _REQUIRED_SOURCE_COLUMNS)
        for name, type_ in _SOURCE_FACT_COLUMNS
    ]


fact_stop_event = Table(
    "fact_stop_event",
    metadata,
    *_source_columns(permissive=False),
    Column("measured_arrivals", SmallInteger, nullable=False),
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
    *_source_columns(permissive=True),
    comment=(
        "Unconstrained landing zone for one load batch, truncated per run. "
        "No NOT NULL, no CHECK, no foreign key and no unique index, by design: "
        "bad rows must land so the loader can diagnose them in SQL."
    ),
)


#: Creation order: dimensions first, so the fact's foreign keys resolve.
ORDERED_TABLES: tuple[Table, ...] = (
    dim_date,
    dim_station,
    dim_relation,
    fact_stop_event,
    stg_fact_stop_event,
)

#: Operational metadata, in its own schema and its own ``MetaData``: dropping and
#: rebuilding Gold must not destroy the record of how it was built.
OPS_SCHEMA = "ops"

ops_metadata = MetaData(schema=OPS_SCHEMA)

#: Longest error message kept; the full traceback belongs in the logs, not here.
ERROR_MAX_LEN = 2000

#: Allowed values of ``load_runs.status``.
RUN_STATUS_RUNNING = "running"
RUN_STATUS_SUCCEEDED = "succeeded"
RUN_STATUS_FAILED = "failed"

load_runs = Table(
    "load_runs",
    ops_metadata,
    Column("id", BigInteger, Identity(), primary_key=True),
    # One invocation groups several rows: load-dims writes one per dimension.
    Column("run_id", Uuid, nullable=False),
    Column("table_name", String(64), nullable=False),
    Column("source_file", String(512), nullable=False),
    # Set for incremental fact loads, null for full-reload dimensions.
    Column("date_key", Date, nullable=True),
    Column("rows_read", BigInteger, nullable=False, server_default=text("0")),
    Column("rows_rejected", BigInteger, nullable=False, server_default=text("0")),
    Column("rows_inserted", BigInteger, nullable=False, server_default=text("0")),
    Column("rows_updated", BigInteger, nullable=False, server_default=text("0")),
    Column("status", String(16), nullable=False, server_default=text(f"'{RUN_STATUS_RUNNING}'")),
    Column("error", String(ERROR_MAX_LEN), nullable=True),
    Column("started_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    CheckConstraint(
        f"status IN ('{RUN_STATUS_RUNNING}', '{RUN_STATUS_SUCCEEDED}', '{RUN_STATUS_FAILED}')",
        name="ck_load_runs_status",
    ),
    CheckConstraint(
        f"(status = '{RUN_STATUS_RUNNING}') = (finished_at IS NULL)",
        name="ck_load_runs_finished_at_matches_status",
    ),
    Index("ix_load_runs_run_id", "run_id"),
    Index("ix_load_runs_table_started", "table_name", "started_at"),
    comment=(
        "One row per table per loader invocation. Written on its own connection, "
        "so a rolled-back load still leaves a record of the attempt."
    ),
)

ORDERED_OPS_TABLES: tuple[Table, ...] = (load_runs,)
