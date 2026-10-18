"""Integration tests for the daily rollups of the fact.

A rollup is only worth having if it is exact: every figure the fact gives at a
coarser grain must come out identical from the rollup.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, Table, func, select, text

from rail_rag.core.exceptions import DatabaseError
from rail_rag.db.models import (
    GOLD_SCHEMA,
    ORDERED_TABLES,
    ROLLUP_TABLES,
    agg_relation_day,
    agg_station_day,
    dim_date,
    dim_relation,
    dim_station,
    fact_stop_event,
    metadata,
)
from rail_rag.db.rollups import StaleRollupsError, assert_rollups_fresh
from rail_rag.db.schema import existing_tables, missing_tables
from rail_rag.ingestion.derivations import refresh_rollups
from rail_rag.observability import span, start_trace
from rail_rag.rag.context import build_context, render_schema
from rail_rag.rag.context.rules import SEMANTIC_RULES
from rail_rag.rag.prompts import build_sql_system
from rail_rag.rag.sql.guard import validate_sql
from rail_rag.rag.sql.policy import SqlPolicy, load_retrieval_config

pytestmark = pytest.mark.integration

POLICY = SqlPolicy(
    allowed_tables=frozenset(
        {"gold.dim_date", "gold.dim_station", "gold.dim_relation", "gold.fact_stop_event"}
    )
)
REPO_RETRIEVAL_CONFIG = Path(__file__).resolve().parents[1] / "config" / "retrieval_config.yaml"

STATIONS = [101, 102, 103]
# Station 101 holds two stop points, so its rollup row sums both.
STOP_POINTS = [("a" * 32, 101), ("b" * 32, 102), ("c" * 32, 103), ("f" * 32, 101)]
RELATIONS = ["d" * 32, "e" * 32]
DAYS = [dt.date(2024, 12, 30), dt.date(2024, 12, 31), dt.date(2025, 1, 1), dt.date(2025, 1, 2)]
MEASURES = ("stop_events", "measured_arrivals", "punctual_arrivals")


def _calendar_row(day: dt.date) -> dict[str, Any]:
    return {
        "date_key": day,
        "year": day.year,
        "quarter": (day.month - 1) // 3 + 1,
        "month": day.month,
        "month_name": day.strftime("%B"),
        "week_of_year": day.isocalendar().week,
        "day_of_week": day.isoweekday(),
        "day_name": day.strftime("%A"),
        "is_weekend": day.isoweekday() in (6, 7),
    }


def _fact_rows(days: list[dt.date]) -> list[dict[str, Any]]:
    """Several trains per station and day, with unmeasured arrivals and weighted events."""
    rows = []
    for day_index, day in enumerate(days):
        for station_index, (stop_point, station) in enumerate(STOP_POINTS):
            for train in range(1, 4):
                punctual = [1, 0, None][(day_index + station_index + train) % 3]
                rows.append(
                    {
                        "date_key": day,
                        "station_key": station,
                        "stop_point_key": stop_point,
                        "relation_key": RELATIONS[(station_index + train) % 2],
                        "train_no": train,
                        "planned_hour": 8,
                        "delay_arr_s": 30,
                        "delay_dep_s": 30,
                        "dwell_delta_s": 0,
                        "punctual_arrivals": punctual,
                        "stop_events": 1 + train % 2,
                        "measured_arrivals": 0 if punctual is None else 1,
                    }
                )
    return rows


def _load(engine: Engine, days: list[dt.date]) -> None:
    with engine.begin() as conn:
        known = set(conn.execute(select(dim_date.c.date_key)).scalars())
        conn.execute(dim_date.insert(), [_calendar_row(d) for d in days if d not in known])
        if not known:
            conn.execute(
                dim_station.insert(),
                [
                    {"station_key": key, "station_name": f"station {key}", "is_passenger": True}
                    for key in STATIONS
                ],
            )
            conn.execute(
                dim_relation.insert(),
                [
                    {"relation_key": key, "relation": f"IC {key[0]}", "operator": "SNCB"}
                    for key in RELATIONS
                ],
            )
        conn.execute(fact_stop_event.insert(), _fact_rows(days))


@pytest.fixture
def loaded(clean_schema: Engine) -> Engine:
    _load(clean_schema, DAYS[:3])
    refresh_rollups(clean_schema)
    return clean_schema


def _rows(engine: Engine, sql: str) -> list[tuple[Any, ...]]:
    with engine.connect() as conn:
        return [tuple(row) for row in conn.execute(text(sql))]


def _sums(table: Table) -> str:
    return ", ".join(f"SUM({m}) AS {m}" for m in MEASURES)


@pytest.mark.parametrize(
    ("rollup", "key"), [(agg_station_day, "station_key"), (agg_relation_day, "relation_key")]
)
def test_rollup_sums_equal_fact_sums_at_every_grain(
    loaded: Engine, rollup: Table, key: str
) -> None:
    fact, agg = f"{GOLD_SCHEMA}.fact_stop_event", f"{GOLD_SCHEMA}.{rollup.name}"
    sums = _sums(rollup)
    for group in ("date_key", key, "date_key, " + key):
        assert _rows(
            loaded, f"SELECT {group}, {sums} FROM {fact} GROUP BY {group} ORDER BY 1, 2"
        ) == (_rows(loaded, f"SELECT {group}, {sums} FROM {agg} GROUP BY {group} ORDER BY 1, 2")), (
            group
        )
    assert _rows(loaded, f"SELECT {sums} FROM {fact}") == _rows(loaded, f"SELECT {sums} FROM {agg}")
    assert _rows(loaded, f"SELECT count(*) FROM {fact}") == _rows(
        loaded, f"SELECT SUM(row_count) FROM {agg}"
    )


def test_the_fixture_exercises_unmeasured_arrivals(loaded: Engine) -> None:
    """Otherwise the NULL handling would pass without being tested."""
    assert _rows(
        loaded, "SELECT count(*) FROM gold.fact_stop_event WHERE punctual_arrivals IS NULL"
    )[0][0]


_STATION_RATE = """
SELECT s.station_name, d.year,
       CAST(SUM({p}) AS DOUBLE PRECISION) / NULLIF(SUM({m}), 0) AS rate
FROM gold.{src} AS f
JOIN gold.dim_station AS s ON f.station_key = s.station_key
JOIN gold.dim_date AS d ON f.date_key = d.date_key
GROUP BY s.station_name, d.year ORDER BY 1, 2
"""
_RELATION_RATE = """
SELECT r.relation, d.year,
       CAST(SUM({p}) AS DOUBLE PRECISION) / NULLIF(SUM({m}), 0) AS rate
FROM gold.{src} AS f
JOIN gold.dim_relation AS r ON f.relation_key = r.relation_key
JOIN gold.dim_date AS d ON f.date_key = d.date_key
GROUP BY r.relation, d.year ORDER BY 1, 2
"""


@pytest.mark.parametrize(
    ("template", "rollup"),
    [(_STATION_RATE, "agg_station_day"), (_RELATION_RATE, "agg_relation_day")],
)
def test_a_reference_shaped_query_gives_the_same_answer_on_both(
    loaded: Engine, template: str, rollup: str
) -> None:
    measures = {"p": "punctual_arrivals", "m": "measured_arrivals"}
    on_fact = _rows(loaded, template.format(src="fact_stop_event", **measures))
    on_rollup = _rows(loaded, template.format(src=rollup, **measures))

    assert on_fact == on_rollup
    assert len(on_fact) > 2


def _content(engine: Engine) -> dict[str, list[tuple[Any, ...]]]:
    return {
        table.name: _rows(engine, f"SELECT * FROM {GOLD_SCHEMA}.{table.name} ORDER BY 1, 2")
        for table in ROLLUP_TABLES
    }


def test_refreshing_twice_gives_the_same_content(loaded: Engine) -> None:
    first = _content(loaded)
    counts = refresh_rollups(loaded)

    assert _content(loaded) == first
    assert counts == {name: len(rows) for name, rows in first.items()}


def test_refresh_reflects_a_newly_loaded_day(loaded: Engine) -> None:
    before = _content(loaded)
    _load(loaded, [DAYS[3]])
    refresh_rollups(loaded)

    after = _content(loaded)
    assert {row[0] for row in after["agg_station_day"]} == set(DAYS)
    assert len(after["agg_station_day"]) == len(before["agg_station_day"]) + len(STATIONS)
    with loaded.connect() as conn:
        total = conn.execute(select(func.sum(fact_stop_event.c.stop_events))).scalar_one()
        rolled = conn.execute(select(func.sum(agg_station_day.c.stop_events))).scalar_one()
    assert total == rolled


def test_an_empty_fact_gives_empty_rollups(clean_schema: Engine) -> None:
    assert refresh_rollups(clean_schema) == {"agg_station_day": 0, "agg_relation_day": 0}


def test_each_rollup_is_keyed_on_its_grain_and_not_partitioned() -> None:
    for table, key in ((agg_station_day, "station_key"), (agg_relation_day, "relation_key")):
        assert [c.name for c in table.primary_key.columns] == ["date_key", key]
        assert {tuple(c.name for c in index.columns) for index in table.indexes} == {(key,)}
        assert table.dialect_options["postgresql"]["partition_by"] is None


def test_db_init_creates_the_rollups(clean_schema: Engine) -> None:
    assert {t.name for t in ROLLUP_TABLES} <= existing_tables(clean_schema)
    assert missing_tables(clean_schema) == set()


def _exposed_policy() -> SqlPolicy:
    return load_retrieval_config(REPO_RETRIEVAL_CONFIG).sql


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_the_rollups_are_part_of_the_gold_metadata_after_the_fact() -> None:
    names = [t.name for t in ORDERED_TABLES]
    assert {t.name for t in ROLLUP_TABLES} <= set(names)
    assert all(names.index(t.name) > names.index("fact_stop_event") for t in ROLLUP_TABLES)
    assert all(f"{GOLD_SCHEMA}.{t.name}" in metadata.tables for t in ROLLUP_TABLES)


def test_the_repository_allow_list_names_both_rollups() -> None:
    allowed = _exposed_policy().allowed_tables
    assert {"gold.agg_station_day", "gold.agg_relation_day"} <= allowed


def test_the_rendered_schema_and_context_include_both_rollups(loaded: Engine) -> None:
    policy = _exposed_policy()
    for rendered in (render_schema(policy), build_context(loaded, policy)):
        assert "TABLE gold.agg_station_day" in rendered
        assert "TABLE gold.agg_relation_day" in rendered
        assert "PRIMARY KEY: (date_key, station_key)" in rendered
        assert "PRIMARY KEY: (date_key, relation_key)" in rendered
        assert "exact daily sum of fact_stop_event" in _flat(rendered)


def test_tables_outside_the_allow_list_stay_hidden(loaded: Engine) -> None:
    rendered = build_context(loaded, _exposed_policy())
    assert "stg_fact_stop_event" not in rendered
    assert "profile_snapshot" not in rendered


def test_the_prompt_carries_the_rollup_and_period_rules(loaded: Engine) -> None:
    system = _flat(build_sql_system(build_context(loaded, _exposed_policy())))

    assert "DAILY ROLLUPS" in system
    assert "Network totals are sums over agg_station_day" in system
    assert "Query fact_stop_event only for what the rollups cannot answer" in system
    assert "date_key >= DATE '2025-01-01' AND date_key < DATE '2026-01-01'" in system
    assert "Use dim_date only to output calendar attributes" in system
    assert "Stop events are SUM(stop_events) on a rollup" in system
    assert "Counting rows is therefore counting stop events" in _flat(SEMANTIC_RULES)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT SUM(punctual_arrivals) AS p FROM gold.agg_station_day"
        " WHERE date_key >= DATE '2025-01-01' AND date_key < DATE '2026-01-01'",
        "SELECT s.station_name, CAST(SUM(a.punctual_arrivals) AS DOUBLE PRECISION)"
        " / NULLIF(SUM(a.measured_arrivals), 0) AS rate"
        " FROM gold.agg_station_day a JOIN gold.dim_station s ON a.station_key = s.station_key"
        " WHERE a.date_key >= DATE '2025-01-01' AND a.date_key < DATE '2026-01-01'"
        " GROUP BY s.station_name",
        "SELECT d.month, SUM(a.stop_events) FROM gold.agg_relation_day a"
        " JOIN gold.dim_date d ON a.date_key = d.date_key"
        " WHERE a.date_key >= DATE '2025-01-01' AND a.date_key < DATE '2026-01-01'"
        " GROUP BY d.month",
    ],
)
def test_the_guard_accepts_rollup_queries(sql: str) -> None:
    assert validate_sql(sql, _exposed_policy()).tables


def _stale(loaded: Engine) -> None:
    _load(loaded, [DAYS[3]])


def test_matching_dates_pass_and_are_recorded_on_the_trace(loaded: Engine) -> None:
    with start_trace("startup") as trace, span("rollups"):
        assert_rollups_fresh(loaded)

    assert trace.spans[0].attributes == {
        "fact_max": DAYS[2].isoformat(),
        "station_rollup_max": DAYS[2].isoformat(),
        "relation_rollup_max": DAYS[2].isoformat(),
        "fresh": True,
    }


def test_a_stale_rollup_raises_an_error_that_names_the_fix(loaded: Engine) -> None:
    _stale(loaded)

    with start_trace("startup") as trace, span("rollups"), pytest.raises(StaleRollupsError) as e:
        assert_rollups_fresh(loaded)

    assert "refresh-rollups" in str(e.value)
    assert DAYS[3].isoformat() in str(e.value)
    assert trace.spans[0].attributes["fresh"] is False


def test_rollups_that_were_never_filled_are_stale(clean_schema: Engine) -> None:
    _load(clean_schema, DAYS[:1])

    with pytest.raises(StaleRollupsError):
        assert_rollups_fresh(clean_schema)


def test_an_empty_database_passes(clean_schema: Engine) -> None:
    assert_rollups_fresh(clean_schema)


def test_a_missing_rollup_table_is_a_clear_database_error(loaded: Engine) -> None:
    with loaded.begin() as conn:
        conn.execute(text(f'DROP TABLE "{GOLD_SCHEMA}"."agg_relation_day"'))

    with pytest.raises(DatabaseError, match="db-init"):
        assert_rollups_fresh(loaded)
