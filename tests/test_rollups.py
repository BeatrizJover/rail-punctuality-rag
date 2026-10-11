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
from rail_rag.db.schema import existing_tables, missing_tables
from rail_rag.ingestion.derivations import refresh_rollups
from rail_rag.rag.context import build_context, render_schema
from rail_rag.rag.exceptions import UnsafeQueryError
from rail_rag.rag.sql.guard import validate_sql
from rail_rag.rag.sql.policy import SqlPolicy, load_retrieval_config

pytestmark = pytest.mark.integration

POLICY = SqlPolicy(
    allowed_tables=frozenset(
        {"gold.dim_date", "gold.dim_station", "gold.dim_relation", "gold.fact_stop_event"}
    )
)
REPO_RETRIEVAL_CONFIG = Path(__file__).resolve().parents[1] / "config" / "retrieval_config.yaml"

STATIONS = ["a" * 32, "b" * 32, "c" * 32]
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
        for station_index, station in enumerate(STATIONS):
            for train in range(1, 4):
                punctual = [1, 0, None][(day_index + station_index + train) % 3]
                rows.append(
                    {
                        "date_key": day,
                        "station_key": station,
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
                [{"station_key": key, "station_name": f"station {key[0]}"} for key in STATIONS],
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


def test_the_rollups_are_invisible_to_the_model(loaded: Engine) -> None:
    names = {t.name for t in ROLLUP_TABLES}
    allowed = load_retrieval_config(REPO_RETRIEVAL_CONFIG).sql.allowed_tables
    rendered = render_schema(POLICY) + build_context(loaded, POLICY)

    assert not names & {t.name for t in ORDERED_TABLES}
    assert not any(f"{GOLD_SCHEMA}.{name}" in metadata.tables for name in names)
    assert not any(name in rendered for name in names)
    assert not any(name in table for table in allowed for name in names)
    for name in names:
        with pytest.raises(UnsafeQueryError):
            validate_sql(f"SELECT * FROM gold.{name}", POLICY)
