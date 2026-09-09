"""Integration tests for the staging validation rules.

Each rule is exercised by staging a deliberately broken row and asserting the rule
catches it with a count and a sample. Marked ``integration``; skipped without a database.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest
from sqlalchemy import Engine, select

from rail_rag.db.models import (
    dim_date,
    dim_relation,
    dim_station,
    stg_fact_stop_event,
)
from rail_rag.ingestion.validation import (
    RULE_DUPLICATE_GRAIN,
    RULE_NULL_KEYS,
    RULE_ORPHAN_DATE,
    RULE_ORPHAN_RELATION,
    RULE_ORPHAN_STATION,
    SAMPLE_SIZE,
    valid_rows_clause,
    validate_staged_fact,
)

pytestmark = pytest.mark.integration

_DATE = dt.date(2026, 8, 23)
_STATION = "a" + "0" * 31
_RELATION = "b" + "0" * 31


def _seed_dimensions(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            dim_date.insert().values(
                date_key=_DATE,
                year=2026,
                quarter=3,
                month=8,
                month_name="August",
                week_of_year=34,
                day_of_week=1,
                day_name="Sunday",
                is_weekend=True,
            )
        )
        conn.execute(dim_station.insert().values(station_key=_STATION, station_name="MOENSBERG"))
        conn.execute(dim_relation.insert().values(relation_key=_RELATION, relation="IC 01"))


def _staged_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "date_key": _DATE,
        "station_key": _STATION,
        "relation_key": _RELATION,
        "train_no": 501,
        "planned_hour": 7,
        "delay_arr_s": 12,
        "delay_dep_s": 30,
        "dwell_delta_s": 0,
        "punctual_arrivals": 1,
        "stop_events": 1,
    }
    row.update(overrides)
    return row


def _stage(engine: Engine, *rows: dict[str, Any]) -> None:
    with engine.begin() as conn:
        conn.execute(stg_fact_stop_event.insert(), list(rows))


def _violations(engine: Engine, service_date: dt.date | None = None) -> dict[str, int]:
    with engine.connect() as conn:
        return {v.rule: v.rows_failed for v in validate_staged_fact(conn, service_date)}


def _valid_count(engine: Engine, service_date: dt.date | None = None) -> int:
    with engine.connect() as conn:
        rows = conn.execute(
            select(stg_fact_stop_event.c.train_no).where(valid_rows_clause(service_date))
        ).all()
    return len(rows)


def test_clean_batch_reports_no_violations(clean_schema: Engine) -> None:
    _seed_dimensions(clean_schema)
    _stage(clean_schema, _staged_row())
    assert _violations(clean_schema) == {}
    assert _valid_count(clean_schema) == 1


def test_null_key_is_caught(clean_schema: Engine) -> None:
    """Staging accepts the null; the rule is what rejects it."""
    _seed_dimensions(clean_schema)
    _stage(clean_schema, _staged_row(station_key=None))
    assert _violations(clean_schema)[RULE_NULL_KEYS] == 1
    assert _valid_count(clean_schema) == 0


def test_duplicate_grain_flags_every_member_of_the_group(clean_schema: Engine) -> None:
    """Both rows are excluded: a duplicate grain means neither is known to be right."""
    _seed_dimensions(clean_schema)
    _stage(clean_schema, _staged_row(), _staged_row(delay_arr_s=999))
    assert _violations(clean_schema)[RULE_DUPLICATE_GRAIN] == 2
    assert _valid_count(clean_schema) == 0


def test_orphan_station_is_caught(clean_schema: Engine) -> None:
    _seed_dimensions(clean_schema)
    _stage(clean_schema, _staged_row(station_key="c" + "0" * 31))
    assert _violations(clean_schema)[RULE_ORPHAN_STATION] == 1


def test_orphan_relation_is_caught(clean_schema: Engine) -> None:
    _seed_dimensions(clean_schema)
    _stage(clean_schema, _staged_row(relation_key="d" + "0" * 31))
    assert _violations(clean_schema)[RULE_ORPHAN_RELATION] == 1


def test_orphan_date_is_caught(clean_schema: Engine) -> None:
    """dim_date spans 2014-2027, so a date outside it is a real upstream signal."""
    _seed_dimensions(clean_schema)
    _stage(clean_schema, _staged_row(date_key=dt.date(2030, 1, 1)))
    assert _violations(clean_schema)[RULE_ORPHAN_DATE] == 1


def test_a_row_can_fail_several_rules_at_once(clean_schema: Engine) -> None:
    _seed_dimensions(clean_schema)
    _stage(clean_schema, _staged_row(station_key="c" + "0" * 31, date_key=dt.date(2030, 1, 1)))
    failed = _violations(clean_schema)
    assert failed[RULE_ORPHAN_STATION] == 1
    assert failed[RULE_ORPHAN_DATE] == 1


def test_sample_is_bounded(clean_schema: Engine) -> None:
    """A 20M-row failure must not build a 20M-row error message."""
    _seed_dimensions(clean_schema)
    orphan = "c" + "0" * 31
    _stage(clean_schema, *[_staged_row(station_key=orphan, train_no=n) for n in range(20)])
    with clean_schema.connect() as conn:
        violations = {v.rule: v for v in validate_staged_fact(conn)}
    assert violations[RULE_ORPHAN_STATION].rows_failed == 20
    assert len(violations[RULE_ORPHAN_STATION].sample) == SAMPLE_SIZE


def test_date_scope_narrows_the_evaluation(clean_schema: Engine) -> None:
    """A bad row on another day is not this run's problem when --date is given."""
    _seed_dimensions(clean_schema)
    _stage(
        clean_schema,
        _staged_row(),
        _staged_row(date_key=dt.date(2030, 1, 1), train_no=777),
    )
    assert _violations(clean_schema, _DATE) == {}
    assert _violations(clean_schema)[RULE_ORPHAN_DATE] == 1
