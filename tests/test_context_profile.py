"""Integration tests for the measured data profile.

The scenario under test is the one that motivates the module: a calendar
dimension spanning years, a fact table spanning days, and a question about the
gap between them.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import Engine

from rail_rag.db.models import dim_date, dim_relation, dim_station, fact_stop_event
from rail_rag.rag.context import build_context, load_profile
from rail_rag.rag.sql.policy import SqlPolicy

pytestmark = pytest.mark.integration

POLICY = SqlPolicy(
    allowed_tables=frozenset(
        {
            "gold.dim_date",
            "gold.dim_station",
            "gold.dim_relation",
            "gold.fact_stop_event",
        }
    )
)

STATION_KEY = "a" * 32
RELATION_KEY = "c" * 32
FIRST_DAY = dt.date(2025, 8, 1)
LAST_DAY = dt.date(2025, 8, 3)


def _calendar_row(day: dt.date) -> dict[str, object]:
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


@pytest.fixture
def narrow_data(clean_schema: Engine) -> Engine:
    """A wide calendar over a deliberately narrow fact window."""
    with clean_schema.begin() as conn:
        days = [FIRST_DAY + dt.timedelta(days=offset) for offset in range(3)]
        calendar = [dt.date(2014, 1, 1), dt.date(2027, 12, 31), *days]
        conn.execute(dim_date.insert(), [_calendar_row(day) for day in calendar])
        conn.execute(
            dim_station.insert(),
            [
                {
                    "station_key": STATION_KEY,
                    "station_name": "Bruxelles-Midi",
                    "observed_stop_events": 900,
                },
                {
                    "station_key": "b" * 32,
                    "station_name": "Gent-Sint-Pieters",
                    "observed_stop_events": 400,
                },
            ],
        )
        conn.execute(
            dim_relation.insert(),
            [{"relation_key": RELATION_KEY, "relation": "IC 01", "operator": "SNCB"}],
        )
        conn.execute(
            fact_stop_event.insert(),
            [
                {
                    "date_key": day,
                    "station_key": STATION_KEY,
                    "relation_key": RELATION_KEY,
                    "train_no": 1000 + index,
                    "planned_hour": 8,
                    "delay_arr_s": 30,
                    "delay_dep_s": 30,
                    "dwell_delta_s": 0,
                    "punctual_arrivals": 1,
                    "stop_events": 1,
                    "measured_arrivals": 1,
                }
                for day in days
                for index in range(2)
            ],
        )
    return clean_schema


def test_profile_measures_the_real_window(narrow_data: Engine) -> None:
    profile = load_profile(narrow_data)
    assert profile.fact_rows == 6
    assert profile.fact_min_date == FIRST_DAY
    assert profile.fact_max_date == LAST_DAY
    assert profile.fact_distinct_days == 3
    assert profile.fact_distinct_stations == 1


def test_profile_reports_the_calendar_separately(narrow_data: Engine) -> None:
    """The two ranges must not be conflated; that conflation is the whole bug."""
    profile = load_profile(narrow_data)
    assert profile.calendar_min_date == dt.date(2014, 1, 1)
    assert profile.calendar_max_date == dt.date(2027, 12, 31)
    assert profile.fact_min_date is not None
    assert profile.calendar_min_date < profile.fact_min_date


def test_coverage_predicates(narrow_data: Engine) -> None:
    profile = load_profile(narrow_data)
    assert profile.covers(dt.date(2025, 8, 2))
    assert not profile.covers(dt.date(2020, 6, 1))
    # A wider question still overlaps and can be answered with a caveat.
    assert profile.overlaps(dt.date(2025, 1, 1), dt.date(2025, 12, 31))
    # A question entirely outside the window cannot.
    assert not profile.overlaps(dt.date(2020, 1, 1), dt.date(2024, 12, 31))


def test_rendered_profile_names_the_gap(narrow_data: Engine) -> None:
    rendered = load_profile(narrow_data).render()
    assert "2025-08-01 to 2025-08-03" in rendered
    assert "WIDER than the data" in rendered
    assert "no data behind it" in rendered


def test_value_samples_teach_the_spelling(narrow_data: Engine) -> None:
    """A model that has not seen the name will invent one that matches nothing."""
    profile = load_profile(narrow_data)
    assert "Bruxelles-Midi" in profile.sample_stations
    # Ordered by observed volume, so the most useful names come first.
    assert profile.sample_stations[0] == "Bruxelles-Midi"
    assert "SNCB" in profile.sample_operators


def test_sample_size_is_respected(narrow_data: Engine) -> None:
    profile = load_profile(narrow_data, sample_size=1)
    assert len(profile.sample_stations) == 1


def test_empty_fact_is_reported_as_empty(clean_schema: Engine) -> None:
    """An empty fact table must be stated, not discovered per query."""
    profile = load_profile(clean_schema)
    assert not profile.has_data
    assert not profile.covers(FIRST_DAY)
    assert not profile.overlaps(FIRST_DAY, LAST_DAY)
    assert "EMPTY" in profile.render()


def test_build_context_assembles_all_three_parts(narrow_data: Engine) -> None:
    context = build_context(narrow_data, POLICY)
    assert "TABLE gold.fact_stop_event" in context
    assert "NULLIF(SUM(measured_arrivals), 0)" in context
    assert "DATA COVERAGE" in context
    assert "stg_fact_stop_event" not in context
