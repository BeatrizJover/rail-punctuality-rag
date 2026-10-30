"""Integration tests for the measured data profile.

The scenario under test is the one that motivates the module: a calendar
dimension spanning years, a fact table spanning days, and a question about the
gap between them.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from sqlalchemy import Engine, text

from rail_rag.db.models import (
    dim_date,
    dim_relation,
    dim_station,
    fact_stop_event,
    metadata,
    profile_snapshot,
)
from rail_rag.db.run_log import LoadCounts, finish_run, new_run_id, start_run
from rail_rag.observability import span, start_trace
from rail_rag.rag.context import DataProfile, build_context, load_profile, render_schema
from rail_rag.rag.exceptions import UnsafeQueryError
from rail_rag.rag.sql.guard import validate_sql
from rail_rag.rag.sql.policy import SqlPolicy, load_retrieval_config

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

REPO_RETRIEVAL_CONFIG = Path(__file__).resolve().parents[1] / "config" / "retrieval_config.yaml"

STATION_KEY = 220
STOP_POINT_KEY = "a" * 32
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
                    "is_passenger": True,
                    "observed_stop_events": 900,
                },
                {
                    "station_key": 455,
                    "station_name": "Gent-Sint-Pieters",
                    "is_passenger": True,
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
                    "stop_point_key": STOP_POINT_KEY,
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
    assert "1 stations" in rendered


def test_the_unresolved_station_is_never_sampled(narrow_data: Engine) -> None:
    """Teaching the model the name 'Unresolved' would invite it into a ranking."""
    with narrow_data.begin() as conn:
        conn.execute(
            dim_station.insert(),
            [
                {
                    "station_key": -1,
                    "station_name": "Unresolved",
                    "is_passenger": False,
                    "observed_stop_events": 10_000,
                }
            ],
        )
    assert "Unresolved" not in load_profile(narrow_data).sample_stations


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


# --- the stored profile: reused while the data is unchanged ------------------


def _profiled(engine: Engine, **kwargs: int) -> tuple[DataProfile, object]:
    """Load a profile the way the pipeline does, and report where it came from."""
    with start_trace("startup") as trace, span("profile"):
        profile = load_profile(engine, **kwargs)
    return profile, trace.spans[0].attributes.get("source")


def _add_fact_day(engine: Engine, day: dt.date) -> None:
    with engine.begin() as conn:
        conn.execute(dim_date.insert(), [_calendar_row(day)])
        conn.execute(
            fact_stop_event.insert(),
            [
                {
                    "date_key": day,
                    "station_key": STATION_KEY,
                    "stop_point_key": STOP_POINT_KEY,
                    "relation_key": RELATION_KEY,
                    "train_no": 1,
                    "planned_hour": 8,
                    "delay_arr_s": 30,
                    "delay_dep_s": 30,
                    "dwell_delta_s": 0,
                    "punctual_arrivals": 1,
                    "stop_events": 1,
                    "measured_arrivals": 1,
                }
            ],
        )


def test_the_first_profile_is_computed_and_the_second_comes_from_the_snapshot(
    narrow_data: Engine,
) -> None:
    computed, first = _profiled(narrow_data)
    stored, second = _profiled(narrow_data)

    assert (first, second) == ("computed", "snapshot")
    assert stored == computed


def test_the_rendered_profile_is_identical_on_both_paths(narrow_data: Engine) -> None:
    computed, _ = _profiled(narrow_data)
    stored, source = _profiled(narrow_data)

    assert source == "snapshot"
    assert stored.render() == computed.render()
    assert build_context(narrow_data, POLICY).endswith(computed.render())


def test_a_new_load_run_invalidates_the_snapshot(narrow_data: Engine) -> None:
    _profiled(narrow_data)

    entry = start_run(narrow_data, run_id=new_run_id(), table_name="dim_date", source_file="x")
    assert _profiled(narrow_data)[1] == "computed"
    assert _profiled(narrow_data)[1] == "snapshot"

    finish_run(narrow_data, entry, LoadCounts())
    assert _profiled(narrow_data)[1] == "computed"
    assert _profiled(narrow_data)[1] == "snapshot"


def test_a_fact_row_outside_the_stored_window_invalidates_the_snapshot(
    narrow_data: Engine,
) -> None:
    _profiled(narrow_data)
    _add_fact_day(narrow_data, LAST_DAY + dt.timedelta(days=1))

    profile, source = _profiled(narrow_data)

    assert source == "computed"
    assert profile.fact_max_date == LAST_DAY + dt.timedelta(days=1)
    assert profile.fact_rows == 7


def test_a_different_sample_size_is_not_served_from_the_snapshot(narrow_data: Engine) -> None:
    _profiled(narrow_data)

    profile, source = _profiled(narrow_data, sample_size=1)

    assert source == "computed"
    assert len(profile.sample_stations) == 1


def test_the_snapshot_is_invisible_to_the_model(narrow_data: Engine) -> None:
    _profiled(narrow_data)
    allowed = load_retrieval_config(REPO_RETRIEVAL_CONFIG).sql.allowed_tables

    assert f"{profile_snapshot.schema}.{profile_snapshot.name}" not in metadata.tables
    assert "profile_snapshot" not in render_schema(POLICY)
    assert "profile_snapshot" not in build_context(narrow_data, POLICY)
    assert not any("profile_snapshot" in name or name.startswith("ops.") for name in allowed)
    with pytest.raises(UnsafeQueryError):
        validate_sql("SELECT * FROM ops.profile_snapshot", POLICY)


def test_an_empty_database_keeps_its_behaviour_on_the_snapshot_path(clean_schema: Engine) -> None:
    computed, first = _profiled(clean_schema)
    stored, second = _profiled(clean_schema)

    assert (first, second) == ("computed", "snapshot")
    assert not stored.has_data
    assert stored == computed
    assert stored.render() == computed.render()
    assert "EMPTY" in stored.render()


def test_dropping_gold_drops_the_snapshot(narrow_data: Engine) -> None:
    from rail_rag.db.schema import create_schema, drop_schema

    _profiled(narrow_data)
    drop_schema(narrow_data)
    create_schema(narrow_data)

    assert _profiled(narrow_data)[1] == "computed"


def test_a_missing_snapshot_table_falls_back_to_computing(narrow_data: Engine) -> None:
    with narrow_data.begin() as conn:
        conn.execute(text(f'DROP TABLE "{profile_snapshot.schema}"."{profile_snapshot.name}"'))

    profile, source = _profiled(narrow_data)

    assert source == "computed"
    assert profile.fact_rows == 6


def test_a_snapshot_with_an_older_format_version_is_recomputed_once(narrow_data: Engine) -> None:
    _profiled(narrow_data)
    with narrow_data.begin() as conn:
        conn.execute(text("UPDATE ops.profile_snapshot SET profile = profile - 'format_version'"))

    assert _profiled(narrow_data)[1] == "computed"
    assert _profiled(narrow_data)[1] == "snapshot"

    with narrow_data.begin() as conn:
        conn.execute(
            text(
                "UPDATE ops.profile_snapshot"
                " SET profile = jsonb_set(profile, '{format_version}', '1')"
            )
        )
    assert _profiled(narrow_data)[1] == "computed"


def test_the_current_format_version_is_stored_in_the_snapshot(narrow_data: Engine) -> None:
    from rail_rag.rag.context.profile import PROFILE_FORMAT_VERSION

    _profiled(narrow_data)
    with narrow_data.connect() as conn:
        stored = conn.execute(text("SELECT profile ->> 'format_version' FROM ops.profile_snapshot"))
        assert stored.scalar_one() == str(PROFILE_FORMAT_VERSION)


def test_operator_and_relation_samples_are_alphabetical(narrow_data: Engine) -> None:
    with narrow_data.begin() as conn:
        conn.execute(
            dim_relation.insert(),
            [
                {"relation_key": f"{i}" * 32, "relation": relation, "operator": operator}
                for i, (relation, operator) in enumerate(
                    [("S 9", "Zeta"), ("IC 20", "Alpha"), ("L 3", "Mu"), ("P 1", "Beta")], start=1
                )
            ],
        )

    profile = load_profile(narrow_data)

    assert list(profile.sample_operators) == sorted(profile.sample_operators)
    assert list(profile.sample_relations) == sorted(profile.sample_relations)
    assert profile.sample_operators[:2] == ("Alpha", "Beta")
    assert profile.sample_relations[:2] == ("IC 01", "IC 20")
