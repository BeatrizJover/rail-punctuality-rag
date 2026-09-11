"""Tests for the post-load derivations and the referential integrity gate."""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import Engine, text

from rail_rag.db.models import GOLD_SCHEMA
from rail_rag.ingestion.derivations import (
    ReferentialIntegrityError,
    assert_referential_integrity,
    derive_station_activity,
)

_SEED = f"""
INSERT INTO {GOLD_SCHEMA}.dim_date
  (date_key, year, quarter, month, month_name, week_of_year, day_of_week, day_name, is_weekend)
VALUES ('2026-08-23', 2026, 3, 8, 'August', 34, 7, 'Sunday', true),
       ('2026-08-24', 2026, 3, 8, 'August', 34, 1, 'Monday', false);
INSERT INTO {GOLD_SCHEMA}.dim_station (station_key, station_name)
VALUES (repeat('a', 32), 'Active-Two-Days'),
       (repeat('b', 32), 'Active-One-Day'),
       (repeat('c', 32), 'Never-Seen');
INSERT INTO {GOLD_SCHEMA}.dim_relation (relation_key) VALUES (repeat('r', 32));
"""

_FACT = f"""
INSERT INTO {GOLD_SCHEMA}.fact_stop_event
  (date_key, station_key, relation_key, train_no, stop_events, measured_arrivals)
VALUES ('2026-08-23', repeat('a', 32), repeat('r', 32), 1, 1, 0),
       ('2026-08-23', repeat('a', 32), repeat('r', 32), 2, 1, 0),
       ('2026-08-24', repeat('a', 32), repeat('r', 32), 3, 1, 0),
       ('2026-08-23', repeat('b', 32), repeat('r', 32), 9, 1, 0);
"""


@pytest.fixture
def seeded(clean_schema: Engine) -> Engine:
    with clean_schema.begin() as conn:
        conn.execute(text(_SEED))
        conn.execute(text(_FACT))
    return clean_schema


def _station(engine: Engine, letter: str) -> tuple[int | None, dt.date | None, dt.date | None]:
    with engine.connect() as conn:
        row = conn.execute(
            text(
                f"SELECT observed_stop_events, first_seen, last_seen FROM {GOLD_SCHEMA}.dim_station"
                " WHERE station_key = :key"
            ),
            {"key": letter * 32},
        ).one()
    return row[0], row[1], row[2]


def test_activity_sums_the_measure_not_the_days(seeded: Engine) -> None:
    """observed_stop_events is SUM(stop_events): three events over two days is 3, not 2."""
    derive_station_activity(seeded)
    events, first, last = _station(seeded, "a")
    assert events == 3
    assert first == dt.date(2026, 8, 23)
    assert last == dt.date(2026, 8, 24)


def test_a_one_day_station_spans_a_single_date(seeded: Engine) -> None:
    derive_station_activity(seeded)
    events, first, last = _station(seeded, "b")
    assert events == 1
    assert first == last == dt.date(2026, 8, 23)


def test_a_station_absent_from_the_fact_is_zero_not_null(seeded: Engine) -> None:
    """Zero observed events is a fact; NULL would read as unknown."""
    derive_station_activity(seeded)
    events, first, last = _station(seeded, "c")
    assert events == 0
    assert first is None
    assert last is None


def test_deriving_twice_is_stable(seeded: Engine) -> None:
    derive_station_activity(seeded)
    first_pass = _station(seeded, "a")
    derive_station_activity(seeded)
    assert _station(seeded, "a") == first_pass


def test_integrity_holds_on_a_clean_load(seeded: Engine) -> None:
    assert_referential_integrity(seeded)  # does not raise


def test_integrity_catches_an_orphan_fact_row(seeded: Engine) -> None:
    """The enforced FK blocks orphans in practice; the assertion is the belt to that
    braces, so the test disables session triggers to fabricate the failure it guards."""
    with seeded.begin() as conn:
        conn.execute(text("SET session_replication_role = replica"))
        conn.execute(
            text(
                f"INSERT INTO {GOLD_SCHEMA}.fact_stop_event"
                " (date_key, station_key, relation_key, train_no, stop_events, measured_arrivals)"
                " VALUES ('2026-08-23', repeat('a', 32), repeat('z', 32), 77, 1, 0)"
            )
        )
        conn.execute(text("SET session_replication_role = DEFAULT"))
    with pytest.raises(ReferentialIntegrityError, match="1 fact row"):
        assert_referential_integrity(seeded)


def test_derivation_reports_the_stations_it_touched(clean_schema: Engine) -> None:
    with clean_schema.begin() as conn:
        conn.execute(text(_SEED))
    assert derive_station_activity(clean_schema) == 3


def test_empty_schema_derivation_is_a_no_op(clean_schema: Engine) -> None:
    assert derive_station_activity(clean_schema) == 0


def test_derivation_orders_the_prompt_sample_by_real_activity(seeded: Engine) -> None:
    """The payoff: load_profile ranks stations by observed_stop_events, so the busiest
    ones teach the model first. Before derivation the column is NULL and the order is
    arbitrary; after it, the two-day station outranks the one-day station."""
    from rail_rag.rag.context.profile import load_profile

    derive_station_activity(seeded)
    profile = load_profile(seeded)
    assert profile.sample_stations[0] == "Active-Two-Days"
    assert "Active-One-Day" in profile.sample_stations
    assert profile.sample_stations.index("Active-Two-Days") < profile.sample_stations.index(
        "Active-One-Day"
    )
