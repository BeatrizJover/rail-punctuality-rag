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
    optimize_fact_for_reads,
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


def _lookup_index_count(engine: Engine, column: str) -> int:
    """Number of child partitions carrying a single-column non-unique index on ``column``."""
    with engine.connect() as conn:
        return int(
            conn.execute(
                text(
                    "SELECT count(*) FROM pg_index i"
                    " JOIN pg_class p ON p.oid = i.indrelid"
                    " JOIN pg_attribute a ON a.attrelid = p.oid AND a.attnum = i.indkey[0]"
                    " WHERE p.relname LIKE 'fact_stop_event\\_%'"
                    " AND NOT i.indisunique AND i.indnatts = 1 AND a.attname = :column"
                ),
                {"column": column},
            ).scalar_one()
        )


def _covering_index_count(engine: Engine, column: str) -> int:
    """Child partitions carrying a covering index keyed on ``column``."""
    with engine.connect() as conn:
        return int(
            conn.execute(
                text(
                    "SELECT count(*) FROM pg_index i"
                    " JOIN pg_class p ON p.oid = i.indrelid"
                    " JOIN pg_attribute a ON a.attrelid = p.oid AND a.attnum = i.indkey[0]"
                    " WHERE p.relname LIKE 'fact_stop_event\\_%'"
                    " AND NOT i.indisunique AND i.indnkeyatts = 1 AND i.indnatts = 4"
                    " AND a.attname = :column"
                ),
                {"column": column},
            ).scalar_one()
        )


def test_optimize_is_idempotent_and_analyzes(seeded: Engine) -> None:
    """The statements are IF NOT EXISTS and VACUUM ANALYZE is safe to repeat, so a
    second pass must be a no-op rather than an error."""
    optimize_fact_for_reads(seeded)
    optimize_fact_for_reads(seeded)
    assert _covering_index_count(seeded, "station_key") == 36
    assert _covering_index_count(seeded, "relation_key") == 36
    with seeded.connect() as conn:
        last_analyze = conn.execute(
            text(
                "SELECT last_analyze FROM pg_stat_all_tables"
                " WHERE schemaname = :schema AND relname = 'fact_stop_event'"
            ),
            {"schema": GOLD_SCHEMA},
        ).scalar_one()
    assert last_analyze is not None


def test_optimize_populates_the_visibility_map(seeded: Engine) -> None:
    """Without VACUUM the visibility map stays empty and an Index Only Scan cannot
    skip the heap, which would silently undo the point of the covering index."""
    optimize_fact_for_reads(seeded)
    with seeded.connect() as conn:
        all_visible = conn.execute(
            text(
                "SELECT bool_and(relallvisible > 0) FROM pg_class"
                " WHERE relkind = 'r' AND relname LIKE 'fact_stop_event\\_%'"
                " AND relpages > 0"
            )
        ).scalar_one()
    assert all_visible


def test_optimize_raises_the_statistics_target_on_the_key_columns(seeded: Engine) -> None:
    """Station traffic is heavily skewed; the default histogram misestimates it."""
    optimize_fact_for_reads(seeded)
    with seeded.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT a.attname, a.attstattarget FROM pg_attribute a"
                " JOIN pg_class p ON p.oid = a.attrelid"
                " JOIN pg_namespace n ON n.oid = p.relnamespace"
                " WHERE p.relname = 'fact_stop_event' AND n.nspname = :schema"
                " AND a.attname IN ('station_key', 'relation_key')"
            ),
            {"schema": GOLD_SCHEMA},
        ).all()
    targets: dict[str, int] = {str(row[0]): int(row[1]) for row in rows}
    assert targets == {"station_key": 500, "relation_key": 500}


def test_optimize_drops_the_superseded_plain_indexes(seeded: Engine) -> None:
    """The covering index answers any lookup the plain one did; keeping both would
    pay twice the write cost and twice the disk for nothing."""
    optimize_fact_for_reads(seeded)
    with seeded.connect() as conn:
        leftovers = conn.execute(
            text(
                "SELECT count(*) FROM pg_class"
                " WHERE relname IN ('ix_fact_stop_event_station_key',"
                " 'ix_fact_stop_event_relation_key')"
            )
        ).scalar_one()
    assert leftovers == 0


def test_optimize_lets_a_station_filter_skip_the_heap(seeded: Engine) -> None:
    """The payoff: the aggregation the RAG generates must be answered by an Index
    Only Scan. A Bitmap Heap Scan here is the 60s timeout reproducing in miniature."""
    optimize_fact_for_reads(seeded)
    with seeded.begin() as conn:
        conn.execute(text("SET LOCAL enable_seqscan = off"))
        plan = "\n".join(
            conn.execute(
                text(
                    "EXPLAIN (VERBOSE, COSTS OFF)"
                    " SELECT SUM(f.stop_events)"
                    f" FROM {GOLD_SCHEMA}.fact_stop_event AS f"
                    " WHERE f.station_key = repeat('a', 32)"
                )
            ).scalars()
        )
    assert "Index Only Scan" in plan
    assert "Heap Scan" not in plan


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
