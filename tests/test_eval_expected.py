"""Integration tests for ground-truth computation and its on-disk cache."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Engine, event

from rail_rag.db.models import dim_station
from rail_rag.eval.cases import CompareSpec, GoldenCase
from rail_rag.eval.expected import cache_key, compute_expected
from rail_rag.rag.exceptions import QueryExecutionError
from rail_rag.rag.sql.policy import SqlPolicy

pytestmark = pytest.mark.integration

POLICY = SqlPolicy(
    allowed_tables=frozenset({"gold.dim_station", "gold.fact_stop_event"}),
    statement_timeout_ms=1000,
)
WINDOW = ("2024-01-01", "2026-09-07")


def _case(sql: str, case_id: str = "stations") -> GoldenCase:
    return GoldenCase(
        id=case_id,
        question="How many stations are there?",
        tags=("ratio",),
        expected_route="data",
        reference_sql=sql,
        compare=CompareSpec(mode="scalar"),
    )


_COUNT_STATIONS = "SELECT COUNT(*) AS n FROM gold.dim_station"


@pytest.fixture
def seeded(clean_schema: Engine) -> Engine:
    with clean_schema.begin() as conn:
        conn.execute(
            dim_station.insert(),
            [
                {"station_key": "a" * 32, "station_name": "Bruxelles-Midi"},
                {"station_key": "b" * 32, "station_name": "Gent-Sint-Pieters"},
            ],
        )
    return clean_schema


class _StatementCounter:
    """Counts the statements an engine sends to the server."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self.count = 0

    def _record(self, *_: object) -> None:
        self.count += 1

    def __enter__(self) -> _StatementCounter:
        event.listen(self._engine, "before_cursor_execute", self._record)
        return self

    def __exit__(self, *_: object) -> None:
        event.remove(self._engine, "before_cursor_execute", self._record)


def test_the_result_is_computed_and_cached(seeded: Engine, tmp_path: Path) -> None:
    cache = tmp_path / "expected"
    result = compute_expected(seeded, _case(_COUNT_STATIONS), POLICY, cache_dir=cache)

    assert result.columns == ["n"]
    assert result.rows == [[2]]
    assert not result.cached
    assert len(list(cache.glob("*.json"))) == 1


def test_a_second_call_is_served_from_the_cache_without_touching_the_database(
    seeded: Engine, tmp_path: Path
) -> None:
    cache = tmp_path / "expected"
    first = compute_expected(seeded, _case(_COUNT_STATIONS), POLICY, cache_dir=cache, window=WINDOW)

    with _StatementCounter(seeded) as counter:
        second = compute_expected(
            seeded, _case(_COUNT_STATIONS), POLICY, cache_dir=cache, window=WINDOW
        )

    assert counter.count == 0
    assert second.cached
    assert (second.columns, second.rows) == (first.columns, first.rows)


def test_without_a_window_only_the_coverage_is_read_on_a_hit(
    seeded: Engine, tmp_path: Path
) -> None:
    cache = tmp_path / "expected"
    compute_expected(seeded, _case(_COUNT_STATIONS), POLICY, cache_dir=cache)

    with _StatementCounter(seeded) as counter:
        again = compute_expected(seeded, _case(_COUNT_STATIONS), POLICY, cache_dir=cache)

    assert again.cached
    assert counter.count == 2  # SET TRANSACTION READ ONLY, then the min/max


def test_changed_sql_gets_its_own_cache_entry(seeded: Engine, tmp_path: Path) -> None:
    cache = tmp_path / "expected"
    compute_expected(seeded, _case(_COUNT_STATIONS), POLICY, cache_dir=cache, window=WINDOW)
    other = compute_expected(
        seeded,
        _case("SELECT COUNT(*) AS n FROM gold.dim_station WHERE station_name > 'C'"),
        POLICY,
        cache_dir=cache,
        window=WINDOW,
    )

    assert not other.cached
    assert other.rows == [[1]]
    assert len(list(cache.glob("*.json"))) == 2


def test_the_cache_key_depends_on_the_sql_and_the_coverage() -> None:
    base = cache_key("SELECT 1", WINDOW)
    assert cache_key("SELECT 2", WINDOW) != base
    assert cache_key("SELECT 1", (WINDOW[0], "2026-09-08")) != base
    assert cache_key("SELECT 1", WINDOW) == base


def test_new_coverage_invalidates_a_cached_result(seeded: Engine, tmp_path: Path) -> None:
    cache = tmp_path / "expected"
    compute_expected(seeded, _case(_COUNT_STATIONS), POLICY, cache_dir=cache, window=WINDOW)
    fresh = compute_expected(
        seeded, _case(_COUNT_STATIONS), POLICY, cache_dir=cache, window=(WINDOW[0], "2026-10-01")
    )
    assert not fresh.cached


def test_a_corrupt_cache_entry_is_recomputed(seeded: Engine, tmp_path: Path) -> None:
    cache = tmp_path / "expected"
    compute_expected(seeded, _case(_COUNT_STATIONS), POLICY, cache_dir=cache, window=WINDOW)
    (entry,) = cache.glob("*.json")
    entry.write_text("{not json", encoding="utf-8")

    again = compute_expected(seeded, _case(_COUNT_STATIONS), POLICY, cache_dir=cache, window=WINDOW)
    assert not again.cached
    assert again.rows == [[2]]


def test_the_reference_runs_under_the_longer_timeout(seeded: Engine, tmp_path: Path) -> None:
    slow = _case("SELECT pg_sleep(1.5) AS n", "slow")
    with pytest.raises(QueryExecutionError):
        compute_expected(
            seeded, slow, POLICY, timeout_ms=500, cache_dir=tmp_path / "a", window=WINDOW
        )
    result = compute_expected(
        seeded, slow, POLICY, timeout_ms=10_000, cache_dir=tmp_path / "b", window=WINDOW
    )
    assert result.elapsed_ms >= 1500


def test_a_case_without_reference_sql_is_refused(seeded: Engine, tmp_path: Path) -> None:
    case = GoldenCase(
        id="behaviour",
        question="q",
        tags=("injection",),
        expected_route="data",
        expected_behaviour="refused",
    )
    with pytest.raises(ValueError, match="no reference_sql"):
        compute_expected(seeded, case, POLICY, cache_dir=tmp_path)
