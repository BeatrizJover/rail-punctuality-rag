"""Tests for the partition-pruning lint and the partition-key mapping."""

from __future__ import annotations

import pytest

from rail_rag.db.partitions import partition_keys
from rail_rag.rag.sql.lint import FACT_WITHOUT_PARTITION_FILTER, lint_sql
from tests.test_sql_guard import PUNCTUALITY_QUERY

_FLAGGED = (FACT_WITHOUT_PARTITION_FILTER,)

_JOIN_DATE = "FROM gold.fact_stop_event f JOIN gold.dim_date d ON d.date_key = f.date_key"


def test_the_partition_keys_come_from_the_models() -> None:
    assert partition_keys()["gold.fact_stop_event"] == "date_key"
    assert "gold.dim_date" not in partition_keys()


# --- must flag -----------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(PUNCTUALITY_QUERY, id="filter-only-through-dim_date"),
        pytest.param("SELECT SUM(f.stop_events) FROM gold.fact_stop_event f", id="no-where"),
        pytest.param(
            f"SELECT 1 {_JOIN_DATE}",
            id="join-equality-only",
        ),
        pytest.param(
            f"SELECT 1 {_JOIN_DATE} WHERE d.year = 2025 AND d.month = 3",
            id="year-month-on-dimension",
        ),
        pytest.param(
            "SELECT 1 FROM (SELECT f.station_key, f.date_key FROM gold.fact_stop_event f) x"
            " WHERE x.date_key >= DATE '2025-03-01'",
            id="outer-filter-fact-unfiltered-in-subquery",
        ),
        pytest.param(
            "SELECT station_key FROM gold.fact_stop_event WHERE date_key = DATE '2025-03-01'"
            " UNION ALL SELECT station_key FROM gold.fact_stop_event",
            id="one-unfiltered-union-branch",
        ),
        pytest.param(
            "SELECT 1 FROM gold.fact_stop_event f"
            " WHERE f.date_key >= DATE '2025-03-01' OR f.station_key = 'x'",
            id="or-with-an-unbounded-side",
        ),
        pytest.param(
            "SELECT 1 FROM gold.fact_stop_event f"
            " WHERE f.date_key >= (SELECT MAX(date_key) FROM gold.dim_date)",
            id="bound-is-a-subquery",
        ),
        pytest.param(
            "SELECT 1 FROM gold.fact_stop_event f JOIN gold.dim_date d ON d.date_key = f.date_key"
            " WHERE date_key >= DATE '2025-03-01'",
            id="unqualified-key-is-ambiguous",
        ),
    ],
)
def test_queries_that_bypass_pruning_are_flagged(sql: str) -> None:
    assert lint_sql(sql) == _FLAGGED


# --- must not flag -------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(
            f"SELECT 1 {_JOIN_DATE} WHERE f.date_key >= DATE '2025-03-01'"
            " AND f.date_key < DATE '2025-04-01'",
            id="half-open-range-with-date-literals",
        ),
        pytest.param(
            "SELECT 1 FROM gold.fact_stop_event f"
            " WHERE f.date_key BETWEEN '2025-03-01' AND '2025-03-31'",
            id="between",
        ),
        pytest.param(
            "SELECT 1 FROM gold.fact_stop_event f"
            " WHERE f.date_key IN ('2025-03-01'::date, '2025-04-01'::date)",
            id="in-list-with-casts",
        ),
        pytest.param(
            "SELECT 1 FROM gold.fact_stop_event WHERE date_key = '2025-03-01'::date",
            id="unqualified-and-sole-table",
        ),
        pytest.param(
            "SELECT 1 FROM gold.fact_stop_event f JOIN gold.dim_station s"
            " ON s.station_key = f.station_key WHERE date_key >= DATE '2025-03-01'",
            id="unqualified-and-no-other-source-has-the-column",
        ),
        pytest.param(
            "SELECT 1 FROM gold.fact_stop_event f WHERE DATE '2025-03-01' <= f.date_key",
            id="constant-on-the-left",
        ),
        pytest.param(
            "SELECT 1 FROM gold.fact_stop_event f"
            " WHERE (f.date_key = DATE '2025-03-01' OR f.date_key = DATE '2025-05-01')",
            id="or-of-bounded-predicates",
        ),
        pytest.param(
            "SELECT 1 FROM gold.dim_station s JOIN gold.fact_stop_event f"
            " ON f.station_key = s.station_key AND f.date_key >= DATE '2025-03-01'",
            id="bound-in-the-join-condition",
        ),
        pytest.param(
            "WITH march AS (SELECT station_key, stop_events FROM gold.fact_stop_event f"
            " WHERE f.date_key >= DATE '2025-03-01' AND f.date_key < DATE '2025-04-01')"
            " SELECT s.station_name, SUM(m.stop_events) FROM march m"
            " JOIN gold.dim_station s ON s.station_key = m.station_key GROUP BY 1",
            id="cte-filters-the-fact-inside",
        ),
        pytest.param(
            "SELECT s.station_name FROM gold.dim_station s JOIN gold.dim_date d ON 1 = 1"
            " WHERE d.month = 3",
            id="dimensions-only",
        ),
    ],
)
def test_queries_that_prune_are_not_flagged(sql: str) -> None:
    assert lint_sql(sql) == ()


def test_each_finding_code_is_reported_once() -> None:
    sql = (
        "SELECT 1 FROM gold.fact_stop_event a"
        " UNION ALL SELECT 1 FROM gold.fact_stop_event b"
        " UNION ALL SELECT 1 FROM gold.fact_stop_event c"
    )
    assert lint_sql(sql) == _FLAGGED
