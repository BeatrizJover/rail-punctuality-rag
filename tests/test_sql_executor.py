"""Integration tests for query execution.

These assert what the static guard cannot: that the *server* refuses writes and
kills slow queries. Both defences must hold on a query that never passed the
guard, which is why the ``SafeQuery`` objects here are built by hand.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest
from sqlalchemy import Engine, event, text

from rail_rag.db.models import dim_relation, dim_station, fact_stop_event
from rail_rag.db.partitions import existing_partitions
from rail_rag.rag.exceptions import QueryExecutionError
from rail_rag.rag.sql.executor import execute_safe_query, explain_safe_query, run_query
from rail_rag.rag.sql.guard import SafeQuery
from rail_rag.rag.sql.policy import SqlPolicy

pytestmark = pytest.mark.integration

POLICY = SqlPolicy(
    allowed_tables=frozenset({"gold.dim_station", "gold.fact_stop_event", "gold.dim_date"}),
    max_rows=200,
    statement_timeout_ms=1000,
    blocked_functions=frozenset({"pg_read_file"}),
)


@pytest.fixture
def seeded(clean_schema: Engine) -> Engine:
    """Two stations, enough to tell an empty result from a populated one."""
    with clean_schema.begin() as conn:
        conn.execute(
            dim_station.insert(),
            [
                {"station_key": "a" * 32, "station_name": "Bruxelles-Midi"},
                {"station_key": "b" * 32, "station_name": "Gent-Sint-Pieters"},
            ],
        )
    return clean_schema


def _unchecked(sql: str, limit: int = 200) -> SafeQuery:
    """Build a SafeQuery without validation, to test the server-side defences alone."""
    return SafeQuery(sql=sql, limit=limit, tables=frozenset())


def test_select_returns_columns_and_rows(seeded: Engine) -> None:
    result = run_query(seeded, "SELECT station_name FROM gold.dim_station ORDER BY 1", POLICY)
    assert result.columns == ["station_name"]
    assert result.rows == [("Bruxelles-Midi",), ("Gent-Sint-Pieters",)]
    assert not result.truncated
    assert not result.is_empty


def test_empty_result_is_not_an_error(seeded: Engine) -> None:
    """An empty answer is a fact about the data, and the caller must be able to say so."""
    result = run_query(
        seeded, "SELECT station_name FROM gold.dim_station WHERE station_name = 'nope'", POLICY
    )
    assert result.is_empty
    assert result.columns == ["station_name"]


def test_server_refuses_a_write_that_bypassed_the_guard(seeded: Engine) -> None:
    """Defence in depth: READ ONLY holds even for a statement the guard never saw.

    RETURNING is deliberate. A bare INSERT produces no columns, so the failure
    would come from reading them rather than from the read-only transaction, and
    the test would pass with the defence removed.
    """
    query = _unchecked(
        "INSERT INTO gold.dim_station (station_key) VALUES ('c') RETURNING station_key"
    )
    with pytest.raises(QueryExecutionError):
        execute_safe_query(seeded, query, POLICY)

    with seeded.connect() as conn:
        remaining = conn.execute(text("SELECT count(*) FROM gold.dim_station")).scalar_one()
    assert remaining == 2


def test_server_refuses_a_delete_that_bypassed_the_guard(seeded: Engine) -> None:
    query = _unchecked("DELETE FROM gold.dim_station RETURNING station_key")
    with pytest.raises(QueryExecutionError):
        execute_safe_query(seeded, query, POLICY)

    with seeded.connect() as conn:
        remaining = conn.execute(text("SELECT count(*) FROM gold.dim_station")).scalar_one()
    assert remaining == 2


def test_server_kills_a_slow_query(seeded: Engine) -> None:
    with pytest.raises(QueryExecutionError):
        execute_safe_query(seeded, _unchecked("SELECT pg_sleep(5)"), POLICY)


def test_timeout_does_not_leak_into_the_next_connection(seeded: Engine) -> None:
    """SET LOCAL dies with the transaction; a pooled connection must come back clean."""
    with pytest.raises(QueryExecutionError):
        execute_safe_query(seeded, _unchecked("SELECT pg_sleep(5)"), POLICY)

    with seeded.connect() as conn:
        setting = conn.execute(text("SHOW statement_timeout")).scalar_one()
    assert setting in {"0", "0ms"}


def test_row_cap_sets_the_truncated_flag(seeded: Engine) -> None:
    """Hitting the cap exactly means the real answer is probably larger."""
    narrow = POLICY.model_copy(update={"max_rows": 1})
    result = run_query(seeded, "SELECT station_name FROM gold.dim_station", narrow)
    assert len(result.rows) == 1
    assert result.truncated


def test_error_message_does_not_leak_the_connection_string(seeded: Engine) -> None:
    """Driver messages can carry the DSN; only the exception class is safe to surface."""
    with pytest.raises(QueryExecutionError) as caught:
        execute_safe_query(seeded, _unchecked("SELECT * FROM gold.does_not_exist"), POLICY)
    message = str(caught.value)
    assert "password" not in message.lower()
    assert "postgresql+psycopg" not in message


def test_punctuality_ratio_query_runs_end_to_end(seeded: Engine) -> None:
    """The canonical business question: SUM/SUM, never an average of percentages."""
    with seeded.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO gold.dim_date (date_key, year, quarter, month, month_name,"
                " week_of_year, day_of_week, day_name, is_weekend)"
                " VALUES (:d, 2024, 3, 8, 'August', 32, 4, 'Thursday', false)"
            ),
            {"d": dt.date(2024, 8, 8)},
        )
        # relation_key is NOT NULL with a foreign key, so the dimension must exist first.
        conn.execute(dim_relation.insert(), [{"relation_key": "c" * 32, "relation": "IC 01"}])
        conn.execute(
            fact_stop_event.insert(),
            [
                {
                    "date_key": dt.date(2024, 8, 8),
                    "station_key": "a" * 32,
                    "relation_key": "c" * 32,
                    "train_no": 100 + i,
                    "planned_hour": 8,
                    "delay_arr_s": 0,
                    "delay_dep_s": 0,
                    "dwell_delta_s": 0,
                    "punctual_arrivals": i % 2,
                    "stop_events": 1,
                    "measured_arrivals": 1,
                }
                for i in range(4)
            ],
        )

    result = run_query(
        seeded,
        "SELECT SUM(punctual_arrivals)::float / NULLIF(SUM(measured_arrivals), 0) AS rate"
        " FROM gold.fact_stop_event",
        POLICY,
    )
    assert result.rows[0][0] == pytest.approx(0.5)


# --- plan estimates ------------------------------------------------------------

_FACT = "gold.fact_stop_event"
_MARCH = (
    "SELECT SUM(f.stop_events) FROM gold.fact_stop_event f"
    " WHERE f.date_key >= DATE '2025-03-01' AND f.date_key < DATE '2025-04-01'"
)
_VIA_DIMENSION = (
    "SELECT SUM(f.stop_events) FROM gold.fact_stop_event f"
    " JOIN gold.dim_date d ON d.date_key = f.date_key WHERE d.year = 2025 AND d.month = 3"
)


def test_a_literal_month_filter_plans_a_single_partition(clean_schema: Engine) -> None:
    estimate = explain_safe_query(clean_schema, _unchecked(_MARCH), POLICY)
    assert estimate.partitions_scanned[_FACT] == 1
    assert estimate.total_cost > 0
    assert estimate.rows >= 1


def test_a_dimension_filter_plans_every_partition(clean_schema: Engine) -> None:
    estimate = explain_safe_query(clean_schema, _unchecked(_VIA_DIMENSION), POLICY)
    total = len(existing_partitions(clean_schema))
    assert total > 1
    assert estimate.partitions_scanned[_FACT] == total


def test_a_query_that_never_touches_the_fact_scans_no_partitions(clean_schema: Engine) -> None:
    estimate = explain_safe_query(clean_schema, _unchecked("SELECT 1 FROM gold.dim_date"), POLICY)
    assert estimate.partitions_scanned[_FACT] == 0


def test_explain_runs_read_only_with_a_timeout_and_rolls_back(clean_schema: Engine) -> None:
    statements: list[str] = []
    transaction_events: list[str] = []

    def record_statement(*args: Any) -> None:
        statements.append(args[2])

    def record_rollback(_: Any) -> None:
        transaction_events.append("rollback")

    def record_commit(_: Any) -> None:
        transaction_events.append("commit")

    event.listen(clean_schema, "before_cursor_execute", record_statement)
    event.listen(clean_schema, "rollback", record_rollback)
    event.listen(clean_schema, "commit", record_commit)
    try:
        explain_safe_query(clean_schema, _unchecked(_MARCH), POLICY)
    finally:
        event.remove(clean_schema, "before_cursor_execute", record_statement)
        event.remove(clean_schema, "rollback", record_rollback)
        event.remove(clean_schema, "commit", record_commit)

    assert statements[0] == "SET TRANSACTION READ ONLY"
    assert statements[1] == f"SET LOCAL statement_timeout = {POLICY.statement_timeout_ms}"
    assert statements[2].startswith("EXPLAIN (FORMAT JSON) ")
    assert not any("ANALYZE" in statement.upper() for statement in statements)
    assert "commit" not in transaction_events
    assert "rollback" in transaction_events


def test_a_failing_plan_surfaces_as_a_query_execution_error(clean_schema: Engine) -> None:
    with pytest.raises(QueryExecutionError, match="Plan failed"):
        explain_safe_query(clean_schema, _unchecked("SELECT * FROM gold.no_such_table"), POLICY)
