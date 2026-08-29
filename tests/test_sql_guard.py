"""Tests for the static SQL guard.

The rejection cases are the point of the module, so they are enumerated rather
than summarised: each one is an attack the generator could plausibly emit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rail_rag.rag.exceptions import RagError, UnsafeQueryError
from rail_rag.rag.sql.guard import validate_sql
from rail_rag.rag.sql.policy import SqlPolicy, load_retrieval_config

#: Resolved from ``__file__``: the autouse settings fixture chdirs into a tmp_path.
REPO_CONFIG = Path(__file__).resolve().parent.parent / "config" / "retrieval_config.yaml"

POLICY = SqlPolicy(
    allowed_tables=frozenset(
        {
            "gold.dim_date",
            "gold.dim_station",
            "gold.dim_relation",
            "gold.fact_stop_event",
        }
    ),
    max_rows=200,
    statement_timeout_ms=5000,
    blocked_functions=frozenset({"pg_sleep", "pg_read_file", "dblink"}),
)

PUNCTUALITY_QUERY = """
SELECT s.station_name,
       SUM(f.punctual_arrivals)::float / NULLIF(SUM(f.measured_arrivals), 0) AS rate
FROM gold.fact_stop_event f
JOIN gold.dim_station s ON s.station_key = f.station_key
JOIN gold.dim_date d ON d.date_key = f.date_key
WHERE d.month = 8
GROUP BY s.station_name
ORDER BY rate ASC
"""


def test_repo_policy_loads_and_excludes_staging() -> None:
    """The committed policy must never expose the unvalidated landing zone."""
    policy = load_retrieval_config(REPO_CONFIG).sql
    assert "gold.fact_stop_event" in policy.allowed_tables
    assert "gold.stg_fact_stop_event" not in policy.allowed_tables
    assert not any(name.startswith("ops.") for name in policy.allowed_tables)


def test_realistic_analytical_query_is_accepted() -> None:
    safe = validate_sql(PUNCTUALITY_QUERY, POLICY)
    assert "gold.fact_stop_event" in safe.tables
    assert "gold.dim_station" in safe.tables
    assert "LIMIT 200" in safe.sql.upper()


def test_cte_alias_is_not_treated_as_a_table() -> None:
    """A CTE name is not a real table; treating it as one rejects valid SQL."""
    sql = """
    WITH busiest AS (
        SELECT station_key, SUM(stop_events) AS n
        FROM gold.fact_stop_event GROUP BY station_key
    )
    SELECT s.station_name, b.n
    FROM busiest b JOIN gold.dim_station s ON s.station_key = b.station_key
    """
    safe = validate_sql(sql, POLICY)
    assert safe.tables == frozenset({"gold.fact_stop_event", "gold.dim_station"})


def test_unqualified_table_resolves_to_the_gold_schema() -> None:
    safe = validate_sql("SELECT * FROM dim_station", POLICY)
    assert safe.tables == frozenset({"gold.dim_station"})


# --- statement shape -------------------------------------------------------


def test_stacked_statements_are_rejected() -> None:
    with pytest.raises(UnsafeQueryError, match="exactly one statement"):
        validate_sql("SELECT 1; DROP TABLE gold.dim_date", POLICY)


def test_statement_hidden_behind_comments_is_rejected() -> None:
    """The reason this module parses instead of pattern-matching."""
    with pytest.raises(UnsafeQueryError, match="exactly one statement"):
        validate_sql("SELECT 1 /* x */ ; /* y */ DROP TABLE gold.dim_date", POLICY)


def test_trailing_semicolon_is_accepted() -> None:
    """Models emit one constantly; rejecting it would be a usability bug."""
    assert validate_sql("SELECT * FROM gold.dim_date;", POLICY).sql


@pytest.mark.parametrize("sql", ["DESCRIBE gold.dim_date", "VALUES (1, 2)"])
def test_non_select_root_is_rejected(sql: str) -> None:
    """These pass the node deny-list, so only the root check stops them."""
    with pytest.raises(UnsafeQueryError, match="Only SELECT"):
        validate_sql(sql, POLICY)


def test_empty_query_is_rejected() -> None:
    with pytest.raises(UnsafeQueryError, match="empty"):
        validate_sql("   ", POLICY)


def test_unparseable_query_is_rejected() -> None:
    with pytest.raises(UnsafeQueryError):
        validate_sql("SELECT FROM WHERE ORDER (", POLICY)


# --- write operations ------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE gold.dim_date SET year = 1",
        "DELETE FROM gold.fact_stop_event",
        "INSERT INTO gold.dim_date (date_key) VALUES ('2024-01-01')",
        "DROP TABLE gold.dim_date",
        "TRUNCATE gold.fact_stop_event",
        "CREATE TABLE evil (id int)",
        "ALTER TABLE gold.dim_date ADD COLUMN x int",
        "GRANT ALL ON gold.dim_date TO PUBLIC",
    ],
)
def test_write_statements_are_rejected(sql: str) -> None:
    with pytest.raises(UnsafeQueryError):
        validate_sql(sql, POLICY)


def test_dml_wrapped_in_a_cte_is_rejected() -> None:
    """Postgres allows writes inside a CTE, so a SELECT root proves nothing."""
    sql = "WITH x AS (DELETE FROM gold.fact_stop_event RETURNING *) SELECT * FROM x"
    with pytest.raises(UnsafeQueryError, match="DELETE"):
        validate_sql(sql, POLICY)


def test_select_into_is_rejected() -> None:
    """``SELECT ... INTO`` creates a table while looking like a read."""
    with pytest.raises(UnsafeQueryError):
        validate_sql("SELECT * INTO evil FROM gold.dim_date", POLICY)


def test_row_locking_is_rejected() -> None:
    with pytest.raises(UnsafeQueryError):
        validate_sql("SELECT * FROM gold.dim_date FOR UPDATE", POLICY)


# --- table allow-list ------------------------------------------------------


def test_staging_table_is_rejected() -> None:
    """Not a security rule: staging holds rows the fact table refused."""
    with pytest.raises(UnsafeQueryError, match="stg_fact_stop_event"):
        validate_sql("SELECT * FROM gold.stg_fact_stop_event", POLICY)


def test_catalog_access_is_rejected() -> None:
    with pytest.raises(UnsafeQueryError, match="not allowed"):
        validate_sql("SELECT * FROM pg_catalog.pg_user", POLICY)


def test_union_smuggling_a_forbidden_table_is_rejected() -> None:
    """The allow-list must apply to every branch, not just the first."""
    sql = "SELECT station_name FROM gold.dim_station UNION ALL SELECT usename FROM pg_user"
    with pytest.raises(UnsafeQueryError, match="not allowed"):
        validate_sql(sql, POLICY)


def test_forbidden_table_in_a_subquery_is_rejected() -> None:
    sql = "SELECT * FROM gold.dim_station WHERE station_key IN (SELECT k FROM secret_table)"
    with pytest.raises(UnsafeQueryError, match="secret_table"):
        validate_sql(sql, POLICY)


# --- functions -------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT pg_sleep(60)",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT * FROM gold.dim_date WHERE date_key > '2024-01-01' AND pg_sleep(1) IS NULL",
    ],
)
def test_blocked_functions_are_rejected(sql: str) -> None:
    with pytest.raises(UnsafeQueryError, match="not allowed"):
        validate_sql(sql, POLICY)


def test_ordinary_aggregates_are_allowed() -> None:
    safe = validate_sql("SELECT SUM(stop_events) FROM gold.fact_stop_event", POLICY)
    assert safe.sql


# --- row limit -------------------------------------------------------------


def test_missing_limit_is_added() -> None:
    assert "LIMIT 200" in validate_sql("SELECT * FROM gold.dim_date", POLICY).sql.upper()


def test_oversized_limit_is_tightened() -> None:
    safe = validate_sql("SELECT * FROM gold.dim_date LIMIT 100000", POLICY)
    assert "LIMIT 200" in safe.sql.upper()
    assert "100000" not in safe.sql


def test_smaller_limit_is_preserved() -> None:
    """The model asking for the top 5 must still get 5, not 200."""
    safe = validate_sql("SELECT * FROM gold.dim_date LIMIT 5", POLICY)
    assert "LIMIT 5" in safe.sql.upper()


def test_non_numeric_limit_is_replaced() -> None:
    safe = validate_sql("SELECT * FROM gold.dim_date LIMIT ALL", POLICY)
    assert "LIMIT 200" in safe.sql.upper()


# --- error hierarchy -------------------------------------------------------


def test_unsafe_query_error_is_an_application_error() -> None:
    assert issubclass(UnsafeQueryError, RagError)
