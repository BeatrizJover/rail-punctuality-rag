"""Tests for the deterministic parts of the context: schema and rules."""

from __future__ import annotations

import pytest

from rail_rag.rag.context.rules import SEMANTIC_RULES
from rail_rag.rag.context.schema import render_schema
from rail_rag.rag.exceptions import RagError
from rail_rag.rag.sql.policy import SqlPolicy

FULL_POLICY = SqlPolicy(
    allowed_tables=frozenset(
        {
            "gold.dim_date",
            "gold.dim_station",
            "gold.dim_relation",
            "gold.fact_stop_event",
        }
    )
)


def test_every_allowed_table_is_rendered() -> None:
    rendered = render_schema(FULL_POLICY)
    for table in FULL_POLICY.allowed_tables:
        assert f"TABLE {table}" in rendered


def test_staging_and_ops_are_never_rendered() -> None:
    """The model cannot ask for a table it has not been shown."""
    rendered = render_schema(FULL_POLICY)
    assert "stg_fact_stop_event" not in rendered
    assert "load_runs" not in rendered


def test_rendering_follows_the_policy_not_the_metadata() -> None:
    """Narrowing the allow-list must narrow the prompt, or the two drift apart."""
    rendered = render_schema(SqlPolicy(allowed_tables=frozenset({"gold.dim_station"})))
    assert "TABLE gold.dim_station" in rendered
    assert "fact_stop_event" not in rendered


def test_policy_naming_an_unknown_table_is_an_error() -> None:
    """A typo in the config must fail loudly, not silently shrink the prompt."""
    policy = SqlPolicy(allowed_tables=frozenset({"gold.dim_stations"}))
    with pytest.raises(RagError, match="absent from the schema"):
        render_schema(policy)


def test_foreign_keys_are_rendered() -> None:
    """Without them the model has to guess the join conditions."""
    rendered = render_schema(FULL_POLICY)
    assert "REFERENCES gold.dim_station.station_key" in rendered
    assert "REFERENCES gold.dim_date.date_key" in rendered


def test_grain_and_ratio_reach_the_prompt_from_the_table_comment() -> None:
    """The comment in models.py is the single source of truth for the grain."""
    rendered = render_schema(FULL_POLICY)
    assert "Grain: one train passing one measuring point" in rendered
    assert "SUM(punctual_arrivals) / SUM(measured_arrivals)" in rendered


def test_nullability_is_rendered() -> None:
    """ptcar_no being nullable is load-bearing: filtering on it drops the network."""
    rendered = render_schema(FULL_POLICY)
    assert "ptcar_no" in rendered
    ptcar_line = next(line for line in rendered.splitlines() if "ptcar_no" in line)
    assert ptcar_line.strip().endswith("NULL")


def test_unique_grain_is_rendered() -> None:
    rendered = render_schema(FULL_POLICY)
    assert "UNIQUE: (date_key, station_key, train_no)" in rendered


def test_check_constraints_are_rendered() -> None:
    rendered = render_schema(FULL_POLICY)
    assert "punctual_arrivals IN (0, 1)" in rendered


# --- semantic rules --------------------------------------------------------


def _flat(text: str) -> str:
    """Collapse wrapping so assertions do not depend on where a line breaks."""
    return " ".join(text.split())


def test_rules_state_the_ratio_formula() -> None:
    """If this ever disappears the model reverts to averaging percentages."""
    assert "NULLIF(SUM(measured_arrivals), 0)" in _flat(SEMANTIC_RULES)
    assert "Never AVG(punctual_arrivals)" in _flat(SEMANTIC_RULES)


def test_rules_state_the_punctuality_threshold() -> None:
    assert "360 seconds" in _flat(SEMANTIC_RULES)


def test_rules_warn_against_inferring_coverage_from_the_calendar() -> None:
    assert "Never infer the available period from dim_date" in _flat(SEMANTIC_RULES)


def test_rules_protect_negative_delays_and_null_ptcar() -> None:
    assert "Negative values are early arrivals" in _flat(SEMANTIC_RULES)
    assert "Never filter on ptcar_no IS NOT NULL" in _flat(SEMANTIC_RULES)
