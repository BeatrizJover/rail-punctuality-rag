"""Tests for the golden-case model, its loader, and the committed golden set itself."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from rail_rag.core.exceptions import ConfigError
from rail_rag.eval.cases import load_golden_set
from rail_rag.rag.sql.guard import validate_sql
from rail_rag.rag.sql.lint import lint_sql
from rail_rag.rag.sql.policy import load_retrieval_config

#: Resolved from ``__file__``: the autouse settings fixture chdirs into a tmp_path.
_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_SET = _ROOT / "eval" / "golden_set.yaml"
POLICY = load_retrieval_config(_ROOT / "config" / "retrieval_config.yaml").sql

_EXPECTED_TAG_COUNTS = {
    "ratio": 5,
    "ranking": 4,
    "time_series": 4,
    "full_period": 2,
    "out_of_coverage": 2,
    "conceptual": 5,
    "multilingual": 2,
    "injection": 2,
}

_DATA_CASE = """
  - id: {id}
    question: How many stations are there?
    tags: [ratio]
    expected_route: data
    reference_sql: SELECT 1 AS n
    compare: {{mode: scalar}}
"""


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "golden.yaml"
    path.write_text(f"cases:{body}", encoding="utf-8")
    return path


# --- the committed golden set ---------------------------------------------------


def test_the_real_golden_set_loads() -> None:
    assert len(load_golden_set(GOLDEN_SET)) == 24


def test_the_tag_counts_match_the_design() -> None:
    cases = load_golden_set(GOLDEN_SET)
    counts = Counter(tag for case in cases for tag in case.tags)
    assert dict(counts) == _EXPECTED_TAG_COUNTS


def test_every_reference_sql_passes_the_guard_under_the_repo_policy() -> None:
    for case in load_golden_set(GOLDEN_SET):
        if case.reference_sql is not None:
            validate_sql(case.reference_sql, POLICY)


def test_only_full_period_references_may_bypass_partition_pruning() -> None:
    for case in load_golden_set(GOLDEN_SET):
        if case.reference_sql is None or "full_period" in case.tags:
            continue
        assert lint_sql(case.reference_sql) == (), case.id


def test_conceptual_and_behaviour_cases_carry_no_reference() -> None:
    for case in load_golden_set(GOLDEN_SET):
        if case.reference_sql is None:
            assert case.compare is None
            assert (case.expected_sources is None) != (case.expected_behaviour is None)


# --- the loader ------------------------------------------------------------------


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, _DATA_CASE.format(id="same") + _DATA_CASE.format(id="same"))
    with pytest.raises(ConfigError, match="Duplicate golden case id 'same'"):
        load_golden_set(path)


def test_a_data_case_without_reference_sql_is_rejected(tmp_path: Path) -> None:
    body = """
  - id: no_reference
    question: How many stations are there?
    tags: [ratio]
    expected_route: data
"""
    with pytest.raises(ConfigError, match="no_reference.*exactly one of"):
        load_golden_set(_write(tmp_path, body))


def test_two_expectation_kinds_are_rejected(tmp_path: Path) -> None:
    body = _DATA_CASE.format(id="both") + "    expected_behaviour: no_data\n"
    with pytest.raises(ConfigError, match="found: reference_sql, expected_behaviour"):
        load_golden_set(_write(tmp_path, body))


def test_reference_sql_without_compare_is_rejected(tmp_path: Path) -> None:
    body = """
  - id: half
    question: q
    tags: [ratio]
    expected_route: data
    reference_sql: SELECT 1
"""
    with pytest.raises(ConfigError, match="given together"):
        load_golden_set(_write(tmp_path, body))


def test_an_unknown_tag_is_rejected(tmp_path: Path) -> None:
    body = _DATA_CASE.format(id="typo").replace("[ratio]", "[ratoi]")
    with pytest.raises(ConfigError, match="unknown tag.*ratoi"):
        load_golden_set(_write(tmp_path, body))


def test_top_k_is_only_allowed_with_the_ordered_mode(tmp_path: Path) -> None:
    body = _DATA_CASE.format(id="bad_k").replace("{mode: scalar}", "{mode: set, top_k: 3}")
    with pytest.raises(ConfigError, match="top_k only applies"):
        load_golden_set(_write(tmp_path, body))


def test_sources_require_a_conceptual_route(tmp_path: Path) -> None:
    body = """
  - id: wrong_route
    question: q
    tags: [conceptual]
    expected_route: data
    expected_sources: [{doc_id: d, heading: h}]
"""
    with pytest.raises(ConfigError, match="expected_route conceptual"):
        load_golden_set(_write(tmp_path, body))


def test_an_unknown_field_is_rejected(tmp_path: Path) -> None:
    body = _DATA_CASE.format(id="extra") + "    surprise: true\n"
    with pytest.raises(ConfigError, match="surprise"):
        load_golden_set(_write(tmp_path, body))


def test_a_missing_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_golden_set(tmp_path / "absent.yaml")


def test_a_file_without_cases_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "golden.yaml"
    path.write_text("other: []\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="non-empty 'cases'"):
        load_golden_set(path)
