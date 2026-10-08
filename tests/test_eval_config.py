"""Tests for the evaluation configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from rail_rag.core.exceptions import ConfigError
from rail_rag.eval.config import EvalConfig, load_eval_config

#: Resolved from ``__file__``: the autouse settings fixture chdirs into a tmp_path.
REPO_CONFIG = Path(__file__).resolve().parent.parent / "config" / "eval_config.yaml"


def test_the_repo_config_loads_with_the_documented_limits() -> None:
    config = load_eval_config(REPO_CONFIG)
    assert config.generation_rpm == 4
    assert config.daily_generation_budget == 15
    assert config.cache is True
    assert config.golden_set == Path("eval/golden_set.yaml")
    assert config.cache_dir == Path(".eval_cache")


def test_the_interval_follows_from_the_requests_per_minute() -> None:
    assert EvalConfig(generation_rpm=4).min_interval_s == 15.0
    assert EvalConfig(generation_rpm=5).min_interval_s == 12.0


def test_the_cache_subdirectories_live_under_the_cache_dir() -> None:
    config = EvalConfig(cache_dir=Path("somewhere"))
    assert config.expected_dir == Path("somewhere/expected")
    assert config.generations_dir == Path("somewhere/generations")
    assert config.usage_dir == Path("somewhere/usage")


def test_an_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "eval.yaml"
    path.write_text("generation_rpm: 4\nsurprise: true\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="surprise"):
        load_eval_config(path)


def test_a_non_positive_limit_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "eval.yaml"
    path.write_text("generation_rpm: 0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="generation_rpm"):
        load_eval_config(path)


def test_a_missing_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_eval_config(tmp_path / "absent.yaml")
