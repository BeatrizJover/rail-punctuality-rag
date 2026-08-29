"""Tests for the YAML model configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rail_rag.core.exceptions import ConfigError
from rail_rag.rag.providers.config import load_model_config

#: Resolved from ``__file__``: the autouse settings fixture chdirs into a tmp_path.
REPO_CONFIG = Path(__file__).resolve().parent.parent / "config" / "model_config.yaml"

_MINIMAL = """
provider: fake
generation:
  model: some-model
embedding:
  model: some-embedding
  dimension: 8
"""


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "model_config.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_repo_configuration_is_valid() -> None:
    """The committed file must always load; it is the default the app boots with."""
    config = load_model_config(REPO_CONFIG)
    assert config.provider
    assert config.embedding.dimension > 0


def test_defaults_are_applied(tmp_path: Path) -> None:
    config = load_model_config(_write(tmp_path, _MINIMAL))
    assert config.generation.temperature == 0.0
    assert config.embedding.batch_size == 32
    assert config.max_retries == 3


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_model_config(tmp_path / "absent.yaml")


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    """``extra=forbid`` turns a typo into a startup failure, not a silent default."""
    with pytest.raises(ConfigError, match="Invalid model configuration"):
        load_model_config(_write(tmp_path, _MINIMAL + "\ntemperatur: 0.7\n"))


def test_out_of_range_temperature_is_rejected(tmp_path: Path) -> None:
    body = _MINIMAL.replace("model: some-model", "model: some-model\n  temperature: 9.0")
    with pytest.raises(ConfigError, match="Invalid model configuration"):
        load_model_config(_write(tmp_path, body))


def test_non_positive_dimension_is_rejected(tmp_path: Path) -> None:
    body = _MINIMAL.replace("dimension: 8", "dimension: 0")
    with pytest.raises(ConfigError, match="Invalid model configuration"):
        load_model_config(_write(tmp_path, body))


def test_non_mapping_document_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_model_config(_write(tmp_path, "- just\n- a list\n"))


def test_malformed_yaml_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Could not read"):
        load_model_config(_write(tmp_path, "provider: [unclosed\n"))


def test_configuration_is_immutable(tmp_path: Path) -> None:
    """Frozen models keep configuration from drifting after the startup checks."""
    config = load_model_config(_write(tmp_path, _MINIMAL))
    with pytest.raises(ValidationError):
        config.generation.model = "other"
