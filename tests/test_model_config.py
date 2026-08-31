"""Tests for the YAML model configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rail_rag.core.exceptions import ConfigError
from rail_rag.rag.providers.config import load_model_config, load_profiles

#: Resolved from ``__file__``: the autouse settings fixture chdirs into a tmp_path.
REPO_CONFIG = Path(__file__).resolve().parent.parent / "config" / "model_config.yaml"

_MINIMAL = """
active: bench
profiles:
  bench:
    provider: fake
    generation:
      model: some-model
    embedding:
      model: some-embedding
      dimension: 8
"""

_TWO_PROFILES = (
    _MINIMAL
    + """  other:
    provider: fake
    generation:
      model: bigger-model
    embedding:
      model: bigger-embedding
      dimension: 16
    chunking:
      max_chars: 4000
      min_chars: 200
"""
)


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "model_config.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_repo_configuration_is_valid() -> None:
    """The committed file must always load; it is the default the app boots with."""
    config = load_model_config(REPO_CONFIG)
    assert config.provider
    assert config.embedding.dimension > 0


def test_repo_configuration_keeps_an_offline_profile() -> None:
    """``fake`` is what lets the stack run with no key; losing it would be silent."""
    profiles = load_profiles(REPO_CONFIG)
    assert "fake" in profiles.names
    assert profiles.select("fake").provider == "fake"


def test_defaults_are_applied(tmp_path: Path) -> None:
    config = load_model_config(_write(tmp_path, _MINIMAL))
    assert config.generation.temperature == 0.0
    assert config.embedding.batch_size == 32
    assert config.max_retries == 3
    assert config.chunking.max_chars == 1500


def test_the_active_profile_is_returned_by_default(tmp_path: Path) -> None:
    config = load_model_config(_write(tmp_path, _TWO_PROFILES))
    assert config.embedding.model == "some-embedding"


def test_a_profile_can_be_selected_by_name(tmp_path: Path) -> None:
    """Switching model is a selection, not an edit that overwrites the old one."""
    path = _write(tmp_path, _TWO_PROFILES)
    config = load_model_config(path, profile="other")
    assert config.embedding.dimension == 16
    assert config.chunking.max_chars == 4000
    assert load_profiles(path).names == ["bench", "other"]


def test_an_unknown_profile_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Unknown profile"):
        load_model_config(_write(tmp_path, _TWO_PROFILES), profile="absent")


def test_an_active_profile_that_does_not_exist_is_rejected(tmp_path: Path) -> None:
    """A typo in ``active`` must fail at load, not fall back to something arbitrary."""
    body = _MINIMAL.replace("active: bench", "active: banch")
    with pytest.raises(ConfigError, match="Invalid model configuration"):
        load_model_config(_write(tmp_path, body))


def test_an_empty_profile_set_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Invalid model configuration"):
        load_model_config(_write(tmp_path, "active: bench\nprofiles: {}\n"))


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_model_config(tmp_path / "absent.yaml")


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    """``extra=forbid`` turns a typo into a startup failure, not a silent default."""
    with pytest.raises(ConfigError, match="Invalid model configuration"):
        load_model_config(_write(tmp_path, _MINIMAL + "\ntemperatur: 0.7\n"))


def test_unknown_key_inside_a_profile_is_rejected(tmp_path: Path) -> None:
    body = _MINIMAL.replace("      dimension: 8", "      dimension: 8\n      dimensionn: 9")
    with pytest.raises(ConfigError, match="Invalid model configuration"):
        load_model_config(_write(tmp_path, body))


def test_out_of_range_temperature_is_rejected(tmp_path: Path) -> None:
    body = _MINIMAL.replace("model: some-model", "model: some-model\n      temperature: 9.0")
    with pytest.raises(ConfigError, match="Invalid model configuration"):
        load_model_config(_write(tmp_path, body))


def test_non_positive_dimension_is_rejected(tmp_path: Path) -> None:
    body = _MINIMAL.replace("dimension: 8", "dimension: 0")
    with pytest.raises(ConfigError, match="Invalid model configuration"):
        load_model_config(_write(tmp_path, body))


def test_inverted_chunk_bounds_are_rejected(tmp_path: Path) -> None:
    body = _MINIMAL + "    chunking:\n      max_chars: 100\n      min_chars: 500\n"
    with pytest.raises(ConfigError, match="Invalid model configuration"):
        load_model_config(_write(tmp_path, body))


def test_non_mapping_document_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_model_config(_write(tmp_path, "- just\n- a list\n"))


def test_malformed_yaml_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Could not read"):
        load_model_config(_write(tmp_path, "active: [unclosed\n"))


def test_configuration_is_immutable(tmp_path: Path) -> None:
    """Frozen models keep configuration from drifting after the startup checks."""
    config = load_model_config(_write(tmp_path, _MINIMAL))
    with pytest.raises(ValidationError):
        config.generation.model = "other"
