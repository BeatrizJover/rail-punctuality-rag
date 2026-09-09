"""The API key is a secret and must behave like one."""

from __future__ import annotations

from pathlib import Path

import pytest

from rail_rag.core.config import Settings


def test_api_key_defaults_to_empty() -> None:
    assert Settings().llm_api_key.get_secret_value() == ""


def test_api_key_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAIL_RAG_LLM_API_KEY", "top-secret")
    assert Settings().llm_api_key.get_secret_value() == "top-secret"


def test_api_key_is_not_leaked_by_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    """``SecretStr`` keeps the key out of tracebacks and log records."""
    monkeypatch.setenv("RAIL_RAG_LLM_API_KEY", "top-secret")
    assert "top-secret" not in repr(Settings())


def test_config_path_has_a_repository_relative_default() -> None:
    assert Settings().llm_config_path == Path("config/model_config.yaml")
