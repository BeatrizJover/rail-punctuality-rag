"""Regression tests for the ``lru_cache`` on :func:`rail_rag.core.config.get_settings`.

The two tests below are a deliberate pair. Each one asserts only that
``get_settings()`` reflects *its own* environment - trivially true in isolation.
Together they pin the cache: whichever runs second fails the moment
``isolated_settings_env`` stops calling ``get_settings.cache_clear()``, because the
cached ``Settings`` from the first test would still be handed out.

Do not merge them into a single test. A cache that is never invalidated is invisible
within one call; the bug only exists across them.
"""

from __future__ import annotations

import pytest

from rail_rag.core.config import get_settings


def test_get_settings_reflects_the_environment_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cached ``Settings`` from an earlier test must not survive into this one."""
    monkeypatch.setenv("RAIL_RAG_DB_NAME", "cache_probe_first")
    assert get_settings().db_name == "cache_probe_first"


def test_get_settings_reflects_the_environment_second(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same assertion, different value: the pair is what makes the cache observable."""
    monkeypatch.setenv("RAIL_RAG_DB_NAME", "cache_probe_second")
    assert get_settings().db_name == "cache_probe_second"


def test_get_settings_still_caches_within_one_test() -> None:
    """Clearing per test must not defeat the point of caching inside a process."""
    assert get_settings() is get_settings()
