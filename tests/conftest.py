"""Shared test fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_settings_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run each test in a clean directory with no ``.env`` and no app vars.

    This keeps ``Settings()`` deterministic regardless of the developer's
    local environment.
    """
    for key in list(os.environ):
        if key.startswith("RAIL_RAG_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
