"""Guards on the DSN that integration tests connect to.

Integration fixtures drop and recreate the Gold schema. The one thing that must
never happen is that they do it to the working database, so the derivation is
asserted rather than trusted.
"""

from __future__ import annotations

import pytest

from tests.conftest import database_required


def test_default_test_dsn_targets_a_dedicated_database(default_test_dsn: str) -> None:
    database = default_test_dsn.rsplit("/", 1)[-1]
    assert database.endswith("_test")


def test_default_test_dsn_uses_the_project_driver(default_test_dsn: str) -> None:
    assert default_test_dsn.startswith("postgresql+psycopg://")


@pytest.mark.parametrize("value", ["1", "true", "YES", " yes "])
def test_database_required_accepts_truthy_values(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_DATABASE_REQUIRED", value)
    assert database_required()


@pytest.mark.parametrize("value", ["", "0", "false", "no"])
def test_database_required_defaults_to_skipping(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_DATABASE_REQUIRED", value)
    assert not database_required()
