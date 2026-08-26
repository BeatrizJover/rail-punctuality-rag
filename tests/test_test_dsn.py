"""Guards on the DSN that integration tests connect to.

Integration fixtures drop and recreate the Gold schema. The one thing that must
never happen is that they do it to the working database, so the derivation is
asserted rather than trusted.
"""

from __future__ import annotations


def test_default_test_dsn_targets_a_dedicated_database(default_test_dsn: str) -> None:
    database = default_test_dsn.rsplit("/", 1)[-1]
    assert database.endswith("_test")


def test_default_test_dsn_uses_the_project_driver(default_test_dsn: str) -> None:
    assert default_test_dsn.startswith("postgresql+psycopg://")
