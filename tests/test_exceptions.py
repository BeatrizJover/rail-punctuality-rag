"""Tests for the exception hierarchy."""

from __future__ import annotations

from rail_rag.core.exceptions import ConfigError, DatabaseError, RailRagError


def test_all_errors_derive_from_base() -> None:
    assert issubclass(ConfigError, RailRagError)
    assert issubclass(DatabaseError, RailRagError)


def test_base_error_is_an_exception() -> None:
    assert issubclass(RailRagError, Exception)
