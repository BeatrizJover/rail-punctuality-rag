"""Tests for structured logging and credential redaction."""

from __future__ import annotations

import logging

from rail_rag.core.logging import RedactingFormatter, configure_logging


def _format(message: str) -> str:
    formatter = RedactingFormatter(fmt="%(message)s")
    record = logging.LogRecord("test", logging.INFO, __file__, 0, message, None, None)
    return formatter.format(record)


def test_dsn_password_is_redacted() -> None:
    out = _format("connecting to postgresql+psycopg://rail_rag:s3cr3t@localhost:5432/rail_rag")
    assert "s3cr3t" not in out
    assert "***" in out


def test_key_value_secret_is_redacted() -> None:
    assert "hunter2" not in _format("auth password=hunter2 ok")
    assert "abc123" not in _format('{"token": "abc123"}')


def test_configure_logging_installs_redacting_handler() -> None:
    configure_logging("DEBUG")
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert any(isinstance(h.formatter, RedactingFormatter) for h in root.handlers)
