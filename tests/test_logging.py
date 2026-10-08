"""Tests for structured logging and credential redaction."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from rail_rag.core.logging import (
    TRACE_LOGGER,
    RedactingFormatter,
    configure_logging,
    configure_trace_sink,
)
from rail_rag.observability import span, start_trace


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


# --- trace sink -----------------------------------------------------------------


@pytest.fixture
def trace_logger() -> Iterator[logging.Logger]:
    """Restore the trace logger afterwards, so later ``caplog`` tests are unaffected."""
    logger = logging.getLogger(TRACE_LOGGER)
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    yield logger
    for handler in logger.handlers:
        if handler not in handlers:
            handler.close()
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


@pytest.fixture
def root_handlers() -> Iterator[None]:
    """``configure_logging`` replaces the root handlers; put the originals back."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def _emit_trace() -> None:
    with start_trace("answer", question="q"), span("retrieve"):
        pass


def test_configuring_the_trace_sink_twice_keeps_one_handler(
    trace_logger: logging.Logger, tmp_path: Path
) -> None:
    configure_trace_sink(tmp_path / "traces.jsonl")
    configure_trace_sink(tmp_path / "traces.jsonl")
    assert len(trace_logger.handlers) == 1
    assert trace_logger.propagate is False
    assert trace_logger.level == logging.INFO


def test_the_sink_creates_its_directory(trace_logger: logging.Logger, tmp_path: Path) -> None:
    target = tmp_path / "nested" / "logs" / "traces.jsonl"
    configure_trace_sink(target)
    assert target.parent.is_dir()


def test_an_emitted_trace_is_one_json_line_in_the_file(
    trace_logger: logging.Logger, tmp_path: Path
) -> None:
    target = tmp_path / "traces.jsonl"
    configure_trace_sink(target)
    _emit_trace()
    _emit_trace()

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["spans"][0]["name"] for line in lines] == ["retrieve", "retrieve"]


def test_traces_stay_off_stdout_when_both_sinks_are_active(
    trace_logger: logging.Logger,
    root_handlers: None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("DEBUG")
    configure_trace_sink(tmp_path / "traces.jsonl")
    _emit_trace()
    assert "retrieve" not in capsys.readouterr().out


def test_traces_reach_the_file_when_the_root_logger_is_at_warning(
    trace_logger: logging.Logger, root_handlers: None, tmp_path: Path
) -> None:
    target = tmp_path / "traces.jsonl"
    logging.getLogger().setLevel(logging.WARNING)
    trace_logger.setLevel(logging.NOTSET)
    configure_trace_sink(target)
    _emit_trace()
    assert len(target.read_text(encoding="utf-8").splitlines()) == 1
