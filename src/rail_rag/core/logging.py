"""Structured logging configuration with credential redaction."""

from __future__ import annotations

import logging
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

TRACE_LOGGER = "rail_rag.trace"
DEFAULT_TRACE_PATH = Path("logs/traces.jsonl")
_TRACE_HANDLER_NAME = "rail_rag.trace.file"
_TRACE_MAX_BYTES = 5 * 1024 * 1024
_TRACE_BACKUPS = 3

# (pattern, replacement) pairs applied to every formatted log message.
_REDACTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # A URL userinfo section, e.g. "scheme://user:password@host" -> mask pass.
    (re.compile(r"(://[^:/@\s]+:)([^@/\s]+)(@)"), r"\1***\3"),
    # "key=value" / "key: value" style secrets.
    (
        re.compile(
            r"(?i)(password|passwd|pwd|secret|token|api[_-]?key)"
            r"(['\"]?\s*[:=]\s*['\"]?)([^'\"\s,;]+)"
        ),
        r"\1\2***",
    ),
)


def redact(message: str) -> str:
    """Return ``message`` with known credential patterns masked."""
    for pattern, replacement in _REDACTION_PATTERNS:
        message = pattern.sub(replacement, message)
    return message


class RedactingFormatter(logging.Formatter):
    """A :class:`logging.Formatter` that masks credentials in the output."""

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger with a redacting stream handler.

    Any previously registered root handlers are removed so the function is
    safe to call more than once (for example, in tests).
    """
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        RedactingFormatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())


def configure_trace_sink(path: Path = DEFAULT_TRACE_PATH) -> None:
    """Send ``rail_rag.trace`` to a rotating JSON-lines file and nowhere else; safe to repeat."""
    trace_logger = logging.getLogger(TRACE_LOGGER)
    # Explicit, because an entrypoint that never configured the root logger would inherit WARNING.
    trace_logger.setLevel(logging.INFO)
    trace_logger.propagate = False
    if any(handler.get_name() == _TRACE_HANDLER_NAME for handler in trace_logger.handlers):
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        path, maxBytes=_TRACE_MAX_BYTES, backupCount=_TRACE_BACKUPS, encoding="utf-8"
    )
    handler.set_name(_TRACE_HANDLER_NAME)
    handler.setFormatter(logging.Formatter("%(message)s"))
    trace_logger.addHandler(handler)
