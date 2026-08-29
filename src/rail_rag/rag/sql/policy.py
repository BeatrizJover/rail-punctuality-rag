"""Execution policy for generated SQL.

Kept in a committed YAML file, like the model configuration: a reviewer must be
able to see in a diff that someone widened the allow-list or raised the timeout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rail_rag.core.exceptions import ConfigError

#: Schema every unqualified table name is resolved against before the allow-list check.
DEFAULT_SCHEMA = "gold"


class SqlPolicy(BaseModel):
    """The limits every generated query is held to."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Fully qualified, lower-case. Staging and ops are deliberately absent.
    allowed_tables: frozenset[str]
    #: Applied to the outermost query, replacing any larger limit the model wrote.
    max_rows: int = Field(default=200, gt=0)
    #: Enforced by the server, so a runaway query dies without the client cooperating.
    statement_timeout_ms: int = Field(default=5000, gt=0)
    #: Function names the model may never call, whatever the surrounding query.
    blocked_functions: frozenset[str] = frozenset()


class RetrievalConfig(BaseModel):
    """Root of ``config/retrieval_config.yaml``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sql: SqlPolicy


def load_retrieval_config(path: Path) -> RetrievalConfig:
    """Read and validate the retrieval configuration file.

    Raises:
        ConfigError: if the file is missing, unreadable, not a mapping, or invalid.
    """
    resolved = path.expanduser().resolve()
    try:
        raw: Any = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Retrieval configuration not found: {resolved}") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not read retrieval configuration at {resolved}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Retrieval configuration at {resolved} must be a mapping")

    try:
        return RetrievalConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid retrieval configuration at {resolved}: {exc}") from exc
