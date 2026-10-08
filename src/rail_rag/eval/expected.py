"""Ground-truth results for golden cases, computed once and cached on disk.

The reference SQL may legitimately scan the whole period, so it runs under a far
longer timeout than the one serving users get. The cache key includes the fact
coverage window, so loading new data invalidates every stored result.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, text

from rail_rag.api.schemas import jsonable_row
from rail_rag.eval.cases import GoldenCase
from rail_rag.rag.sql.executor import execute_safe_query
from rail_rag.rag.sql.guard import validate_sql
from rail_rag.rag.sql.policy import SqlPolicy

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(".eval_cache/expected")
DEFAULT_TIMEOUT_MS = 300_000

#: First and last ``date_key`` of the fact, as ISO strings.
Window = tuple[str, str]


@dataclass(frozen=True)
class ExpectedResult:
    """The reference result of one case, in the JSON-safe form stored on disk."""

    columns: list[str]
    rows: list[list[Any]]
    truncated: bool
    elapsed_ms: float
    cached: bool


def fact_window(engine: Engine) -> Window:
    """Read the fact's coverage with ``min``/``max`` only, which the index answers directly."""
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        first, last = conn.execute(
            text("SELECT min(date_key), max(date_key) FROM gold.fact_stop_event")
        ).one()
        conn.rollback()
    return (str(first), str(last))


def compute_expected(
    engine: Engine,
    case: GoldenCase,
    policy: SqlPolicy,
    *,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    window: Window | None = None,
) -> ExpectedResult:
    """Return the reference result for ``case``, from the cache when it is current.

    ``window`` lets a caller that runs many cases read the coverage once; when it
    is given and the result is cached, the database is not touched at all.

    Raises:
        ValueError: if the case has no reference SQL.
        UnsafeQueryError: if the reference SQL fails validation.
        QueryExecutionError: if it fails or exceeds ``timeout_ms``.
    """
    if case.reference_sql is None:
        raise ValueError(f"Case {case.id} has no reference_sql")
    safe = validate_sql(case.reference_sql, policy)

    key = cache_key(case.reference_sql, window if window is not None else fact_window(engine))
    path = cache_dir / f"{key}.json"
    cached = _read(path)
    if cached is not None:
        return cached

    generous = policy.model_copy(update={"statement_timeout_ms": timeout_ms})
    result = execute_safe_query(engine, safe, generous)
    stored = ExpectedResult(
        columns=list(result.columns),
        rows=[jsonable_row(row) for row in result.rows],
        truncated=result.truncated,
        elapsed_ms=result.elapsed_ms,
        cached=False,
    )
    _write(path, stored)
    logger.info("computed expected result for %s in %.0f ms", case.id, stored.elapsed_ms)
    return stored


def cache_key(reference_sql: str, window: Window) -> str:
    """Digest of the SQL and the coverage it was computed against."""
    payload = f"{reference_sql}\n{window[0]}..{window[1]}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _read(path: Path) -> ExpectedResult | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return ExpectedResult(
            columns=list(raw["columns"]),
            rows=[list(row) for row in raw["rows"]],
            truncated=bool(raw["truncated"]),
            elapsed_ms=float(raw["elapsed_ms"]),
            cached=True,
        )
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("ignoring unreadable expected-result cache entry %s", path.name)
        return None


def _write(path: Path, result: ExpectedResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "columns": result.columns,
        "rows": result.rows,
        "truncated": result.truncated,
        "elapsed_ms": result.elapsed_ms,
    }
    scratch = path.with_suffix(f".{os.getpid()}.tmp")
    scratch.write_text(json.dumps(payload), encoding="utf-8")
    scratch.replace(path)
