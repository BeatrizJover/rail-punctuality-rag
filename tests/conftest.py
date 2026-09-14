"""Shared test fixtures."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote_plus

import pytest
from dotenv import dotenv_values
from reportlab.lib.colors import black
from reportlab.pdfgen.canvas import Canvas
from sqlalchemy import Engine, create_engine
from sqlalchemy.exc import OperationalError

from rail_rag.core.config import get_settings
from rail_rag.db.schema import create_schema, drop_schema
from rail_rag.rag.store.models import KbSchema, build_kb_schema
from rail_rag.rag.store.schema import create_kb_schema, drop_kb_schema

_TEST_DSN_VAR = "TEST_DATABASE_URL"
_TEST_DB_SUFFIX = "_test"
_REPO_ROOT = Path(__file__).resolve().parent.parent
_UNREACHABLE_SIGNALS = (
    "connection refused",
    "could not connect",
    "could not translate host",
    "failed to resolve host",
    "timeout expired",
)
#: Small on purpose: the tests assert on ranking, not on embedding quality.
_KB_TEST_DIMENSION = 8


def _default_test_dsn() -> str:
    """Derive the integration-test DSN from the repository's own ``.env``."""
    values = dotenv_values(_REPO_ROOT / ".env")
    user = values.get("RAIL_RAG_DB_USER") or "rail_rag"
    password = values.get("RAIL_RAG_DB_PASSWORD") or ""
    host = values.get("RAIL_RAG_DB_HOST") or "localhost"
    port = values.get("RAIL_RAG_DB_PORT") or "5432"
    name = (values.get("RAIL_RAG_DB_NAME") or "rail_rag") + _TEST_DB_SUFFIX
    return f"postgresql+psycopg://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{name}"


@pytest.fixture(autouse=True)
def isolated_settings_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run each test in a clean directory with no ``.env`` and no app vars."""
    for key in list(os.environ):
        if key.startswith("RAIL_RAG_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def default_test_dsn() -> str:
    """Expose the derived DSN so its guarantees can be asserted in a test."""
    return _default_test_dsn()


@pytest.fixture(scope="session")
def postgres_engine() -> Iterator[Engine]:
    """Yield an engine against a live PostgreSQL, or skip the test.

    Skipping rather than failing keeps ``pytest`` green on a clean checkout and
    in CI, where no database is provisioned.
    """
    dsn = os.environ.get(_TEST_DSN_VAR) or _default_test_dsn()
    engine = create_engine(dsn, pool_pre_ping=True, future=True)
    try:
        with engine.connect():
            pass
    except OperationalError as exc:
        engine.dispose()
        if not any(signal in str(exc).lower() for signal in _UNREACHABLE_SIGNALS):
            raise
        pytest.skip(f"No PostgreSQL reachable; set {_TEST_DSN_VAR} or start docker compose")
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def clean_schema(postgres_engine: Engine) -> Iterator[Engine]:
    """Provide an engine whose Gold schema has just been created from scratch."""
    drop_schema(postgres_engine)
    create_schema(postgres_engine)
    try:
        yield postgres_engine
    finally:
        drop_schema(postgres_engine)


@pytest.fixture(scope="session")
def gold_parquet_dir() -> Path:
    """Directory of the four real Gold samples, converted from CSV to Parquet.

    These are a faithful mirror of Betty's manual extraction: same 100 rows, same
    nulls, native Parquet types. They are three independent cuts, so fact keys may
    not resolve against the dimensions here - use the synthetic generator for a
    referentially coherent set.
    """
    return Path(__file__).resolve().parent / "fixtures" / "gold_parquet"


@pytest.fixture(scope="session")
def synthetic_gold_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A referentially coherent Gold export: every fact key resolves to a dimension."""
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    from generate_sample_gold import generate

    out_dir = tmp_path_factory.mktemp("synthetic_gold")
    generate(out_dir, stations=20, days=3)
    return out_dir


@pytest.fixture
def kb_schema(postgres_engine: Engine) -> Iterator[KbSchema]:
    """A knowledge-base schema created from scratch at a small test dimension."""
    kb = build_kb_schema(_KB_TEST_DIMENSION)
    drop_kb_schema(postgres_engine)
    create_kb_schema(postgres_engine, kb)
    try:
        yield kb
    finally:
        drop_kb_schema(postgres_engine)


# --- synthetic glossary PDFs ------------------------------------------------
#
# Built rather than committed, because the point of a fixture is that a reviewer
# can see what it asserts. A binary under tests/fixtures says nothing about which
# page carries the decoy underline or where the running header sits.
#
# The tables are drawn the way the real publisher draws them: one thin FILLED
# RECTANGLE per column under each row, and no vertical marks at all. A fixture
# drawn with a full grid would pass against a parser that could never read the
# real document.

#: Matches the real document closely enough for the geometry to be comparable.
PAGE_WIDTH = 842.0
PAGE_HEIGHT = 595.0
RULE_THICKNESS = 0.6

_HEADER_TOP = 30.0
_STRAY_TOP = 86.0
_HEADING_TOP = 110.0
_TABLE_TOP = 140.0
_ROW_HEIGHT = 36.0
_FOOTER_TOP = 560.0


@dataclass(frozen=True)
class FakePage:
    """One page of a synthetic glossary."""

    #: Emitted as table rows. Include the header row explicitly when the test
    #: needs the parser to recognise and drop one.
    rows: list[list[str]] = field(default_factory=list)
    #: Repeated on every page by default, so it becomes running furniture.
    running: tuple[str, ...] = ("NETWORK STATEMENT", "Version: 01/01/2026")
    #: Printed above the table; the parser matches it against the registry.
    heading: str | None = None
    #: Prose outside the table, which a two-type contract would silently drop.
    stray: str | None = None
    #: A hyperlink underline: thin, wide, horizontal, aligned with nothing.
    decoy_underline: bool = False
    #: When false, no rules are drawn and the page carries no measurable table.
    draw_rules: bool = True


def write_glossary_pdf(
    path: Path,
    pages: list[FakePage],
    *,
    column_edges: tuple[float, ...] = (71.0, 241.0, 610.0, 780.0),
) -> Path:
    """Draw a synthetic glossary and return the path it was written to."""
    canvas = Canvas(str(path), pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
    canvas.setFillColor(black)

    def text(content: str, x: float, top: float, size: float = 10.0) -> None:
        canvas.setFont("Helvetica", size)
        canvas.drawString(x, PAGE_HEIGHT - top - size, content)

    def rule(x0: float, x1: float, top: float) -> None:
        canvas.rect(
            x0,
            PAGE_HEIGHT - top - RULE_THICKNESS,
            x1 - x0,
            RULE_THICKNESS,
            stroke=0,
            fill=1,
        )

    for index, page in enumerate(pages, start=1):
        for offset, line in enumerate(page.running):
            text(line, 600.0, _HEADER_TOP + offset * 18.0)
        text(str(index), 410.0, _FOOTER_TOP)

        if page.stray is not None:
            text(page.stray, column_edges[0], _STRAY_TOP)
        if page.heading is not None:
            text(page.heading, column_edges[0], _HEADING_TOP, size=13.0)
        if page.decoy_underline:
            # Sits between the heading and the table, spanning no column.
            rule(column_edges[1] + 40.0, column_edges[-1] - 30.0, _HEADING_TOP + 17.0)

        for row_index, row in enumerate(page.rows):
            top = _TABLE_TOP + row_index * _ROW_HEIGHT
            if page.draw_rules:
                for left, right in zip(column_edges, column_edges[1:], strict=False):
                    rule(left, right, top)
            for cell, left in zip(row, column_edges, strict=False):
                text(cell, left + 5.0, top + 10.0)
        if page.rows and page.draw_rules:
            bottom = _TABLE_TOP + len(page.rows) * _ROW_HEIGHT
            for left, right in zip(column_edges, column_edges[1:], strict=False):
                rule(left, right, bottom)

        canvas.showPage()

    canvas.save()
    return path
