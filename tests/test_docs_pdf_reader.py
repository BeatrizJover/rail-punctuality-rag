"""Tests for the geometry-driven PDF reader.

Every fixture is drawn the way the real publisher draws: filled rectangles, one
per column under each row, no vertical marks. A fixture with a proper grid would
be green against a parser that cannot read the actual document, which is the
failure this file exists to prevent.
"""

from __future__ import annotations

from pathlib import Path

import pdfplumber
import pytest
from pydantic import HttpUrl
from tests.conftest import FakePage, write_glossary_pdf

from rail_rag.ingestion.docs.exceptions import ParseError
from rail_rag.ingestion.docs.pdf_reader import (
    Grid,
    assert_grid_coverage,
    column_edges,
    extract_grid_rows,
    page_grid,
    page_lines,
    page_rules,
    parse_document,
    row_edges,
    running_lines,
)
from rail_rag.ingestion.docs.sources import DocumentSource, ParseSpec, SectionSpec

_EDGES = (71.0, 241.0, 610.0, 780.0)
_TWO_EDGES = (71.0, 188.0, 542.0)


def _source(*sections: SectionSpec) -> DocumentSource:
    return DocumentSource(
        source_id="example_glossary",
        title="Example Glossary",
        publisher="Example Publisher",
        landing_page=HttpUrl("https://example.org/landing"),
        canonical_url=HttpUrl("https://example.org/files/example_20260101.pdf"),
        version="20260101",
        language="en",
        document_type="glossary",
        sha256="a" * 64,
        parse=ParseSpec(sections=list(sections)),
    )


_DEFINITIONS = SectionSpec(heading="Definitions", columns=["term", "definition", "source"])


def _two_page_glossary(tmp_path: Path, **overrides: object) -> Path:
    first = FakePage(
        heading="Definitions",
        rows=[
            ["Term", "Definition", "Source"],
            ["Applicant", "A railway undertaking.", "Railway Codex"],
            ["Charge", "Price demanded for services.", "RNE"],
        ],
        **overrides,  # type: ignore[arg-type]
    )
    second = FakePage(
        rows=[
            ["Term", "Definition", "Source"],
            ["Shunting", "Moving a rail vehicle.", "Eurostat"],
        ]
    )
    return write_glossary_pdf(tmp_path / "glossary.pdf", [first, second], column_edges=_EDGES)


# --- measuring the marks ---------------------------------------------------


def test_the_row_separators_are_found_as_rules(tmp_path: Path) -> None:
    path = _two_page_glossary(tmp_path)
    with pdfplumber.open(path) as pdf:
        rules = page_rules(pdf.pages[0])
    # Three data rows plus a closing separator, one segment per column.
    assert len(rules) == 4 * (len(_EDGES) - 1)


def test_the_column_edges_come_from_the_rule_endpoints(tmp_path: Path) -> None:
    """There is not one vertical mark in the document; the horizontal ones carry
    the column geometry in their own endpoints."""
    path = _two_page_glossary(tmp_path)
    with pdfplumber.open(path) as pdf:
        edges = column_edges(list(pdf.pages))

    assert len(edges) == len(_EDGES)
    assert all(abs(found - wanted) < 1.0 for found, wanted in zip(edges, _EDGES, strict=True))


def test_a_hyperlink_underline_is_not_a_column_edge(tmp_path: Path) -> None:
    """It is thin, wide and horizontal, so only alignment separates it from a rule."""
    path = _two_page_glossary(tmp_path, decoy_underline=True)
    with pdfplumber.open(path) as pdf:
        edges = column_edges(list(pdf.pages))

    assert len(edges) == len(_EDGES)


def test_a_hyperlink_underline_is_not_a_row_edge(tmp_path: Path) -> None:
    """Regression guard: on the real document this underline sits between the
    heading and the table, and counting it as a row swallows the heading."""
    path = _two_page_glossary(tmp_path, decoy_underline=True)
    with pdfplumber.open(path) as pdf:
        page = pdf.pages[0]
        rows = row_edges(page, column_edges(list(pdf.pages)))
        headings = [line for line in page_lines(page) if line.text == "Definitions"]

    assert len(rows) == 4
    assert headings[0].top < min(rows)


def test_a_page_without_rules_has_no_grid(tmp_path: Path) -> None:
    path = write_glossary_pdf(
        tmp_path / "flat.pdf",
        [FakePage(heading="Definitions", rows=[["a", "b", "c"]], draw_rules=False)],
        column_edges=_EDGES,
    )
    with pdfplumber.open(path) as pdf:
        assert page_grid(pdf.pages[0], _EDGES) is None


def test_pages_that_share_no_grid_are_rejected(tmp_path: Path) -> None:
    path = write_glossary_pdf(
        tmp_path / "flat.pdf",
        [FakePage(heading="Definitions", rows=[["a", "b", "c"]], draw_rules=False)],
        column_edges=_EDGES,
    )
    with pdfplumber.open(path) as pdf, pytest.raises(ParseError, match="No table grid"):
        column_edges(list(pdf.pages))


# --- page furniture --------------------------------------------------------


def test_a_numbered_footer_is_recognised_despite_the_number(tmp_path: Path) -> None:
    """Masking digits is what makes one comparison catch both a constant header
    and a page number that changes on every page."""
    path = _two_page_glossary(tmp_path)
    with pdfplumber.open(path) as pdf:
        outside = [
            [line for line in page_lines(page) if line.top < 130.0 or line.top > 500.0]
            for page in pdf.pages
        ]
    furniture = running_lines(outside)

    assert "NETWORK STATEMENT" in furniture
    assert "#" in furniture


def test_a_heading_is_not_mistaken_for_furniture(tmp_path: Path) -> None:
    """It appears on one page out of two, well under the quorum."""
    path = _two_page_glossary(tmp_path)
    with pdfplumber.open(path) as pdf:
        outside = [[line for line in page_lines(page) if line.top < 130.0] for page in pdf.pages]

    assert "Definitions" not in running_lines(outside)


# --- the coverage assertion ------------------------------------------------


def test_text_inside_the_grid_that_no_cell_returned_is_an_error(tmp_path: Path) -> None:
    """The defence: a word falling between two cells is dropped by the extractor
    without a word, and the whole pipeline is built to refuse that."""
    path = _two_page_glossary(tmp_path)
    with pdfplumber.open(path) as pdf:
        page = pdf.pages[0]
        grid = page_grid(page, _EDGES)
        assert grid is not None
        complete = extract_grid_rows(page, grid)

        assert_grid_coverage(page, grid, complete)  # the honest result passes

        truncated = [row[:-1] for row in complete]
        with pytest.raises(ParseError, match="not returned by any cell"):
            assert_grid_coverage(page, grid, truncated)


def test_the_coverage_error_names_the_page(tmp_path: Path) -> None:
    path = _two_page_glossary(tmp_path)
    with pdfplumber.open(path) as pdf:
        page = pdf.pages[1]
        grid = page_grid(page, _EDGES)
        assert grid is not None
        with pytest.raises(ParseError, match="Page 2"):
            assert_grid_coverage(page, grid, [])


# --- the whole document ----------------------------------------------------


def test_a_document_becomes_a_stream_of_blocks(tmp_path: Path) -> None:
    blocks = parse_document(_two_page_glossary(tmp_path), _source(_DEFINITIONS))
    kinds = [block.type for block in blocks]

    assert kinds.count("heading") == 1
    assert kinds.count("table_row") == 3


def test_the_repeated_header_row_is_not_emitted_as_data(tmp_path: Path) -> None:
    """It is printed again at the top of every page; three data rows, not five."""
    blocks = parse_document(_two_page_glossary(tmp_path), _source(_DEFINITIONS))
    rows = [block for block in blocks if block.type == "table_row"]

    assert all(row.cells["term"] != "Term" for row in rows)
    assert [row.cells["term"] for row in rows] == ["Applicant", "Charge", "Shunting"]


def test_cells_are_keyed_by_the_declared_column_names(tmp_path: Path) -> None:
    blocks = parse_document(_two_page_glossary(tmp_path), _source(_DEFINITIONS))
    first = next(block for block in blocks if block.type == "table_row")

    assert set(first.cells) == {"term", "definition", "source"}
    assert first.cells["source"] == "Railway Codex"


def test_running_furniture_is_not_emitted(tmp_path: Path) -> None:
    blocks = parse_document(_two_page_glossary(tmp_path), _source(_DEFINITIONS))
    texts = [block.text for block in blocks if block.type in {"heading", "paragraph"}]

    assert "NETWORK STATEMENT" not in texts
    assert "1" not in texts


def test_prose_outside_the_table_survives(tmp_path: Path) -> None:
    """A contract of only headings and rows would drop this line in silence."""
    path = write_glossary_pdf(
        tmp_path / "glossary.pdf",
        [
            FakePage(
                heading="Definitions",
                stray="A more elaborate glossary is available elsewhere.",
                rows=[["Applicant", "A railway undertaking.", "Codex"]],
            ),
            FakePage(rows=[["Charge", "Price demanded.", "RNE"]]),
        ],
        column_edges=_EDGES,
    )
    blocks = parse_document(path, _source(_DEFINITIONS))
    paragraphs = [block for block in blocks if block.type == "paragraph"]

    assert len(paragraphs) == 1
    assert paragraphs[0].text.startswith("A more elaborate glossary")


def test_prose_above_a_new_heading_belongs_to_the_outgoing_section(tmp_path: Path) -> None:
    """On the real document the closing note of Definitions is printed on the same
    page the abbreviations start on, physically above that heading."""
    path = write_glossary_pdf(
        tmp_path / "glossary.pdf",
        [
            FakePage(
                heading="Definitions",
                rows=[["Applicant", "A railway undertaking.", "Codex"]],
            ),
            FakePage(
                heading="Explanation of abbreviations",
                stray="A more elaborate glossary is available elsewhere.",
                rows=[["Applicant", "A railway undertaking.", "Codex"]],
            ),
        ],
        column_edges=_EDGES,
    )
    second = SectionSpec(
        heading="Explanation of abbreviations", columns=["term", "definition", "source"]
    )
    blocks = parse_document(path, _source(_DEFINITIONS, second))
    paragraph = next(block for block in blocks if block.type == "paragraph")

    assert paragraph.page == 2
    assert paragraph.section == "Definitions"


def test_a_continuation_row_keeps_its_empty_cell(tmp_path: Path) -> None:
    """The abbreviations table continues an entry with a blank first column."""
    path = write_glossary_pdf(
        tmp_path / "abbr.pdf",
        [
            FakePage(heading="Definitions", rows=[["RT", "Real-Time"], ["", "Reservable Tracks"]]),
            FakePage(rows=[["RU", "Railway Undertaking"]]),
        ],
        column_edges=_TWO_EDGES,
    )
    source = _source(SectionSpec(heading="Definitions", columns=["abbreviation", "expansion"]))
    rows = [block for block in parse_document(path, source) if block.type == "table_row"]

    assert rows[1].cells == {"abbreviation": "", "expansion": "Reservable Tracks"}


def test_a_column_count_the_registry_does_not_declare_is_rejected(tmp_path: Path) -> None:
    """The measured grid is the truth; the declaration is the assertion against it."""
    path = _two_page_glossary(tmp_path)
    source = _source(SectionSpec(heading="Definitions", columns=["term", "definition"]))

    with pytest.raises(ParseError, match="3 measured column"):
        parse_document(path, source)


def test_a_page_before_every_heading_is_rejected(tmp_path: Path) -> None:
    """Nothing can be filed under a section that has not started."""
    path = write_glossary_pdf(
        tmp_path / "headless.pdf",
        [
            FakePage(rows=[["Applicant", "A railway undertaking.", "Codex"]]),
            FakePage(heading="Definitions", rows=[["Charge", "Price demanded.", "RNE"]]),
        ],
        column_edges=_EDGES,
    )
    with pytest.raises(ParseError, match="precedes every declared section heading"):
        parse_document(path, _source(_DEFINITIONS))


def test_a_source_without_a_parse_block_cannot_be_parsed(tmp_path: Path) -> None:
    path = _two_page_glossary(tmp_path)
    bare = _source(_DEFINITIONS).model_copy(update={"parse": None})

    with pytest.raises(Exception, match="declares no 'parse' block"):
        parse_document(path, bare)


def test_every_row_carries_a_box_inside_the_measured_grid(tmp_path: Path) -> None:
    blocks = parse_document(_two_page_glossary(tmp_path), _source(_DEFINITIONS))
    rows = [block for block in blocks if block.type == "table_row"]

    for row in rows:
        x0, top, x1, bottom = row.bbox
        assert abs(x0 - _EDGES[0]) < 1.0
        assert abs(x1 - _EDGES[-1]) < 1.0
        assert bottom > top


def test_the_grid_bbox_spans_its_own_edges() -> None:
    grid = Grid(columns=[71.0, 241.0, 610.0], rows=[140.0, 176.0])

    assert grid.bbox == (71.0, 140.0, 610.0, 176.0)
    assert grid.column_count == 2
