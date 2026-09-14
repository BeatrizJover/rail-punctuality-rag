"""Reconstruction of a PDF's structure from the marks the publisher actually left.

The source glossary draws its tables with thin filled rectangles and nothing else:
no vertical rules anywhere, and on the later pages no header band either. Asking
``pdfplumber`` to find the tables therefore returns a phantom nine-column strip on
the first page and nothing at all on the last three. So the tables are not
*found*, they are *measured*: each row separator is drawn as one segment per
column, which means the horizontal marks carry the column geometry in their own
endpoints.

That geometry is taken by consensus across the pages of a section rather than page
by page. A hyperlink underline is also a thin wide rectangle, and on two pages it
sits exactly where a row separator would - close enough to swallow a heading into
the table. Consensus discards it without a rule written specially for hyperlinks:
a mark is a row separator only if it spans a column every other page agrees on.

Nothing here writes files or reads configuration. The functions take a page and
return values, so each measurement is testable on a two-page fixture that draws
its table the same way the real publisher does.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import pdfplumber

from rail_rag.ingestion.docs.canonical import Block, Heading, Paragraph, TableRow
from rail_rag.ingestion.docs.exceptions import ParseError
from rail_rag.ingestion.docs.sources import DocumentSource, SectionSpec

#: A rule is a drawn mark, not a shape: anything thicker is a filled panel.
MAX_RULE_THICKNESS = 2.0
#: Below this, a mark is an underline fragment or a bullet, not a table edge.
MIN_RULE_WIDTH = 40.0
#: Coordinates of the same intended edge differ by fractions of a point.
EDGE_TOLERANCE = 2.0
#: Words within this vertical distance belong to the same line of text.
LINE_TOLERANCE = 3.0
#: Share of a section's pages that must agree on a column edge.
COLUMN_QUORUM = 0.5
#: Share of pages a line must repeat on to count as a running header or footer.
RUNNING_QUORUM = 0.8

#: Page numbers change; the line they sit on does not. Masking digits lets one
#: comparison catch both a constant footer and a numbered one.
_DIGITS = re.compile(r"\d+")


@dataclass(frozen=True)
class Rule:
    """One horizontal mark: a row separator, or something pretending to be one."""

    x0: float
    x1: float
    y: float


@dataclass(frozen=True)
class TextLine:
    """A run of words sharing a baseline, with the box that encloses them."""

    text: str
    bbox: tuple[float, float, float, float]
    top: float


@dataclass(frozen=True)
class Grid:
    """The measured table geometry of one page."""

    columns: Sequence[float]
    rows: Sequence[float]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """The rectangle the table occupies."""
        return (self.columns[0], self.rows[0], self.columns[-1], self.rows[-1])

    @property
    def column_count(self) -> int:
        return len(self.columns) - 1


def _cluster(values: Iterable[float], tolerance: float = EDGE_TOLERANCE) -> list[float]:
    """Collapse near-identical coordinates into their mean."""
    groups: list[list[float]] = []
    for value in sorted(values):
        if groups and value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [sum(group) / len(group) for group in groups]


def page_rules(page: Any) -> list[Rule]:
    """Every horizontal mark on a page that is thin and wide enough to be an edge.

    Both filled rectangles and stroked lines qualify: which one a producer emits is
    an accident of the tool that made the PDF, not a property of the table.
    """
    found: list[Rule] = []
    for shape in (*page.rects, *page.lines):
        height = abs(float(shape["bottom"]) - float(shape["top"]))
        width = abs(float(shape["x1"]) - float(shape["x0"]))
        if height <= MAX_RULE_THICKNESS and width >= MIN_RULE_WIDTH:
            found.append(
                Rule(
                    x0=float(shape["x0"]),
                    x1=float(shape["x1"]),
                    y=(float(shape["top"]) + float(shape["bottom"])) / 2,
                )
            )
    return found


def column_edges(pages: Sequence[Any]) -> list[float]:
    """The vertical edges the pages of a section agree on.

    Raises:
        ParseError: if the pages share no grid at all.
    """
    votes: Counter[float] = Counter()
    for page in pages:
        rules = page_rules(page)
        at_height: dict[float, set[float]] = {}
        for edge in _cluster([rule.x0 for rule in rules] + [rule.x1 for rule in rules]):
            heights = {
                rule.y
                for rule in rules
                if abs(rule.x0 - edge) <= EDGE_TOLERANCE or abs(rule.x1 - edge) <= EDGE_TOLERANCE
            }
            at_height[round(edge, 1)] = heights
        # A column edge is drawn again under every row. A hyperlink underline is
        # drawn once, so a single height is what disqualifies it - no rule about
        # hyperlinks, and none needed.
        votes.update(edge for edge, heights in at_height.items() if len(heights) >= 2)

    quorum = max(1, int(len(pages) * COLUMN_QUORUM))
    agreed = _cluster([edge for edge, count in votes.items() if count >= quorum])
    if len(agreed) < 2:
        raise ParseError(
            f"No table grid is shared by these {len(pages)} page(s): "
            f"{len(agreed)} agreed edge(s) out of {len(votes)} candidate(s)"
        )
    return agreed


def row_edges(page: Any, columns: Sequence[float]) -> list[float]:
    """The horizontal edges of a page, keeping only marks that span a real column.

    A hyperlink underline is thin, wide, and horizontal. What it is not is aligned
    with a column, so alignment is what separates the two.
    """
    bands = list(pairwise(columns))
    aligned = [
        rule.y
        for rule in page_rules(page)
        if any(
            abs(rule.x0 - left) <= EDGE_TOLERANCE and abs(rule.x1 - right) <= EDGE_TOLERANCE
            for left, right in bands
        )
    ]
    return _cluster(aligned)


def page_grid(page: Any, columns: Sequence[float]) -> Grid | None:
    """The table geometry of one page, or ``None`` if the page carries no table."""
    rows = row_edges(page, columns)
    if len(rows) < 2:
        return None
    return Grid(columns=list(columns), rows=rows)


def page_lines(page: Any) -> list[TextLine]:
    """Group the words of a page into lines of text, top to bottom."""
    by_line: list[list[dict[str, Any]]] = []
    for word in sorted(page.extract_words(), key=lambda w: (float(w["top"]), float(w["x0"]))):
        if by_line and float(word["top"]) - float(by_line[-1][0]["top"]) <= LINE_TOLERANCE:
            by_line[-1].append(word)
        else:
            by_line.append([word])

    lines: list[TextLine] = []
    for group in by_line:
        ordered = sorted(group, key=lambda w: float(w["x0"]))
        lines.append(
            TextLine(
                text=" ".join(str(word["text"]) for word in ordered),
                bbox=(
                    min(float(w["x0"]) for w in group),
                    min(float(w["top"]) for w in group),
                    max(float(w["x1"]) for w in group),
                    max(float(w["bottom"]) for w in group),
                ),
                top=min(float(w["top"]) for w in group),
            )
        )
    return lines


def _normalise(text: str) -> str:
    """Collapse whitespace and mask digits, so a numbered footer compares equal."""
    return _DIGITS.sub("#", " ".join(text.split()))


def running_lines(lines_per_page: Sequence[Sequence[TextLine]]) -> set[str]:
    """The normalised lines that repeat on enough pages to be page furniture.

    Only lines outside the tables should be offered here: the table's own header
    row repeats on every page too, and it is content.
    """
    votes: Counter[str] = Counter()
    for lines in lines_per_page:
        votes.update({_normalise(line.text) for line in lines})
    quorum = max(2, int(len(lines_per_page) * RUNNING_QUORUM))
    return {text for text, count in votes.items() if count >= quorum}


def _inside(
    bbox: tuple[float, float, float, float], box: tuple[float, float, float, float]
) -> bool:
    x0, top, x1, bottom = bbox
    bx0, btop, bx1, bbottom = box
    centre_x, centre_y = (x0 + x1) / 2, (top + bottom) / 2
    return bx0 <= centre_x <= bx1 and btop <= centre_y <= bbottom


def extract_grid_rows(page: Any, grid: Grid) -> list[list[str]]:
    """Read the cells of a measured grid, preserving line breaks inside a cell."""
    table = page.extract_table(
        {
            "vertical_strategy": "explicit",
            "horizontal_strategy": "explicit",
            "explicit_vertical_lines": list(grid.columns),
            "explicit_horizontal_lines": list(grid.rows),
        }
    )
    return [[(cell or "").strip() for cell in row] for row in (table or [])]


def assert_grid_coverage(page: Any, grid: Grid, rows: Sequence[Sequence[str]]) -> None:
    """Every character drawn inside the grid must appear in an extracted cell.

    A word that falls between two cells is dropped by the extractor without a
    word, which is the failure this pipeline exists to make impossible. Characters
    are compared as a multiset because cell order and whitespace are not the point.

    Raises:
        ParseError: if the grid swallowed text it did not return.
    """
    drawn: Counter[str] = Counter()
    for word in page.extract_words():
        box = (
            float(word["x0"]),
            float(word["top"]),
            float(word["x1"]),
            float(word["bottom"]),
        )
        if _inside(box, grid.bbox):
            drawn.update("".join(str(word["text"]).split()))

    returned: Counter[str] = Counter()
    for row in rows:
        for cell in row:
            returned.update("".join(cell.split()))

    missing = drawn - returned
    if missing:
        sample = "".join(sorted(missing.elements()))[:60]
        raise ParseError(
            f"Page {page.page_number}: {sum(missing.values())} character(s) inside the "
            f"table were not returned by any cell: {sample!r}"
        )


def _header_row_matches(row: Sequence[str], spec: SectionSpec) -> bool:
    return [cell.strip().casefold() for cell in row] == [
        name.strip().casefold() for name in spec.columns
    ]


def _rows_to_blocks(
    page_number: int,
    spec: SectionSpec,
    grid: Grid,
    rows: Sequence[Sequence[str]],
) -> list[Block]:
    """Name the cells of each data row, dropping a repeated header row."""
    if grid.column_count != len(spec.columns):
        raise ParseError(
            f"Page {page_number}: the table has {grid.column_count} measured column(s) but "
            f"section {spec.heading!r} declares {len(spec.columns)}: {spec.columns}"
        )

    blocks: list[Block] = []
    for index, row in enumerate(rows):
        if index == 0 and _header_row_matches(row, spec):
            continue
        if not any(cell.strip() for cell in row):
            continue
        blocks.append(
            TableRow(
                page=page_number,
                section=spec.heading,
                cells=dict(zip(spec.columns, row, strict=True)),
                bbox=(grid.columns[0], grid.rows[index], grid.columns[-1], grid.rows[index + 1]),
            )
        )
    return blocks


def heading_line(page: Any, headings: Sequence[str]) -> tuple[str, TextLine] | None:
    """The declared heading printed on this page, with the line that carries it."""
    wanted = {_normalise(name).casefold(): name for name in headings}
    for line in page_lines(page):
        found = wanted.get(_normalise(line.text).casefold())
        if found is not None:
            return found, line
    return None


def parse_document(path: Path, source: DocumentSource) -> list[Block]:
    """Turn a pinned PDF into an ordered stream of canonical blocks.

    Raises:
        ParseError: if the document does not have the structure the registry
            declares, or if any text on any page would be dropped.
    """
    spec_of = source.parse_spec()

    with pdfplumber.open(path) as pdf:
        pages = list(pdf.pages)
        if not pages:
            raise ParseError(f"{path} has no pages")

        # First pass: which section each page belongs to.
        section_of_page: list[str] = []
        #: Where a section starts mid-page, the text above the heading still
        #: belongs to the section that is ending.
        preceding: list[tuple[str, float] | None] = []
        current: str | None = None
        for page in pages:
            found = heading_line(page, spec_of.headings)
            if found is not None and current is not None and found[0] != current:
                preceding.append((current, found[1].top))
            else:
                preceding.append(None)
            if found is not None:
                current = found[0]
            if current is None:
                raise ParseError(
                    f"Page {page.page_number} precedes every declared section heading "
                    f"({', '.join(spec_of.headings)}); the document layout changed"
                )
            section_of_page.append(current)

        # Second pass: the grid of each section, agreed across its own pages.
        grids: dict[int, Grid | None] = {}
        for heading in dict.fromkeys(section_of_page):
            owned = [p for p, name in zip(pages, section_of_page, strict=True) if name == heading]
            columns = column_edges(owned)
            for page in owned:
                grids[page.page_number] = page_grid(page, columns)

        # Third pass: page furniture, judged only on text outside the tables.
        outside: list[list[TextLine]] = []
        for page in pages:
            grid = grids[page.page_number]
            outside.append(
                [
                    line
                    for line in page_lines(page)
                    if grid is None or not _inside(line.bbox, grid.bbox)
                ]
            )
        furniture = running_lines(outside)

        blocks: list[Block] = []
        for page, lines, heading, before in zip(
            pages, outside, section_of_page, preceding, strict=True
        ):
            spec = spec_of.select(heading)
            grid = grids[page.page_number]
            if grid is None:
                raise ParseError(
                    f"Page {page.page_number} of section {heading!r} carries no table; "
                    f"the document layout changed"
                )

            for line in lines:
                if _normalise(line.text) in furniture:
                    continue
                owner = heading
                if before is not None and line.top < before[1]:
                    owner = before[0]
                if _normalise(line.text).casefold() == _normalise(heading).casefold():
                    blocks.append(
                        Heading(
                            page=page.page_number,
                            section=heading,
                            text=line.text,
                            bbox=line.bbox,
                        )
                    )
                else:
                    blocks.append(
                        Paragraph(
                            page=page.page_number,
                            section=owner,
                            text=line.text,
                            bbox=line.bbox,
                        )
                    )

            rows = extract_grid_rows(page, grid)
            assert_grid_coverage(page, grid, rows)
            blocks.extend(_rows_to_blocks(page.page_number, spec, grid, rows))

    return blocks
