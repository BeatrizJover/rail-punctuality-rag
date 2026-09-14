"""The canonical representation of a parsed document, and its JSONL encoding.

A parsed PDF is stored as one JSON object per line, every line a block. The
stream is deliberately homogeneous: no header record, no trailing summary. A
reader that has to special-case line zero pays for that special case in every
consumer, forever, so the provenance of a parse lives in a sibling
``.meta.json`` instead - the same split the fetcher already makes between the
artefact and its manifest.

Three block types cover this corpus. ``table_row`` and ``heading`` carry the
glossary itself; ``paragraph`` exists because the source document puts prose
outside its tables, and a contract with nowhere to put that prose would drop it
silently. Silence is the failure mode this pipeline is built to avoid.

Every block carries the ``section`` it was found under rather than leaving the
reader to infer it from page numbers. A row on page 7 belongs to ``Definitions``
because the heading on page 1 governs until the next one, and that arithmetic is
done once here rather than in every consumer downstream.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from rail_rag.ingestion.docs.exceptions import ParseError

#: Bumped whenever the block shape changes. A consumer comparing this against the
#: value in a ``.meta.json`` knows whether a re-parse is due without diffing bytes.
PARSER_VERSION = "1"

#: Repository-relative default; the artefacts themselves are git-ignored.
DEFAULT_PARSED_DIR = Path("data/docs/parsed")

_RawBBox = tuple[float, float, float, float]


def _positive_area(box: _RawBBox) -> _RawBBox:
    """Reject a box that encloses nothing.

    ``pdfplumber`` reports ``(x0, top, x1, bottom)`` with the origin at the top of
    the page, so ``top < bottom`` on a well-formed box. A zero-area box means the
    geometry was guessed rather than measured, and a guessed bbox is worse than no
    bbox: it looks like provenance.
    """
    x0, top, x1, bottom = box
    if x1 <= x0 or bottom <= top:
        raise ValueError(f"bbox encloses no area: {box}")
    return box


#: ``(x0, top, x1, bottom)`` in PDF user space, as ``pdfplumber`` reports it.
BBox = Annotated[_RawBBox, AfterValidator(_positive_area)]

_CONFIG = ConfigDict(extra="forbid", frozen=True)


class Heading(BaseModel):
    """A section title, and the point at which ``section`` starts applying."""

    model_config = _CONFIG

    type: Literal["heading"] = "heading"
    page: int = Field(ge=1)
    section: str = Field(min_length=1)
    text: str = Field(min_length=1)
    bbox: BBox


class Paragraph(BaseModel):
    """Prose found outside any table."""

    model_config = _CONFIG

    type: Literal["paragraph"] = "paragraph"
    page: int = Field(ge=1)
    section: str = Field(min_length=1)
    text: str = Field(min_length=1)
    bbox: BBox


class TableRow(BaseModel):
    """One row of a table, with its columns reconnected under declared names.

    An individual cell may be empty: the source glossary continues one entry onto
    a second row with a blank first column, and rewriting that to look complete
    would invent data. A row in which *every* cell is empty is different - that is
    extraction noise, not content, and it is rejected.
    """

    model_config = _CONFIG

    type: Literal["table_row"] = "table_row"
    page: int = Field(ge=1)
    section: str = Field(min_length=1)
    cells: dict[str, str] = Field(min_length=1)
    bbox: BBox

    @model_validator(mode="after")
    def _cells_are_named_and_not_all_empty(self) -> Self:
        blank = [name for name in self.cells if not name.strip()]
        if blank:
            raise ValueError("every cell must be keyed by a declared column name")
        if not any(value.strip() for value in self.cells.values()):
            raise ValueError("a row with no content in any cell is extraction noise")
        return self


#: Discriminated on ``type``: a malformed line names the field that failed, not
#: three unrelated failures from trying every member of the union in turn.
Block = Annotated[Heading | Paragraph | TableRow, Field(discriminator="type")]

_BLOCK_ADAPTER: TypeAdapter[Block] = TypeAdapter(Block)


class ParseManifest(BaseModel):
    """Provenance of one parsed artefact.

    ``source_sha256`` is the digest of the *input* PDF, so a consumer can tell
    whether a JSONL file still corresponds to the bytes it was derived from.
    """

    model_config = _CONFIG

    source_id: str
    version: str
    source_sha256: str
    parser_version: str
    block_count: int = Field(ge=1)
    blocks_per_section: dict[str, int] = Field(min_length=1)
    parsed_at: dt.datetime

    @model_validator(mode="after")
    def _section_counts_agree_with_the_total(self) -> Self:
        total = sum(self.blocks_per_section.values())
        if total != self.block_count:
            raise ValueError(
                f"blocks_per_section sums to {total}, block_count declares {self.block_count}"
            )
        return self


def artefact_paths(
    source_id: str, version: str, dest_dir: Path = DEFAULT_PARSED_DIR
) -> tuple[Path, Path]:
    """Where the parsed stream and its manifest live, as a pair."""
    stem = dest_dir / f"{source_id}__{version}"
    return stem.with_suffix(".jsonl"), stem.with_suffix(".meta.json")


def encode_block(block: Block) -> str:
    """Render one block as a single JSONL line, without its newline."""
    return block.model_dump_json()


def decode_block(line: str, *, source: str = "<string>", line_number: int = 1) -> Block:
    """Parse one JSONL line back into a block.

    Raises:
        ParseError: if the line is not JSON, or not a valid block.
    """
    try:
        return _BLOCK_ADAPTER.validate_json(line)
    except ValidationError as exc:
        raise ParseError(f"{source}:{line_number} is not a valid block: {exc}") from exc


def write_blocks(path: Path, blocks: Iterable[Block]) -> int:
    """Write a block stream to disk and return how many lines were written.

    The parent directory is created; the file is replaced, not appended to, so a
    re-parse never interleaves two runs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for block in blocks:
            handle.write(encode_block(block))
            handle.write("\n")
            written += 1
    return written


def read_blocks(path: Path) -> Iterator[Block]:
    """Stream the blocks of a parsed artefact, one line at a time.

    Raises:
        ParseError: if the file is unreadable or any line is not a valid block.
    """
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ParseError(f"Could not read parsed document {path}: {exc}") from exc
    with handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            yield decode_block(line, source=str(path), line_number=number)
