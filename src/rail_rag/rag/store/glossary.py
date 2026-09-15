"""Fold a parsed glossary artefact into the passages the retriever understands."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from rail_rag.core.exceptions import ConfigError
from rail_rag.ingestion.docs.canonical import Block, Heading, Paragraph, TableRow, read_blocks
from rail_rag.ingestion.docs.sources import ParseSpec
from rail_rag.rag.exceptions import RagError
from rail_rag.rag.store.chunking import Chunk

#: Joins two rows folded into one entry. A single newline is already taken: the
#: parser uses it for a line wrapped inside one cell.
ENTRY_SEPARATOR = "\n\n"

#: Joins the labelled fields of one row, for the same reason.
FIELD_SEPARATOR = "\n\n"

#: Separates the section from the term in a heading, as ``kb-search`` prints it.
HEADING_SEPARATOR = " — "

#: Artefact stems are ``{source_id}__{version}``; the version is not part of the doc_id.
_STEM_SEPARATOR = "__"

_WHITESPACE = re.compile(r"\s+")


@dataclass
class _Entry:
    """One glossary entry under construction, before it becomes a ``Chunk``."""

    section: str
    heading: str
    parts: list[str] = field(default_factory=list)
    from_table: bool = False


def _flatten(text: str) -> str:
    """Collapse internal whitespace, for text that has to sit on one line."""
    return _WHITESPACE.sub(" ", text).strip()


def _render_row(row: TableRow, columns: list[str]) -> str:
    """Render the non-key columns of a row, labelled only when the section declares several."""
    remaining = columns[1:]
    values = [(name, row.cells.get(name, "").strip()) for name in remaining]
    present = [(name, value) for name, value in values if value]
    if not present:
        return ""
    if len(remaining) == 1:
        return present[0][1]
    return FIELD_SEPARATOR.join(f"{name}: {value}" for name, value in present)


def _columns_for(spec: ParseSpec, section: str) -> list[str]:
    """The declared columns of a section.

    Raises:
        RagError: if the section is not declared in the registry.
    """
    try:
        return spec.select(section).columns
    except ConfigError as exc:
        raise RagError(
            f"Section {section!r} appears in the parsed document but is not declared"
            " in the registry. The artefact and config/sources.yaml disagree;"
            " re-run 'docs-parse' after reconciling them."
        ) from exc


def glossary_chunks(blocks: Iterable[Block], spec: ParseSpec, *, doc_id: str) -> list[Chunk]:
    """Fold a stream of canonical blocks into retrievable passages.

    Headings are dropped; prose outside the tables becomes a passage of its own.

    Raises:
        RagError: if a section is not declared, if a row is missing its key
            column, or if a continuation row has no entry to continue.
    """
    entries: list[_Entry] = []
    for block in blocks:
        if isinstance(block, Heading):
            continue
        if isinstance(block, Paragraph):
            entries.append(
                _Entry(
                    section=block.section,
                    heading=_flatten(block.section),
                    parts=[block.text.strip()],
                )
            )
            continue
        columns = _columns_for(spec, block.section)
        key_column = columns[0]
        if key_column not in block.cells:
            raise RagError(
                f"A row in {block.section!r} on page {block.page} has no"
                f" {key_column!r} cell. The parsed artefact does not match the"
                " columns declared in config/sources.yaml."
            )
        key = _flatten(block.cells[key_column])
        body = _render_row(block, columns)
        if key:
            # A row with a key and nothing else still becomes a chunk: the term
            # itself is content, and dropping it would be a silent loss.
            entries.append(
                _Entry(
                    section=block.section,
                    heading=f"{_flatten(block.section)}{HEADING_SEPARATOR}{key}",
                    parts=[body or key],
                    from_table=True,
                )
            )
            continue
        if not entries or not entries[-1].from_table or entries[-1].section != block.section:
            raise RagError(
                f"A row in {block.section!r} on page {block.page} has an empty"
                f" {key_column!r} but no preceding entry in the same section to"
                " continue."
            )
        if body:
            entries[-1].parts.append(body)
    return [
        Chunk(
            doc_id=doc_id,
            chunk_index=index,
            heading=entry.heading,
            content=ENTRY_SEPARATOR.join(entry.parts),
        )
        for index, entry in enumerate(entries)
    ]


def doc_id_for(path: Path) -> str:
    """The source_id an artefact was written under, inverse of ``artefact_paths``.

    Raises:
        RagError: if the filename does not follow that convention.
    """
    source_id, separator, _version = path.stem.partition(_STEM_SEPARATOR)
    if not separator or not source_id:
        raise RagError(
            f"{path.name} is not a parsed artefact: the filename must be"
            f" '{{source_id}}{_STEM_SEPARATOR}{{version}}.jsonl'."
        )
    return source_id


def load_glossary(path: Path, spec: ParseSpec, *, doc_id: str | None = None) -> list[Chunk]:
    """Read one parsed glossary from disk and chunk it.

    Raises:
        ParseError: if the artefact is unreadable or malformed.
        RagError: if its blocks do not match the declared sections.
    """
    return glossary_chunks(read_blocks(path), spec, doc_id=doc_id or doc_id_for(path))
