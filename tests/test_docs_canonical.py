"""Tests for the canonical block contract and its JSONL encoding.

What is being protected here is an *artefact format*. Once a parsed file exists on
disk, a change to these models is a change to data someone already has, so the
round trip and the discriminator are asserted directly rather than through the
parser that will produce them.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rail_rag.ingestion.docs.canonical import (
    PARSER_VERSION,
    Block,
    Heading,
    Paragraph,
    ParseManifest,
    TableRow,
    artefact_paths,
    decode_block,
    encode_block,
    read_blocks,
    write_blocks,
)
from rail_rag.ingestion.docs.exceptions import DocumentError, ParseError

_BBOX = (108.0, 180.0, 1212.0, 330.0)


def _heading(section: str = "Definitions") -> Heading:
    return Heading(page=1, section=section, text=section, bbox=_BBOX)


def _row(**cells: str) -> TableRow:
    payload = cells or {
        "term": "Applicant",
        "definition": "A railway undertaking or an international grouping...",
        "source": "Article 3, 11 of the Railway Codex",
    }
    return TableRow(page=3, section="Definitions", cells=payload, bbox=_BBOX)


def _paragraph() -> Paragraph:
    return Paragraph(
        page=16,
        section="Definitions",
        text="A more elaborate glossary is available on the website of RailNetEurope.",
        bbox=_BBOX,
    )


# --- the discriminator -----------------------------------------------------


def test_the_type_tag_is_written_without_being_passed() -> None:
    """It is a constant of the class, not something a caller can get wrong."""
    assert json.loads(encode_block(_row()))["type"] == "table_row"


@pytest.mark.parametrize("block", [_heading(), _row(), _paragraph()])
def test_every_block_survives_a_round_trip(block: Block) -> None:
    assert decode_block(encode_block(block)) == block


def test_the_decoded_class_is_chosen_by_the_type_tag() -> None:
    """Without the discriminator a heading and a paragraph are interchangeable:
    same fields, same types. The tag is the only thing telling them apart."""
    assert isinstance(decode_block(encode_block(_paragraph())), Paragraph)
    assert isinstance(decode_block(encode_block(_heading())), Heading)


def test_an_unknown_type_tag_is_a_loud_error() -> None:
    line = json.dumps({"type": "footnote", "page": 1, "section": "S", "text": "x", "bbox": _BBOX})
    with pytest.raises(ParseError, match="not a valid block"):
        decode_block(line)


def test_a_missing_type_tag_is_a_loud_error() -> None:
    line = json.dumps({"page": 1, "section": "Definitions", "text": "x", "bbox": list(_BBOX)})
    with pytest.raises(ParseError, match="not a valid block"):
        decode_block(line)


def test_a_parse_error_is_a_document_error() -> None:
    """The CLI catches the family, not each member."""
    with pytest.raises(DocumentError):
        decode_block("{not json")


def test_the_error_names_the_file_and_the_line() -> None:
    with pytest.raises(ParseError, match=r"glossary\.jsonl:42"):
        decode_block("{}", source="glossary.jsonl", line_number=42)


# --- the invariants --------------------------------------------------------


def test_an_extra_field_is_rejected() -> None:
    """A future field must be added here, not smuggled in by a producer."""
    with pytest.raises(ValidationError):
        Heading(page=1, section="S", text="S", bbox=_BBOX, confidence=0.9)  # type: ignore[call-arg]


def test_a_block_is_immutable() -> None:
    block = _row()
    with pytest.raises(ValidationError):
        block.page = 9


def test_page_numbers_start_at_one() -> None:
    """Zero would mean the producer used a list index instead of a page number."""
    with pytest.raises(ValidationError):
        Paragraph(page=0, section="S", text="x", bbox=_BBOX)


@pytest.mark.parametrize(
    "bbox",
    [
        (108.0, 180.0, 108.0, 330.0),
        (108.0, 330.0, 1212.0, 330.0),
        (1212.0, 180.0, 108.0, 330.0),
    ],
)
def test_a_bbox_enclosing_no_area_is_rejected(bbox: tuple[float, float, float, float]) -> None:
    """A guessed bbox is worse than none: it looks like it was measured."""
    with pytest.raises(ValidationError, match="encloses no area"):
        Heading(page=1, section="S", text="S", bbox=bbox)


def test_a_continuation_row_with_one_empty_cell_is_accepted() -> None:
    """Regression guard: the abbreviations table continues 'RT' onto a second row
    with a blank first column. Rejecting it would discard real content; filling it
    in would invent data."""
    row = _row(abbreviation="", expansion="Reservable Tracks")
    assert row.cells["abbreviation"] == ""


def test_a_row_with_every_cell_empty_is_rejected() -> None:
    """That is extraction noise, not a continuation."""
    with pytest.raises(ValidationError, match="extraction noise"):
        _row(abbreviation="", expansion="   ")


def test_a_row_with_no_cells_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TableRow(page=3, section="Definitions", cells={}, bbox=_BBOX)


def test_an_unnamed_column_is_rejected() -> None:
    """Column names come from the registry; an empty key means they were not applied."""
    with pytest.raises(ValidationError, match="declared column name"):
        _row(**{"": "Applicant", "definition": "..."})


# --- the file on disk ------------------------------------------------------


def test_the_stream_is_one_json_object_per_line(tmp_path: Path) -> None:
    path = tmp_path / "glossary.jsonl"
    written = write_blocks(path, [_heading(), _row(), _paragraph()])

    lines = path.read_text(encoding="utf-8").splitlines()
    assert written == 3
    assert len(lines) == 3
    assert [json.loads(line)["type"] for line in lines] == ["heading", "table_row", "paragraph"]


def test_there_is_no_header_record_on_the_first_line(tmp_path: Path) -> None:
    """Provenance lives in the sibling manifest; line zero is a block like any other."""
    path = tmp_path / "glossary.jsonl"
    write_blocks(path, [_heading(), _row()])

    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["type"] == "heading"


def test_blocks_are_read_back_in_written_order(tmp_path: Path) -> None:
    path = tmp_path / "glossary.jsonl"
    blocks: list[Block] = [_heading(), _row(), _paragraph()]
    write_blocks(path, blocks)

    assert list(read_blocks(path)) == blocks


def test_a_rewrite_replaces_the_previous_run(tmp_path: Path) -> None:
    """Appending would interleave two parses into one unusable file."""
    path = tmp_path / "glossary.jsonl"
    write_blocks(path, [_heading(), _row(), _paragraph()])
    write_blocks(path, [_heading()])

    assert len(list(read_blocks(path))) == 1


def test_the_destination_directory_is_created(tmp_path: Path) -> None:
    path = tmp_path / "parsed" / "glossary.jsonl"
    write_blocks(path, [_heading()])
    assert path.is_file()


def test_a_blank_line_is_skipped(tmp_path: Path) -> None:
    """Trailing newlines are an artefact of text files, not a missing block."""
    path = tmp_path / "glossary.jsonl"
    path.write_text(encode_block(_heading()) + "\n\n", encoding="utf-8")

    assert len(list(read_blocks(path))) == 1


def test_a_corrupt_line_names_its_position(tmp_path: Path) -> None:
    path = tmp_path / "glossary.jsonl"
    path.write_text(f"{encode_block(_heading())}\nnot json\n", encoding="utf-8")

    with pytest.raises(ParseError, match="glossary.jsonl:2"):
        list(read_blocks(path))


def test_a_missing_file_is_a_loud_error(tmp_path: Path) -> None:
    with pytest.raises(ParseError, match="Could not read"):
        list(read_blocks(tmp_path / "absent.jsonl"))


def test_the_two_artefacts_sit_side_by_side(tmp_path: Path) -> None:
    stream, manifest = artefact_paths("infrabel_ns_glossary", "20260402", tmp_path)

    assert stream.name == "infrabel_ns_glossary__20260402.jsonl"
    assert manifest.name == "infrabel_ns_glossary__20260402.meta.json"
    assert stream.parent == manifest.parent


# --- the manifest ----------------------------------------------------------


def _manifest(**overrides: object) -> ParseManifest:
    payload: dict[str, object] = {
        "source_id": "infrabel_ns_glossary",
        "version": "20260402",
        "source_sha256": "a" * 64,
        "parser_version": PARSER_VERSION,
        "block_count": 5,
        "blocks_per_section": {"Definitions": 3, "Explanation of abbreviations": 2},
        "parsed_at": dt.datetime(2026, 9, 14, tzinfo=dt.UTC),
    }
    return ParseManifest.model_validate(payload | overrides)


def test_the_manifest_records_the_digest_of_its_input() -> None:
    """It is the PDF's digest, not the JSONL's: it answers 'was this derived from
    the bytes I still have?'."""
    assert _manifest().source_sha256 == "a" * 64


def test_section_counts_must_agree_with_the_total() -> None:
    """A disagreement means blocks were emitted under a section nobody counted."""
    with pytest.raises(ValidationError, match="block_count declares 9"):
        _manifest(block_count=9)


def test_the_manifest_survives_a_round_trip() -> None:
    manifest = _manifest()
    assert ParseManifest.model_validate_json(manifest.model_dump_json()) == manifest
