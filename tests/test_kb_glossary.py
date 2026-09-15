"""Tests for folding a parsed glossary into retrievable passages."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from rail_rag.ingestion.docs.canonical import Block, TableRow, decode_block, write_blocks
from rail_rag.ingestion.docs.exceptions import ParseError
from rail_rag.ingestion.docs.sources import ParseSpec, SectionSpec, SourceRegistry, load_registry
from rail_rag.rag.exceptions import RagError
from rail_rag.rag.store.glossary import (
    doc_id_for,
    glossary_chunks,
    load_declared_glossaries,
    load_glossary,
)

DOC_ID = "infrabel_ns_glossary"

SPEC = ParseSpec(
    sections=[
        SectionSpec(heading="Definitions", columns=["term", "definition", "source"]),
        SectionSpec(heading="Explanation of abbreviations", columns=["abbreviation", "expansion"]),
    ]
)

# --- real lines from the artefact ------------------------------------------

ABBREV_HEADING = (
    '{"type":"heading","page":16,"section":"Explanation of abbreviations",'
    '"text":"Explanation of abbreviations","bbox":[70.92,120.282,285.832,138.282]}'
)
RT_FIRST_SENSE = (
    '{"type":"table_row","page":18,"section":"Explanation of abbreviations",'
    '"cells":{"abbreviation":"RT","expansion":"Real-Time"},'
    '"bbox":[70.9,134.879,542.4,160.439]}'
)
RT_SECOND_SENSE = (
    '{"type":"table_row","page":18,"section":"Explanation of abbreviations",'
    '"cells":{"abbreviation":"","expansion":"Reservable Tracks"},'
    '"bbox":[70.9,160.439,542.4,186.12]}'
)
RU_ROW = (
    '{"type":"table_row","page":18,"section":"Explanation of abbreviations",'
    '"cells":{"abbreviation":"RU","expansion":"Railway Undertaking"},'
    '"bbox":[70.9,186.12,542.4,211.68]}'
)
RID_ROW = (
    '{"type":"table_row","page":17,"section":"Explanation of abbreviations",'
    '"cells":{"abbreviation":"RID","expansion":"R\\u00e8glement concernant le transport '
    "International ferroviaire de marchandises\\nDangereuses (Regulation on the "
    'International Carriage of Dangerous Goods\\nby Rail)"},'
    '"bbox":[70.9,404.519,542.4,457.079]}'
)
DEFINITIONS_HEADING = (
    '{"type":"heading","page":1,"section":"Definitions","text":"Definitions",'
    '"bbox":[70.92,90.882,152.856,108.882]}'
)
APPLICANT_ROW = (
    '{"type":"table_row","page":1,"section":"Definitions","cells":{"term":"Applicant",'
    '"definition":"A railway undertaking or an international grouping of railway '
    "undertakings or\\nother persons or legal entities, such as the competent authorities "
    "under\\nRegulation (EC) 1370/2007 and shippers, freight forwarders and "
    "combined\\ntransport operators, with a public-service or commercial interest in "
    'procuring\\ninfrastructure capacity.",'
    '"source":"Article 3, 11\\u00b0 of the Railway Codex"},'
    '"bbox":[70.8,311.279,779.6,390.6]}'
)
WRAPPED_TERM_ROW = (
    '{"type":"table_row","page":3,"section":"Definitions",'
    '"cells":{"term":"Commercial passenger transport\\nservices (HkvNPso);",'
    '"definition":"Services for the national or international commercial transport of '
    "passengers.\\nSuch transport is not designed for high speed and is not provided under "
    "a public\\nservice contract concluded between a railway undertaking and a competent"
    '\\nauthority.","source":"Infrabel (market segment within\\nthe framework of the '
    'user charge)"},"bbox":[70.8,227.64,779.6,293.52]}'
)
RAILNETEUROPE_PARAGRAPH = (
    '{"type":"paragraph","page":16,"section":"Definitions",'
    '"text":"A more elaborate glossary is available on the website of RailNetEurope.",'
    '"bbox":[70.8,90.0,779.6,110.0]}'
)


def _blocks(*lines: str) -> list[Block]:
    return [decode_block(line) for line in lines]


# --- the two-sense abbreviation --------------------------------------------


def test_a_second_sense_folds_into_the_entry_it_belongs_to() -> None:
    """``RT`` is two rows because it means two things, not because one ran over."""
    chunks = glossary_chunks(
        _blocks(ABBREV_HEADING, RT_FIRST_SENSE, RT_SECOND_SENSE, RU_ROW), SPEC, doc_id=DOC_ID
    )
    assert len(chunks) == 2
    assert chunks[0].heading == "Explanation of abbreviations — RT"
    assert chunks[0].content == "Real-Time\n\nReservable Tracks"


def test_the_two_senses_are_separated_by_a_blank_line() -> None:
    """A space would produce the nonsense string 'Real-Time Reservable Tracks'."""
    chunks = glossary_chunks(_blocks(RT_FIRST_SENSE, RT_SECOND_SENSE), SPEC, doc_id=DOC_ID)
    assert "Real-Time Reservable Tracks" not in chunks[0].content
    assert chunks[0].content.splitlines() == ["Real-Time", "", "Reservable Tracks"]


def test_folding_removes_a_position_rather_than_leaving_a_gap() -> None:
    """``chunk_index`` counts entries, not the rectangles the publisher drew."""
    chunks = glossary_chunks(_blocks(RT_FIRST_SENSE, RT_SECOND_SENSE, RU_ROW), SPEC, doc_id=DOC_ID)
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert chunks[1].heading == "Explanation of abbreviations — RU"


def test_a_continuation_with_nothing_to_continue_is_a_loud_error() -> None:
    with pytest.raises(RagError, match="no preceding entry"):
        glossary_chunks(_blocks(RT_SECOND_SENSE), SPEC, doc_id=DOC_ID)


def test_a_continuation_does_not_attach_across_a_section_boundary() -> None:
    """The blank cell means 'the row above'; across tables that claim is false."""
    with pytest.raises(RagError, match="no preceding entry"):
        glossary_chunks(_blocks(APPLICANT_ROW, RT_SECOND_SENSE), SPEC, doc_id=DOC_ID)


# --- how a row is rendered -------------------------------------------------


def test_a_two_column_section_is_rendered_without_labels() -> None:
    """'expansion: Real-Time' would put a config key into the vector."""
    chunks = glossary_chunks(_blocks(RU_ROW), SPEC, doc_id=DOC_ID)
    assert chunks[0].content == "Railway Undertaking"


def test_a_three_column_section_labels_its_fields() -> None:
    """Unlabelled, the legal citation reads as the last sentence of the definition."""
    chunks = glossary_chunks(_blocks(APPLICANT_ROW), SPEC, doc_id=DOC_ID)
    assert chunks[0].content.startswith("definition: A railway undertaking")
    assert "\n\nsource: Article 3, 11° of the Railway Codex" in chunks[0].content


def test_a_newline_inside_a_cell_is_left_alone() -> None:
    """It is the publisher's line wrap, and in a definition it carries structure."""
    chunks = glossary_chunks(_blocks(RID_ROW), SPEC, doc_id=DOC_ID)
    assert "marchandises\nDangereuses" in chunks[0].content


def test_a_term_wrapped_across_two_lines_is_flattened_in_the_heading() -> None:
    """A heading is printed on one line and prefixed to the embedding text."""
    chunks = glossary_chunks(_blocks(WRAPPED_TERM_ROW), SPEC, doc_id=DOC_ID)
    assert chunks[0].heading == ("Definitions — Commercial passenger transport services (HkvNPso);")
    assert "\n" not in (chunks[0].heading or "")


# --- the other block types -------------------------------------------------


def test_a_heading_block_produces_no_chunk() -> None:
    """Every block already carries its ``section``; the heading adds nothing."""
    assert glossary_chunks(_blocks(DEFINITIONS_HEADING), SPEC, doc_id=DOC_ID) == []


def test_prose_outside_the_tables_survives_as_its_own_passage() -> None:
    """Dropping it would be the silent loss the block contract exists to prevent."""
    chunks = glossary_chunks(_blocks(RAILNETEUROPE_PARAGRAPH), SPEC, doc_id=DOC_ID)
    assert len(chunks) == 1
    assert chunks[0].heading == "Definitions"
    assert chunks[0].content.endswith("RailNetEurope.")


# --- what the entry inherits for free --------------------------------------


def test_the_embedding_text_carries_section_and_term() -> None:
    """Nothing is computed here: ``Chunk`` already prepends the heading."""
    chunks = glossary_chunks(_blocks(RT_FIRST_SENSE, RT_SECOND_SENSE), SPEC, doc_id=DOC_ID)
    assert chunks[0].embedding_text.startswith("Explanation of abbreviations — RT\n\n")
    assert "Reservable Tracks" in chunks[0].embedding_text


def test_the_hash_moves_when_a_folded_sense_changes() -> None:
    """Why folding is safe for quota: the digest covers the whole entry."""
    folded = glossary_chunks(_blocks(RT_FIRST_SENSE, RT_SECOND_SENSE), SPEC, doc_id=DOC_ID)
    alone = glossary_chunks(_blocks(RT_FIRST_SENSE), SPEC, doc_id=DOC_ID)
    assert folded[0].content_hash != alone[0].content_hash


# --- the contract with the registry ----------------------------------------


def test_a_section_the_registry_does_not_declare_is_refused() -> None:
    spec = ParseSpec(sections=[SectionSpec(heading="Definitions", columns=["term", "definition"])])
    with pytest.raises(RagError, match="not declared"):
        glossary_chunks(_blocks(RU_ROW), spec, doc_id=DOC_ID)


def test_a_row_missing_its_key_column_is_refused() -> None:
    """The keys come from the registry; a missing one means they were not applied."""
    row = TableRow(
        page=18,
        section="Explanation of abbreviations",
        cells={"expansion": "Railway Undertaking"},
        bbox=(70.9, 186.12, 542.4, 211.68),
    )
    with pytest.raises(RagError, match="'abbreviation' cell"):
        glossary_chunks([row], SPEC, doc_id=DOC_ID)


# --- reading the artefact --------------------------------------------------


def test_the_doc_id_is_the_source_id_without_the_version() -> None:
    """A new edition must update the terms in place, not shadow them."""
    assert doc_id_for(Path("data/docs/parsed/infrabel_ns_glossary__20260402.jsonl")) == DOC_ID
    assert doc_id_for(Path("infrabel_ns_glossary__20270101.jsonl")) == DOC_ID


def test_a_filename_outside_the_convention_is_refused() -> None:
    with pytest.raises(RagError, match="not a parsed artefact"):
        doc_id_for(Path("glossary.jsonl"))


def test_the_artefact_is_read_and_chunked_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "infrabel_ns_glossary__20260402.jsonl"
    write_blocks(path, _blocks(ABBREV_HEADING, RT_FIRST_SENSE, RT_SECOND_SENSE, RU_ROW))
    chunks = load_glossary(path, SPEC)
    assert [chunk.doc_id for chunk in chunks] == [DOC_ID, DOC_ID]
    assert chunks[0].content == "Real-Time\n\nReservable Tracks"


def test_a_missing_artefact_is_a_loud_error(tmp_path: Path) -> None:
    with pytest.raises(ParseError, match="Could not read"):
        load_glossary(tmp_path / "infrabel_ns_glossary__20260402.jsonl", SPEC)


# --- resolving the registry ------------------------------------------------

_DIGEST = "a" * 64

GLOSSARY_ENTRY = f"""
sources:
  - source_id: infrabel_ns_glossary
    title: Network Statement
    publisher: Infrabel
    landing_page: https://example.org/landing
    canonical_url: https://example.org/files/ns_20260402.pdf
    version: "20260402"
    language: en
    document_type: glossary
    sha256: "{_DIGEST}"
    parse:
      sections:
        - heading: Definitions
          columns: [term, definition, source]
        - heading: Explanation of abbreviations
          columns: [abbreviation, expansion]
"""

UNDESCRIBED_ENTRY = f"""  - source_id: example_prose
    title: Example Prose
    publisher: Example Publisher
    landing_page: https://example.org/landing
    canonical_url: https://example.org/files/prose.pdf
    version: "1.0"
    language: fr
    document_type: prose
    sha256: "{_DIGEST}"
"""


def _registry(tmp_path: Path, body: str) -> SourceRegistry:
    path = tmp_path / "sources.yaml"
    path.write_text(body, encoding="utf-8")
    return load_registry(path)


def _parsed_dir(tmp_path: Path, *lines: str, version: str = "20260402") -> Path:
    directory = tmp_path / "parsed"
    write_blocks(directory / f"{DOC_ID}__{version}.jsonl", _blocks(*lines))
    return directory


def test_a_declared_document_is_chunked(tmp_path: Path) -> None:
    parsed = _parsed_dir(tmp_path, ABBREV_HEADING, RT_FIRST_SENSE, RT_SECOND_SENSE, RU_ROW)
    chunks = load_declared_glossaries(_registry(tmp_path, GLOSSARY_ENTRY), parsed)
    assert [chunk.doc_id for chunk in chunks] == [DOC_ID, DOC_ID]
    assert chunks[0].content == "Real-Time\n\nReservable Tracks"


def test_a_source_without_a_parse_block_is_skipped(tmp_path: Path) -> None:
    """Fetching and parsing are separate capabilities; an undescribed source has no chunks."""
    parsed = _parsed_dir(tmp_path, RU_ROW)
    registry = _registry(tmp_path, GLOSSARY_ENTRY + UNDESCRIBED_ENTRY)
    assert {chunk.doc_id for chunk in load_declared_glossaries(registry, parsed)} == {DOC_ID}


def test_a_declared_artefact_that_was_never_parsed_names_the_command(tmp_path: Path) -> None:
    with pytest.raises(ParseError, match=f"docs-parse --source {DOC_ID}"):
        load_declared_glossaries(_registry(tmp_path, GLOSSARY_ENTRY), tmp_path / "parsed")


def test_the_declared_version_decides_which_artefact_is_read(tmp_path: Path) -> None:
    """A JSONL from an older version would collide on (doc_id, chunk_index)."""
    parsed = _parsed_dir(tmp_path, RU_ROW, version="20250101")
    with pytest.raises(ParseError, match="20260402"):
        load_declared_glossaries(_registry(tmp_path, GLOSSARY_ENTRY), parsed)


def test_an_oversized_entry_is_logged_rather_than_split(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Half a definition reads as a whole one, so the warning is the whole remedy."""
    parsed = _parsed_dir(tmp_path, APPLICANT_ROW)
    with caplog.at_level(logging.WARNING, logger="rail_rag.rag.store.glossary"):
        chunks = load_declared_glossaries(_registry(tmp_path, GLOSSARY_ENTRY), parsed, max_chars=20)
    assert len(chunks) == 1
    assert len(chunks[0].content) > 20
    assert "Applicant" in caplog.text
