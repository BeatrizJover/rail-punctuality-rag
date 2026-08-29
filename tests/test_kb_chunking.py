"""Tests for corpus chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from rail_rag.rag.exceptions import RagError
from rail_rag.rag.store.chunking import Chunk, chunk_markdown, load_corpus

#: Resolved from ``__file__``: the autouse settings fixture chdirs into a tmp_path.
CORPUS_DIR = Path(__file__).resolve().parent.parent / "docs" / "knowledge"

DOCUMENT = """\
# Title

## First section

Some text about the first topic, long enough to stand on its own as a chunk
rather than being folded into a neighbour.

## Second section

Different text about a different topic, also long enough to survive the
minimum-size merge that folds fragments together.
"""


def test_sections_become_separate_chunks() -> None:
    chunks = chunk_markdown(DOCUMENT, "doc")
    assert len(chunks) == 2
    assert chunks[0].heading == "First section"
    assert chunks[1].heading == "Second section"


def test_chunk_index_is_sequential_within_a_document() -> None:
    chunks = chunk_markdown(DOCUMENT, "doc")
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]


def test_heading_only_sections_are_dropped() -> None:
    """The document title has no body and is not retrievable on its own."""
    chunks = chunk_markdown(DOCUMENT, "doc")
    assert all(chunk.heading != "Title" for chunk in chunks)


def test_oversized_section_splits_on_paragraph_boundaries() -> None:
    """A boundary the author chose beats one chosen by character count."""
    paragraph = "word " * 60
    body = f"## Big\n\n{paragraph}\n\n{paragraph}\n\n{paragraph}"
    chunks = chunk_markdown(body, "doc", max_chars=400)
    assert len(chunks) > 1
    assert all(len(chunk.content) <= 400 for chunk in chunks)
    # Splitting on paragraphs means no chunk ends mid-word.
    assert all(not chunk.content.endswith("wo") for chunk in chunks)


def test_paragraph_with_no_boundary_is_hard_split() -> None:
    body = "## Big\n\n" + ("x" * 900)
    chunks = chunk_markdown(body, "doc", max_chars=300, min_chars=0)
    assert len(chunks) == 3
    assert all(len(chunk.content) <= 300 for chunk in chunks)


def test_short_fragments_are_merged_into_the_previous_chunk() -> None:
    """A two-line fragment carries too little context to retrieve on its own."""
    body = "## Long one\n\n" + ("context " * 40) + "\n\n## Tiny\n\nShort.\n"
    chunks = chunk_markdown(body, "doc", min_chars=120)
    assert len(chunks) == 1
    assert "Short." in chunks[0].content


def test_content_hash_is_stable_and_sensitive() -> None:
    """Stability is what lets a rebuild skip unchanged chunks and save quota."""
    first = Chunk("doc", 0, "H", "identical text")
    second = Chunk("doc", 5, "OTHER", "identical text")
    third = Chunk("doc", 0, "H", "different text")
    assert first.content_hash == second.content_hash
    assert first.content_hash != third.content_hash


def test_embedding_text_carries_the_heading() -> None:
    """Without it, a passage like "It is nullable" is unretrievable."""
    chunk = Chunk("doc", 0, "Why ptcar_no is null", "Because the daily feed omits it.")
    assert chunk.embedding_text.startswith("Why ptcar_no is null")
    assert "daily feed" in chunk.embedding_text


def test_embedding_text_without_a_heading_is_just_content() -> None:
    assert Chunk("doc", 0, None, "body").embedding_text == "body"


# --- the real corpus -------------------------------------------------------


def test_real_corpus_chunks_cleanly() -> None:
    chunks = load_corpus(CORPUS_DIR)
    assert len(chunks) >= 15
    assert all(chunk.content.strip() for chunk in chunks)
    assert all(chunk.heading for chunk in chunks)
    assert all(len(chunk.content) <= 1500 for chunk in chunks)


def test_real_corpus_covers_the_questions_it_exists_for() -> None:
    """These are the conceptual questions SQL cannot answer."""
    text = " ".join(chunk.embedding_text for chunk in load_corpus(CORPUS_DIR)).lower()
    for topic in ("ptcar_no", "360 seconds", "concat_ws", "star schema", "read only"):
        assert topic.lower() in text


def test_documents_are_identified_by_their_stem() -> None:
    doc_ids = {chunk.doc_id for chunk in load_corpus(CORPUS_DIR)}
    assert "01-data-source" in doc_ids
    assert all(not doc_id.endswith(".md") for doc_id in doc_ids)


def test_missing_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(RagError, match="not found"):
        load_corpus(tmp_path / "absent")


def test_empty_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(RagError, match="No markdown"):
        load_corpus(tmp_path)
