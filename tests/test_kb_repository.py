"""Tests for knowledge-base persistence: upsert, invalidation and search."""

from __future__ import annotations

import pytest
from sqlalchemy import Engine, select

from rail_rag.core.exceptions import DatabaseError
from rail_rag.rag.exceptions import RagError
from rail_rag.rag.store.chunking import Chunk
from rail_rag.rag.store.models import KbSchema, build_kb_schema
from rail_rag.rag.store.repository import (
    kb_stats,
    pending_chunks,
    save_embeddings,
    search,
    sync_chunks,
)
from rail_rag.rag.store.schema import drop_kb_schema

pytestmark = pytest.mark.integration

MODEL = "fake-embedding"


def _chunk(doc_id: str, index: int, heading: str | None, content: str) -> Chunk:
    return Chunk(doc_id=doc_id, chunk_index=index, heading=heading, content=content)


def _corpus() -> list[Chunk]:
    return [
        _chunk("data-source", 0, "Two exports", "The daily feed carries no ptcar_no."),
        _chunk("data-source", 1, "Station identity", "Identity is an md5 of the name."),
        _chunk("punctuality", 0, "Threshold", "A train is punctual below 360 seconds."),
    ]


def _unit(kb: KbSchema, index: int) -> list[float]:
    """A basis vector, so cosine distances in the assertions are exactly 0 or 1."""
    vector = [0.0] * kb.dimension
    vector[index] = 1.0
    return vector


def _embed_all(engine: Engine, kb: KbSchema, *, model: str = MODEL) -> None:
    pending = pending_chunks(engine, kb, model_name=model)
    save_embeddings(
        engine, kb, [(row.id, _unit(kb, i)) for i, row in enumerate(pending)], model_name=model
    )


def test_first_sync_inserts_everything(postgres_engine: Engine, kb_schema: KbSchema) -> None:
    report = sync_chunks(postgres_engine, kb_schema, _corpus())
    assert (report.inserted, report.updated, report.unchanged, report.deleted) == (3, 0, 0, 0)
    assert kb_stats(postgres_engine, kb_schema).total == 3


def test_resyncing_an_identical_corpus_touches_nothing(
    postgres_engine: Engine, kb_schema: KbSchema
) -> None:
    """The whole point of the hash: a rebuild must not spend quota for free."""
    sync_chunks(postgres_engine, kb_schema, _corpus())
    _embed_all(postgres_engine, kb_schema)
    report = sync_chunks(postgres_engine, kb_schema, _corpus())
    assert (report.inserted, report.updated, report.unchanged) == (0, 0, 3)
    assert pending_chunks(postgres_engine, kb_schema, model_name=MODEL) == []


def test_changed_content_discards_the_stale_vector(
    postgres_engine: Engine, kb_schema: KbSchema
) -> None:
    sync_chunks(postgres_engine, kb_schema, _corpus())
    _embed_all(postgres_engine, kb_schema)

    edited = _corpus()
    edited[2] = _chunk("punctuality", 0, "Threshold", "A train is punctual below six minutes.")
    report = sync_chunks(postgres_engine, kb_schema, edited)

    assert (report.updated, report.unchanged) == (1, 2)
    pending = pending_chunks(postgres_engine, kb_schema, model_name=MODEL)
    assert [row.chunk.doc_id for row in pending] == ["punctuality"]
    assert kb_stats(postgres_engine, kb_schema).embedded == 2


def test_changed_heading_also_discards_the_vector(
    postgres_engine: Engine, kb_schema: KbSchema
) -> None:
    """The heading is part of the embedded text, so it must be part of the digest."""
    sync_chunks(postgres_engine, kb_schema, _corpus())
    _embed_all(postgres_engine, kb_schema)

    renamed = _corpus()
    renamed[2] = _chunk("punctuality", 0, "Punctuality threshold", renamed[2].content)
    report = sync_chunks(postgres_engine, kb_schema, renamed)

    assert report.updated == 1
    assert [
        row.chunk.doc_id for row in pending_chunks(postgres_engine, kb_schema, model_name=MODEL)
    ] == ["punctuality"]


def test_removed_document_is_deleted(postgres_engine: Engine, kb_schema: KbSchema) -> None:
    sync_chunks(postgres_engine, kb_schema, _corpus())
    report = sync_chunks(postgres_engine, kb_schema, _corpus()[:2])
    assert report.deleted == 1
    assert kb_stats(postgres_engine, kb_schema).total == 2


def test_shortened_document_loses_its_trailing_chunks(
    postgres_engine: Engine, kb_schema: KbSchema
) -> None:
    """A section deleted from a file must not survive as an orphan passage."""
    sync_chunks(postgres_engine, kb_schema, _corpus())
    shorter = [_corpus()[0], _corpus()[2]]
    report = sync_chunks(postgres_engine, kb_schema, shorter)
    assert report.deleted == 1
    stored = pending_chunks(postgres_engine, kb_schema, model_name=MODEL)
    assert [(row.chunk.doc_id, row.chunk.chunk_index) for row in stored] == [
        ("data-source", 0),
        ("punctuality", 0),
    ]


def test_empty_corpus_empties_the_table(postgres_engine: Engine, kb_schema: KbSchema) -> None:
    """No policy here: refusing an empty corpus is ``load_corpus``'s job."""
    sync_chunks(postgres_engine, kb_schema, _corpus())
    assert sync_chunks(postgres_engine, kb_schema, []).deleted == 3
    assert kb_stats(postgres_engine, kb_schema).total == 0


def test_duplicate_positions_are_rejected(postgres_engine: Engine, kb_schema: KbSchema) -> None:
    duplicated = [_chunk("a", 0, None, "one"), _chunk("a", 0, None, "two")]
    with pytest.raises(RagError, match="Duplicate"):
        sync_chunks(postgres_engine, kb_schema, duplicated)


def test_pending_is_ordered_and_carries_the_embedding_text(
    postgres_engine: Engine, kb_schema: KbSchema
) -> None:
    sync_chunks(postgres_engine, kb_schema, _corpus())
    pending = pending_chunks(postgres_engine, kb_schema, model_name=MODEL)
    assert [(row.chunk.doc_id, row.chunk.chunk_index) for row in pending] == [
        ("data-source", 0),
        ("data-source", 1),
        ("punctuality", 0),
    ]
    assert pending[0].chunk.embedding_text.startswith("Two exports")


def test_a_different_model_makes_every_chunk_pending_again(
    postgres_engine: Engine, kb_schema: KbSchema
) -> None:
    """A model swap inside one dimension must self-heal without --force."""
    sync_chunks(postgres_engine, kb_schema, _corpus())
    _embed_all(postgres_engine, kb_schema)
    assert pending_chunks(postgres_engine, kb_schema, model_name=MODEL) == []
    assert len(pending_chunks(postgres_engine, kb_schema, model_name="other-embedding")) == 3


def test_force_ignores_the_hash_and_the_model(postgres_engine: Engine, kb_schema: KbSchema) -> None:
    sync_chunks(postgres_engine, kb_schema, _corpus())
    _embed_all(postgres_engine, kb_schema)
    assert len(pending_chunks(postgres_engine, kb_schema, model_name=MODEL, force=True)) == 3


def test_save_embeddings_records_the_model(postgres_engine: Engine, kb_schema: KbSchema) -> None:
    sync_chunks(postgres_engine, kb_schema, _corpus())
    _embed_all(postgres_engine, kb_schema)
    column = kb_schema.kb_chunk.c.embedding_model
    with postgres_engine.connect() as conn:
        stored = set(conn.execute(select(column)).scalars())
    assert stored == {MODEL}


def test_save_embeddings_rejects_a_wrong_width(
    postgres_engine: Engine, kb_schema: KbSchema
) -> None:
    """Caught before the driver turns it into an opaque type error."""
    sync_chunks(postgres_engine, kb_schema, _corpus())
    row = pending_chunks(postgres_engine, kb_schema, model_name=MODEL)[0]
    with pytest.raises(RagError, match="expects 8"):
        save_embeddings(postgres_engine, kb_schema, [(row.id, [1.0, 0.0])], model_name=MODEL)


def test_search_ranks_by_cosine_distance(postgres_engine: Engine, kb_schema: KbSchema) -> None:
    sync_chunks(postgres_engine, kb_schema, _corpus())
    _embed_all(postgres_engine, kb_schema)
    results = search(postgres_engine, kb_schema, _unit(kb_schema, 2), top_k=3)
    assert [row.doc_id for row in results][0] == "punctuality"
    assert results[0].similarity == pytest.approx(1.0)
    assert results[1].similarity == pytest.approx(0.0)


def test_search_honours_top_k_and_cites_the_source(
    postgres_engine: Engine, kb_schema: KbSchema
) -> None:
    sync_chunks(postgres_engine, kb_schema, _corpus())
    _embed_all(postgres_engine, kb_schema)
    results = search(postgres_engine, kb_schema, _unit(kb_schema, 0), top_k=1)
    assert len(results) == 1
    assert (results[0].doc_id, results[0].heading) == ("data-source", "Two exports")


def test_search_skips_unembedded_rows(postgres_engine: Engine, kb_schema: KbSchema) -> None:
    """A NULL vector has no distance; ranking it would surface unrelated text."""
    sync_chunks(postgres_engine, kb_schema, _corpus())
    row = pending_chunks(postgres_engine, kb_schema, model_name=MODEL)[0]
    save_embeddings(postgres_engine, kb_schema, [(row.id, _unit(kb_schema, 0))], model_name=MODEL)
    results = search(postgres_engine, kb_schema, _unit(kb_schema, 0), top_k=10)
    assert [item.doc_id for item in results] == ["data-source"]


def test_search_rejects_a_wrong_width(postgres_engine: Engine, kb_schema: KbSchema) -> None:
    with pytest.raises(RagError, match="expects 8"):
        search(postgres_engine, kb_schema, [1.0, 0.0], top_k=3)


def test_search_on_a_missing_table_is_a_database_error(postgres_engine: Engine) -> None:
    """No knowledge base at all must not read as an empty one."""
    kb = build_kb_schema(8)
    drop_kb_schema(postgres_engine)
    with pytest.raises(DatabaseError):
        search(postgres_engine, kb, _unit(kb, 0), top_k=1)
