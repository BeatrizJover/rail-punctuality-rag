"""Tests for the answer pipeline.

Every test runs offline: ``FakeGenerator`` provides canned replies and
``FakeEmbedder`` provides stable vectors.  The database is only needed for the
integration tests that create a schema and execute real SQL.
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from dataclasses import dataclass

import pytest
from sqlalchemy import Engine

from rail_rag.rag.exceptions import AnswerError
from rail_rag.rag.pipeline import Answer, AnswerPipeline, Source, _parse_sql, _sources_from
from rail_rag.rag.prompts import NO_SQL
from rail_rag.rag.providers.fake import FakeEmbedder, FakeGenerator
from rail_rag.rag.router import Route, classify_lexically
from rail_rag.rag.sql.policy import SqlPolicy
from rail_rag.rag.store.chunking import Chunk
from rail_rag.rag.store.models import KbSchema
from rail_rag.rag.store.repository import pending_chunks, save_embeddings, sync_chunks

_DATA_Q = "How many stations are there?"
_CONCEPTUAL_Q = "How is punctuality defined?"
_SUPERLATIVE_Q = "Which station had the worst punctuality?"


def test_the_fixture_questions_route_without_a_model_call() -> None:
    assert classify_lexically(_DATA_Q) is Route.DATA
    assert classify_lexically(_CONCEPTUAL_Q) is Route.CONCEPTUAL
    assert classify_lexically(_SUPERLATIVE_Q) is Route.DATA


# --- unit tests for helpers (no database) -------------------------------------


@dataclass(frozen=True)
class StubPassage:
    doc_id: str
    heading: str | None
    content: str
    similarity: float = 0.9


def test_sources_preserve_order_and_content() -> None:
    passages = [
        StubPassage("01-data-source", "Why a hash", "explanation"),
        StubPassage("03-punctuality", None, "threshold"),
    ]
    sources = _sources_from(passages)
    assert sources == (
        Source("01-data-source", "Why a hash"),
        Source("03-punctuality", None),
    )


def test_sources_from_empty_passages() -> None:
    assert _sources_from([]) == ()


def test_answer_defaults_are_empty() -> None:
    answer = Answer(text="x", route=Route.CONCEPTUAL)
    assert answer.sql is None
    assert answer.result is None
    assert answer.sources == ()


# --- integration tests: real Postgres, fake LLM -------------------------------


_ALLOWED = frozenset(
    {"gold.dim_date", "gold.dim_station", "gold.dim_relation", "gold.fact_stop_event"}
)


def _build(
    engine: Engine,
    kb: KbSchema,
    responses: list[str],
    *,
    external_ids: Collection[str] = (),
) -> tuple[AnswerPipeline, FakeGenerator]:
    """Construct a pipeline against a live schema with a scripted generator."""
    generator = FakeGenerator(responses)
    embedder = FakeEmbedder(dimension=kb.dimension, model_name="fake-embedding")
    pipeline = AnswerPipeline(
        engine,
        generator,
        embedder,
        kb,
        SqlPolicy(allowed_tables=_ALLOWED),
        external_ids=external_ids,
    )
    return pipeline, generator


@pytest.mark.integration
def test_a_data_question_generates_validates_and_executes_sql(
    clean_schema: Engine, kb_schema: KbSchema
) -> None:
    # dim_station is empty, so the query must not depend on rows existing.
    pipeline, generator = _build(
        clean_schema, kb_schema, ["```sql\nSELECT 1 AS n\n```", "The answer is 1."]
    )
    answer = pipeline.answer(_DATA_Q)
    assert answer.route is Route.DATA
    assert answer.sql is not None
    assert answer.result is not None
    assert answer.text == "The answer is 1."
    # SQL generation then narration: routing was lexical, so no third call.
    assert len(generator.calls) == 2


@pytest.mark.integration
def test_a_conceptual_question_skips_sql(clean_schema: Engine, kb_schema: KbSchema) -> None:
    pipeline, generator = _build(clean_schema, kb_schema, ["Punctuality is under 6 minutes."])
    answer = pipeline.answer(_CONCEPTUAL_Q)
    assert answer.route is Route.CONCEPTUAL
    assert answer.sql is None
    assert answer.result is None
    assert len(generator.calls) == 1


@pytest.mark.integration
def test_the_generator_declining_with_no_sql_falls_back_to_conceptual(
    clean_schema: Engine, kb_schema: KbSchema
) -> None:
    pipeline, generator = _build(clean_schema, kb_schema, [NO_SQL, "It is a modelling decision."])
    answer = pipeline.answer(_SUPERLATIVE_Q)
    assert answer.route is Route.CONCEPTUAL
    assert answer.sql is None
    assert answer.text == "It is a modelling decision."
    # SQL attempt (declined) then the conceptual answer.
    assert len(generator.calls) == 2


@pytest.mark.integration
def test_a_rejected_query_is_retried_once(clean_schema: Engine, kb_schema: KbSchema) -> None:
    bad = "```sql\nSELECT * FROM gold.stg_fact_stop_event\n```"
    good = "```sql\nSELECT 1 AS n\n```"
    pipeline, generator = _build(clean_schema, kb_schema, [bad, good, "ok"])
    answer = pipeline.answer(_DATA_Q)
    assert answer.route is Route.DATA
    assert answer.sql is not None
    assert "stg_fact_stop_event" not in answer.sql
    # First SQL, repair, narration.
    assert len(generator.calls) == 3


@pytest.mark.integration
def test_two_rejections_raise_answer_error(clean_schema: Engine, kb_schema: KbSchema) -> None:
    bad = "```sql\nSELECT * FROM gold.stg_fact_stop_event\n```"
    pipeline, _ = _build(clean_schema, kb_schema, [bad, bad])
    with pytest.raises(AnswerError, match="rejected twice"):
        pipeline.answer(_DATA_Q)


@pytest.mark.integration
def test_an_empty_result_on_an_empty_table_is_stated_not_narrated(
    clean_schema: Engine, kb_schema: KbSchema
) -> None:
    pipeline, generator = _build(
        clean_schema, kb_schema, ["```sql\nSELECT * FROM gold.fact_stop_event\n```"]
    )
    answer = pipeline.answer(_DATA_Q)
    assert answer.route is Route.DATA
    assert answer.result is not None
    assert answer.result.is_empty
    assert "empty" in answer.text.lower()
    # No narration call: the message is deterministic.
    assert len(generator.calls) == 1


@pytest.mark.integration
def test_an_empty_question_raises_answer_error(clean_schema: Engine, kb_schema: KbSchema) -> None:
    pipeline, _ = _build(clean_schema, kb_schema, ["unused"])
    with pytest.raises(AnswerError, match="empty"):
        pipeline.answer("   ")


# --- scoped retrieval: the SQL path sees internal documentation only ----------

_INTERNAL_DOC = "02-data-model"
_EXTERNAL_DOC = "external-glossary"
_INTERNAL_BODY = "relation_direction is modelled on dim_relation."
_EXTERNAL_BODY = "Train path means the infrastructure capacity needed."


def _populate_kb(engine: Engine, kb: KbSchema) -> None:
    """Two embedded documents, one of which the pipeline will be told is external."""
    chunks = [
        Chunk(doc_id=_INTERNAL_DOC, chunk_index=0, heading="Relations", content=_INTERNAL_BODY),
        Chunk(doc_id=_EXTERNAL_DOC, chunk_index=0, heading="Train path", content=_EXTERNAL_BODY),
    ]
    sync_chunks(engine, kb, chunks)
    embedder = FakeEmbedder(dimension=kb.dimension, model_name="fake-embedding")
    save_embeddings(
        engine,
        kb,
        [
            (row.id, embedder.embed_query(row.chunk.embedding_text))
            for row in pending_chunks(engine, kb, model_name="fake-embedding")
        ],
        model_name="fake-embedding",
    )


@pytest.mark.integration
def test_the_data_path_never_sees_an_external_passage(
    clean_schema: Engine, kb_schema: KbSchema
) -> None:
    """A join invented from a glossary the schema does not match is a fabricated answer."""
    _populate_kb(clean_schema, kb_schema)
    pipeline, generator = _build(
        clean_schema,
        kb_schema,
        ["```sql\nSELECT 1 AS n\n```", "The answer is 1."],
        external_ids=[_EXTERNAL_DOC],
    )
    answer = pipeline.answer(_DATA_Q)

    sql_prompt, narration_prompt = generator.calls[0][1], generator.calls[1][1]
    assert _INTERNAL_BODY in sql_prompt
    assert _EXTERNAL_BODY not in sql_prompt
    assert _EXTERNAL_BODY not in narration_prompt
    assert answer.route is Route.DATA
    # Scoping must not buy itself an extra model call.
    assert len(generator.calls) == 2


@pytest.mark.integration
def test_a_data_answer_cites_only_internal_sources(
    clean_schema: Engine, kb_schema: KbSchema
) -> None:
    """Citing a passage the data path never read would make the citation a lie."""
    _populate_kb(clean_schema, kb_schema)
    pipeline, _ = _build(
        clean_schema,
        kb_schema,
        ["```sql\nSELECT 1 AS n\n```", "The answer is 1."],
        external_ids=[_EXTERNAL_DOC],
    )
    answer = pipeline.answer(_DATA_Q)
    assert {source.doc_id for source in answer.sources} == {_INTERNAL_DOC}


@pytest.mark.integration
def test_the_no_sql_fallback_keeps_every_source(clean_schema: Engine, kb_schema: KbSchema) -> None:
    """The router's asymmetry survives scoping: the conceptual path keeps the glossary."""
    _populate_kb(clean_schema, kb_schema)
    pipeline, generator = _build(
        clean_schema,
        kb_schema,
        [NO_SQL, "It is a modelling decision."],
        external_ids=[_EXTERNAL_DOC],
    )
    answer = pipeline.answer(_SUPERLATIVE_Q)

    assert _EXTERNAL_BODY in generator.calls[1][1]
    assert {source.doc_id for source in answer.sources} == {_INTERNAL_DOC, _EXTERNAL_DOC}


# --- the output contract: a broken reply must leave evidence -------------------


@pytest.mark.integration
def test_a_reply_that_breaks_the_contract_is_logged(
    clean_schema: Engine, kb_schema: KbSchema, caplog: pytest.LogCaptureFixture
) -> None:
    """Diagnosing a broken output contract must not cost a second call to the provider."""
    pipeline, _ = _build(clean_schema, kb_schema, ["I would rather explain the schema instead."])
    with caplog.at_level(logging.INFO, logger="rail_rag.rag.pipeline"), pytest.raises(AnswerError):
        pipeline.answer(_DATA_Q)
    assert "rather explain the schema" in caplog.text


def test_a_long_rejected_reply_is_truncated_in_the_log(caplog: pytest.LogCaptureFixture) -> None:
    """A model that answers with an essay must stay readable in a terminal."""
    reply = "Not a query. " * 200
    with caplog.at_level(logging.INFO, logger="rail_rag.rag.pipeline"), pytest.raises(AnswerError):
        _parse_sql(reply)
    assert "chars in total" in caplog.text
    assert len(caplog.text) < len(reply)


def test_a_valid_reply_is_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    """The log line is evidence of a failure; on the happy path it would be noise."""
    with caplog.at_level(logging.INFO, logger="rail_rag.rag.pipeline"):
        assert _parse_sql("```sql\nSELECT 1\n```") == "SELECT 1"
    assert caplog.text == ""
