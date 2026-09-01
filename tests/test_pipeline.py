"""Tests for the answer pipeline.

Every test runs offline: ``FakeGenerator`` provides canned replies and
``FakeEmbedder`` provides stable vectors.  The database is only needed for the
integration tests that create a schema and execute real SQL.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from sqlalchemy import Engine

from rail_rag.rag.exceptions import AnswerError
from rail_rag.rag.pipeline import Answer, AnswerPipeline, Source, _sources_from
from rail_rag.rag.prompts import NO_SQL
from rail_rag.rag.providers.fake import FakeEmbedder, FakeGenerator
from rail_rag.rag.router import Route, classify_lexically
from rail_rag.rag.sql.policy import SqlPolicy
from rail_rag.rag.store.models import KbSchema

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
    engine: Engine, kb: KbSchema, responses: list[str]
) -> tuple[AnswerPipeline, FakeGenerator]:
    """Construct a pipeline against a live schema with a scripted generator."""
    generator = FakeGenerator(responses)
    embedder = FakeEmbedder(dimension=kb.dimension, model_name="fake-embedding")
    pipeline = AnswerPipeline(engine, generator, embedder, kb, SqlPolicy(allowed_tables=_ALLOWED))
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
