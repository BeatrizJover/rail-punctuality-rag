"""Tests for the answer pipeline.

Every test runs offline: ``FakeGenerator`` provides canned replies and
``FakeEmbedder`` provides stable vectors.  The database is only needed for the
integration tests that create a schema and execute real SQL.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Collection
from dataclasses import dataclass, replace
from typing import Any

import pytest
from sqlalchemy import Engine

from rail_rag.observability import Span, start_trace
from rail_rag.rag.exceptions import AnswerError
from rail_rag.rag.pipeline import Answer, AnswerPipeline, Source, _parse_sql, _sources_from
from rail_rag.rag.prompts import NO_SQL
from rail_rag.rag.providers.fake import FakeEmbedder, FakeGenerator
from rail_rag.rag.router import Route, classify_lexically
from rail_rag.rag.sql.executor import QueryResult
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
    assert answer.trace is None


def test_the_trace_does_not_affect_answer_equality_or_repr() -> None:
    bare = Answer(text="x", route=Route.CONCEPTUAL)
    with start_trace("answer") as trace:
        pass
    assert replace(bare, trace=trace) == bare
    assert "Trace" not in repr(replace(bare, trace=trace))


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
    min_similarity: float = 0.0,
    explain_plans: bool = False,
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
        min_similarity=min_similarity,
        explain_plans=explain_plans,
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


# --- similarity floor: weak passages never reach the model -------------------

#: Only an exact match clears it; the fake scores unrelated text anywhere in [-1, 1].
_HIGH_FLOOR = 0.99


@pytest.mark.integration
def test_the_floor_reaches_the_data_path(clean_schema: Engine, kb_schema: KbSchema) -> None:
    _populate_kb(clean_schema, kb_schema)
    pipeline, generator = _build(
        clean_schema,
        kb_schema,
        ["```sql\nSELECT 1 AS n\n```", "The answer is 1."],
        min_similarity=_HIGH_FLOOR,
    )
    answer = pipeline.answer(_DATA_Q)

    assert answer.route is Route.DATA
    assert _INTERNAL_BODY not in generator.calls[0][1]
    assert "no relevant documentation" in generator.calls[0][1]
    assert answer.sources == ()


@pytest.mark.integration
def test_the_floor_reaches_the_conceptual_path(clean_schema: Engine, kb_schema: KbSchema) -> None:
    """An unsupported question must be answered as unsupported, not from noise."""
    _populate_kb(clean_schema, kb_schema)
    pipeline, generator = _build(
        clean_schema, kb_schema, ["Not in the documentation."], min_similarity=_HIGH_FLOOR
    )
    answer = pipeline.answer(_CONCEPTUAL_Q)

    assert answer.route is Route.CONCEPTUAL
    assert _INTERNAL_BODY not in generator.calls[0][1]
    assert _EXTERNAL_BODY not in generator.calls[0][1]
    assert "no relevant documentation" in generator.calls[0][1]
    assert answer.sources == ()


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


# --- tracing: every construction and every answer leaves one structured trace --

_BAD_SQL = "```sql\nSELECT * FROM gold.stg_fact_stop_event\n```"
_GOOD_SQL = "```sql\nSELECT 1 AS n\n```"


def _span_names(answer: Answer) -> list[str]:
    assert answer.trace is not None
    return [s.name for s in answer.trace.spans]


def _span(answer: Answer, name: str) -> Span:
    assert answer.trace is not None
    (found,) = [s for s in answer.trace.spans if s.name == name]
    return found


def _emitted(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [json.loads(r.getMessage()) for r in caplog.records if r.name == "rail_rag.trace"]


@pytest.mark.integration
def test_the_data_path_traces_each_stage_in_order(
    clean_schema: Engine, kb_schema: KbSchema
) -> None:
    pipeline, _ = _build(clean_schema, kb_schema, [_GOOD_SQL, "The answer is 1."])
    answer = pipeline.answer(_DATA_Q)

    assert _span_names(answer) == [
        "retrieve",
        "route",
        "sql_generation",
        "guard",
        "sql_checks",
        "execute",
        "narration",
    ]
    assert answer.trace is not None
    assert answer.trace.kind == "answer"
    assert answer.trace.route == "data"
    assert answer.trace.error_class is None
    assert answer.trace.total_ms >= 0
    guard, execute = _span(answer, "guard"), _span(answer, "execute")
    assert guard.attributes == {"rejected": False}
    assert execute.attributes == {"rows": 1, "truncated": False, "sql": answer.sql}
    assert answer.sql is not None
    assert "SELECT 1" in answer.sql


@pytest.mark.integration
def test_the_conceptual_path_traces_without_sql_stages(
    clean_schema: Engine, kb_schema: KbSchema
) -> None:
    pipeline, _ = _build(clean_schema, kb_schema, ["Punctuality is under 6 minutes."])
    answer = pipeline.answer(_CONCEPTUAL_Q)

    assert _span_names(answer) == ["retrieve", "route", "conceptual_answer"]
    assert answer.trace is not None
    assert answer.trace.route == "conceptual"


@pytest.mark.integration
def test_a_declined_query_reports_the_conceptual_fallback_as_its_route(
    clean_schema: Engine, kb_schema: KbSchema
) -> None:
    pipeline, _ = _build(clean_schema, kb_schema, [NO_SQL, "It is a modelling decision."])
    answer = pipeline.answer(_SUPERLATIVE_Q)

    assert _span_names(answer) == ["retrieve", "route", "sql_generation", "conceptual_answer"]
    assert answer.trace is not None
    assert answer.trace.route == "conceptual"


@pytest.mark.integration
def test_the_repair_path_traces_both_guard_attempts(
    clean_schema: Engine, kb_schema: KbSchema
) -> None:
    pipeline, _ = _build(clean_schema, kb_schema, [_BAD_SQL, _GOOD_SQL, "ok"])
    answer = pipeline.answer(_DATA_Q)

    assert _span_names(answer) == [
        "retrieve",
        "route",
        "sql_generation",
        "guard",
        "sql_repair",
        "guard",
        "sql_checks",
        "execute",
        "narration",
    ]
    assert answer.trace is not None
    guards = [s.attributes["rejected"] for s in answer.trace.spans if s.name == "guard"]
    assert guards == [True, False]
    assert answer.trace.error_class is None


@pytest.mark.integration
def test_two_rejections_emit_a_trace_naming_the_error_and_the_guard(
    clean_schema: Engine, kb_schema: KbSchema, caplog: pytest.LogCaptureFixture
) -> None:
    pipeline, _ = _build(clean_schema, kb_schema, [_BAD_SQL, _BAD_SQL])
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="rail_rag.trace"), pytest.raises(AnswerError):
        pipeline.answer(_DATA_Q)

    (record,) = _emitted(caplog)
    assert record["kind"] == "answer"
    assert record["error_class"] == "AnswerError"
    assert record["error_span"] == "guard"
    assert [s["name"] for s in record["spans"]][-3:] == ["guard", "sql_repair", "guard"]


@pytest.mark.integration
def test_an_empty_question_still_emits_a_trace(
    clean_schema: Engine, kb_schema: KbSchema, caplog: pytest.LogCaptureFixture
) -> None:
    pipeline, _ = _build(clean_schema, kb_schema, ["unused"])
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="rail_rag.trace"), pytest.raises(AnswerError):
        pipeline.answer("   ")

    (record,) = _emitted(caplog)
    assert (record["error_class"], record["error_span"]) == ("AnswerError", None)


@pytest.mark.integration
def test_an_empty_table_answer_has_no_narration_span(
    clean_schema: Engine, kb_schema: KbSchema
) -> None:
    pipeline, _ = _build(
        clean_schema, kb_schema, ["```sql\nSELECT * FROM gold.fact_stop_event\n```"]
    )
    answer = pipeline.answer(_DATA_Q)
    assert _span_names(answer)[-1] == "execute"


@pytest.mark.integration
def test_construction_emits_a_startup_trace_with_profile_and_context(
    clean_schema: Engine, kb_schema: KbSchema, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="rail_rag.trace"):
        _build(clean_schema, kb_schema, ["unused"])

    (record,) = _emitted(caplog)
    assert record["kind"] == "startup"
    assert [s["name"] for s in record["spans"]] == ["profile", "partitions", "context", "retriever"]
    assert record["question_hash"] is None


@pytest.mark.integration
def test_the_emitted_answer_trace_never_contains_the_question(
    clean_schema: Engine, kb_schema: KbSchema, caplog: pytest.LogCaptureFixture
) -> None:
    pipeline, _ = _build(clean_schema, kb_schema, [_GOOD_SQL, "The answer is 1."])
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="rail_rag.trace"):
        answer = pipeline.answer(_DATA_Q)

    assert answer.trace is not None
    (line,) = [r.getMessage() for r in caplog.records if r.name == "rail_rag.trace"]
    assert _DATA_Q not in line
    assert json.loads(line)["request_id"] == answer.trace.request_id


def _slow_execute(elapsed_ms: float) -> Any:
    def execute(*_: Any) -> QueryResult:
        return QueryResult(["n"], [(1,)], truncated=False, elapsed_ms=elapsed_ms)

    return execute


@pytest.mark.integration
def test_a_query_at_eighty_percent_of_the_timeout_is_flagged(
    clean_schema: Engine, kb_schema: KbSchema, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, _ = _build(clean_schema, kb_schema, [_GOOD_SQL, "ok"])
    # The default policy allows 5000 ms.
    monkeypatch.setattr("rail_rag.rag.pipeline.execute_safe_query", _slow_execute(4000.0))
    answer = pipeline.answer(_DATA_Q)

    assert answer.trace is not None
    assert _span(answer, "execute").attributes["near_timeout"] is True


@pytest.mark.integration
def test_a_query_under_the_threshold_is_not_flagged(
    clean_schema: Engine, kb_schema: KbSchema, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, _ = _build(clean_schema, kb_schema, [_GOOD_SQL, "ok"])
    monkeypatch.setattr("rail_rag.rag.pipeline.execute_safe_query", _slow_execute(3999.0))
    answer = pipeline.answer(_DATA_Q)

    assert answer.trace is not None
    assert "near_timeout" not in _span(answer, "execute").attributes


# --- SQL quality signals: lint findings and plan estimates --------------------

_VIA_DIMENSION_SQL = (
    "```sql\nSELECT SUM(f.stop_events) AS n FROM gold.fact_stop_event f"
    " JOIN gold.dim_date d ON d.date_key = f.date_key WHERE d.month = 3\n```"
)
_PRUNED_SQL = (
    "```sql\nSELECT SUM(f.stop_events) AS n FROM gold.fact_stop_event f"
    " WHERE f.date_key >= DATE '2025-03-01' AND f.date_key < DATE '2025-04-01'\n```"
)


@pytest.mark.integration
def test_sql_checks_sits_between_the_guard_and_execute(
    clean_schema: Engine, kb_schema: KbSchema
) -> None:
    pipeline, _ = _build(clean_schema, kb_schema, [_PRUNED_SQL, "ok"], explain_plans=True)
    answer = pipeline.answer(_DATA_Q)

    names = _span_names(answer)
    assert names.index("guard") + 1 == names.index("sql_checks")
    assert names.index("sql_checks") + 1 == names.index("execute")
    checks = _span(answer, "sql_checks").attributes
    assert checks["lint"] == []
    assert checks["partitions_scanned"] == 1
    assert checks["partitions_total"] == 36
    assert checks["plan_total_cost"] > 0
    assert checks["plan_rows"] >= 1


@pytest.mark.integration
def test_sql_checks_follows_the_repair_when_there_is_one(
    clean_schema: Engine, kb_schema: KbSchema
) -> None:
    pipeline, _ = _build(clean_schema, kb_schema, [_BAD_SQL, _GOOD_SQL, "ok"])
    names = _span_names(pipeline.answer(_DATA_Q))
    assert names[-4:] == ["guard", "sql_checks", "execute", "narration"]
    assert names[-5] == "sql_repair"


@pytest.mark.integration
def test_a_dimension_filter_is_flagged_and_logged_and_scans_every_partition(
    clean_schema: Engine, kb_schema: KbSchema, caplog: pytest.LogCaptureFixture
) -> None:
    pipeline, _ = _build(clean_schema, kb_schema, [_VIA_DIMENSION_SQL, "ok"], explain_plans=True)
    with caplog.at_level(logging.WARNING, logger="rail_rag.rag.pipeline"):
        answer = pipeline.answer(_DATA_Q)

    checks = _span(answer, "sql_checks").attributes
    assert checks["lint"] == ["fact_without_partition_filter"]
    assert checks["partitions_scanned"] == checks["partitions_total"] == 36
    assert "fact_without_partition_filter" in caplog.text


@pytest.mark.integration
def test_without_explain_only_the_lint_is_recorded(
    clean_schema: Engine, kb_schema: KbSchema
) -> None:
    pipeline, _ = _build(clean_schema, kb_schema, [_VIA_DIMENSION_SQL, "ok"])
    answer = pipeline.answer(_DATA_Q)
    assert _span(answer, "sql_checks").attributes == {"lint": ["fact_without_partition_filter"]}


@pytest.mark.integration
def test_a_failing_plan_check_is_recorded_and_the_answer_completes(
    clean_schema: Engine, kb_schema: KbSchema, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_: Any) -> Any:
        raise RuntimeError("planner exploded")

    monkeypatch.setattr("rail_rag.rag.pipeline.explain_safe_query", broken)
    pipeline, _ = _build(
        clean_schema, kb_schema, [_GOOD_SQL, "The answer is 1."], explain_plans=True
    )
    answer = pipeline.answer(_DATA_Q)

    assert answer.text == "The answer is 1."
    assert answer.trace is not None
    assert answer.trace.error_class is None
    assert _span(answer, "sql_checks").attributes == {"lint": [], "plan_error": "RuntimeError"}
    assert "narration" in _span_names(answer)


@pytest.mark.integration
def test_a_failing_lint_never_fails_the_answer(
    clean_schema: Engine, kb_schema: KbSchema, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_: Any) -> Any:
        raise ValueError("cannot scope this")

    monkeypatch.setattr("rail_rag.rag.pipeline.lint_sql", broken)
    pipeline, _ = _build(clean_schema, kb_schema, [_GOOD_SQL, "The answer is 1."])
    answer = pipeline.answer(_DATA_Q)

    assert answer.text == "The answer is 1."
    assert _span(answer, "sql_checks").attributes == {"lint": [], "lint_error": "ValueError"}


@pytest.mark.integration
def test_the_startup_trace_records_the_partition_count(
    clean_schema: Engine, kb_schema: KbSchema, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="rail_rag.trace"):
        _build(clean_schema, kb_schema, ["unused"])

    (record,) = _emitted(caplog)
    (partitions,) = [s for s in record["spans"] if s["name"] == "partitions"]
    assert partitions["attributes"] == {"count": 36}
