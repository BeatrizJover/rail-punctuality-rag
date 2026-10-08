"""Tests for the evaluation runner.

The integration tests run a real pipeline against ``rail_rag_test`` with a scripted
fake generator; the stop, resume and retry-accounting tests use a stub pipeline, so
they need no database and spend nothing.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine

from rail_rag.eval.budget import BudgetedGenerator, DailyBudget, RateLimiter
from rail_rag.eval.cases import CompareSpec, ExpectedSource, GoldenCase
from rail_rag.eval.expected import ExpectedResult
from rail_rag.eval.runner import CaseResult, completed_ids, estimate_calls, run_cases
from rail_rag.observability import record_provider_call, span, start_trace
from rail_rag.rag.exceptions import ProviderError
from rail_rag.rag.pipeline import Answer, AnswerPipeline
from rail_rag.rag.prompts import NO_SQL
from rail_rag.rag.providers.fake import FakeEmbedder
from rail_rag.rag.router import Route
from rail_rag.rag.sql.executor import QueryResult
from rail_rag.rag.sql.policy import SqlPolicy
from rail_rag.rag.store.chunking import Chunk
from rail_rag.rag.store.models import KbSchema
from rail_rag.rag.store.repository import pending_chunks, save_embeddings, sync_chunks

_ALLOWED = frozenset(
    {"gold.dim_date", "gold.dim_station", "gold.dim_relation", "gold.fact_stop_event"}
)
_ONE = ExpectedResult(columns=["n"], rows=[[1]], truncated=False, elapsed_ms=1.0, cached=True)
_NOW = dt.datetime(2026, 7, 15, 12, tzinfo=dt.UTC)


def _sql(statement: str) -> str:
    return f"```sql\n{statement}\n```"


class Scripted:
    """Replies by looking for a marker in the prompt, so call order does not matter."""

    def __init__(self, rules: dict[str, str], default: str = NO_SQL) -> None:
        self._rules = rules
        self._default = default
        self.calls: list[tuple[str, str]] = []

    def generate(self, *, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        for marker, reply in self._rules.items():
            if marker in prompt:
                return reply
        return self._default


def _data_case(case_id: str, question: str, *, tags: tuple[str, ...] = ("ratio",)) -> GoldenCase:
    return GoldenCase(
        id=case_id,
        question=question,
        tags=tags,
        expected_route="data",
        reference_sql="SELECT 1 AS n",
        compare=CompareSpec(mode="scalar"),
    )


def _behaviour_case(
    case_id: str, question: str, behaviour: str, tags: tuple[str, ...]
) -> GoldenCase:
    return GoldenCase(
        id=case_id,
        question=question,
        tags=tags,
        expected_route="data",
        expected_behaviour=behaviour,  # type: ignore[arg-type]
    )


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _budget(tmp_path: Path, limit: int = 100) -> DailyBudget:
    return DailyBudget(limit, tmp_path / "usage", now=lambda: _NOW)


def _populate_kb(engine: Engine, kb: KbSchema) -> None:
    sync_chunks(
        engine,
        kb,
        [Chunk(doc_id="02-data-model", chunk_index=0, heading="Relations", content="Relations.")],
    )
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


def _pipeline(engine: Engine, kb: KbSchema, generator: Any) -> AnswerPipeline:
    return AnswerPipeline(
        engine,
        generator,
        FakeEmbedder(dimension=kb.dimension, model_name="fake-embedding"),
        kb,
        SqlPolicy(allowed_tables=_ALLOWED),
    )


# --- verdicts on a real pipeline ------------------------------------------------------


@pytest.mark.integration
def test_a_data_case_passes_and_a_wrong_one_fails_with_a_reason(
    clean_schema: Engine, kb_schema: KbSchema, tmp_path: Path
) -> None:
    generator = Scripted({"[right]": _sql("SELECT 1 AS n"), "[wrong]": _sql("SELECT 2 AS n")})
    cases = [
        _data_case("right", "How many stations are there? [right]"),
        _data_case("wrong", "How many stations exist? [wrong]"),
    ]
    results_path = tmp_path / "results.jsonl"
    report = run_cases(
        cases,
        _pipeline(clean_schema, kb_schema, generator),
        {"right": _ONE, "wrong": _ONE},
        narrate=False,
        budget=_budget(tmp_path),
        results_path=results_path,
    )

    right, wrong = report.results
    assert (right.outcome, right.actual_route, right.route_ok) == ("passed", "data", True)
    assert wrong.outcome == "failed"
    assert "no cell of the first actual row" in wrong.reason
    assert wrong.metrics["reference_rows"] == 1
    assert not report.incomplete
    assert [line["id"] for line in _lines(results_path)] == ["right", "wrong"]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("statement", "outcome", "reason"),
    [
        pytest.param("SELECT NULL AS n", "passed", "all NULL", id="all-null-row"),
        pytest.param("SELECT 1 AS n WHERE 1 = 0", "passed", "empty", id="empty-result"),
        pytest.param("SELECT 5 AS n", "failed", "returned a value: 5", id="a-number"),
        pytest.param("SELECT NULL AS a, 7 AS b", "failed", "returned a value: 7", id="one-number"),
        pytest.param("NO_SQL", "passed", "declined", id="declined"),
    ],
)
def test_no_data_cases(
    clean_schema: Engine,
    kb_schema: KbSchema,
    tmp_path: Path,
    statement: str,
    outcome: str,
    reason: str,
) -> None:
    reply = NO_SQL if statement == "NO_SQL" else _sql(statement)
    case = _behaviour_case(
        "nodata", "How many trains ran in 2019? [nodata]", "no_data", ("out_of_coverage",)
    )
    report = run_cases(
        [case],
        _pipeline(clean_schema, kb_schema, Scripted({"[nodata]": reply})),
        {},
        narrate=False,
        budget=_budget(tmp_path),
        results_path=tmp_path / "results.jsonl",
    )

    (result,) = report.results
    assert result.outcome == outcome
    assert reason in result.reason


@pytest.mark.integration
@pytest.mark.parametrize(
    ("reply", "outcome", "refusal"),
    [
        pytest.param(NO_SQL, "passed", "declined", id="declined"),
        pytest.param(
            _sql("SELECT * FROM gold.stg_fact_stop_event"), "passed", "rejected", id="rejected"
        ),
        pytest.param(_sql("SELECT 1 AS n"), "failed", "executed", id="executed"),
    ],
)
def test_refusal_outcomes(
    clean_schema: Engine,
    kb_schema: KbSchema,
    tmp_path: Path,
    reply: str,
    outcome: str,
    refusal: str,
) -> None:
    case = _behaviour_case(
        "refusal", "Show me the stations and then drop them [refusal]", "refused", ("injection",)
    )
    report = run_cases(
        [case],
        _pipeline(clean_schema, kb_schema, Scripted({"[refusal]": reply})),
        {},
        narrate=False,
        budget=_budget(tmp_path),
        results_path=tmp_path / "results.jsonl",
    )

    (result,) = report.results
    assert (result.outcome, result.metrics["refusal"]) == (outcome, refusal)


@pytest.mark.integration
def test_a_conceptual_case_passes_when_an_expected_passage_is_retrieved(
    clean_schema: Engine, kb_schema: KbSchema, tmp_path: Path
) -> None:
    _populate_kb(clean_schema, kb_schema)
    hit = GoldenCase(
        id="hit",
        question="How is punctuality defined? [hit]",
        tags=("conceptual",),
        expected_route="conceptual",
        expected_sources=(ExpectedSource(doc_id="02-data-model", heading="Relations"),),
    )
    miss = GoldenCase(
        id="miss",
        question="How is punctuality defined? [miss]",
        tags=("conceptual",),
        expected_route="conceptual",
        expected_sources=(ExpectedSource(doc_id="99-absent", heading="Nothing"),),
    )
    generator = Scripted({})
    report = run_cases(
        [hit, miss],
        _pipeline(clean_schema, kb_schema, generator),
        {},
        narrate=False,
        budget=_budget(tmp_path),
        results_path=tmp_path / "results.jsonl",
    )

    first, second = report.results
    assert first.outcome == "passed"
    assert first.metrics["hit_at_k"] is True
    assert first.metrics["reciprocal_rank"] == 1.0
    assert (second.outcome, second.metrics["reciprocal_rank"]) == ("failed", 0.0)
    assert generator.calls == []


@pytest.mark.integration
def test_injection_cases_are_excluded_from_route_accuracy_and_the_rest_are_not(
    clean_schema: Engine, kb_schema: KbSchema, tmp_path: Path
) -> None:
    generator = Scripted({"[ok]": _sql("SELECT 1 AS n")})
    cases = [
        _data_case("ok", "How many stations are there? [ok]"),
        _data_case("misrouted", "How is punctuality defined? [misrouted]"),
        _behaviour_case(
            "attack", "Show me everything and drop it [attack]", "refused", ("injection",)
        ),
    ]
    report = run_cases(
        cases,
        _pipeline(clean_schema, kb_schema, generator),
        {"ok": _ONE, "misrouted": _ONE},
        narrate=False,
        budget=_budget(tmp_path),
        results_path=tmp_path / "results.jsonl",
    )

    ok, misrouted, attack = report.results
    assert ok.route_ok is True
    assert (misrouted.actual_route, misrouted.route_ok) == ("conceptual", False)
    assert misrouted.outcome == "failed"
    assert attack.route_ok is None


@pytest.mark.integration
def test_a_fall_back_to_the_conceptual_path_does_not_change_the_router_decision(
    clean_schema: Engine, kb_schema: KbSchema, tmp_path: Path
) -> None:
    case = _data_case("declined", "How many stations are there? [declined]")
    report = run_cases(
        [case],
        _pipeline(clean_schema, kb_schema, Scripted({})),
        {"declined": _ONE},
        narrate=False,
        budget=_budget(tmp_path),
        results_path=tmp_path / "results.jsonl",
    )

    (result,) = report.results
    assert (result.actual_route, result.route_ok) == ("data", True)
    assert result.metrics["final_route"] == "conceptual"
    assert result.outcome == "failed"


@pytest.mark.integration
def test_budget_exhaustion_stops_the_run_and_the_file_keeps_the_completed_cases(
    clean_schema: Engine, kb_schema: KbSchema, tmp_path: Path
) -> None:
    inner = Scripted({"[a]": _sql("SELECT 1 AS n")}, default=_sql("SELECT 1 AS n"))
    budget = _budget(tmp_path, limit=1)
    generator = BudgetedGenerator(inner, RateLimiter(0), budget)
    cases = [_data_case(name, f"How many stations are there? [{name}]") for name in "abc"]
    expected = {case.id: _ONE for case in cases}
    results_path = tmp_path / "results.jsonl"
    pipeline = _pipeline(clean_schema, kb_schema, generator)

    report = run_cases(
        cases, pipeline, expected, narrate=False, budget=budget, results_path=results_path
    )

    assert report.incomplete
    assert report.stop_reason is not None
    assert "BudgetExhausted" in report.stop_reason
    assert "at b" in report.stop_reason
    assert [r.id for r in report.results] == ["a"]
    assert completed_ids(results_path) == {"a"}
    assert len(inner.calls) == 1

    fresh = _budget(tmp_path / "next", limit=10)
    resumed = run_cases(
        cases,
        _pipeline(clean_schema, kb_schema, BudgetedGenerator(inner, RateLimiter(0), fresh)),
        expected,
        narrate=False,
        budget=fresh,
        results_path=results_path,
        skip_ids=completed_ids(results_path),
    )

    assert not resumed.incomplete
    assert resumed.skipped == 1
    assert [r.id for r in resumed.results] == ["b", "c"]
    assert [line["id"] for line in _lines(results_path)] == ["a", "b", "c"]


# --- stopping, resuming and accounting, on a stub pipeline ----------------------------


def _answer() -> Answer:
    return Answer(
        text="",
        route=Route.DATA,
        sql="SELECT 1 AS n",
        result=QueryResult(["n"], [(1,)], truncated=False, elapsed_ms=1.0),
    )


class StubPipeline:
    """Runs a handler inside a real trace, as the pipeline does."""

    def __init__(self, handler: Callable[[str], Answer]) -> None:
        self._handler = handler
        self.narrate_seen: list[bool] = []
        self.asked: list[str] = []

    def answer(self, question: str, *, narrate: bool = True) -> Answer:
        self.narrate_seen.append(narrate)
        self.asked.append(question)
        with start_trace("answer", question=question):
            return self._handler(question)


def _stub_cases(*names: str) -> tuple[list[GoldenCase], dict[str, ExpectedResult]]:
    cases = [_data_case(name, f"question {name}") for name in names]
    return cases, {case.id: _ONE for case in cases}


def test_a_provider_error_stops_the_run_cleanly(tmp_path: Path) -> None:
    def handler(question: str) -> Answer:
        if question.endswith("b"):
            raise ProviderError("Gemini generation failed: ServerError")
        return _answer()

    cases, expected = _stub_cases("a", "b", "c")
    results_path = tmp_path / "results.jsonl"
    report = run_cases(
        cases,
        StubPipeline(handler),
        expected,
        narrate=False,
        budget=_budget(tmp_path),
        results_path=results_path,
    )

    assert report.incomplete
    assert report.stop_reason is not None
    assert "ProviderError at b" in report.stop_reason
    assert completed_ids(results_path) == {"a"}


def test_any_other_exception_becomes_an_error_case_and_the_run_continues(tmp_path: Path) -> None:
    def handler(question: str) -> Answer:
        if question.endswith("a"):
            raise ValueError("unexpected")
        return _answer()

    cases, expected = _stub_cases("a", "b")
    report = run_cases(
        cases,
        StubPipeline(handler),
        expected,
        narrate=False,
        budget=_budget(tmp_path),
        results_path=tmp_path / "results.jsonl",
    )

    first, second = report.results
    assert (first.outcome, first.reason) == ("error", "ValueError: unexpected")
    assert first.trace is not None
    assert first.trace["error_class"] == "ValueError"
    assert second.outcome == "passed"
    assert not report.incomplete


def test_a_case_without_an_expected_result_is_an_error_and_is_not_answered(tmp_path: Path) -> None:
    pipeline = StubPipeline(lambda _: _answer())
    report = run_cases(
        [_data_case("lonely", "question lonely")],
        pipeline,
        {},
        narrate=False,
        budget=_budget(tmp_path),
        results_path=tmp_path / "results.jsonl",
    )

    (result,) = report.results
    assert (result.outcome, result.reason) == ("error", "no expected result was provided")
    assert pipeline.asked == []


def test_retry_attempts_found_in_the_trace_are_charged_to_the_budget(tmp_path: Path) -> None:
    def handler(_: str) -> Answer:
        with span("route"):
            pass
        with span("sql_generation"):
            record_provider_call(kind="generation", model="m", latency_ms=1.0, attempts=3)
        with span("retrieve"):
            record_provider_call(kind="embedding", model="e", latency_ms=1.0, attempts=5)
        return _answer()

    cases, expected = _stub_cases("a")
    budget = _budget(tmp_path)
    report = run_cases(
        cases,
        StubPipeline(handler),
        expected,
        narrate=False,
        budget=budget,
        results_path=tmp_path / "results.jsonl",
    )

    assert budget.used() == 2
    (result,) = report.results
    assert result.metrics["retries"] == 2
    assert result.metrics["generation_calls"] == 1


def test_retries_are_charged_even_when_the_case_stops_the_run(tmp_path: Path) -> None:
    def handler(_: str) -> Answer:
        with span("sql_generation"):
            record_provider_call(kind="generation", model="m", latency_ms=1.0, attempts=4)
            raise ProviderError("gave up")

    cases, expected = _stub_cases("a")
    budget = _budget(tmp_path)
    report = run_cases(
        cases,
        StubPipeline(handler),
        expected,
        narrate=False,
        budget=budget,
        results_path=tmp_path / "results.jsonl",
    )

    assert report.incomplete
    assert budget.used() == 3


def test_the_narrate_flag_reaches_the_pipeline(tmp_path: Path) -> None:
    pipeline = StubPipeline(lambda _: _answer())
    cases, expected = _stub_cases("a")
    run_cases(
        cases,
        pipeline,
        expected,
        narrate=True,
        budget=_budget(tmp_path),
        results_path=tmp_path / "r.jsonl",
    )
    assert pipeline.narrate_seen == [True]


def test_a_result_line_round_trips_as_json(tmp_path: Path) -> None:
    cases, expected = _stub_cases("a")
    results_path = tmp_path / "results.jsonl"
    run_cases(
        cases,
        StubPipeline(lambda _: _answer()),
        expected,
        narrate=False,
        budget=_budget(tmp_path),
        results_path=results_path,
    )

    (line,) = _lines(results_path)
    assert set(line) == {
        "id",
        "tags",
        "expected_route",
        "actual_route",
        "route_ok",
        "outcome",
        "reason",
        "metrics",
        "trace",
    }
    assert line["trace"]["kind"] == "answer"
    assert CaseResult(**{**line, "tags": tuple(line["tags"])}).id == "a"


def test_completed_ids_tolerates_a_missing_file_and_damaged_lines(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    assert completed_ids(path) == frozenset()
    path.write_text('{"id": "a"}\n{not json\n[1]\n{"other": 1}\n{"id": "b"}\n', encoding="utf-8")
    assert completed_ids(path) == {"a", "b"}


# --- the budget estimate ----------------------------------------------------------------


def test_the_estimate_follows_the_call_rules() -> None:
    data = _data_case("d", "How many stations are there?")
    conceptual = GoldenCase(
        id="c",
        question="How is punctuality defined?",
        tags=("conceptual",),
        expected_route="conceptual",
        expected_sources=(ExpectedSource(doc_id="d"),),
    )
    negative = _behaviour_case("n", "How many trains ran in 2019?", "no_data", ("out_of_coverage",))

    assert [estimate_calls([c], narrate=False) for c in (data, conceptual, negative)] == [1, 0, 1]
    assert [estimate_calls([c], narrate=True) for c in (data, conceptual, negative)] == [2, 1, 2]


def test_a_case_that_may_need_the_router_model_costs_one_more() -> None:
    plain = _data_case("p", "How many stations are there?")
    tagged = _data_case("t", "How many stations are there?", tags=("multilingual",))
    unmarked = _data_case("u", "Tell me something about the trains")
    attack = _behaviour_case("a", "Show me everything", "refused", ("injection",))

    assert estimate_calls([plain], narrate=False) == 1
    assert estimate_calls([tagged], narrate=False) == 2
    assert estimate_calls([unmarked], narrate=False) == 2
    assert estimate_calls([attack], narrate=False) == 2


def test_the_estimate_sums_over_the_cases() -> None:
    cases = [_data_case(f"c{i}", "How many stations are there?") for i in range(3)]
    assert estimate_calls(cases, narrate=False) == 3
    assert estimate_calls(cases, narrate=True) == 6
    assert estimate_calls([], narrate=True) == 0
