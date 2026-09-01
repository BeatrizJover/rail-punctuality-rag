"""Tests for the HTTP layer.

The unit tests serve a stub pipeline, so they exercise routing, status codes and
the wire contract with no database, corpus or API key. The integration tests use
a real engine to check that readiness reports the truth about the schema.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from rail_rag.api import main
from rail_rag.api.main import create_app
from rail_rag.api.state import AppState
from rail_rag.core.exceptions import DatabaseError
from rail_rag.rag.exceptions import AnswerError
from rail_rag.rag.pipeline import Answer, Source
from rail_rag.rag.router import Route
from rail_rag.rag.sql.executor import QueryResult
from rail_rag.rag.store.models import build_kb_schema

_DIMENSION = 8


@dataclass
class StubPipeline:
    """Returns a canned answer, or raises whatever it was given."""

    answer_to_return: Answer | None = None
    error: Exception | None = None
    seen: list[str] | None = None

    def answer(self, question: str) -> Answer:
        if self.seen is not None:
            self.seen.append(question)
        if self.error is not None:
            raise self.error
        assert self.answer_to_return is not None
        return self.answer_to_return


def _conceptual_answer() -> Answer:
    return Answer(
        text="Infrabel considers a train punctual below 360 seconds.",
        route=Route.CONCEPTUAL,
        sources=(Source(doc_id="03-punctuality", heading="The threshold"),),
    )


def _data_answer() -> Answer:
    return Answer(
        text="There are 2 stations.",
        route=Route.DATA,
        sql="SELECT station_name, first_seen, rate FROM gold.dim_station LIMIT 200",
        result=QueryResult(
            columns=("station_name", "first_seen", "rate"),
            rows=(("Brussel-Zuid", dt.date(2024, 5, 1), Decimal("0.91")),),
            truncated=False,
            elapsed_ms=2.3,
        ),
        sources=(Source(doc_id="02-data-model", heading=None),),
    )


def _client(pipeline: StubPipeline, engine: Engine | None = None) -> TestClient:
    state = AppState(
        engine=engine,  # type: ignore[arg-type]
        pipeline=pipeline,
        kb=build_kb_schema(_DIMENSION),
        environment="test",
    )
    return TestClient(create_app(state))


def test_health_reports_the_environment_without_touching_the_database() -> None:
    with _client(StubPipeline(answer_to_return=_conceptual_answer())) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "environment": "test"}


def test_ask_returns_a_conceptual_answer_with_no_sql() -> None:
    with _client(StubPipeline(answer_to_return=_conceptual_answer())) as client:
        response = client.post("/ask", json={"question": "How is punctuality defined?"})
    body = response.json()
    assert response.status_code == 200
    assert body["route"] == "conceptual"
    assert body["sql"] is None
    assert body["result"] is None
    assert body["sources"] == [{"doc_id": "03-punctuality", "heading": "The threshold"}]


def test_ask_serialises_dates_and_decimals_in_the_result() -> None:
    """Postgres types that JSON cannot carry natively must be normalised."""
    with _client(StubPipeline(answer_to_return=_data_answer())) as client:
        response = client.post("/ask", json={"question": "How many stations are there?"})
    body = response.json()
    assert response.status_code == 200
    assert body["route"] == "data"
    assert body["sql"] is not None
    assert body["result"]["rows"] == [["Brussel-Zuid", "2024-05-01", 0.91]]
    assert body["result"]["truncated"] is False


def test_ask_passes_the_question_through_unchanged() -> None:
    seen: list[str] = []
    pipeline = StubPipeline(answer_to_return=_conceptual_answer(), seen=seen)
    with _client(pipeline) as client:
        client.post("/ask", json={"question": "  Why is ptcar_no null?  "})
    assert seen == ["  Why is ptcar_no null?  "]


def test_an_unanswerable_question_is_a_client_error_not_a_server_error() -> None:
    pipeline = StubPipeline(error=AnswerError("The query was rejected twice"))
    with _client(pipeline) as client:
        response = client.post("/ask", json={"question": "drop everything"})
    assert response.status_code == 422
    assert "rejected twice" in response.json()["detail"]


def test_a_dependency_failure_is_reported_as_unavailable() -> None:
    pipeline = StubPipeline(error=DatabaseError("Could not reach the database"))
    with _client(pipeline) as client:
        response = client.post("/ask", json={"question": "How many stations?"})
    assert response.status_code == 503


def test_an_empty_question_is_rejected_before_reaching_the_pipeline() -> None:
    seen: list[str] = []
    pipeline = StubPipeline(answer_to_return=_conceptual_answer(), seen=seen)
    with _client(pipeline) as client:
        response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422
    assert seen == []


def test_an_unknown_field_is_rejected() -> None:
    """``extra="forbid"`` keeps a typo from being silently ignored."""
    with _client(StubPipeline(answer_to_return=_conceptual_answer())) as client:
        response = client.post("/ask", json={"question": "hi", "shwo_sql": True})
    assert response.status_code == 422


@pytest.mark.integration
def test_ready_reports_a_problem_against_a_real_database(postgres_engine: Engine) -> None:
    """The stub schema is 8-wide; a real knowledge base is not."""
    pipeline = StubPipeline(answer_to_return=_conceptual_answer())
    with _client(pipeline, engine=postgres_engine) as client:
        response = client.get("/ready")
    body = response.json()
    assert body["ready"] is False
    assert response.status_code == 503
    assert body["detail"] is not None


def test_ready_names_the_missing_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing Gold table must be reported by name, not as an opaque false."""
    monkeypatch.setattr(main, "missing_tables", lambda _engine: {"dim_station"})
    pipeline = StubPipeline(answer_to_return=_conceptual_answer())
    with _client(pipeline) as client:
        response = client.get("/ready")
    body = response.json()
    assert response.status_code == 503
    assert body["missing_tables"] == ["dim_station"]
    assert "db-init" in (body["detail"] or "")
    assert body["knowledge_base"] is False
