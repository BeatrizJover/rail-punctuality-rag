"""Tests for per-request tracing; a fake clock keeps every duration exact."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor

import pytest

from rail_rag.observability import (
    annotate,
    capture_traces,
    record_provider_call,
    span,
    start_trace,
)
from rail_rag.observability.trace import Trace


class FakeClock:
    """A clock that only moves when the test says so."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _trace_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "rail_rag.trace"]


def test_span_durations_are_measured_with_the_injected_clock() -> None:
    clock = FakeClock()
    with start_trace("answer", clock=clock) as trace:
        with span("retrieve"):
            clock.advance(0.25)
        with span("route"):
            clock.advance(0.5)
        clock.advance(0.125)

    assert [(s.name, s.duration_ms) for s in trace.spans] == [("retrieve", 250.0), ("route", 500.0)]
    assert trace.total_ms == 875.0


def test_attributes_and_provider_calls_land_on_the_innermost_span() -> None:
    with start_trace("answer") as trace:
        with span("execute", table="t"):
            annotate(rows=3, truncated=False)
            record_provider_call(
                kind="generation",
                model="m",
                latency_ms=12.3456,
                attempts=2,
                prompt_tokens=10,
                total_tokens=30,
            )
        annotate(route="data")

    (executed,) = trace.spans
    assert executed.attributes == {"table": "t", "rows": 3, "truncated": False}
    (call,) = executed.provider_calls
    assert (call.kind, call.model, call.attempts) == ("generation", "m", 2)
    assert (call.prompt_tokens, call.output_tokens, call.thought_tokens) == (10, None, None)
    assert call.total_tokens == 30
    assert call.latency_ms == 12.346
    assert trace.route == "data"


def test_the_helpers_do_nothing_outside_a_trace(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="rail_rag.trace"), span("retrieve", a=1):
        annotate(route="data", rows=1)
        record_provider_call(kind="embedding", model="m", latency_ms=1.0, attempts=1)
    assert _trace_lines(caplog) == []


def test_a_provider_call_outside_any_span_is_dropped_not_raised() -> None:
    with start_trace("answer") as trace:
        record_provider_call(kind="embedding", model="m", latency_ms=1.0, attempts=1)
    assert trace.spans == []


def test_an_escaping_exception_is_recorded_reraised_unchanged_and_emitted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    boom = ValueError("boom")
    with (
        caplog.at_level(logging.INFO, logger="rail_rag.trace"),
        pytest.raises(ValueError) as caught,
        start_trace("answer") as trace,
    ):
        with span("retrieve"):
            pass
        with span("execute"):
            raise boom

    assert caught.value is boom
    assert (trace.error_class, trace.error_span) == ("ValueError", "execute")
    (line,) = _trace_lines(caplog)
    record = json.loads(line)
    assert (record["error_class"], record["error_span"]) == ("ValueError", "execute")


def test_the_failing_span_follows_an_exception_translated_outside_it() -> None:
    with pytest.raises(RuntimeError), start_trace("answer") as trace:
        try:
            with span("guard"):
                raise KeyError("inner")
        except KeyError as exc:
            raise RuntimeError("translated") from exc
    assert (trace.error_class, trace.error_span) == ("RuntimeError", "guard")


def test_an_exception_caught_inside_the_trace_is_not_an_error() -> None:
    with start_trace("answer") as trace:
        try:
            with span("guard"):
                raise KeyError("recovered")
        except KeyError:
            pass
    assert trace.error_class is None
    assert trace.error_span is None


def test_an_error_outside_every_span_has_no_failing_span() -> None:
    with pytest.raises(ValueError), start_trace("answer") as trace:
        raise ValueError("early")
    assert (trace.error_class, trace.error_span) == ("ValueError", None)


def test_the_trace_is_emitted_as_one_valid_json_line(caplog: pytest.LogCaptureFixture) -> None:
    with (
        caplog.at_level(logging.INFO, logger="rail_rag.trace"),
        start_trace("answer", question="How many stations?"),
        span("retrieve"),
    ):
        pass

    (line,) = _trace_lines(caplog)
    assert "\n" not in line
    record = json.loads(line)
    assert record["kind"] == "answer"
    assert record["spans"][0]["name"] == "retrieve"
    assert record["started_at"].endswith("+00:00")


def test_the_raw_question_is_never_emitted(caplog: pytest.LogCaptureFixture) -> None:
    question = "Which station had the worst punctuality in secret-marker?"
    with (
        caplog.at_level(logging.INFO, logger="rail_rag.trace"),
        start_trace("answer", question=question) as trace,
        span("retrieve"),
    ):
        pass

    (line,) = _trace_lines(caplog)
    assert question not in line
    assert "secret-marker" not in line
    record = json.loads(line)
    assert len(record["question_hash"]) == 12
    assert record["question_length"] == len(question)
    assert trace.question_hash == record["question_hash"]


def test_a_startup_trace_carries_no_question() -> None:
    with start_trace("startup") as trace:
        pass
    assert (trace.question_hash, trace.question_length) == (None, None)


def test_concurrent_traces_never_share_state() -> None:
    def work(label: str) -> Trace:
        with start_trace("answer") as trace, span(label):
            annotate(label=label)
        return trace

    with ThreadPoolExecutor(max_workers=4) as pool:
        traces = list(pool.map(work, [f"s{i}" for i in range(16)]))

    for index, trace in enumerate(traces):
        assert [s.name for s in trace.spans] == [f"s{index}"]
        assert trace.spans[0].attributes == {"label": f"s{index}"}
    assert len({t.request_id for t in traces}) == len(traces)


def test_a_nested_trace_restores_the_outer_one() -> None:
    with start_trace("answer") as outer:
        with start_trace("startup") as inner, span("profile"):
            pass
        with span("retrieve"):
            pass

    assert [s.name for s in inner.spans] == ["profile"]
    assert [s.name for s in outer.spans] == ["retrieve"]


# --- capture_traces ------------------------------------------------------------------


def test_capture_collects_the_traces_finished_inside_it() -> None:
    with capture_traces() as traces:
        with start_trace("answer", question="one"), span("retrieve"):
            pass
        with start_trace("answer", question="two"):
            pass

    assert [t.question_length for t in traces] == [3, 3]
    assert [s.name for s in traces[0].spans] == ["retrieve"]


def test_capture_also_collects_the_trace_of_a_request_that_raised() -> None:
    with (
        capture_traces() as traces,
        pytest.raises(ValueError),
        start_trace("answer"),
        span("execute"),
    ):
        raise ValueError("boom")

    (trace,) = traces
    assert (trace.error_class, trace.error_span) == ("ValueError", "execute")


def test_traces_outside_a_capture_are_not_collected() -> None:
    with start_trace("answer"):
        pass
    with capture_traces() as traces:
        pass
    assert traces == []


def test_nested_captures_each_see_the_traces_finished_inside_them() -> None:
    with capture_traces() as outer:
        with start_trace("answer"):
            pass
        with capture_traces() as inner, start_trace("answer"):
            pass

    assert len(inner) == 1
    assert len(outer) == 2


def test_a_finished_capture_stops_collecting() -> None:
    with capture_traces() as traces:
        pass
    with start_trace("answer"):
        pass
    assert traces == []
