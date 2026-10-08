"""Tests for report rendering, which works from a run's files alone."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rail_rag.eval.report import render_report, write_report


def _span(name: str, ms: float = 1.0, **attributes: Any) -> dict[str, Any]:
    return {"name": name, "duration_ms": ms, "attributes": attributes, "provider_calls": []}


def _call(**fields: Any) -> dict[str, Any]:
    return {"kind": "generation", "model": "m", "latency_ms": 1.0, "attempts": 1, **fields}


def _result(
    case_id: str,
    total_ms: float,
    spans: list[dict[str, Any]],
    *,
    outcome: str = "passed",
    route_ok: bool | None = True,
    expected_route: str = "data",
    actual_route: str | None = "data",
    reason: str = "match",
    metrics: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    error_class: str | None = None,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "tags": tags or ["ratio"],
        "expected_route": expected_route,
        "actual_route": actual_route,
        "route_ok": route_ok,
        "outcome": outcome,
        "reason": reason,
        "metrics": metrics or {"retries": 0},
        "trace": {
            "kind": "answer",
            "total_ms": total_ms,
            "error_class": error_class,
            "spans": spans,
        },
    }


def _case(case_id: str, kind: str, question: str, *, limited: bool = False) -> dict[str, Any]:
    return {
        "id": case_id,
        "question": question,
        "tags": [],
        "kind": kind,
        "known_limitation": limited,
        "expected_route": "data",
    }


def _meta(**overrides: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "run_id": "20261008T120000Z-sql",
        "layer": "sql",
        "tags": [],
        "case_ids": ["a", "b", "c", "d", "e", "f", "g", "pending_one"],
        "cases": [
            _case("a", "data", "What was the rate at A?"),
            _case("b", "data", "What was the rate at B?"),
            _case("c", "conceptual", "What is a train path?"),
            _case("d", "conceptual", "Why is ptcar_no null?"),
            _case("e", "data", "Wat was de punctualiteit?", limited=True),
            _case("f", "refused", "Drop everything"),
            _case("g", "data", "What delay threshold makes a train count as punctual?"),
            _case("pending_one", "data", "Not asked yet"),
        ],
        "provenance": {
            "commit": "abc1234",
            "dirty": True,
            "model": "fake-generation",
            "temperature": 0.0,
            "max_output_tokens": 4096,
            "embedding_model": "fake-embedding",
            "system_prompt_sha": "0123456789ab",
            "golden_set_sha": "ba9876543210",
        },
        "data": {
            "first": "2024-01-01",
            "last": "2026-09-07",
            "freshness_days": 31,
            "latest_load": {"table": "fact_stop_event", "finished_at": "2026-09-08T03:00:00+00:00"},
        },
        "startup_trace": {
            "total_ms": 300000.0,
            "spans": [
                _span("profile", 250000.0),
                _span("partitions", 12.0, count=36),
                _span("context", 3.0),
                _span("retriever", 20.0),
            ],
        },
        "status": {"incomplete": True, "stop_reason": "BudgetExhausted at pending_one"},
        "budget": {"used_today": 7, "limit": 15},
    }
    return meta | overrides


def _results() -> list[dict[str, Any]]:
    clean = [
        _span("retrieve", 5.0),
        _span("route", 1.0),
        _span("sql_generation", 2.0, cache="hit"),
        _span("guard", 1.0, rejected=False),
        _span("sql_checks", 3.0, lint=[], partitions_scanned=1, partitions_total=36),
        _span("execute", 100.0),
    ]
    flagged = [
        _span("retrieve", 5.0),
        _span("route", 1.0),
        _span("sql_generation", 2.0, cache="miss"),
        _span("guard", 1.0, rejected=True),
        _span("sql_repair", 2.0),
        _span("guard", 1.0, rejected=False),
        _span(
            "sql_checks",
            3.0,
            lint=["fact_without_partition_filter"],
            partitions_scanned=36,
            partitions_total=36,
        ),
        _span("execute", 300.0, near_timeout=True),
    ]
    gen = [_span("sql_generation", 4.0)]
    gen[0]["provider_calls"] = [_call(prompt_tokens=100, output_tokens=10, thought_tokens=50)]
    return [
        _result("a", 200.0, clean),
        _result(
            "b",
            400.0,
            flagged,
            outcome="failed",
            reason="no column matches",
            metrics={"retries": 2},
        ),
        _result(
            "c",
            100.0,
            [_span("retrieve"), _span("route")],
            expected_route="conceptual",
            actual_route="conceptual",
            tags=["conceptual"],
            metrics={"hit_at_k": True, "reciprocal_rank": 1.0},
        ),
        _result(
            "d",
            300.0,
            [_span("retrieve"), _span("route")],
            outcome="failed",
            expected_route="conceptual",
            actual_route="conceptual",
            tags=["conceptual"],
            metrics={"hit_at_k": False, "reciprocal_rank": 0.0},
        ),
        _result(
            "e",
            500.0,
            gen,
            outcome="failed",
            route_ok=False,
            actual_route="conceptual",
            tags=["multilingual"],
        ),
        _result("f", 600.0, [_span("retrieve"), _span("route")], route_ok=None, tags=["injection"]),
        _result(
            "g",
            700.0,
            [_span("retrieve"), _span("route")],
            outcome="failed",
            route_ok=False,
            actual_route="conceptual",
            reason="no query was executed",
        ),
    ]


def _write(
    tmp_path: Path, results: list[dict[str, Any]], meta: dict[str, Any]
) -> tuple[Path, Path]:
    results_path, meta_path = tmp_path / "run.jsonl", tmp_path / "run.meta.json"
    results_path.write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return results_path, meta_path


def _render(tmp_path: Path, **meta: Any) -> Any:
    return render_report(*_write(tmp_path, _results(), _meta(**meta)))


def test_an_incomplete_run_says_so_with_the_reason_and_the_pending_ids(tmp_path: Path) -> None:
    rendered = _render(tmp_path)

    assert "**INCOMPLETE**" in rendered.markdown
    assert "BudgetExhausted at pending_one" in rendered.markdown
    assert "Pending: pending_one" in rendered.markdown
    assert rendered.data["status"]["pending"] == ["pending_one"]
    assert not rendered.data["status"]["complete"]


def test_a_run_with_no_pending_case_is_complete(tmp_path: Path) -> None:
    meta = _meta(case_ids=["a", "b"], status={"incomplete": False, "stop_reason": None})
    rendered = render_report(*_write(tmp_path, _results()[:2], meta))

    assert "**COMPLETE**" in rendered.markdown
    assert "INCOMPLETE" not in rendered.markdown


def test_headline_numbers_leave_out_known_limitations_and_injections(tmp_path: Path) -> None:
    headline = _render(tmp_path).data["headline"]

    # a, b, c, d, g count; e is a known limitation; f is an injection (no route verdict).
    assert headline["route_accuracy"] == {"value": 0.8, "num": 4, "n": 5}
    assert headline["execution_accuracy"]["n"] == 3
    assert headline["execution_accuracy"]["num"] == 1
    assert headline["hit_at_4"] == {"value": 0.5, "num": 1, "n": 2}
    assert headline["mrr"] == {"value": 0.5, "n": 2}
    assert headline["correct_refusal_rate"] == {"value": 1.0, "num": 1, "n": 1}
    assert headline["excluded_known_limitations"] == 1


def test_guard_repair_lint_partition_and_timeout_figures(tmp_path: Path) -> None:
    headline = _render(tmp_path).data["headline"]

    assert headline["guard_rejection_rate"] == {"value": 0.5, "num": 1, "n": 2}
    assert headline["repair_rate"] == {"value": 0.5, "num": 1, "n": 2}
    assert headline["lint_flag_rate"] == {"value": 0.5, "num": 1, "n": 2}
    assert headline["partitions_scanned_median"] == 18.5
    assert headline["partitions_total"] == 36
    assert headline["near_timeout"] == 1


def test_errors_are_counted_by_class(tmp_path: Path) -> None:
    results = [
        _result(
            "a",
            10.0,
            [_span("route")],
            outcome="error",
            reason="ValueError: x",
            error_class="ValueError",
        ),
        _result("b", 10.0, [_span("route")], outcome="error", reason="AnswerError: y"),
    ]
    meta = _meta(case_ids=["a", "b"], cases=[_case("a", "data", "q"), _case("b", "data", "q")])
    rendered = render_report(*_write(tmp_path, results, meta))
    assert rendered.data["headline"]["errors_by_class"] == {"ValueError": 1, "AnswerError": 1}


def test_percentiles_are_computed_on_known_values(tmp_path: Path) -> None:
    system = _render(tmp_path).data["system"]

    # Totals are 200, 400, 100, 300, 500, 600, 700 ms.
    assert system["end_to_end_ms"]["p50"] == 400.0
    assert system["end_to_end_ms"]["p95"] == 670.0
    # Two execute spans: 100 ms and 300 ms.
    assert system["span_ms"]["execute"]["p50"] == 200.0
    assert system["span_ms"]["execute"]["p95"] == 290.0
    assert system["span_ms"]["execute"]["n"] == 2


def test_system_figures_count_cache_retries_tokens_and_budget(tmp_path: Path) -> None:
    system = _render(tmp_path).data["system"]

    assert (system["cache_hits"], system["cache_misses"]) == (1, 1)
    assert system["retries"] == 2
    assert system["tokens_per_generation_call"] == {
        "calls": 1,
        "prompt": 100.0,
        "output": 10.0,
        "thought": 50.0,
    }
    assert system["budget"] == {"used_today": 7, "limit": 15}


def test_routing_findings_list_every_misrouted_case_with_both_routes(tmp_path: Path) -> None:
    rendered = _render(tmp_path)
    findings = {f["id"]: f for f in rendered.data["routing_findings"]}

    assert set(findings) == {"e", "g"}
    assert findings["g"]["question"] == "What delay threshold makes a train count as punctual?"
    assert (findings["g"]["expected_route"], findings["g"]["actual_route"]) == (
        "data",
        "conceptual",
    )
    assert findings["e"]["known_limitation"] is True
    assert "What delay threshold makes a train count as punctual?" in rendered.markdown


def test_known_limitations_are_listed_separately(tmp_path: Path) -> None:
    rendered = _render(tmp_path)
    assert [item["id"] for item in rendered.data["known_limitations"]] == ["e"]
    assert "## Known limitations" in rendered.markdown


def test_provenance_carries_the_commit_and_the_model(tmp_path: Path) -> None:
    markdown = _render(tmp_path).markdown

    assert "abc1234 (dirty)" in markdown
    assert "fake-generation" in markdown
    assert "0123456789ab" in markdown
    assert "ba9876543210" in markdown


def test_the_data_and_startup_sections_come_from_the_meta(tmp_path: Path) -> None:
    markdown = _render(tmp_path).markdown

    assert "2024-01-01 to 2026-09-07" in markdown
    assert "Pipeline construction took 300000.0 ms." in markdown
    assert "partitions" in markdown
    assert "count=36" in markdown


def test_a_missing_startup_trace_is_reported_not_hidden(tmp_path: Path) -> None:
    markdown = _render(tmp_path, startup_trace=None).markdown
    assert "No startup trace was recorded." in markdown


def test_the_case_table_has_a_row_per_result(tmp_path: Path) -> None:
    rows = _render(tmp_path).data["cases"]
    assert [r["id"] for r in rows] == ["a", "b", "c", "d", "e", "f", "g"]
    b = rows[1]
    assert (b["partitions"], b["lint"], b["cache"], b["execute_ms"]) == (
        "36/36",
        ["fact_without_partition_filter"],
        "miss",
        300.0,
    )


def test_pipes_in_text_do_not_break_the_tables(tmp_path: Path) -> None:
    results = [_result("a", 1.0, [_span("route")], reason="a | b")]
    meta = _meta(case_ids=["a"], cases=[_case("a", "data", "q")])
    markdown = render_report(*_write(tmp_path, results, meta)).markdown
    assert "a \\| b" in markdown


def test_the_report_is_written_as_markdown_and_json(tmp_path: Path) -> None:
    rendered = _render(tmp_path)
    markdown_path, json_path = write_report(rendered, tmp_path / "reports" / "eval", "run-1")

    assert markdown_path.read_text(encoding="utf-8") == rendered.markdown
    assert json.loads(json_path.read_text(encoding="utf-8"))["status"]["complete"] is False


def test_an_id_recorded_twice_counts_once(tmp_path: Path) -> None:
    results = [_result("a", 1.0, [_span("route")], outcome="failed"), *_results()[:1]]
    meta = _meta(case_ids=["a"], cases=[_case("a", "data", "q")])
    rendered = render_report(*_write(tmp_path, results, meta))
    assert rendered.data["status"]["recorded"] == 1
    assert rendered.data["cases"][0]["outcome"] == "passed"


def test_an_empty_fact_table_is_reported_as_such(tmp_path: Path) -> None:
    markdown = _render(
        tmp_path, data={"first": "None", "last": "None", "freshness_days": None}
    ).markdown
    assert "the fact table is empty" in markdown
