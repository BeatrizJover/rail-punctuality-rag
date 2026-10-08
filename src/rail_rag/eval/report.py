"""Renders an evaluation run as markdown and JSON, from its files alone.

The inputs are the run's results file and its meta file, never in-memory state, so
a report can be regenerated later, or for a run that stopped halfway, and always
describes what is on disk. Headline numbers leave out ``known_limitation`` cases,
which are listed on their own.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rail_rag.eval.compare import percentile

_REASON_CHARS = 70
Record = dict[str, Any]


@dataclass(frozen=True)
class RenderedReport:
    """The same report in the two forms it is written in."""

    markdown: str
    data: Record


def render_report(results_path: Path, meta_path: Path) -> RenderedReport:
    """Build the report for one run."""
    meta: Record = json.loads(meta_path.read_text(encoding="utf-8"))
    results = _read_results(results_path)
    data = _build(results, meta)
    return RenderedReport(markdown=_markdown(data), data=data)


def write_report(rendered: RenderedReport, reports_dir: Path, run_id: str) -> tuple[Path, Path]:
    """Write ``<run_id>.md`` and ``<run_id>.json`` and return their paths."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = reports_dir / f"{run_id}.md"
    json_path = reports_dir / f"{run_id}.json"
    markdown_path.write_text(rendered.markdown, encoding="utf-8")
    json_path.write_text(json.dumps(rendered.data, indent=2, default=str) + "\n", encoding="utf-8")
    return markdown_path, json_path


def _read_results(path: Path) -> list[Record]:
    """Results in file order, keeping the last line when an id was recorded twice."""
    latest: dict[str, Record] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    for line in lines:
        try:
            record = json.loads(line)
            latest[str(record["id"])] = record
        except (ValueError, KeyError, TypeError):
            continue
    return list(latest.values())


# --- computing the report -------------------------------------------------------------


def _build(results: list[Record], meta: Record) -> Record:
    cases: dict[str, Record] = {str(c["id"]): c for c in meta.get("cases", [])}
    limited = {cid for cid, case in cases.items() if case.get("known_limitation")}
    counted = [r for r in results if r["id"] not in limited]
    done = {r["id"] for r in results}
    pending = [cid for cid in meta.get("case_ids", []) if cid not in done]

    return {
        "run_id": meta.get("run_id"),
        "status": {
            "complete": not pending,
            "stop_reason": meta.get("status", {}).get("stop_reason"),
            "pending": pending,
            "recorded": len(results),
            "selected": len(meta.get("case_ids", [])),
        },
        "provenance": {**meta.get("provenance", {}), "layer": meta.get("layer")}
        | {"tags": meta.get("tags", [])},
        "data": meta.get("data", {}),
        "startup": _startup(meta.get("startup_trace")),
        "headline": _headline(counted, cases) | {"excluded_known_limitations": len(limited & done)},
        "system": _system(results, meta),
        "cases": [_case_row(r) for r in results],
        "known_limitations": [
            {"id": r["id"], "outcome": r["outcome"], "reason": r["reason"]}
            for r in results
            if r["id"] in limited
        ],
        "routing_findings": [
            {
                "id": r["id"],
                "question": cases.get(r["id"], {}).get("question"),
                "expected_route": r["expected_route"],
                "actual_route": r["actual_route"],
                "known_limitation": r["id"] in limited,
            }
            for r in results
            if r.get("route_ok") is False
        ],
    }


def _spans(record: Record) -> list[Record]:
    trace = record.get("trace")
    return list(trace["spans"]) if trace else []


def _span(record: Record, name: str) -> Record | None:
    return next((s for s in _spans(record) if s["name"] == name), None)


def _rate(flags: Sequence[bool]) -> Record:
    n = len(flags)
    num = sum(flags)
    return {"value": num / n if n else None, "num": num, "n": n}


def _median(values: Sequence[float]) -> float | None:
    return percentile(values, 50) if values else None


def _headline(counted: list[Record], cases: dict[str, Record]) -> Record:
    def of_kind(kind: str) -> list[Record]:
        return [r for r in counted if cases.get(r["id"], {}).get("kind") == kind]

    conceptual = of_kind("conceptual")
    guarded = [r for r in counted if _span(r, "guard") is not None]
    checked = [r for r in counted if _span(r, "sql_checks") is not None]
    scanned = [
        a["partitions_scanned"]
        for r in checked
        if isinstance(
            (a := (_span(r, "sql_checks") or {}).get("attributes", {})).get("partitions_scanned"),
            int,
        )
    ]
    totals = [
        a["partitions_total"]
        for r in checked
        if isinstance(
            (a := (_span(r, "sql_checks") or {}).get("attributes", {})).get("partitions_total"), int
        )
    ]
    errors = Counter(
        (r.get("trace") or {}).get("error_class") or str(r["reason"]).split(":", 1)[0]
        for r in counted
        if r["outcome"] == "error"
    )
    return {
        "route_accuracy": _rate([r["route_ok"] for r in counted if r.get("route_ok") is not None]),
        "execution_accuracy": _rate([r["outcome"] == "passed" for r in of_kind("data")]),
        "hit_at_4": _rate([bool(r["metrics"].get("hit_at_k")) for r in conceptual]),
        "mrr": {
            "value": (
                sum(r["metrics"].get("reciprocal_rank", 0.0) for r in conceptual) / len(conceptual)
                if conceptual
                else None
            ),
            "n": len(conceptual),
        },
        "correct_refusal_rate": _rate([r["outcome"] == "passed" for r in of_kind("refused")]),
        "no_data_rate": _rate([r["outcome"] == "passed" for r in of_kind("no_data")]),
        "guard_rejection_rate": _rate(
            [
                any(
                    s["attributes"].get("rejected") is True
                    for s in _spans(r)
                    if s["name"] == "guard"
                )
                for r in guarded
            ]
        ),
        "repair_rate": _rate([_span(r, "sql_repair") is not None for r in guarded]),
        "lint_flag_rate": _rate(
            [
                bool((_span(r, "sql_checks") or {}).get("attributes", {}).get("lint"))
                for r in checked
            ]
        ),
        "partitions_scanned_median": _median(scanned),
        "partitions_total": max(totals) if totals else None,
        "near_timeout": sum(
            1
            for r in counted
            if (_span(r, "execute") or {}).get("attributes", {}).get("near_timeout")
        ),
        "errors_by_class": dict(errors),
    }


def _stats(values: Sequence[float]) -> Record:
    return {
        "n": len(values),
        "p50": percentile(values, 50) if values else None,
        "p95": percentile(values, 95) if values else None,
    }


def _system(results: list[Record], meta: Record) -> Record:
    per_span: dict[str, list[float]] = {}
    for record in results:
        for span in _spans(record):
            per_span.setdefault(span["name"], []).append(span["duration_ms"])
    calls = [
        call
        for record in results
        for span in _spans(record)
        for call in span["provider_calls"]
        if call["kind"] == "generation"
    ]

    def mean(name: str) -> float | None:
        values = [c[name] for c in calls if c.get(name) is not None]
        return sum(values) / len(values) if values else None

    cache = Counter(
        s["attributes"]["cache"] for r in results for s in _spans(r) if "cache" in s["attributes"]
    )
    return {
        "end_to_end_ms": _stats([r["trace"]["total_ms"] for r in results if r.get("trace")]),
        "span_ms": {name: _stats(values) for name, values in per_span.items()},
        "tokens_per_generation_call": {
            "calls": len(calls),
            "prompt": mean("prompt_tokens"),
            "output": mean("output_tokens"),
            "thought": mean("thought_tokens"),
        },
        "generation_calls": len(calls),
        "cache_hits": cache.get("hit", 0),
        "cache_misses": cache.get("miss", 0),
        "retries": sum(int(r["metrics"].get("retries") or 0) for r in results),
        "budget": meta.get("budget", {}),
    }


def _startup(trace: Record | None) -> Record | None:
    if not trace:
        return None
    return {
        "total_ms": trace["total_ms"],
        "spans": [
            {"name": s["name"], "duration_ms": s["duration_ms"], "attributes": s["attributes"]}
            for s in trace["spans"]
        ],
    }


def _case_row(record: Record) -> Record:
    checks = (_span(record, "sql_checks") or {}).get("attributes", {})
    scanned, total = checks.get("partitions_scanned"), checks.get("partitions_total")
    caches = sorted(
        {s["attributes"]["cache"] for s in _spans(record) if "cache" in s["attributes"]}
    )
    return {
        "id": record["id"],
        "tags": record["tags"],
        "route_ok": record.get("route_ok"),
        "outcome": record["outcome"],
        "reason": record["reason"],
        "total_ms": (record.get("trace") or {}).get("total_ms"),
        "execute_ms": (_span(record, "execute") or {}).get("duration_ms"),
        "partitions": None if scanned is None else f"{scanned}/{total}",
        "lint": checks.get("lint") or [],
        "cache": "+".join(caches) or None,
    }


# --- rendering ------------------------------------------------------------------------


def _markdown(data: Record) -> str:
    parts = [f"# Evaluation run {data['run_id']}", ""]
    parts += _status_section(data["status"])
    parts += _provenance_section(data["provenance"])
    parts += _data_section(data["data"])
    parts += _startup_section(data["startup"])
    parts += _headline_section(data["headline"])
    parts += _system_section(data["system"])
    parts += _cases_section(data["cases"])
    parts += _limitations_section(data["known_limitations"])
    parts += _routing_section(data["routing_findings"])
    return "\n".join(parts).rstrip() + "\n"


def _status_section(status: Record) -> list[str]:
    if status["complete"]:
        return [
            "## Status",
            "",
            f"**COMPLETE** — {status['recorded']} of {status['selected']} cases.",
            "",
        ]
    lines = [
        "## Status",
        "",
        f"**INCOMPLETE** — {status['recorded']} of {status['selected']} cases recorded.",
        "",
    ]
    if status["stop_reason"]:
        lines += [f"Stopped: {status['stop_reason']}", ""]
    lines += [f"Pending: {', '.join(status['pending'])}", ""]
    return lines


def _provenance_section(provenance: Record) -> list[str]:
    commit = provenance.get("commit", "unknown") + (" (dirty)" if provenance.get("dirty") else "")
    rows = [
        ("Commit", commit),
        ("Generation model", provenance.get("model")),
        ("Temperature", provenance.get("temperature")),
        ("Max output tokens", provenance.get("max_output_tokens")),
        ("Embedding model", provenance.get("embedding_model")),
        ("SQL system prompt (sha256)", provenance.get("system_prompt_sha")),
        ("Golden set (sha256)", provenance.get("golden_set_sha")),
        ("Layer", provenance.get("layer")),
        ("Tags", ", ".join(provenance.get("tags") or []) or "all"),
    ]
    return ["## Provenance", "", *_table(("Item", "Value"), rows), ""]


def _data_section(data: Record) -> list[str]:
    load = data.get("latest_load")
    rows = [
        (
            "Coverage",
            "the fact table is empty"
            if data.get("first") in (None, "None")
            else f"{data.get('first')} to {data.get('last')}",
        ),
        ("Freshness (days)", data.get("freshness_days")),
        (
            "Latest successful load",
            "unavailable" if not load else f"{load['table']} at {load['finished_at']}",
        ),
    ]
    return ["## Data", "", *_table(("Item", "Value"), rows), ""]


def _startup_section(startup: Record | None) -> list[str]:
    if startup is None:
        return ["## Startup", "", "No startup trace was recorded.", ""]
    rows = [
        (s["name"], f"{s['duration_ms']:.1f}", _attrs(s["attributes"])) for s in startup["spans"]
    ]
    return [
        "## Startup",
        "",
        f"Pipeline construction took {startup['total_ms']:.1f} ms.",
        "",
        *_table(("Span", "ms", "Attributes"), rows),
        "",
    ]


def _pct(rate: Record) -> str:
    if not rate["n"]:
        return "n/a (0 cases)"
    return f"{rate['value'] * 100:.1f}% ({rate['num']}/{rate['n']})"


def _headline_section(h: Record) -> list[str]:
    scanned = h["partitions_scanned_median"]
    rows = [
        ("Route accuracy (injection excluded)", _pct(h["route_accuracy"])),
        ("Execution accuracy (data cases)", _pct(h["execution_accuracy"])),
        ("Hit@4 (conceptual)", _pct(h["hit_at_4"])),
        (
            "MRR (conceptual)",
            "n/a (0 cases)"
            if h["mrr"]["value"] is None
            else f"{h['mrr']['value']:.3f} ({h['mrr']['n']} cases)",
        ),
        ("Correct-refusal rate", _pct(h["correct_refusal_rate"])),
        ("No-data rate", _pct(h["no_data_rate"])),
        ("Guard rejection rate (of queries reaching the guard)", _pct(h["guard_rejection_rate"])),
        ("Repair rate (of queries reaching the guard)", _pct(h["repair_rate"])),
        ("Lint flag rate (of checked queries)", _pct(h["lint_flag_rate"])),
        (
            "Partitions scanned / total (median)",
            "n/a" if scanned is None else f"{scanned:g} / {h['partitions_total']}",
        ),
        ("Near-timeout queries", h["near_timeout"]),
        ("Errors by class", _attrs(h["errors_by_class"]) or "none"),
    ]
    note = h["excluded_known_limitations"]
    lines = ["## Headline metrics", ""]
    if note:
        lines += [f"{note} known-limitation case(s) are excluded from every number below.", ""]
    return [*lines, *_table(("Metric", "Value"), rows), ""]


def _ms(value: float | None) -> str:
    return "–" if value is None else f"{value:.1f}"


def _system_section(system: Record) -> list[str]:
    e2e = system["end_to_end_ms"]
    latency = [("end to end", e2e["n"], _ms(e2e["p50"]), _ms(e2e["p95"]))]
    latency += [
        (name, s["n"], _ms(s["p50"]), _ms(s["p95"])) for name, s in system["span_ms"].items()
    ]
    tokens = system["tokens_per_generation_call"]
    budget = system["budget"]
    rows = [
        ("Generation calls", system["generation_calls"]),
        ("Cache hits / misses", f"{system['cache_hits']} / {system['cache_misses']}"),
        ("Retries", system["retries"]),
        (
            "Mean tokens per generation call (prompt / output / thought)",
            " / ".join(
                "–" if tokens[k] is None else f"{tokens[k]:.0f}"
                for k in ("prompt", "output", "thought")
            ),
        ),
        (
            "Budget used today",
            f"{budget.get('used_today')} of {budget.get('limit')}" if budget else "unknown",
        ),
    ]
    return [
        "## System",
        "",
        "Latency in ms; cache hits make a case look faster than a real call would be.",
        "",
        *_table(("Span", "n", "p50", "p95"), latency),
        "",
        *_table(("Item", "Value"), rows),
        "",
    ]


def _cases_section(rows: Iterable[Record]) -> list[str]:
    table = [
        (
            r["id"],
            ", ".join(r["tags"]),
            {True: "yes", False: "NO", None: "–"}[r["route_ok"]],
            r["outcome"],
            _short(r["reason"]),
            _ms(r["total_ms"]),
            _ms(r["execute_ms"]),
            r["partitions"] or "–",
            ", ".join(r["lint"]) or "–",
            r["cache"] or "–",
        )
        for r in rows
    ]
    headers = (
        "Case",
        "Tags",
        "Route ok",
        "Outcome",
        "Reason",
        "ms",
        "Execute ms",
        "Partitions",
        "Lint",
        "Cache",
    )
    return ["## Cases", "", *_table(headers, table), ""]


def _limitations_section(items: list[Record]) -> list[str]:
    if not items:
        return ["## Known limitations", "", "No known-limitation case in this run.", ""]
    rows = [(i["id"], i["outcome"], _short(i["reason"])) for i in items]
    return ["## Known limitations", "", *_table(("Case", "Outcome", "Reason"), rows), ""]


def _routing_section(items: list[Record]) -> list[str]:
    if not items:
        return ["## Routing findings", "", "Every routed case went down the expected path.", ""]
    rows = [
        (
            i["id"] + (" (known limitation)" if i["known_limitation"] else ""),
            i["question"] or "–",
            i["expected_route"],
            i["actual_route"] or "–",
        )
        for i in items
    ]
    return [
        "## Routing findings",
        "",
        *_table(("Case", "Question", "Expected route", "Actual route"), rows),
        "",
    ]


def _short(text: str) -> str:
    return text if len(text) <= _REASON_CHARS else text[: _REASON_CHARS - 1] + "…"


def _attrs(attributes: Record) -> str:
    return ", ".join(f"{k}={v}" for k, v in attributes.items())


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return lines
