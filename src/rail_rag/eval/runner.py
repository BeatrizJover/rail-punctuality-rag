"""Runs golden cases through the pipeline within a request budget, one result line per case.

The runner never computes ground truth and never spends quota it was not given:
cases are answered sequentially, each result is appended to disk as soon as it
exists, and the run stops cleanly when the budget or the provider gives out, so
a later run can resume where this one stopped.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from rail_rag.eval.budget import BudgetExhausted, DailyBudget
from rail_rag.eval.cases import GoldenCase
from rail_rag.eval.compare import compare_results, hit_at_k, reciprocal_rank
from rail_rag.eval.expected import ExpectedResult
from rail_rag.observability import Trace, capture_traces
from rail_rag.rag.exceptions import AnswerError, ProviderError
from rail_rag.rag.pipeline import Answer
from rail_rag.rag.router import classify_lexically

logger = logging.getLogger(__name__)

#: The pipeline's default ``top_k``, which bounds how many passages a conceptual answer cites.
HIT_K = 4

#: Cases whose wording can defeat the lexical router and force a model call.
_ROUTER_FALLBACK_TAGS = frozenset({"multilingual", "injection"})

Outcome = Literal["passed", "failed", "error"]


class Answerer(Protocol):
    """What the runner needs from the pipeline."""

    def answer(self, question: str, *, narrate: bool = True) -> Answer: ...


@dataclass(frozen=True)
class CaseResult:
    """The verdict on one case, with the evidence it was based on."""

    id: str
    tags: tuple[str, ...]
    expected_route: str
    #: The router's decision, which a later fall-back to the other path does not change.
    actual_route: str | None
    #: ``None`` when the case is not about routing (injection attempts).
    route_ok: bool | None
    outcome: Outcome
    reason: str
    metrics: dict[str, Any]
    trace: dict[str, Any] | None

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), default=str)


@dataclass(frozen=True)
class RunReport:
    """What one invocation of :func:`run_cases` did."""

    results: tuple[CaseResult, ...]
    skipped: int
    incomplete: bool
    stop_reason: str | None


def estimate_calls(cases: Sequence[GoldenCase], narrate: bool) -> int:
    """Upper bound on generation requests, ignoring SQL repairs and provider retries."""
    total = 0
    for case in cases:
        total += 0 if case.expected_sources is not None else 1
        if narrate:
            total += 1
        if _may_call_the_router_model(case):
            total += 1
    return total


def run_cases(
    cases: Sequence[GoldenCase],
    pipeline: Answerer,
    expected: Mapping[str, ExpectedResult],
    *,
    narrate: bool,
    budget: DailyBudget,
    results_path: Path,
    skip_ids: Collection[str] = frozenset(),
) -> RunReport:
    """Answer and judge each case in turn, appending every verdict to ``results_path``."""
    results: list[CaseResult] = []
    skipped = 0
    for case in cases:
        if case.id in skip_ids:
            skipped += 1
            continue
        try:
            result = _run_case(case, pipeline, expected, narrate, budget)
        except (BudgetExhausted, ProviderError) as exc:
            reason = f"{type(exc).__name__} at {case.id}: {exc}"
            logger.warning("evaluation stopped: %s", reason)
            return RunReport(tuple(results), skipped, True, reason)
        _append(results_path, result)
        results.append(result)
    return RunReport(tuple(results), skipped, False, None)


def completed_ids(results_path: Path) -> frozenset[str]:
    """Ids of the cases already recorded in a results file."""
    try:
        lines = results_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return frozenset()
    found: set[str] = set()
    for line in lines:
        try:
            found.add(str(json.loads(line)["id"]))
        except (ValueError, KeyError, TypeError):
            continue
    return frozenset(found)


def _may_call_the_router_model(case: GoldenCase) -> bool:
    return bool(_ROUTER_FALLBACK_TAGS & set(case.tags)) or classify_lexically(case.question) is None


def _run_case(
    case: GoldenCase,
    pipeline: Answerer,
    expected: Mapping[str, ExpectedResult],
    narrate: bool,
    budget: DailyBudget,
) -> CaseResult:
    reference = expected.get(case.id)
    if case.reference_sql is not None and reference is None:
        return _result(case, None, None, "error", "no expected result was provided", {})

    answer: Answer | None = None
    failure: Exception | None = None
    with capture_traces() as traces:
        try:
            answer = pipeline.answer(case.question, narrate=narrate)
        except (BudgetExhausted, ProviderError):
            raise
        except Exception as exc:
            failure = exc
        finally:
            _reconcile(budget, traces)

    trace = traces[-1] if traces else None
    outcome, reason, extra = _judge(case, reference, answer, failure, trace)
    metrics = {**_trace_metrics(trace), **extra}
    if answer is not None:
        metrics["final_route"] = answer.route.value
    return _result(case, trace, answer, outcome, reason, metrics)


def _result(
    case: GoldenCase,
    trace: Trace | None,
    answer: Answer | None,
    outcome: Outcome,
    reason: str,
    metrics: dict[str, Any],
) -> CaseResult:
    actual = _router_decision(trace, answer)
    return CaseResult(
        id=case.id,
        tags=case.tags,
        expected_route=case.expected_route,
        actual_route=actual,
        route_ok=None if "injection" in case.tags else actual == case.expected_route,
        outcome=outcome,
        reason=reason,
        metrics=metrics,
        trace=trace.to_dict() if trace is not None else None,
    )


def _router_decision(trace: Trace | None, answer: Answer | None) -> str | None:
    """Which path the router chose; entering SQL generation means it chose the data path."""
    if trace is None:
        return answer.route.value if answer is not None else None
    names = [span.name for span in trace.spans]
    if "route" not in names:
        return None
    return "data" if "sql_generation" in names else "conceptual"


def _judge(
    case: GoldenCase,
    reference: ExpectedResult | None,
    answer: Answer | None,
    failure: Exception | None,
    trace: Trace | None,
) -> tuple[Outcome, str, dict[str, Any]]:
    if case.expected_behaviour == "refused":
        return _judge_refusal(trace, failure)
    if failure is not None or answer is None:
        return "error", f"{type(failure).__name__}: {failure}", {}
    if case.reference_sql is not None and reference is not None:
        return _judge_data(case, reference, answer)
    if case.expected_sources is not None:
        return _judge_sources(case, answer)
    return _judge_no_data(answer)


def _judge_data(
    case: GoldenCase, reference: ExpectedResult, answer: Answer
) -> tuple[Outcome, str, dict[str, Any]]:
    if answer.result is None or case.compare is None:
        return (
            "failed",
            f"no query was executed (the answer took the {answer.route.value} path)",
            {},
        )
    passed, reason = compare_results(reference.rows, answer.result.rows, case.compare)
    extra = {"reference_rows": len(reference.rows), "actual_rows": len(answer.result.rows)}
    return ("passed" if passed else "failed"), reason, extra


def _judge_sources(case: GoldenCase, answer: Answer) -> tuple[Outcome, str, dict[str, Any]]:
    wanted = {(s.doc_id, s.heading) for s in case.expected_sources or ()}
    returned = [(s.doc_id, s.heading) for s in answer.sources]
    hit = hit_at_k(wanted, returned, HIT_K)
    extra = {
        "hit_at_k": hit,
        "reciprocal_rank": reciprocal_rank(wanted, returned),
        "sources_returned": len(returned),
    }
    if hit:
        return "passed", "an expected passage was retrieved", extra
    return "failed", f"no expected passage among the {len(returned)} retrieved", extra


def _judge_no_data(answer: Answer) -> tuple[Outcome, str, dict[str, Any]]:
    if answer.result is None:
        return "passed", "the model declined to write a query", {}
    cells = [cell for row in answer.result.rows for cell in row]
    value = next((cell for cell in cells if cell is not None), None)
    if value is None:
        return "passed", "the result is empty or all NULL", {}
    return "failed", f"the query returned a value: {value!r}", {}


def _judge_refusal(
    trace: Trace | None, failure: Exception | None
) -> tuple[Outcome, str, dict[str, Any]]:
    """A refusal holds when no SQL reached the database, whichever way it was stopped."""
    if trace is None:
        return "error", "the run produced no trace", {}
    if any(span.name == "execute" for span in trace.spans):
        return "failed", "the SQL was executed", {"refusal": "executed"}
    if failure is not None and not isinstance(failure, AnswerError):
        return "error", f"{type(failure).__name__}: {failure}", {}
    rejected = any(
        span.name == "guard" and span.attributes.get("rejected") is True for span in trace.spans
    )
    kind = "rejected" if rejected else "declined"
    return "passed", f"no SQL was executed ({kind})", {"refusal": kind}


def _trace_metrics(trace: Trace | None) -> dict[str, Any]:
    if trace is None:
        return {}
    calls = [call for span in trace.spans for call in span.provider_calls]
    generations = [call for call in calls if call.kind == "generation"]

    def tokens(name: str) -> int | None:
        values = [v for v in (getattr(call, name) for call in generations) if v is not None]
        return sum(values) if values else None

    checks = next((span.attributes for span in trace.spans if span.name == "sql_checks"), {})
    return {
        "total_ms": trace.total_ms,
        "generation_calls": len(generations),
        "retries": _retries(trace),
        "prompt_tokens": tokens("prompt_tokens"),
        "output_tokens": tokens("output_tokens"),
        "thought_tokens": tokens("thought_tokens"),
        "total_tokens": tokens("total_tokens"),
        "lint": checks.get("lint"),
        "partitions_scanned": checks.get("partitions_scanned"),
    }


def _retries(trace: Trace) -> int:
    return sum(
        max(call.attempts - 1, 0)
        for span in trace.spans
        for call in span.provider_calls
        if call.kind == "generation"
    )


def _reconcile(budget: DailyBudget, traces: Sequence[Trace]) -> None:
    """Charge the retry attempts, which the reservation made per call did not see."""
    budget.add(sum(_retries(trace) for trace in traces))


def _append(path: Path, result: CaseResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(result.to_json() + "\n")
