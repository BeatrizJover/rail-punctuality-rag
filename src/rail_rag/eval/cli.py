"""The ``eval`` command: choose cases, run them inside the budget, write the run's files and report.

Everything that needs a database or a provider arrives through :class:`EvalBackend`,
so a dry run can be shown never to touch either.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import Engine, select, text
from sqlalchemy.exc import SQLAlchemyError

from rail_rag.core.exceptions import ConfigError
from rail_rag.db.models import RUN_STATUS_SUCCEEDED, load_runs
from rail_rag.eval.budget import BudgetedGenerator, DailyBudget, RateLimiter
from rail_rag.eval.cache import CachingGenerator, GeneratorFingerprint
from rail_rag.eval.cases import GoldenCase, load_golden_set
from rail_rag.eval.config import EvalConfig
from rail_rag.eval.expected import ExpectedResult, Window, cache_key, compute_expected, fact_window
from rail_rag.eval.report import render_report, write_report
from rail_rag.eval.runner import Answerer, completed_ids, estimate_calls, run_cases
from rail_rag.observability import capture_traces
from rail_rag.rag.providers.base import TextGenerator
from rail_rag.rag.providers.config import ModelConfig
from rail_rag.rag.sql.policy import SqlPolicy

logger = logging.getLogger(__name__)

EXIT_OK = 0
#: The run stopped before every selected case was answered; ``--resume`` continues it.
EXIT_INCOMPLETE = 2

DEFAULT_REPORTS_DIR = Path("reports/eval")
_REPO_ROOT = Path(__file__).resolve().parents[3]

Layer = Literal["sql", "e2e"]
DEFAULT_LAYER: Layer = "sql"
GeneratorWrapper = Callable[[TextGenerator, ModelConfig], TextGenerator]
Echo = Callable[[str], None]


@dataclass(frozen=True)
class EvalOptions:
    """What the command line asked for."""

    layer: Layer | None = None
    tags: tuple[str, ...] = ()
    dry_run: bool = False
    resume: str | None = None
    no_cache: bool = False


@dataclass(frozen=True)
class Assembly:
    """A built pipeline and the facts about it that the report records."""

    pipeline: Answerer
    model: ModelConfig
    system_prompt: str


@dataclass(frozen=True)
class EvalBackend:
    """The pieces of a real run, supplied by the composition root."""

    policy: SqlPolicy
    open_engine: Callable[[], Engine]
    build: Callable[[Engine, GeneratorWrapper], Assembly]


@dataclass(frozen=True)
class _Plan:
    run_id: str
    layer: Layer
    tags: tuple[str, ...]
    cases: list[GoldenCase]
    results_path: Path
    meta_path: Path
    prior_meta: dict[str, Any] | None

    @property
    def narrate(self) -> bool:
        return self.layer == "e2e"


def run_eval(
    options: EvalOptions,
    config: EvalConfig,
    backend: EvalBackend,
    *,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    echo: Echo = print,
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
    limiter: RateLimiter | None = None,
) -> int:
    """Run (or, with ``--dry-run``, only plan) an evaluation and return the exit code."""
    plan = _plan(options, config, now())
    budget = DailyBudget(config.daily_generation_budget, config.usage_dir, now=now)
    if options.dry_run:
        _dry_run(plan, config, budget, options, echo)
        return EXIT_OK
    return _execute(plan, options, config, backend, budget, reports_dir, echo, now, limiter)


# --- planning -------------------------------------------------------------------------


def _plan(options: EvalOptions, config: EvalConfig, started: dt.datetime) -> _Plan:
    runs_dir = config.cache_dir / "runs"
    golden = {case.id: case for case in load_golden_set(config.golden_set)}

    if options.resume is not None:
        run_id = options.resume
        results_path, meta_path = runs_dir / f"{run_id}.jsonl", runs_dir / f"{run_id}.meta.json"
        if not meta_path.exists():
            raise ConfigError(f"No run {run_id!r} to resume under {runs_dir}")
        meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
        layer: Layer = meta["layer"]
        tags = tuple(meta["tags"])
        if options.layer not in (None, layer) or (options.tags and options.tags != tags):
            raise ConfigError(
                f"Run {run_id} used layer {layer!r} and tags {list(tags)}; "
                "--layer and --tag cannot change them on resume"
            )
        missing = [cid for cid in meta["case_ids"] if cid not in golden]
        if missing:
            raise ConfigError(f"The golden set no longer contains: {', '.join(missing)}")
        return _Plan(
            run_id,
            layer,
            tags,
            [golden[c] for c in meta["case_ids"]],
            results_path,
            meta_path,
            meta,
        )

    layer = options.layer or DEFAULT_LAYER
    selected = [
        case for case in golden.values() if not options.tags or set(options.tags) & set(case.tags)
    ]
    if not selected:
        raise ConfigError(f"No golden case carries any of the tags: {', '.join(options.tags)}")
    run_id = f"{started:%Y%m%dT%H%M%SZ}-{layer}"
    return _Plan(
        run_id,
        layer,
        options.tags,
        selected,
        runs_dir / f"{run_id}.jsonl",
        runs_dir / f"{run_id}.meta.json",
        None,
    )


def _kind(case: GoldenCase) -> str:
    if case.reference_sql is not None:
        return "data"
    return "conceptual" if case.expected_sources is not None else str(case.expected_behaviour)


def _pending(plan: _Plan) -> list[GoldenCase]:
    done = completed_ids(plan.results_path)
    return [case for case in plan.cases if case.id not in done]


# --- dry run --------------------------------------------------------------------------


def _dry_run(
    plan: _Plan, config: EvalConfig, budget: DailyBudget, options: EvalOptions, echo: Echo
) -> None:
    pending = _pending(plan)
    estimate = estimate_calls(pending, plan.narrate)
    data_cases = [case for case in pending if case.reference_sql is not None]
    worst = estimate + len(data_cases)
    remaining = budget.remaining()
    kinds = ", ".join(f"{n} {k}" for k, n in sorted(_count([_kind(c) for c in pending]).items()))

    echo("Evaluation dry run: no API call, no pipeline, no database connection.")
    echo(f"Run: {plan.run_id}" + (" (resuming)" if options.resume else ""))
    echo(
        f"Layer: {plan.layer} ({'full answers' if plan.narrate else 'SQL only, no narration'});"
        f" tags: {', '.join(plan.tags) or 'all'};"
        f" generation cache: {'on' if config.cache and not options.no_cache else 'off'}"
    )
    echo(f"Cases to run: {len(pending)} of {len(plan.cases)} selected ({kinds or 'none'})")
    for case in pending:
        echo(f"  - {case.id} [{_kind(case)}; {', '.join(case.tags)}]")
    echo(f"Estimated generation calls: {estimate}")
    echo(f"Worst case (estimate + 1 per data case for a SQL repair): {worst}")
    echo(f"Budget remaining today: {remaining} of {budget.limit}")
    if worst <= remaining:
        echo("Verdict: even the worst case fits today's budget.")
    elif estimate <= remaining:
        echo("Verdict: the estimate fits, the worst case may stop early; --resume continues it.")
    else:
        echo("Verdict: the estimate exceeds today's budget; expect the run to stop and resume.")

    window = _recorded_window(config.cache_dir)
    if window is None:
        echo(
            "Reference results: cannot tell which are cached, because no earlier run recorded"
            " the coverage window; they are computed (read-only) at the start of a real run."
        )
        return
    absent = [
        c.id
        for c in data_cases
        if c.reference_sql is None
        or not (config.expected_dir / f"{cache_key(c.reference_sql, window)}.json").exists()
    ]
    echo(
        f"Reference results (coverage {window[0]} to {window[1]} as of the last run):"
        f" {len(data_cases) - len(absent)} cached, {len(absent)} missing"
    )
    for case_id in absent:
        echo(f"  - missing: {case_id}")


def _count(items: Sequence[str]) -> dict[str, int]:
    return {item: items.count(item) for item in set(items)}


# --- running --------------------------------------------------------------------------


def _execute(
    plan: _Plan,
    options: EvalOptions,
    config: EvalConfig,
    backend: EvalBackend,
    budget: DailyBudget,
    reports_dir: Path,
    echo: Echo,
    now: Callable[[], dt.datetime],
    limiter: RateLimiter | None,
) -> int:
    pending = _pending(plan)
    use_cache = config.cache and not options.no_cache
    estimate = estimate_calls(pending, plan.narrate)
    echo(f"Run {plan.run_id}: {len(pending)} case(s), layer {plan.layer}.")
    if estimate > budget.remaining():
        echo(
            f"Warning: estimated {estimate} generation calls against {budget.remaining()}"
            " remaining today; the run will stop early and can be resumed."
        )

    engine = backend.open_engine()
    window = fact_window(engine)
    _record_window(config.cache_dir, window)
    expected = _load_expected(pending, engine, backend.policy, config, window, echo)

    spacing = limiter if limiter is not None else RateLimiter(config.min_interval_s)

    def wrap(generator: TextGenerator, model: ModelConfig) -> TextGenerator:
        guarded = BudgetedGenerator(generator, spacing, budget)
        if not use_cache:
            return guarded
        fingerprint = GeneratorFingerprint(
            model.generation.model, model.generation.temperature, model.generation.max_output_tokens
        )
        return CachingGenerator(guarded, config.generations_dir, fingerprint)

    echo("Building the pipeline...")
    with capture_traces() as built:
        assembly = backend.build(engine, wrap)
    startup = next((t for t in built if t.kind == "startup"), None)

    meta = _initial_meta(plan, options, config, assembly, window, engine, now, startup, budget)
    _write_json(plan.meta_path, meta)

    report = run_cases(
        plan.cases,
        assembly.pipeline,
        expected,
        narrate=plan.narrate,
        budget=budget,
        results_path=plan.results_path,
        skip_ids=completed_ids(plan.results_path),
    )

    meta["updated_at"] = now().isoformat()
    meta["status"] = {"incomplete": report.incomplete, "stop_reason": report.stop_reason}
    meta["budget"] = {"used_today": budget.used(), "limit": budget.limit}
    meta["sessions"].append(
        {
            "at": now().isoformat(),
            "answered": len(report.results),
            "stop_reason": report.stop_reason,
        }
    )
    _write_json(plan.meta_path, meta)

    rendered = render_report(plan.results_path, plan.meta_path)
    markdown_path, json_path = write_report(rendered, reports_dir, plan.run_id)
    complete = rendered.data["status"]["complete"]
    echo(f"{'COMPLETE' if complete else 'INCOMPLETE'}: {len(report.results)} case(s) answered now.")
    if report.stop_reason:
        echo(f"Stopped: {report.stop_reason}")
        echo(f"Continue with: manage.py eval --resume {plan.run_id}")
    echo(f"Report: {markdown_path} and {json_path}")
    return EXIT_OK if complete else EXIT_INCOMPLETE


def _load_expected(
    cases: Sequence[GoldenCase],
    engine: Engine,
    policy: SqlPolicy,
    config: EvalConfig,
    window: Window,
    echo: Echo,
) -> dict[str, ExpectedResult]:
    expected: dict[str, ExpectedResult] = {}
    for case in cases:
        if case.reference_sql is None:
            continue
        result = compute_expected(
            engine, case, policy, cache_dir=config.expected_dir, window=window
        )
        expected[case.id] = result
        origin = "cached" if result.cached else f"computed in {result.elapsed_ms / 1000:.1f} s"
        echo(f"Reference {case.id}: {len(result.rows)} row(s), {origin}")
    return expected


# --- run files ------------------------------------------------------------------------


def _initial_meta(
    plan: _Plan,
    options: EvalOptions,
    config: EvalConfig,
    assembly: Assembly,
    window: Window,
    engine: Engine,
    now: Callable[[], dt.datetime],
    startup: Any,
    budget: DailyBudget,
) -> dict[str, Any]:
    """Meta for this session; a resumed run keeps its identity but gets a fresh startup trace."""
    current = now()
    prior = plan.prior_meta or {}
    model = assembly.model
    commit, dirty = _git_state()
    return {
        "run_id": plan.run_id,
        "layer": plan.layer,
        "tags": list(plan.tags),
        "use_cache": config.cache and not options.no_cache,
        "created_at": prior.get("created_at", current.isoformat()),
        "updated_at": current.isoformat(),
        "case_ids": [case.id for case in plan.cases],
        "cases": [
            {
                "id": case.id,
                "question": case.question,
                "tags": list(case.tags),
                "kind": _kind(case),
                "known_limitation": case.known_limitation,
                "expected_route": case.expected_route,
            }
            for case in plan.cases
        ],
        "provenance": {
            "commit": commit,
            "dirty": dirty,
            "model": model.generation.model,
            "temperature": model.generation.temperature,
            "max_output_tokens": model.generation.max_output_tokens,
            "embedding_model": model.embedding.model,
            "system_prompt_sha": _sha(assembly.system_prompt.encode()),
            "golden_set_sha": _sha(config.golden_set.read_bytes()),
        },
        "data": {
            "first": window[0],
            "last": window[1],
            "freshness_days": _freshness(window[1], current),
            "latest_load": _latest_load(engine),
        },
        "startup_trace": startup.to_dict() if startup is not None else prior.get("startup_trace"),
        "status": {"incomplete": True, "stop_reason": "the run is in progress or was interrupted"},
        "budget": {"used_today": budget.used(), "limit": budget.limit},
        "sessions": list(prior.get("sessions", [])),
    }


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:12]


def _freshness(last: str, today: dt.datetime) -> int | None:
    try:
        return (today.date() - dt.date.fromisoformat(last)).days
    except ValueError:
        return None


def _latest_load(engine: Engine) -> dict[str, str] | None:
    statement = (
        select(load_runs.c.table_name, load_runs.c.finished_at)
        .where(load_runs.c.status == RUN_STATUS_SUCCEEDED)
        .order_by(load_runs.c.finished_at.desc())
        .limit(1)
    )
    try:
        with engine.connect() as conn:
            conn.execute(text("SET TRANSACTION READ ONLY"))
            row = conn.execute(statement).first()
            conn.rollback()
    except SQLAlchemyError:
        return None
    return None if row is None else {"table": row[0], "finished_at": row[1].isoformat()}


def _git_state() -> tuple[str, bool]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()

    try:
        return git("rev-parse", "--short", "HEAD"), bool(git("status", "--porcelain"))
    except (OSError, subprocess.SubprocessError):
        return "unknown", False


def _recorded_window(cache_dir: Path) -> Window | None:
    try:
        raw = json.loads((cache_dir / "window.json").read_text(encoding="utf-8"))
        return (str(raw["first"]), str(raw["last"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _record_window(cache_dir: Path, window: Window) -> None:
    _write_json(cache_dir / "window.json", {"first": window[0], "last": window[1]})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(f".{os.getpid()}.tmp")
    scratch.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    scratch.replace(path)
