"""Tests for the ``eval`` command: parsing, the dry run, and complete runs.

The dry-run tests hand the command a backend that fails if anything touches it, which
is how "no pipeline and no connection" is shown rather than assumed.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine

from rail_rag.core.exceptions import ConfigError
from rail_rag.eval.budget import DailyBudget, RateLimiter
from rail_rag.eval.cases import load_golden_set
from rail_rag.eval.cli import (
    EXIT_INCOMPLETE,
    EXIT_OK,
    Assembly,
    EvalBackend,
    EvalOptions,
    GeneratorWrapper,
    run_eval,
)
from rail_rag.eval.config import EvalConfig
from rail_rag.eval.expected import cache_key
from rail_rag.rag.pipeline import AnswerPipeline
from rail_rag.rag.providers.config import load_model_config
from rail_rag.rag.providers.fake import FakeEmbedder
from rail_rag.rag.sql.policy import SqlPolicy
from rail_rag.rag.store.models import KbSchema
from tests.test_eval_runner import Scripted, _populate_kb, _sql

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import manage  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
_GOLDEN = _ROOT / "eval" / "golden_set.yaml"
_NOW = dt.datetime(2026, 10, 8, 12, tzinfo=dt.UTC)
_ALLOWED = frozenset(
    {"gold.dim_date", "gold.dim_station", "gold.dim_relation", "gold.fact_stop_event"}
)


def _config(tmp_path: Path, **fields: Any) -> EvalConfig:
    return EvalConfig(
        golden_set=fields.pop("golden_set", _GOLDEN), cache_dir=tmp_path / "cache", **fields
    )


def _fail(*_: Any) -> Any:
    raise AssertionError("the dry run must not reach for a database or a pipeline")


_UNTOUCHABLE = EvalBackend(
    policy=SqlPolicy(allowed_tables=_ALLOWED), open_engine=_fail, build=_fail
)


def _dry(tmp_path: Path, options: EvalOptions, **config: Any) -> list[str]:
    lines: list[str] = []
    code = run_eval(
        options, _config(tmp_path, **config), _UNTOUCHABLE, echo=lines.append, now=lambda: _NOW
    )
    assert code == EXIT_OK
    return lines


# --- parsing ------------------------------------------------------------------------


def test_the_defaults_are_a_full_selection_on_the_sql_layer() -> None:
    args = manage.build_parser().parse_args(["eval"])
    options = manage._eval_options(args)
    assert options == EvalOptions()


def test_the_flags_are_parsed() -> None:
    args = manage.build_parser().parse_args(
        ["eval", "--layer", "e2e", "--dry-run", "--no-cache", "--resume", "r1", "--profile", "fake"]
    )
    options = manage._eval_options(args)
    assert (options.layer, options.dry_run, options.no_cache, options.resume) == (
        "e2e",
        True,
        True,
        "r1",
    )
    assert args.profile == "fake"


def test_tags_are_repeatable() -> None:
    args = manage.build_parser().parse_args(["eval", "--tag", "ratio", "--tag", "ranking"])
    assert manage._eval_options(args).tags == ("ratio", "ranking")


def test_an_unknown_layer_is_refused() -> None:
    with pytest.raises(SystemExit):
        manage.build_parser().parse_args(["eval", "--layer", "nope"])


def test_the_default_layer_is_sql(tmp_path: Path) -> None:
    lines = _dry(tmp_path, EvalOptions(dry_run=True))
    assert any(line.startswith("Layer: sql (SQL only, no narration)") for line in lines)


def test_resume_requires_an_existing_run(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="No run 'missing'"):
        run_eval(EvalOptions(resume="missing"), _config(tmp_path), _UNTOUCHABLE, now=lambda: _NOW)


def test_a_tag_that_selects_nothing_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="No golden case carries"):
        run_eval(EvalOptions(tags=("nope",)), _config(tmp_path), _UNTOUCHABLE, now=lambda: _NOW)


# --- the dry run ----------------------------------------------------------------------


def test_the_dry_run_shows_the_estimate_the_worst_case_and_the_budget(tmp_path: Path) -> None:
    lines = _dry(tmp_path, EvalOptions(dry_run=True))

    assert "Cases to run: 24 of 24 selected" in "\n".join(lines)
    assert "Estimated generation calls: 26" in lines
    assert "Worst case (estimate + 1 per data case for a SQL repair): 41" in lines
    assert "Budget remaining today: 15 of 15" in lines


def test_the_estimate_grows_with_narration(tmp_path: Path) -> None:
    lines = _dry(tmp_path, EvalOptions(layer="e2e", dry_run=True))
    assert "Estimated generation calls: 50" in lines


def test_the_dry_run_reads_the_budget_already_spent_today(tmp_path: Path) -> None:
    config = _config(tmp_path)
    spent = DailyBudget(15, config.usage_dir, now=lambda: _NOW)
    for _ in range(4):
        spent.reserve()

    lines = _dry(tmp_path, EvalOptions(dry_run=True))
    assert "Budget remaining today: 11 of 15" in lines


def test_the_dry_run_says_when_the_estimate_exceeds_the_budget(tmp_path: Path) -> None:
    lines = _dry(tmp_path, EvalOptions(dry_run=True), daily_generation_budget=10)
    assert any("exceeds today's budget" in line for line in lines)


def test_tags_narrow_the_dry_run(tmp_path: Path) -> None:
    lines = _dry(tmp_path, EvalOptions(dry_run=True, tags=("conceptual",)))
    listed = [line for line in lines if line.startswith("  - concept_")]
    assert len(listed) == 5
    assert not any("ratio_" in line for line in lines)


def test_without_an_earlier_run_the_dry_run_cannot_tell_what_is_cached(tmp_path: Path) -> None:
    lines = _dry(tmp_path, EvalOptions(dry_run=True))
    assert any("cannot tell which are cached" in line for line in lines)


def test_the_dry_run_lists_the_data_cases_with_no_cached_reference(tmp_path: Path) -> None:
    config = _config(tmp_path)
    window = ("2024-01-01", "2026-09-07")
    config.cache_dir.mkdir(parents=True)
    (config.cache_dir / "window.json").write_text(
        json.dumps({"first": window[0], "last": window[1]}), encoding="utf-8"
    )
    data_cases = [c for c in load_golden_set(_GOLDEN) if c.reference_sql]
    config.expected_dir.mkdir(parents=True)
    for case in data_cases[:2]:
        assert case.reference_sql is not None
        (config.expected_dir / f"{cache_key(case.reference_sql, window)}.json").write_text("{}")

    lines = _dry(tmp_path, EvalOptions(dry_run=True))

    assert any("2 cached, 13 missing" in line for line in lines)
    missing = [line.split("missing: ")[1] for line in lines if "  - missing: " in line]
    assert missing == [c.id for c in data_cases[2:]]


def test_the_dry_run_leaves_no_trace_on_disk(tmp_path: Path) -> None:
    _dry(tmp_path, EvalOptions(dry_run=True))
    assert not (tmp_path / "cache").exists()


def test_the_dry_run_does_not_read_the_coverage_or_compute_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("fact_window", "compute_expected", "run_cases"):
        monkeypatch.setattr(f"rail_rag.eval.cli.{name}", _fail)
    _dry(tmp_path, EvalOptions(dry_run=True))


# --- complete runs --------------------------------------------------------------------

_GOLDEN_FIXTURE = """
cases:
  - id: data_ok
    question: How many stations are there? [ok]
    tags: [ratio]
    expected_route: data
    reference_sql: SELECT 1 AS n
    compare: {mode: scalar}

  - id: data_wrong
    question: How many stations exist? [wrong]
    tags: [ratio]
    expected_route: data
    known_limitation: true
    reference_sql: SELECT 1 AS n WHERE true
    compare: {mode: scalar}

  - id: concept_threshold
    question: What delay threshold makes a train count as punctual?
    tags: [conceptual]
    expected_route: conceptual
    expected_sources:
      - {doc_id: 02-data-model, heading: Relations}
"""


@pytest.fixture
def golden(tmp_path: Path) -> Path:
    path = tmp_path / "golden.yaml"
    path.write_text(_GOLDEN_FIXTURE, encoding="utf-8")
    return path


def _backend(engine: Engine, kb: KbSchema, generator: Scripted) -> EvalBackend:
    model = load_model_config(_ROOT / "config" / "model_config.yaml", profile="fake")

    def build(engine: Engine, wrap: GeneratorWrapper) -> Assembly:
        pipeline = AnswerPipeline(
            engine,
            wrap(generator, model),
            FakeEmbedder(dimension=kb.dimension, model_name="fake-embedding"),
            kb,
            SqlPolicy(allowed_tables=_ALLOWED),
        )
        return Assembly(pipeline, model, pipeline.system_prompt)

    return EvalBackend(
        policy=SqlPolicy(allowed_tables=_ALLOWED), open_engine=lambda: engine, build=build
    )


def _run(
    tmp_path: Path, backend: EvalBackend, options: EvalOptions, config: EvalConfig
) -> tuple[int, list[str]]:
    lines: list[str] = []
    code = run_eval(
        options,
        config,
        backend,
        reports_dir=tmp_path / "reports",
        echo=lines.append,
        now=lambda: _NOW,
        limiter=RateLimiter(0),
    )
    return code, lines


_REPLIES = {"[ok]": _sql("SELECT 1 AS n"), "[wrong]": _sql("SELECT 2 AS n")}
_RUN_ID = "20261008T120000Z-sql"


@pytest.mark.integration
def test_a_stopped_run_resumes_to_a_complete_report(
    clean_schema: Engine, kb_schema: KbSchema, tmp_path: Path, golden: Path
) -> None:
    _populate_kb(clean_schema, kb_schema)
    backend = _backend(clean_schema, kb_schema, Scripted(_REPLIES))
    cache = tmp_path / "cache"

    code, lines = _run(
        tmp_path,
        backend,
        EvalOptions(),
        _config(tmp_path, golden_set=golden, daily_generation_budget=1),
    )

    assert code == EXIT_INCOMPLETE
    assert any(line.startswith("INCOMPLETE") for line in lines)
    assert any(f"--resume {_RUN_ID}" in line for line in lines)
    results_path = cache / "runs" / f"{_RUN_ID}.jsonl"
    meta_path = cache / "runs" / f"{_RUN_ID}.meta.json"
    assert len(results_path.read_text(encoding="utf-8").splitlines()) == 1
    stopped = (tmp_path / "reports" / f"{_RUN_ID}.md").read_text(encoding="utf-8")
    assert "**INCOMPLETE**" in stopped
    assert "Pending: data_wrong, concept_threshold" in stopped
    assert "BudgetExhausted" in stopped

    code, lines = _run(
        tmp_path,
        backend,
        EvalOptions(resume=_RUN_ID),
        _config(tmp_path, golden_set=golden, daily_generation_budget=10),
    )

    assert code == EXIT_OK
    assert any(line.startswith("COMPLETE") for line in lines)
    ids = [json.loads(x)["id"] for x in results_path.read_text(encoding="utf-8").splitlines()]
    assert ids == ["data_ok", "data_wrong", "concept_threshold"]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert [s["answered"] for s in meta["sessions"]] == [1, 2]
    assert meta["layer"] == "sql"

    report = (tmp_path / "reports" / f"{_RUN_ID}.md").read_text(encoding="utf-8")
    assert "**COMPLETE** — 3 of 3 cases." in report
    assert "## Startup" in report
    for span in ("profile", "partitions", "context", "retriever"):
        assert span in report
    assert "Pending:" not in report
    document = json.loads((tmp_path / "reports" / f"{_RUN_ID}.json").read_text(encoding="utf-8"))
    assert document["status"]["complete"] is True
    assert document["startup"]["spans"][0]["name"] == "profile"
    assert (cache / "window.json").exists()
    assert len(list((cache / "expected").glob("*.json"))) == 2


@pytest.mark.integration
def test_the_report_excludes_the_known_limitation_and_surfaces_the_misrouting(
    clean_schema: Engine, kb_schema: KbSchema, tmp_path: Path, golden: Path
) -> None:
    _populate_kb(clean_schema, kb_schema)
    backend = _backend(clean_schema, kb_schema, Scripted(_REPLIES))
    code, _ = _run(tmp_path, backend, EvalOptions(), _config(tmp_path, golden_set=golden))

    assert code == EXIT_OK
    data = json.loads((tmp_path / "reports" / f"{_RUN_ID}.json").read_text(encoding="utf-8"))
    headline = data["headline"]
    assert headline["execution_accuracy"] == {"value": 1.0, "num": 1, "n": 1}
    assert headline["excluded_known_limitations"] == 1
    assert [i["id"] for i in data["known_limitations"]] == ["data_wrong"]
    (finding,) = data["routing_findings"]
    assert finding["id"] == "concept_threshold"
    assert (finding["expected_route"], finding["actual_route"]) == ("conceptual", "data")
    assert data["provenance"]["model"] == "fake-generation"
    assert data["provenance"]["commit"]


@pytest.mark.integration
def test_the_e2e_layer_narrates(
    clean_schema: Engine, kb_schema: KbSchema, tmp_path: Path, golden: Path
) -> None:
    _populate_kb(clean_schema, kb_schema)
    backend = _backend(clean_schema, kb_schema, Scripted(_REPLIES))
    code, _ = _run(
        tmp_path,
        backend,
        EvalOptions(layer="e2e", tags=("ratio",)),
        _config(tmp_path, golden_set=golden),
    )

    assert code == EXIT_OK
    run_id = "20261008T120000Z-e2e"
    lines = (
        (tmp_path / "cache" / "runs" / f"{run_id}.jsonl").read_text(encoding="utf-8").splitlines()
    )
    spans = [s["name"] for s in json.loads(lines[0])["trace"]["spans"]]
    assert spans[-1] == "narration"
    assert len(lines) == 2


@pytest.mark.integration
def test_no_cache_bypasses_the_generation_cache(
    clean_schema: Engine, kb_schema: KbSchema, tmp_path: Path, golden: Path
) -> None:
    _populate_kb(clean_schema, kb_schema)
    generator = Scripted(_REPLIES)
    backend = _backend(clean_schema, kb_schema, generator)
    config = _config(tmp_path, golden_set=golden)

    _run(tmp_path, backend, EvalOptions(tags=("ratio",), no_cache=True), config)

    assert not config.generations_dir.exists()
    assert len(generator.calls) == 2


@pytest.mark.integration
def test_the_generation_cache_is_used_by_default_and_a_second_run_costs_nothing(
    clean_schema: Engine, kb_schema: KbSchema, tmp_path: Path, golden: Path
) -> None:
    _populate_kb(clean_schema, kb_schema)
    generator = Scripted(_REPLIES)
    backend = _backend(clean_schema, kb_schema, generator)
    config = _config(tmp_path, golden_set=golden)
    options = EvalOptions(tags=("ratio",))

    _run(tmp_path, backend, options, config)
    spent = len(generator.calls)
    later = _NOW + dt.timedelta(seconds=5)
    lines: list[str] = []
    run_eval(
        options,
        config,
        backend,
        reports_dir=tmp_path / "reports",
        echo=lines.append,
        now=lambda: later,
        limiter=RateLimiter(0),
    )

    assert spent == 2
    assert len(generator.calls) == 2
    assert DailyBudget(15, config.usage_dir, now=lambda: _NOW).used() == 2


@pytest.mark.integration
def test_resuming_with_a_different_layer_is_refused(
    clean_schema: Engine, kb_schema: KbSchema, tmp_path: Path, golden: Path
) -> None:
    _populate_kb(clean_schema, kb_schema)
    backend = _backend(clean_schema, kb_schema, Scripted(_REPLIES))
    config = _config(tmp_path, golden_set=golden, daily_generation_budget=1)
    _run(tmp_path, backend, EvalOptions(), config)

    with pytest.raises(ConfigError, match="cannot change them on resume"):
        _run(tmp_path, backend, EvalOptions(resume=_RUN_ID, layer="e2e"), config)
