"""Tests for the generation cache and its place in front of the budget."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from rail_rag.eval.budget import BudgetedGenerator, DailyBudget, RateLimiter
from rail_rag.eval.cache import CachingGenerator, GeneratorFingerprint
from rail_rag.observability import span, start_trace
from rail_rag.rag.providers.fake import FakeGenerator

_FINGERPRINT = GeneratorFingerprint(model="m", temperature=0.0, max_output_tokens=1024)


def _entries(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.json"))


def test_a_hit_replays_the_reply_without_calling_the_inner_generator(tmp_path: Path) -> None:
    inner = FakeGenerator(["first", "second"])
    generator = CachingGenerator(inner, tmp_path, _FINGERPRINT)

    assert generator.generate(system="s", prompt="p") == "first"
    assert generator.generate(system="s", prompt="p") == "first"
    assert len(inner.calls) == 1


def test_the_cache_survives_a_new_instance(tmp_path: Path) -> None:
    CachingGenerator(FakeGenerator(["stored"]), tmp_path, _FINGERPRINT).generate(
        system="s", prompt="p"
    )
    inner = FakeGenerator(["unused"])
    assert (
        CachingGenerator(inner, tmp_path, _FINGERPRINT).generate(system="s", prompt="p") == "stored"
    )
    assert inner.calls == []


def test_a_hit_costs_no_budget_and_no_waiting(tmp_path: Path) -> None:
    waited: list[float] = []
    budget = DailyBudget(
        5, tmp_path / "usage", now=lambda: dt.datetime(2026, 7, 15, 12, tzinfo=dt.UTC)
    )
    limiter = RateLimiter(15, clock=lambda: 0.0, sleep=waited.append)
    generator = CachingGenerator(
        BudgetedGenerator(FakeGenerator(["reply"]), limiter, budget),
        tmp_path / "generations",
        _FINGERPRINT,
    )

    generator.generate(system="s", prompt="p")
    for _ in range(3):
        generator.generate(system="s", prompt="p")

    assert budget.used() == 1
    assert waited == []


def test_each_part_of_the_key_changes_it(tmp_path: Path) -> None:
    inner = FakeGenerator(["a", "b", "c", "d", "e"])
    base = CachingGenerator(inner, tmp_path, _FINGERPRINT)
    base.generate(system="s", prompt="p")
    base.generate(system="other", prompt="p")
    base.generate(system="s", prompt="other")
    for changed in (
        GeneratorFingerprint(model="m2", temperature=0.0, max_output_tokens=1024),
        GeneratorFingerprint(model="m", temperature=0.5, max_output_tokens=1024),
        GeneratorFingerprint(model="m", temperature=0.0, max_output_tokens=2048),
    ):
        CachingGenerator(inner, tmp_path, changed).generate(system="s", prompt="p")

    assert len(inner.calls) == 6
    assert len(_entries(tmp_path)) == 6


def test_the_system_prompt_and_the_prompt_cannot_be_confused(tmp_path: Path) -> None:
    inner = FakeGenerator(["a", "b"])
    generator = CachingGenerator(inner, tmp_path, _FINGERPRINT)
    generator.generate(system="ab", prompt="c")
    generator.generate(system="a", prompt="bc")
    assert len(inner.calls) == 2


def test_the_span_records_a_miss_then_a_hit(tmp_path: Path) -> None:
    generator = CachingGenerator(FakeGenerator(["reply"]), tmp_path, _FINGERPRINT)
    with start_trace("answer") as trace:
        with span("first"):
            generator.generate(system="s", prompt="p")
        with span("second"):
            generator.generate(system="s", prompt="p")

    assert [s.attributes["cache"] for s in trace.spans] == ["miss", "hit"]


def test_outside_a_trace_the_cache_still_works(tmp_path: Path) -> None:
    generator = CachingGenerator(FakeGenerator(["reply"]), tmp_path, _FINGERPRINT)
    assert generator.generate(system="s", prompt="p") == "reply"


def test_a_corrupt_entry_is_replaced(tmp_path: Path) -> None:
    CachingGenerator(FakeGenerator(["first"]), tmp_path, _FINGERPRINT).generate(
        system="s", prompt="p"
    )
    (entry,) = _entries(tmp_path)
    entry.write_text("{not json", encoding="utf-8")

    inner = FakeGenerator(["second"])
    assert (
        CachingGenerator(inner, tmp_path, _FINGERPRINT).generate(system="s", prompt="p") == "second"
    )
    assert len(inner.calls) == 1
