"""Tests for rate limiting, the persistent daily budget, and the budgeted generator."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from rail_rag.eval.budget import BudgetedGenerator, BudgetExhausted, DailyBudget, RateLimiter
from rail_rag.rag.providers.fake import FakeGenerator


class FakeTime:
    """A clock that only moves when something sleeps or the test advances it."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class Instant:
    """A settable wall clock for the budget."""

    def __init__(self, value: dt.datetime) -> None:
        self.value = value

    def __call__(self) -> dt.datetime:
        return self.value


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> dt.datetime:
    return dt.datetime(year, month, day, hour, minute, tzinfo=dt.UTC)


# --- RateLimiter -----------------------------------------------------------------


def test_the_first_call_does_not_wait() -> None:
    time = FakeTime()
    RateLimiter(15, clock=time.clock, sleep=time.sleep).wait()
    assert time.slept == []


def test_a_call_that_follows_too_soon_waits_for_the_rest_of_the_interval() -> None:
    time = FakeTime()
    limiter = RateLimiter(15, clock=time.clock, sleep=time.sleep)
    limiter.wait()
    time.now += 4
    limiter.wait()
    assert time.slept == [11]


def test_a_call_after_the_interval_does_not_wait() -> None:
    time = FakeTime()
    limiter = RateLimiter(15, clock=time.clock, sleep=time.sleep)
    limiter.wait()
    time.now += 20
    limiter.wait()
    assert time.slept == []


def test_the_interval_is_measured_from_the_previous_call_not_the_first() -> None:
    time = FakeTime()
    limiter = RateLimiter(10, clock=time.clock, sleep=time.sleep)
    for _ in range(3):
        limiter.wait()
    assert time.slept == [10, 10]


# --- DailyBudget -----------------------------------------------------------------


def test_the_count_persists_across_instances(tmp_path: Path) -> None:
    now = Instant(_utc(2026, 7, 15, 12))
    DailyBudget(5, tmp_path, now=now).reserve()
    DailyBudget(5, tmp_path, now=now).reserve()

    later = DailyBudget(5, tmp_path, now=now)
    assert later.used() == 2
    assert later.remaining() == 3


def test_reserving_past_the_limit_raises(tmp_path: Path) -> None:
    budget = DailyBudget(2, tmp_path, now=Instant(_utc(2026, 7, 15, 12)))
    budget.reserve()
    budget.reserve()
    with pytest.raises(BudgetExhausted, match="2 requests"):
        budget.reserve()
    assert budget.used() == 2


def test_add_charges_unreserved_requests_and_may_overshoot(tmp_path: Path) -> None:
    budget = DailyBudget(3, tmp_path, now=Instant(_utc(2026, 7, 15, 12)))
    budget.reserve()
    budget.add(4)
    assert budget.used() == 5
    assert budget.remaining() == 0
    budget.add(0)
    assert budget.used() == 5


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param(_utc(2026, 7, 15, 6, 59), _utc(2026, 7, 15, 7, 0), id="summer-time-07:00-utc"),
        pytest.param(
            _utc(2026, 12, 15, 7, 59), _utc(2026, 12, 15, 8, 0), id="winter-time-08:00-utc"
        ),
    ],
)
def test_the_quota_day_rolls_over_at_midnight_in_los_angeles(
    tmp_path: Path, before: dt.datetime, after: dt.datetime
) -> None:
    now = Instant(before)
    budget = DailyBudget(2, tmp_path, now=now)
    budget.reserve()
    budget.reserve()
    assert budget.remaining() == 0

    now.value = after
    assert budget.remaining() == 2
    budget.reserve()
    assert budget.used() == 1

    now.value = before
    assert budget.used() == 2


def test_the_utc_date_does_not_decide_the_quota_day(tmp_path: Path) -> None:
    now = Instant(_utc(2026, 7, 15, 3))
    budget = DailyBudget(1, tmp_path, now=now)
    budget.reserve()
    now.value = _utc(2026, 7, 15, 6)
    assert budget.remaining() == 0


def test_a_corrupt_usage_file_counts_as_unused(tmp_path: Path) -> None:
    budget = DailyBudget(2, tmp_path, now=Instant(_utc(2026, 7, 15, 12)))
    budget.reserve()
    (entry,) = tmp_path.glob("*.json")
    entry.write_text("{not json", encoding="utf-8")
    assert budget.used() == 0


# --- BudgetedGenerator -----------------------------------------------------------


def test_the_budgeted_generator_reserves_waits_and_then_calls(tmp_path: Path) -> None:
    time = FakeTime()
    inner = FakeGenerator(["one", "two"])
    budget = DailyBudget(5, tmp_path, now=Instant(_utc(2026, 7, 15, 12)))
    generator = BudgetedGenerator(
        inner, RateLimiter(15, clock=time.clock, sleep=time.sleep), budget
    )

    assert generator.generate(system="s", prompt="a") == "one"
    assert generator.generate(system="s", prompt="b") == "two"
    assert budget.used() == 2
    assert time.slept == [15]


def test_an_exhausted_budget_raises_before_the_inner_generator_is_called(tmp_path: Path) -> None:
    time = FakeTime()
    inner = FakeGenerator(["one"])
    budget = DailyBudget(1, tmp_path, now=Instant(_utc(2026, 7, 15, 12)))
    generator = BudgetedGenerator(
        inner, RateLimiter(15, clock=time.clock, sleep=time.sleep), budget
    )

    generator.generate(system="s", prompt="a")
    with pytest.raises(BudgetExhausted):
        generator.generate(system="s", prompt="b")

    assert len(inner.calls) == 1
    assert time.slept == []
