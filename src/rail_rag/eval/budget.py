"""Keeping an evaluation run inside the provider's request quota.

The quota day follows the provider's reset at midnight Pacific time, so the
counter is keyed by that date rather than by the local one.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from zoneinfo import ZoneInfo

from rail_rag.core.exceptions import RailRagError
from rail_rag.rag.providers.base import TextGenerator

QUOTA_TIMEZONE = ZoneInfo("America/Los_Angeles")


class BudgetExhausted(RailRagError):
    """Raised before a call that would exceed the day's request allowance."""


class RateLimiter:
    """Spaces calls at least ``min_interval_s`` apart."""

    def __init__(
        self,
        min_interval_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._min_interval_s = min_interval_s
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        """Block until the next call is allowed, then record it."""
        if self._last is not None:
            remaining = self._min_interval_s - (self._clock() - self._last)
            if remaining > 0:
                self._sleep(remaining)
        self._last = self._clock()


class DailyBudget:
    """A persistent count of generation requests per quota day."""

    def __init__(
        self,
        limit: int,
        path_dir: Path,
        *,
        now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
    ) -> None:
        self._limit = limit
        self._dir = path_dir
        self._now = now

    @property
    def limit(self) -> int:
        return self._limit

    def used(self) -> int:
        try:
            return int(json.loads(self._path().read_text(encoding="utf-8"))["used"])
        except (OSError, ValueError, KeyError, TypeError):
            return 0

    def remaining(self) -> int:
        return max(0, self._limit - self.used())

    def reserve(self) -> None:
        """Count one request, or raise :class:`BudgetExhausted` when none is left."""
        if self.remaining() <= 0:
            raise BudgetExhausted(
                f"The daily generation budget of {self._limit} requests is used up"
            )
        self._store(self.used() + 1)

    def add(self, count: int) -> None:
        """Count requests that were made without a reservation, such as retries."""
        if count > 0:
            self._store(self.used() + count)

    def _path(self) -> Path:
        return self._dir / f"{self._now().astimezone(QUOTA_TIMEZONE).date().isoformat()}.json"

    def _store(self, used: int) -> None:
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        scratch = path.with_suffix(f".{os.getpid()}.tmp")
        scratch.write_text(json.dumps({"used": used}), encoding="utf-8")
        scratch.replace(path)


class BudgetedGenerator:
    """:class:`~rail_rag.rag.providers.base.TextGenerator` that spends budget before it calls."""

    def __init__(self, inner: TextGenerator, limiter: RateLimiter, budget: DailyBudget) -> None:
        self._inner = inner
        self._limiter = limiter
        self._budget = budget

    def generate(self, *, system: str, prompt: str) -> str:
        self._budget.reserve()
        self._limiter.wait()
        return self._inner.generate(system=system, prompt=prompt)
