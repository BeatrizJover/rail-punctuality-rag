"""Per-request tracing: stage timings, provider calls and errors, one JSON line each.

The active trace lives in a :class:`~contextvars.ContextVar`, so concurrent
requests never share one, and every helper is a no-op when none is active:
code that runs outside a traced request behaves exactly as it did before.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

logger = logging.getLogger("rail_rag.trace")

Clock = Callable[[], float]

_HASH_CHARS = 12
_MS_PER_S = 1000


@dataclass
class ProviderCall:
    """One generation or embedding call, retries included."""

    kind: Literal["generation", "embedding"]
    model: str
    latency_ms: float
    attempts: int
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    thought_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class Span:
    """One timed stage of a request."""

    name: str
    duration_ms: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    provider_calls: list[ProviderCall] = field(default_factory=list)


@dataclass
class Trace:
    """Everything recorded about one pipeline construction or one answered question."""

    request_id: str
    kind: Literal["startup", "answer"]
    started_at: datetime
    question_hash: str | None = None
    question_length: int | None = None
    route: str | None = None
    error_class: str | None = None
    error_span: str | None = None
    total_ms: float = 0.0
    spans: list[Span] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialise to a single line, so a log file is one record per trace."""
        record = asdict(self)
        record["started_at"] = self.started_at.isoformat()
        return json.dumps(record, separators=(",", ":"), default=str)


class _Active:
    """The open trace plus the bookkeeping that only matters while it is open."""

    def __init__(self, trace: Trace, clock: Clock) -> None:
        self.trace = trace
        self.clock = clock
        self.open_spans: list[Span] = []
        # Keeps the exception alive, so its identity cannot be recycled mid-request.
        self._failures: list[tuple[BaseException, str]] = []

    def note_failure(self, exc: BaseException, span_name: str) -> None:
        """Remember the innermost span an exception passed through."""
        if not any(seen is exc for seen, _ in self._failures):
            self._failures.append((exc, span_name))

    def failing_span(self, exc: BaseException) -> str | None:
        """Find the span behind ``exc``, following the chain of exceptions it was raised from."""
        visited: set[int] = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            for seen, name in self._failures:
                if seen is current:
                    return name
            current = current.__cause__ or current.__context__
        return None


_active: ContextVar[_Active | None] = ContextVar("rail_rag_active_trace", default=None)


@contextmanager
def start_trace(
    kind: Literal["startup", "answer"],
    *,
    question: str | None = None,
    clock: Clock = time.perf_counter,
) -> Iterator[Trace]:
    """Open a trace and emit it as one JSON line when the block exits.

    An escaping exception is recorded on the trace and re-raised unchanged.
    The question is reduced to a short hash and a length, never stored.
    """
    trace = Trace(request_id=uuid.uuid4().hex, kind=kind, started_at=datetime.now(UTC))
    if question is not None:
        trace.question_hash = hashlib.sha256(question.encode()).hexdigest()[:_HASH_CHARS]
        trace.question_length = len(question)

    active = _Active(trace, clock)
    token = _active.set(active)
    started = clock()
    try:
        yield trace
    except BaseException as exc:
        trace.error_class = type(exc).__name__
        trace.error_span = active.failing_span(exc)
        raise
    finally:
        trace.total_ms = _elapsed_ms(clock, started)
        _active.reset(token)
        _emit(trace)


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[None]:
    """Time a stage of the active trace; does nothing when there is none."""
    active = _active.get()
    if active is None:
        yield
        return

    current = Span(name=name, attributes=dict(attrs))
    active.trace.spans.append(current)
    active.open_spans.append(current)
    started = active.clock()
    try:
        yield
    except BaseException as exc:
        active.note_failure(exc, name)
        raise
    finally:
        current.duration_ms = _elapsed_ms(active.clock, started)
        active.open_spans.pop()


def annotate(*, route: str | None = None, **attrs: Any) -> None:
    """Set the trace's route and add attributes to the innermost open span."""
    active = _active.get()
    if active is None:
        return
    if route is not None:
        active.trace.route = route
    if attrs and active.open_spans:
        active.open_spans[-1].attributes.update(attrs)


def record_provider_call(
    *,
    kind: Literal["generation", "embedding"],
    model: str,
    latency_ms: float,
    attempts: int,
    prompt_tokens: int | None = None,
    output_tokens: int | None = None,
    thought_tokens: int | None = None,
    total_tokens: int | None = None,
) -> None:
    """Attach a provider call to the innermost open span; does nothing without one."""
    active = _active.get()
    if active is None or not active.open_spans:
        return
    active.open_spans[-1].provider_calls.append(
        ProviderCall(
            kind=kind,
            model=model,
            latency_ms=round(latency_ms, 3),
            attempts=attempts,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            thought_tokens=thought_tokens,
            total_tokens=total_tokens,
        )
    )


def _elapsed_ms(clock: Clock, started: float) -> float:
    return round((clock() - started) * _MS_PER_S, 3)


def _emit(trace: Trace) -> None:
    # Observability must never turn a good answer, or the original error, into a different failure.
    try:
        logger.info(trace.to_json())
    except Exception:
        logger.exception("could not emit trace %s", trace.request_id)
