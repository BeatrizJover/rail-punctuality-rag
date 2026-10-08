"""Tests for how the Gemini adapter reports its calls to the active trace.

The SDK client is stubbed, so nothing here touches the network or a real key.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from rail_rag.observability import span, start_trace
from rail_rag.rag.exceptions import ProviderError
from rail_rag.rag.providers import EmbeddingConfig, GenerationConfig, ModelConfig
from rail_rag.rag.providers import gemini as gemini_module
from rail_rag.rag.providers.gemini import GeminiEmbedder, GeminiGenerator

_DIMENSION = 4


class _Throttled(Exception):
    """Mimics the SDK's quota error, which carries the HTTP status as ``code``."""

    code = 429


class _StubModels:
    """Plays back scripted outcomes: an exception is raised, anything else is returned."""

    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = outcomes
        self.calls = 0

    def _next(self) -> Any:
        outcome = self._outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def generate_content(self, **_: Any) -> Any:
        return self._next()

    def embed_content(self, **_: Any) -> Any:
        return self._next()


def _config(max_retries: int = 3) -> ModelConfig:
    return ModelConfig(
        provider="gemini",
        generation=GenerationConfig(model="gen-model"),
        embedding=EmbeddingConfig(model="emb-model", dimension=_DIMENSION),
        max_retries=max_retries,
    )


def _stub_client(monkeypatch: pytest.MonkeyPatch, outcomes: list[Any]) -> _StubModels:
    models = _StubModels(outcomes)
    monkeypatch.setattr(
        gemini_module, "_client", lambda api_key: SimpleNamespace(models=models), raising=True
    )
    return models


def _response(text: str = "answer", *, usage: dict[str, int] | None = None) -> Any:
    metadata = SimpleNamespace(**usage) if usage else None
    return SimpleNamespace(text=text, usage_metadata=metadata)


def _traced(call: Callable[[], Any]) -> Any:
    with start_trace("answer") as trace, span("sql_generation"):
        call()
    return trace.spans[0].provider_calls


def test_generation_tokens_come_from_usage_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    usage = {
        "prompt_token_count": 120,
        "candidates_token_count": 15,
        "thoughts_token_count": 400,
        "total_token_count": 535,
    }
    _stub_client(monkeypatch, [_response(usage=usage)])
    generator = GeminiGenerator(_config(), "key")

    (call,) = _traced(lambda: generator.generate(system="s", prompt="p"))

    assert (call.kind, call.model, call.attempts) == ("generation", "gen-model", 1)
    assert (call.prompt_tokens, call.output_tokens) == (120, 15)
    assert (call.thought_tokens, call.total_tokens) == (400, 535)
    assert call.latency_ms >= 0


def test_missing_usage_metadata_leaves_every_token_count_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_client(monkeypatch, [_response()])
    generator = GeminiGenerator(_config(), "key")

    (call,) = _traced(lambda: generator.generate(system="s", prompt="p"))

    assert (call.prompt_tokens, call.output_tokens) == (None, None)
    assert (call.thought_tokens, call.total_tokens) == (None, None)


def test_a_partial_usage_metadata_keeps_what_is_present(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_client(monkeypatch, [_response(usage={"prompt_token_count": 7, "total_token_count": 9})])
    generator = GeminiGenerator(_config(), "key")

    (call,) = _traced(lambda: generator.generate(system="s", prompt="p"))

    assert (call.prompt_tokens, call.output_tokens, call.total_tokens) == (7, None, 9)


def test_a_throttled_call_that_then_succeeds_records_two_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    models = _stub_client(monkeypatch, [_Throttled(), _response(usage={"total_token_count": 5})])
    generator = GeminiGenerator(_config(), "key")

    (call,) = _traced(lambda: generator.generate(system="s", prompt="p"))

    assert call.attempts == 2
    assert call.total_tokens == 5
    assert models.calls == 2
    assert len(sleeps) == 1


def test_a_call_that_exhausts_its_retries_is_still_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("time.sleep", lambda _: None)
    _stub_client(monkeypatch, [_Throttled(), _Throttled()])
    generator = GeminiGenerator(_config(max_retries=1), "key")

    with (
        pytest.raises(ProviderError, match="generation failed"),
        start_trace("answer") as trace,
        span("sql_generation"),
    ):
        generator.generate(system="s", prompt="p")

    (call,) = trace.spans[0].provider_calls
    assert call.attempts == 2
    assert call.total_tokens is None
    assert (trace.error_class, trace.error_span) == ("ProviderError", "sql_generation")


def test_each_embedding_call_is_recorded_with_no_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    vector = SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1] * _DIMENSION)])
    _stub_client(monkeypatch, [vector, vector])
    embedder = GeminiEmbedder(_config(), "key")

    calls = _traced(lambda: embedder.embed_documents(["a", "b"]))

    assert [(c.kind, c.model, c.attempts) for c in calls] == [("embedding", "emb-model", 1)] * 2
    assert all(c.total_tokens is None for c in calls)


def test_the_adapter_behaves_as_before_without_a_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_client(monkeypatch, [_response("plain")])
    assert GeminiGenerator(_config(), "key").generate(system="s", prompt="p") == "plain"
