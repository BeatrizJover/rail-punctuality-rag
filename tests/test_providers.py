"""Tests for the provider contracts, the offline fake and the factory."""

from __future__ import annotations

import math

import pytest
from pydantic import SecretStr

from rail_rag.rag.exceptions import ProviderError, RagError
from rail_rag.rag.providers import (
    EmbeddingConfig,
    GenerationConfig,
    ModelConfig,
    available_providers,
    build_embedder,
    build_generator,
)
from rail_rag.rag.providers.base import Embedder, TextGenerator
from rail_rag.rag.providers.fake import DEFAULT_RESPONSE, FakeEmbedder, FakeGenerator


def _config(provider: str, dimension: int = 8) -> ModelConfig:
    return ModelConfig(
        provider=provider,
        generation=GenerationConfig(model="g"),
        embedding=EmbeddingConfig(model="e", dimension=dimension),
    )


def test_fake_provider_satisfies_the_protocols() -> None:
    assert isinstance(FakeGenerator(), TextGenerator)
    assert isinstance(FakeEmbedder(), Embedder)


def test_fake_generator_returns_canned_answers_in_order() -> None:
    generator = FakeGenerator(["first", "second"])
    assert generator.generate(system="s", prompt="a") == "first"
    assert generator.generate(system="s", prompt="b") == "second"
    # Exhausted responses repeat, so a test never fails on call count alone.
    assert generator.generate(system="s", prompt="c") == "second"
    assert generator.calls[0] == ("s", "a")


def test_fake_generator_has_a_default_response() -> None:
    assert FakeGenerator().generate(system="s", prompt="p") == DEFAULT_RESPONSE


def test_fake_embedder_is_deterministic_across_instances() -> None:
    """Digest-derived vectors survive a restart; ``hash()`` would not."""
    first = FakeEmbedder(dimension=16).embed_query("Brussels-Central")
    second = FakeEmbedder(dimension=16).embed_query("Brussels-Central")
    assert first == second


def test_fake_embedder_separates_distinct_texts() -> None:
    embedder = FakeEmbedder(dimension=16)
    assert embedder.embed_query("Gent") != embedder.embed_query("Liege")


def test_fake_embedder_respects_dimension_and_normalises() -> None:
    vector = FakeEmbedder(dimension=32).embed_query("any")
    assert len(vector) == 32
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)


def test_fake_embedder_preserves_batch_order() -> None:
    embedder = FakeEmbedder(dimension=8)
    texts = ["a", "b", "c"]
    batch = embedder.embed_documents(texts)
    assert len(batch) == 3
    assert batch == [embedder.embed_query(text) for text in texts]


def test_fake_embedder_rejects_non_positive_dimension() -> None:
    with pytest.raises(ValueError, match="dimension"):
        FakeEmbedder(dimension=0)


def test_factory_builds_the_fake_provider() -> None:
    config = _config("fake", dimension=16)
    assert isinstance(build_generator(config, SecretStr("")), TextGenerator)
    embedder = build_embedder(config, SecretStr(""))
    # Dimension and model name come from configuration, not from the adapter.
    assert embedder.dimension == 16
    assert embedder.model_name == "e"


def test_factory_rejects_an_unknown_provider() -> None:
    with pytest.raises(ProviderError, match="Unknown provider"):
        build_generator(_config("nope"), SecretStr(""))
    with pytest.raises(ProviderError, match="Unknown provider"):
        build_embedder(_config("nope"), SecretStr(""))


def test_gemini_without_an_api_key_fails_fast() -> None:
    """A missing key must surface as a provider error, not an SDK stack trace."""
    with pytest.raises(ProviderError, match="RAIL_RAG_LLM_API_KEY"):
        build_generator(_config("gemini"), SecretStr(""))


def test_provider_errors_are_application_errors() -> None:
    """The CLI already catches ``RailRagError``; RAG failures must be caught too."""
    assert issubclass(ProviderError, RagError)


def test_available_providers_lists_the_offline_option() -> None:
    assert "fake" in available_providers()
    assert "gemini" in available_providers()
