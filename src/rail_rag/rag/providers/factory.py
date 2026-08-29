"""Build providers from configuration.

A registry keyed by ``provider`` name is the whole extension point: adding
Anthropic later means one adapter module and two entries here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from pydantic import SecretStr

from rail_rag.rag.exceptions import ProviderError
from rail_rag.rag.providers.base import Embedder, TextGenerator
from rail_rag.rag.providers.config import ModelConfig
from rail_rag.rag.providers.fake import FakeEmbedder, FakeGenerator

GeneratorBuilder = Callable[[ModelConfig, SecretStr], TextGenerator]
EmbedderBuilder = Callable[[ModelConfig, SecretStr], Embedder]

_B = TypeVar("_B")


def _fake_generator(config: ModelConfig, api_key: SecretStr) -> TextGenerator:
    return FakeGenerator()


def _fake_embedder(config: ModelConfig, api_key: SecretStr) -> Embedder:
    return FakeEmbedder(dimension=config.embedding.dimension, model_name=config.embedding.model)


def _gemini_generator(config: ModelConfig, api_key: SecretStr) -> TextGenerator:
    from rail_rag.rag.providers.gemini import GeminiGenerator

    return GeminiGenerator(config, api_key.get_secret_value())


def _gemini_embedder(config: ModelConfig, api_key: SecretStr) -> Embedder:
    from rail_rag.rag.providers.gemini import GeminiEmbedder

    return GeminiEmbedder(config, api_key.get_secret_value())


_GENERATORS: dict[str, GeneratorBuilder] = {
    "fake": _fake_generator,
    "gemini": _gemini_generator,
}

_EMBEDDERS: dict[str, EmbedderBuilder] = {
    "fake": _fake_embedder,
    "gemini": _gemini_embedder,
}


def available_providers() -> list[str]:
    """Return the provider names that can be selected in the configuration."""
    return sorted(_GENERATORS)


def build_generator(config: ModelConfig, api_key: SecretStr) -> TextGenerator:
    """Instantiate the configured text generator.

    Raises:
        ProviderError: if the provider is unknown or cannot be constructed.
    """
    return _lookup(_GENERATORS, config.provider)(config, api_key)


def build_embedder(config: ModelConfig, api_key: SecretStr) -> Embedder:
    """Instantiate the configured embedder.

    Raises:
        ProviderError: if the provider is unknown or cannot be constructed.
    """
    return _lookup(_EMBEDDERS, config.provider)(config, api_key)


def _lookup(registry: dict[str, _B], provider: str) -> _B:
    try:
        return registry[provider]
    except KeyError as exc:
        raise ProviderError(
            f"Unknown provider {provider!r}; available: {', '.join(available_providers())}"
        ) from exc
