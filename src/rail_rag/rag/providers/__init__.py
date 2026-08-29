"""Model providers behind a common contract."""

from rail_rag.rag.providers.base import Embedder, TextGenerator
from rail_rag.rag.providers.config import (
    EmbeddingConfig,
    GenerationConfig,
    ModelConfig,
    load_model_config,
)
from rail_rag.rag.providers.factory import available_providers, build_embedder, build_generator

__all__ = [
    "Embedder",
    "EmbeddingConfig",
    "GenerationConfig",
    "ModelConfig",
    "TextGenerator",
    "available_providers",
    "build_embedder",
    "build_generator",
    "load_model_config",
]
