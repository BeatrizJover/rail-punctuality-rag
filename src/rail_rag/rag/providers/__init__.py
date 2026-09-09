"""Model providers behind a common contract."""

from rail_rag.rag.providers.base import Embedder, TextGenerator
from rail_rag.rag.providers.config import (
    ChunkingConfig,
    EmbeddingConfig,
    GenerationConfig,
    ModelConfig,
    ProfileSet,
    load_model_config,
    load_profiles,
)
from rail_rag.rag.providers.factory import available_providers, build_embedder, build_generator

__all__ = [
    "ChunkingConfig",
    "Embedder",
    "EmbeddingConfig",
    "GenerationConfig",
    "ModelConfig",
    "ProfileSet",
    "TextGenerator",
    "available_providers",
    "build_embedder",
    "build_generator",
    "load_model_config",
    "load_profiles",
]
