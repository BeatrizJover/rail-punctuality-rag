"""Declarative model configuration, loaded from a committed YAML file.

The split is deliberate: parameters that belong in version control (provider,
model names, dimensionality, timeouts) live in YAML where a reviewer can see
them change; the API key is an environment variable and never touches the repo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rail_rag.core.exceptions import ConfigError


class GenerationConfig(BaseModel):
    """Parameters for the text generation model."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    model: str
    #: Zero by default: text-to-SQL wants the same query for the same question.
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, gt=0)
    timeout_s: float = Field(default=60.0, gt=0)


class EmbeddingConfig(BaseModel):
    """Parameters for the embedding model."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    model: str
    #: Must match the ``vector(n)`` column in the ``rag`` schema; checked at startup.
    dimension: int = Field(gt=0)
    batch_size: int = Field(default=32, gt=0)
    timeout_s: float = Field(default=60.0, gt=0)


class ModelConfig(BaseModel):
    """Everything the provider factory needs, minus the credentials."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    provider: str
    generation: GenerationConfig
    embedding: EmbeddingConfig
    #: Free-tier quotas are per-minute, so a burst of embeddings will hit 429.
    max_retries: int = Field(default=3, ge=0)
    retry_backoff_s: float = Field(default=2.0, gt=0)


def load_model_config(path: Path) -> ModelConfig:
    """Read and validate the model configuration file.

    Raises:
        ConfigError: if the file is missing, unreadable, not a mapping, or invalid.
    """
    resolved = path.expanduser().resolve()
    try:
        raw: Any = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Model configuration not found: {resolved}") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not read model configuration at {resolved}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Model configuration at {resolved} must be a mapping")

    try:
        return ModelConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid model configuration at {resolved}: {exc}") from exc
