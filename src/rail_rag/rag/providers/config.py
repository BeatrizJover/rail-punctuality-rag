"""Declarative model configuration, loaded from a committed YAML file.

The split is deliberate: parameters that belong in version control (provider,
model names, dimensionality, timeouts) live in YAML where a reviewer can see
them change; the API key is an environment variable and never touches the repo.

The file holds named profiles rather than one flat configuration. Gemini's free
tier is a prototyping bench, not the destination, and overwriting its settings
to try another model would delete from version control the exact combination
that was measured. Switching is one line; the profile that was replaced stays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

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


class ChunkingConfig(BaseModel):
    """How the corpus is cut before it reaches the embedding model.

    It belongs to the profile because chunk size is a property of the model that
    consumes it: a larger input window justifies a coarser cut.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    #: Mirrors the defaults in :mod:`rail_rag.rag.store.chunking`, which owns them.
    max_chars: int = Field(default=1500, gt=0)
    min_chars: int = Field(default=120, ge=0)

    @model_validator(mode="after")
    def _minimum_below_maximum(self) -> Self:
        if self.min_chars >= self.max_chars:
            raise ValueError("min_chars must be below max_chars")
        return self


class ModelConfig(BaseModel):
    """Everything the provider factory needs, minus the credentials."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    provider: str
    generation: GenerationConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig = ChunkingConfig()
    #: Free-tier quotas are per-minute, so a burst of embeddings will hit 429.
    max_retries: int = Field(default=3, ge=0)
    retry_backoff_s: float = Field(default=2.0, gt=0)


class ProfileSet(BaseModel):
    """Every configuration the repository knows about, plus which one is live."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    active: str
    profiles: dict[str, ModelConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _active_is_defined(self) -> Self:
        if self.active not in self.profiles:
            raise ValueError(
                f"active profile {self.active!r} is not one of {sorted(self.profiles)}"
            )
        return self

    @property
    def names(self) -> list[str]:
        """The selectable profile names."""
        return sorted(self.profiles)

    def select(self, name: str | None = None) -> ModelConfig:
        """Return one profile, defaulting to the active one.

        Raises:
            ConfigError: if the requested profile is not defined.
        """
        chosen = name or self.active
        try:
            return self.profiles[chosen]
        except KeyError as exc:
            raise ConfigError(
                f"Unknown profile {chosen!r}; defined: {', '.join(self.names)}"
            ) from exc


def load_profiles(path: Path) -> ProfileSet:
    """Read and validate every profile in the configuration file.

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
        return ProfileSet.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid model configuration at {resolved}: {exc}") from exc


def load_model_config(path: Path, *, profile: str | None = None) -> ModelConfig:
    """Read the configuration file and return one profile, the active one by default.

    Raises:
        ConfigError: if the file is invalid or the requested profile is undefined.
    """
    return load_profiles(path).select(profile)
