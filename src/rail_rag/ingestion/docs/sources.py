"""Declarative registry of the external documents the knowledge base ingests.

Third-party documentation is not committed to this repository, so what must be
reproducible is the *reference* to it: where it came from, which version it is,
and the digest the download has to produce. The reference lives in version
control where a reviewer sees it change; the bytes live under ``data/docs/raw``
and are ignored by Git.

Entries are a list, not a mapping keyed by ``source_id``. A YAML mapping keeps
the last of two duplicate keys without a word, and a registry that drops a
source silently is worse than one that refuses to load.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError, model_validator

from rail_rag.core.exceptions import ConfigError

#: Lowercase hex, as produced by ``hashlib.sha256().hexdigest()``.
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class DocumentSource(BaseModel):
    """One external document, pinned to a version and a digest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    title: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    landing_page: HttpUrl
    canonical_url: HttpUrl
    #: Published in the file name upstream. Constrained because it becomes a path.
    version: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    language: str = Field(pattern=r"^[a-z]{2}$")
    #: Free-form on purpose: a future web source must not need a schema change.
    document_type: str = Field(min_length=1)
    #: Reviewed in a commit. A silently republished file fails the fetch.
    sha256: str = Field(pattern=_SHA256_PATTERN)

    @property
    def file_extension(self) -> str:
        """The suffix the artefact is stored under, taken from the canonical URL."""
        return Path(self.canonical_url.path or "").suffix

    @property
    def file_stem(self) -> str:
        """Identity of the stored artefact: the source and the version it pins."""
        return f"{self.source_id}__{self.version}"

    @model_validator(mode="after")
    def _url_carries_a_file_extension(self) -> Self:
        if not self.file_extension:
            raise ValueError("canonical_url must point at a file with an extension")
        return self


class SourceRegistry(BaseModel):
    """Every external document the project knows how to fetch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sources: list[DocumentSource] = Field(min_length=1)

    @model_validator(mode="after")
    def _identifiers_are_unique(self) -> Self:
        counted = Counter(source.source_id for source in self.sources)
        repeated = sorted(name for name, total in counted.items() if total > 1)
        if repeated:
            raise ValueError(f"duplicate source_id: {', '.join(repeated)}")
        return self

    @property
    def names(self) -> list[str]:
        """The selectable source identifiers."""
        return sorted(source.source_id for source in self.sources)

    def select(self, source_id: str) -> DocumentSource:
        """Return one source by identifier.

        Raises:
            ConfigError: if the requested source is not declared.
        """
        for source in self.sources:
            if source.source_id == source_id:
                return source
        raise ConfigError(f"Unknown source {source_id!r}; declared: {', '.join(self.names)}")


def load_registry(path: Path) -> SourceRegistry:
    """Read and validate every source in the registry file.

    Raises:
        ConfigError: if the file is missing, unreadable, not a mapping, or invalid.
    """
    resolved = path.expanduser().resolve()
    try:
        raw: Any = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Source registry not found: {resolved}") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not read source registry at {resolved}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Source registry at {resolved} must be a mapping")

    try:
        return SourceRegistry.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid source registry at {resolved}: {exc}") from exc


def load_source(path: Path, source_id: str) -> DocumentSource:
    """Read the registry file and return one declared source.

    Raises:
        ConfigError: if the file is invalid or the source is not declared.
    """
    return load_registry(path).select(source_id)
