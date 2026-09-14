"""Acquisition and parsing of external documentation sources."""

from rail_rag.ingestion.docs.exceptions import DocumentError, FetchError
from rail_rag.ingestion.docs.fetcher import (
    DEFAULT_RAW_DIR,
    FetchManifest,
    FetchOutcome,
    artefact_path,
    fetch_source,
    manifest_path,
    probe_digest,
    sha256_of,
)
from rail_rag.ingestion.docs.sources import (
    DocumentSource,
    SourceRegistry,
    load_registry,
    load_source,
)

__all__ = [
    "DEFAULT_RAW_DIR",
    "DocumentError",
    "DocumentSource",
    "FetchError",
    "FetchManifest",
    "FetchOutcome",
    "SourceRegistry",
    "artefact_path",
    "fetch_source",
    "load_registry",
    "load_source",
    "manifest_path",
    "probe_digest",
    "sha256_of",
]
