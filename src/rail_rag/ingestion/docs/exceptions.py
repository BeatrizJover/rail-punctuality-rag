"""Exceptions raised by the external-document ingestion pipeline.

They live beside the subpackage rather than in :mod:`rail_rag.core.exceptions`
for the same reason as :mod:`rail_rag.rag.exceptions`: several sibling modules
share them. A malformed ``sources.yaml`` is deliberately not one of them - an
invalid registry is a ``ConfigError``, like every other configuration file.
"""

from __future__ import annotations

from rail_rag.core.exceptions import RailRagError


class DocumentError(RailRagError):
    """Base class for every failure in the external-document pipeline."""


class FetchError(DocumentError):
    """Raised when an external document cannot be downloaded, or fails its digest."""


class ParseError(DocumentError):
    """Raised when a downloaded document does not yield the structure it declares."""
