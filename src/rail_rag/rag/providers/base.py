"""Provider-agnostic contracts for text generation and embeddings.

Nothing outside :mod:`rail_rag.rag.providers` imports a vendor SDK: the rest of
the application depends only on the two protocols defined here, so swapping
Gemini for another provider is an added adapter, not a refactor.

Both protocols are synchronous. The database layer is synchronous SQLAlchemy
Core, and FastAPI already runs ``def`` endpoints in a worker thread, so an
async contract would buy nothing and colour half the codebase.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class TextGenerator(Protocol):
    """Turns a system instruction plus a prompt into a single text answer."""

    def generate(self, *, system: str, prompt: str) -> str:
        """Return the model's answer, with no streaming and no tool calls.

        Raises:
            ProviderError: if the provider cannot be reached or returns no text.
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    """Maps text to dense vectors, with distinct document and query modes."""

    @property
    def dimension(self) -> int:
        """Vector length, which the ``rag`` schema pins into its ``vector(n)`` column."""
        ...

    @property
    def model_name(self) -> str:
        """Identifier stored alongside each vector, so a model swap is detectable."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed corpus chunks for storage, preserving input order.

        Raises:
            ProviderError: if the provider fails or returns a malformed batch.
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a user question for similarity search.

        Kept separate from :meth:`embed_documents` because providers optimise the
        two asymmetrically; using one mode for both measurably degrades recall.

        Raises:
            ProviderError: if the provider fails or returns a malformed vector.
        """
        ...
