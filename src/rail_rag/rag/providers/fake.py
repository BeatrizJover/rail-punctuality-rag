"""A deterministic, offline provider.

It ships in ``src`` rather than in ``tests`` on purpose: with
``provider: fake`` the whole application boots, answers and is exercised
end to end with no API key and no network, which is what makes the RAG layer
testable in CI and reviewable by someone who has no credentials.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Sequence

DEFAULT_RESPONSE = "fake-provider response"


class FakeGenerator:
    """Returns canned answers and records what it was asked."""

    def __init__(self, responses: Sequence[str] | None = None) -> None:
        self._responses = list(responses) if responses else [DEFAULT_RESPONSE]
        self.calls: list[tuple[str, str]] = []

    def generate(self, *, system: str, prompt: str) -> str:
        """Return the next canned answer, repeating the last one once exhausted."""
        self.calls.append((system, prompt))
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[index]


class FakeEmbedder:
    """Hashes text into a stable unit vector of the configured dimension."""

    def __init__(self, dimension: int = 768, model_name: str = "fake-embedding") -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        """Derive a unit vector from a digest, so results survive process restarts."""
        # ``hash()`` is salted per process; a digest keeps fixtures reproducible.
        digest = hashlib.sha256(f"{self._model_name}:{text}".encode()).digest()
        rng = random.Random(int.from_bytes(digest, "big"))
        raw = [rng.gauss(0.0, 1.0) for _ in range(self._dimension)]
        norm = math.sqrt(sum(value * value for value in raw)) or 1.0
        return [value / norm for value in raw]
