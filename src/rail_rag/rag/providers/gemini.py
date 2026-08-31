"""Google AI Studio (Gemini) adapter.

The SDK is imported inside ``__init__`` rather than at module scope so that
``rail_rag`` stays importable — and the fake provider stays usable — on an
installation without the optional ``gemini`` extra.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

from rail_rag.rag.exceptions import ProviderError
from rail_rag.rag.providers.config import ModelConfig

logger = logging.getLogger(__name__)

_INSTALL_HINT = 'Gemini provider requires the optional extra: pip install -e ".[gemini]"'

#: Transient HTTP statuses: quota exhausted, backend unavailable, gateway timeout.
_RETRYABLE_STATUS = frozenset({429, 500, 503, 504})

#: The SDK expresses request timeouts in milliseconds.
_MS_PER_S = 1000


def _client(api_key: str) -> Any:
    """Build a Gemini client, or fail with an actionable message."""
    if not api_key:
        raise ProviderError("Gemini provider requires RAIL_RAG_LLM_API_KEY to be set")
    try:
        from google import genai
    except ImportError as exc:
        raise ProviderError(_INSTALL_HINT) from exc
    return genai.Client(api_key=api_key)


def _retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    return isinstance(code, int) and code in _RETRYABLE_STATUS


class _Retrying:
    """Shared exponential backoff around a single SDK call."""

    def __init__(self, max_retries: int, backoff_s: float) -> None:
        self._max_retries = max_retries
        self._backoff_s = backoff_s

    def run(self, what: str, call: Any) -> Any:
        last: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return call()
            except Exception as exc:
                if not _retryable(exc) or attempt == self._max_retries:
                    raise ProviderError(f"Gemini {what} failed: {type(exc).__name__}") from exc
                last = exc
                delay = self._backoff_s * (2**attempt)
                logger.warning("Gemini %s throttled, retrying in %.1fs", what, delay)
                time.sleep(delay)
        raise ProviderError(f"Gemini {what} failed: {type(last).__name__}")


class GeminiGenerator:
    """:class:`~rail_rag.rag.providers.base.TextGenerator` backed by Gemini."""

    def __init__(self, config: ModelConfig, api_key: str) -> None:
        from google.genai import types

        self._types = types
        self._client = _client(api_key)
        self._config = config.generation
        self._retrying = _Retrying(config.max_retries, config.retry_backoff_s)

    def _http_options(self) -> Any:
        return self._types.HttpOptions(timeout=int(self._config.timeout_s * _MS_PER_S))

    def generate(self, *, system: str, prompt: str) -> str:
        request = self._types.GenerateContentConfig(
            system_instruction=system,
            temperature=self._config.temperature,
            max_output_tokens=self._config.max_output_tokens,
            http_options=self._http_options(),
        )
        response = self._retrying.run(
            "generation",
            lambda: self._client.models.generate_content(
                model=self._config.model, contents=prompt, config=request
            ),
        )
        text = getattr(response, "text", None)
        if not text:
            # An empty body usually means a safety block or a truncated response.
            raise ProviderError("Gemini returned no text for the request")
        return str(text)


class GeminiEmbedder:
    """:class:`~rail_rag.rag.providers.base.Embedder` backed by Gemini."""

    #: Gemini distinguishes the two retrieval roles; using one for both hurts recall.
    _DOCUMENT_TASK = "RETRIEVAL_DOCUMENT"
    _QUERY_TASK = "RETRIEVAL_QUERY"

    def __init__(self, config: ModelConfig, api_key: str) -> None:
        from google.genai import types

        self._types = types
        self._client = _client(api_key)
        self._config = config.embedding
        self._retrying = _Retrying(config.max_retries, config.retry_backoff_s)

    @property
    def dimension(self) -> int:
        return self._config.dimension

    @property
    def model_name(self) -> str:
        return self._config.model

    def _http_options(self) -> Any:
        return self._types.HttpOptions(timeout=int(self._config.timeout_s * _MS_PER_S))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._config.batch_size):
            batch = list(texts[start : start + self._config.batch_size])
            vectors.extend(self._embed(batch, self._DOCUMENT_TASK))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], self._QUERY_TASK)[0]

    def _embed(self, batch: list[str], task_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in batch:
            request = self._types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self._config.dimension,
                http_options=self._http_options(),
            )
            response = self._retrying.run(
                "embedding",
                lambda text=text, request=request: self._client.models.embed_content(
                    model=self._config.model, contents=text, config=request
                ),
            )
            embeddings = getattr(response, "embeddings", None) or []
            if len(embeddings) != 1:
                raise ProviderError(
                    f"Gemini returned {len(embeddings)} vectors for one input"
                )
            values = getattr(embeddings[0], "values", None)
            if values is None or len(values) != self._config.dimension:
                raise ProviderError(
                    f"Gemini returned a vector of length {len(values or [])} "
                    f"for model {self._config.model}"
                )
            vectors.append([float(value) for value in values])
        return vectors
