"""A disk cache for generator replies, so re-running an evaluation costs no quota."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from rail_rag.observability import annotate
from rail_rag.rag.providers.base import TextGenerator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeneratorFingerprint:
    """The settings that change what a model replies to the same prompt."""

    model: str
    temperature: float
    max_output_tokens: int


class CachingGenerator:
    """:class:`~rail_rag.rag.providers.base.TextGenerator` that replays stored replies."""

    def __init__(
        self, inner: TextGenerator, cache_dir: Path, fingerprint: GeneratorFingerprint
    ) -> None:
        self._inner = inner
        self._dir = cache_dir
        self._fingerprint = fingerprint

    def generate(self, *, system: str, prompt: str) -> str:
        path = self._dir / f"{self._key(system, prompt)}.json"
        cached = _read(path)
        if cached is not None:
            annotate(cache="hit")
            return cached
        annotate(cache="miss")
        reply = self._inner.generate(system=system, prompt=prompt)
        _write(path, reply)
        return reply

    def _key(self, system: str, prompt: str) -> str:
        payload = json.dumps([asdict(self._fingerprint), system, prompt], sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


def _read(path: Path) -> str | None:
    try:
        return str(json.loads(path.read_text(encoding="utf-8"))["reply"])
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("ignoring unreadable generation cache entry %s", path.name)
        return None


def _write(path: Path, reply: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(f".{os.getpid()}.tmp")
    scratch.write_text(json.dumps({"reply": reply}), encoding="utf-8")
    scratch.replace(path)
