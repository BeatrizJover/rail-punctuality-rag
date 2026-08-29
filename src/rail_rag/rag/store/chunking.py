"""Split the corpus into retrievable passages.

Chunking is structural, not blind. The corpus is documentation the project
wrote about itself: every section already answers one question, so a section is
already the right retrieval unit. Cutting on a fixed character window instead
would routinely split a definition from the sentence that qualifies it, and the
retrieved half reads as complete while being wrong.

Sections that outgrow the size limit fall back to paragraph boundaries, which is
still a boundary the author chose. A hard character cut is the last resort, used
only when a single paragraph is itself too long.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from rail_rag.rag.exceptions import RagError

#: Roughly 250 words, comfortably inside any embedding model's input window while
#: still holding a whole section of the corpus.
DEFAULT_MAX_CHARS = 1500

#: Below this a chunk carries too little context to retrieve usefully, so it is
#: merged into the previous one instead of standing alone.
DEFAULT_MIN_CHARS = 120

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage, tied back to where it came from."""

    doc_id: str
    chunk_index: int
    heading: str | None
    content: str

    @property
    def content_hash(self) -> str:
        """Digest of the text, so an unchanged chunk need not be re-embedded."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @property
    def embedding_text(self) -> str:
        """The heading is prepended so the vector carries its own topic.

        A passage reading "It is nullable on the daily feed" is unretrievable on
        its own; prefixed with its heading it is not.
        """
        return f"{self.heading}\n\n{self.content}" if self.heading else self.content


def chunk_markdown(
    text: str,
    doc_id: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[Chunk]:
    """Split one markdown document into chunks."""
    sections = _split_sections(text)
    pieces: list[tuple[str | None, str]] = []
    for heading, body in sections:
        cleaned = body.strip()
        if not cleaned:
            continue
        for part in _split_to_size(cleaned, max_chars):
            pieces.append((heading, part))

    merged = _merge_short_pieces(pieces, min_chars)
    return [
        Chunk(doc_id=doc_id, chunk_index=index, heading=heading, content=content)
        for index, (heading, content) in enumerate(merged)
    ]


def load_corpus(
    directory: Path,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[Chunk]:
    """Read every markdown file in ``directory`` and chunk it.

    Raises:
        RagError: if the directory is missing or contains no markdown.
    """
    if not directory.is_dir():
        raise RagError(f"Knowledge base directory not found: {directory}")
    paths = sorted(directory.glob("*.md"))
    if not paths:
        raise RagError(f"No markdown documents in {directory}")

    chunks: list[Chunk] = []
    for path in paths:
        content = path.read_text(encoding="utf-8")
        chunks.extend(chunk_markdown(content, path.stem, max_chars=max_chars, min_chars=min_chars))
    return chunks


def _split_sections(text: str) -> list[tuple[str | None, str]]:
    """Split on markdown headings, keeping each heading with its body."""
    sections: list[tuple[str | None, list[str]]] = []
    current_heading: str | None = None
    current_body: list[str] = []

    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            if current_body or current_heading is not None:
                sections.append((current_heading, current_body))
            current_heading = match.group(2).strip()
            current_body = []
        else:
            current_body.append(line)
    sections.append((current_heading, current_body))
    return [(heading, "\n".join(body)) for heading, body in sections]


def _split_to_size(body: str, max_chars: int) -> list[str]:
    """Break an oversized section on paragraph boundaries, then on length."""
    if len(body) <= max_chars:
        return [body]

    parts: list[str] = []
    buffer = ""
    for paragraph in _PARAGRAPH_BREAK.split(body):
        candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        if buffer:
            parts.append(buffer)
        buffer = paragraph if len(paragraph) <= max_chars else ""
        if not buffer:
            parts.extend(_hard_split(paragraph, max_chars))
    if buffer:
        parts.append(buffer)
    return parts


def _hard_split(paragraph: str, max_chars: int) -> list[str]:
    """Last resort for a paragraph with no internal boundary to use."""
    return [paragraph[start : start + max_chars] for start in range(0, len(paragraph), max_chars)]


def _merge_short_pieces(
    pieces: list[tuple[str | None, str]], min_chars: int
) -> list[tuple[str | None, str]]:
    """Fold fragments too small to retrieve into the piece before them."""
    merged: list[tuple[str | None, str]] = []
    for heading, content in pieces:
        if merged and len(content) < min_chars:
            previous_heading, previous_content = merged[-1]
            merged[-1] = (previous_heading, f"{previous_content}\n\n{content}")
        else:
            merged.append((heading, content))
    return merged
