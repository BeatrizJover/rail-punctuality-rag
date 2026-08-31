"""Tests for the corpus-to-vectors build, exercised with the offline embedder."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from sqlalchemy import Engine

from rail_rag.core.exceptions import DatabaseError
from rail_rag.rag.exceptions import ProviderError, RagError
from rail_rag.rag.providers.fake import FakeEmbedder
from rail_rag.rag.store.builder import build_knowledge_base
from rail_rag.rag.store.models import KbSchema, build_kb_schema
from rail_rag.rag.store.repository import kb_stats
from rail_rag.rag.store.schema import drop_kb_schema

pytestmark = pytest.mark.integration


class CountingEmbedder(FakeEmbedder):
    """A fake that records how many documents it was asked to embed."""

    def __init__(self, dimension: int = 8) -> None:
        super().__init__(dimension=dimension, model_name="fake-embedding")
        self.batches: list[int] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(len(texts))
        return super().embed_documents(texts)


class FlakyEmbedder(CountingEmbedder):
    """Fails once a given number of batches have succeeded."""

    def __init__(self, fail_after: int, dimension: int = 8) -> None:
        super().__init__(dimension=dimension)
        self._fail_after = fail_after

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if len(self.batches) >= self._fail_after:
            raise ProviderError("quota exhausted")
        return super().embed_documents(texts)


def _corpus_dir(tmp_path: Path, documents: int = 2) -> Path:
    directory = tmp_path / "knowledge"
    directory.mkdir()
    for index in range(documents):
        (directory / f"{index:02d}-doc.md").write_text(
            f"# Section {index}\n\n"
            + " ".join(f"Sentence {index} number {n} about punctuality." for n in range(8))
            + f"\n\n## Second section {index}\n\n"
            + " ".join(f"Another sentence {index} number {n} about stations." for n in range(8))
            + "\n",
            encoding="utf-8",
        )
    return directory


def test_build_chunks_syncs_and_embeds(
    postgres_engine: Engine, kb_schema: KbSchema, tmp_path: Path
) -> None:
    embedder = CountingEmbedder()
    report = build_knowledge_base(
        postgres_engine, kb_schema, embedder, _corpus_dir(tmp_path), batch_size=32
    )
    stats = kb_stats(postgres_engine, kb_schema)
    assert report.sync.inserted == stats.total > 0
    assert report.embedded == stats.total
    assert stats.pending == 0


def test_a_second_build_spends_no_quota(
    postgres_engine: Engine, kb_schema: KbSchema, tmp_path: Path
) -> None:
    """The reason ``content_hash`` exists at all."""
    corpus = _corpus_dir(tmp_path)
    build_knowledge_base(postgres_engine, kb_schema, CountingEmbedder(), corpus)
    second = CountingEmbedder()
    report = build_knowledge_base(postgres_engine, kb_schema, second, corpus)
    assert second.batches == []
    assert (report.embedded, report.batches) == (0, 0)
    assert report.sync.unchanged == kb_stats(postgres_engine, kb_schema).total


def test_force_re_embeds_everything(
    postgres_engine: Engine, kb_schema: KbSchema, tmp_path: Path
) -> None:
    corpus = _corpus_dir(tmp_path)
    build_knowledge_base(postgres_engine, kb_schema, CountingEmbedder(), corpus)
    embedder = CountingEmbedder()
    report = build_knowledge_base(postgres_engine, kb_schema, embedder, corpus, force=True)
    assert report.embedded == kb_stats(postgres_engine, kb_schema).total
    assert sum(embedder.batches) == report.embedded


def test_batch_size_splits_the_provider_calls(
    postgres_engine: Engine, kb_schema: KbSchema, tmp_path: Path
) -> None:
    embedder = CountingEmbedder()
    report = build_knowledge_base(
        postgres_engine, kb_schema, embedder, _corpus_dir(tmp_path), batch_size=2
    )
    assert report.batches == len(embedder.batches) > 1
    assert max(embedder.batches) <= 2


def test_a_failed_batch_keeps_the_vectors_already_paid_for(
    postgres_engine: Engine, kb_schema: KbSchema, tmp_path: Path
) -> None:
    """Committing per batch is what makes a mid-build 429 survivable."""
    with pytest.raises(ProviderError):
        build_knowledge_base(
            postgres_engine,
            kb_schema,
            FlakyEmbedder(fail_after=1),
            _corpus_dir(tmp_path),
            batch_size=1,
        )
    stats = kb_stats(postgres_engine, kb_schema)
    assert stats.embedded == 1
    assert stats.pending > 0


def test_a_resumed_build_finishes_the_remainder(
    postgres_engine: Engine, kb_schema: KbSchema, tmp_path: Path
) -> None:
    corpus = _corpus_dir(tmp_path)
    with pytest.raises(ProviderError):
        build_knowledge_base(
            postgres_engine, kb_schema, FlakyEmbedder(fail_after=1), corpus, batch_size=1
        )
    embedder = CountingEmbedder()
    report = build_knowledge_base(postgres_engine, kb_schema, embedder, corpus, batch_size=1)
    assert report.sync.unchanged == kb_stats(postgres_engine, kb_schema).total
    assert kb_stats(postgres_engine, kb_schema).pending == 0


def test_an_edited_document_re_embeds_only_its_own_chunks(
    postgres_engine: Engine, kb_schema: KbSchema, tmp_path: Path
) -> None:
    corpus = _corpus_dir(tmp_path)
    build_knowledge_base(postgres_engine, kb_schema, CountingEmbedder(), corpus)
    target = corpus / "00-doc.md"
    target.write_text(
        target.read_text(encoding="utf-8") + "\n\nA new closing paragraph about delays.\n",
        encoding="utf-8",
    )

    embedder = CountingEmbedder()
    report = build_knowledge_base(postgres_engine, kb_schema, embedder, corpus)
    assert report.embedded == report.sync.needs_embedding
    assert 0 < report.embedded < kb_stats(postgres_engine, kb_schema).total


def test_a_mismatched_embedder_is_refused_before_any_call(
    postgres_engine: Engine, kb_schema: KbSchema, tmp_path: Path
) -> None:
    """Cheap check first: a wrong width must not cost a single request."""
    embedder = CountingEmbedder(dimension=kb_schema.dimension + 1)
    with pytest.raises(RagError, match="dimensions"):
        build_knowledge_base(postgres_engine, kb_schema, embedder, _corpus_dir(tmp_path))
    assert embedder.batches == []


def test_building_without_a_knowledge_base_fails_fast(
    postgres_engine: Engine, tmp_path: Path
) -> None:
    kb = build_kb_schema(8)
    drop_kb_schema(postgres_engine)
    embedder = CountingEmbedder()
    with pytest.raises(DatabaseError, match="kb-init"):
        build_knowledge_base(postgres_engine, kb, embedder, _corpus_dir(tmp_path))
    assert embedder.batches == []


def test_an_empty_corpus_directory_is_refused(
    postgres_engine: Engine, kb_schema: KbSchema, tmp_path: Path
) -> None:
    """Deleting the whole knowledge base by accident should take more than one typo."""
    empty = tmp_path / "empty"
    empty.mkdir()
    build_knowledge_base(postgres_engine, kb_schema, CountingEmbedder(), _corpus_dir(tmp_path))
    before = kb_stats(postgres_engine, kb_schema).total
    with pytest.raises(RagError, match="No markdown"):
        build_knowledge_base(postgres_engine, kb_schema, CountingEmbedder(), empty)
    assert kb_stats(postgres_engine, kb_schema).total == before


def test_non_positive_batch_size_is_refused(
    postgres_engine: Engine, kb_schema: KbSchema, tmp_path: Path
) -> None:
    with pytest.raises(RagError, match="batch_size"):
        build_knowledge_base(
            postgres_engine, kb_schema, CountingEmbedder(), _corpus_dir(tmp_path), batch_size=0
        )


def test_chunk_size_reaches_the_chunker(
    postgres_engine: Engine, kb_schema: KbSchema, tmp_path: Path
) -> None:
    """Chunking lives in the profile, so a coarser model can ask for coarser cuts."""
    corpus = _corpus_dir(tmp_path, documents=1)
    build_knowledge_base(postgres_engine, kb_schema, CountingEmbedder(), corpus, max_chars=1500)
    coarse = kb_stats(postgres_engine, kb_schema).total
    build_knowledge_base(
        postgres_engine, kb_schema, CountingEmbedder(), corpus, max_chars=200, min_chars=10
    )
    assert kb_stats(postgres_engine, kb_schema).total > coarse
