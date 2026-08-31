"""Tests for question-to-passage retrieval."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Engine

from rail_rag.core.exceptions import DatabaseError
from rail_rag.rag.exceptions import RagError
from rail_rag.rag.providers.fake import FakeEmbedder
from rail_rag.rag.store.builder import build_knowledge_base
from rail_rag.rag.store.chunking import load_corpus
from rail_rag.rag.store.models import KbSchema, build_kb_schema
from rail_rag.rag.store.retriever import Retriever
from rail_rag.rag.store.schema import drop_kb_schema

pytestmark = pytest.mark.integration

_DOCUMENTS = {
    "punctuality": "# Punctuality threshold\n\nA train counts as punctual below 360 seconds.\n",
    "stations": "# Station identity\n\nIdentity is an md5 of the normalised station name.\n",
    "coverage": "# Coverage window\n\nThe calendar spans 2014 to 2027; the fact table does not.\n",
}


def _embedder(dimension: int = 8) -> FakeEmbedder:
    return FakeEmbedder(dimension=dimension, model_name="fake-embedding")


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "knowledge"
    directory.mkdir()
    for name, body in _DOCUMENTS.items():
        (directory / f"{name}.md").write_text(body, encoding="utf-8")
    return directory


@pytest.fixture
def populated(postgres_engine: Engine, kb_schema: KbSchema, corpus_dir: Path) -> KbSchema:
    build_knowledge_base(postgres_engine, kb_schema, _embedder(), corpus_dir)
    return kb_schema


def test_retrieve_puts_the_matching_passage_first(
    postgres_engine: Engine, populated: KbSchema, corpus_dir: Path
) -> None:
    """The fake embeds a query exactly like the document, so the match is exact."""
    target = next(chunk for chunk in load_corpus(corpus_dir) if chunk.doc_id == "punctuality")
    retriever = Retriever(postgres_engine, populated, _embedder())
    results = retriever.retrieve(target.embedding_text)
    assert results[0].doc_id == "punctuality"
    assert results[0].similarity == pytest.approx(1.0)


def test_retrieve_cites_document_and_heading(
    postgres_engine: Engine, populated: KbSchema, corpus_dir: Path
) -> None:
    target = next(chunk for chunk in load_corpus(corpus_dir) if chunk.doc_id == "stations")
    results = Retriever(postgres_engine, populated, _embedder()).retrieve(target.embedding_text)
    assert (results[0].doc_id, results[0].heading) == ("stations", "Station identity")
    assert results[0].content


def test_top_k_bounds_the_result(postgres_engine: Engine, populated: KbSchema) -> None:
    retriever = Retriever(postgres_engine, populated, _embedder(), top_k=2)
    assert len(retriever.retrieve("anything at all")) == 2
    assert len(retriever.retrieve("anything at all", top_k=1)) == 1


def test_an_empty_question_costs_nothing(postgres_engine: Engine, populated: KbSchema) -> None:
    """Embedding whitespace spends quota to rank passages arbitrarily."""

    class RefusingEmbedder(FakeEmbedder):
        def embed_query(self, text: str) -> list[float]:
            raise AssertionError("the provider must not be called")

    retriever = Retriever(
        postgres_engine,
        populated,
        RefusingEmbedder(dimension=populated.dimension, model_name="fake-embedding"),
    )
    assert retriever.retrieve("   \n  ") == []


def test_an_empty_knowledge_base_returns_nothing(
    postgres_engine: Engine, kb_schema: KbSchema
) -> None:
    """No passages is a valid answer; it must not be an exception."""
    assert Retriever(postgres_engine, kb_schema, _embedder()).retrieve("anything") == []


def test_a_mismatched_embedder_is_refused_at_construction(
    postgres_engine: Engine, kb_schema: KbSchema
) -> None:
    """Cosine distance between two models' vectors is valid arithmetic and meaningless."""
    wrong = FakeEmbedder(dimension=kb_schema.dimension + 1, model_name="other")
    with pytest.raises(RagError, match="dimensions"):
        Retriever(postgres_engine, kb_schema, wrong)


def test_a_missing_knowledge_base_is_refused_at_construction(
    postgres_engine: Engine,
) -> None:
    kb = build_kb_schema(8)
    drop_kb_schema(postgres_engine)
    with pytest.raises(DatabaseError, match="kb-init"):
        Retriever(postgres_engine, kb, _embedder())


def test_non_positive_top_k_is_refused(postgres_engine: Engine, kb_schema: KbSchema) -> None:
    with pytest.raises(RagError, match="top_k"):
        Retriever(postgres_engine, kb_schema, _embedder(), top_k=0)
