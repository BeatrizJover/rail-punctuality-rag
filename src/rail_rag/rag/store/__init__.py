"""Persistence for the knowledge base."""

from rail_rag.rag.store.chunking import Chunk, chunk_markdown, load_corpus
from rail_rag.rag.store.models import RAG_SCHEMA, KbSchema, build_kb_schema
from rail_rag.rag.store.schema import (
    create_kb_schema,
    drop_kb_schema,
    kb_schema_exists,
    stored_dimension,
    verify_dimension,
)

__all__ = [
    "RAG_SCHEMA",
    "Chunk",
    "KbSchema",
    "build_kb_schema",
    "chunk_markdown",
    "create_kb_schema",
    "drop_kb_schema",
    "kb_schema_exists",
    "load_corpus",
    "stored_dimension",
    "verify_dimension",
]
