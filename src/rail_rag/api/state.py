"""The long-lived objects the API serves from.

Everything expensive — the engine, the profile measurement, the knowledge-base
dimension check, the rendered schema context — is paid for once here rather than
per request. A pipeline rebuilt on every question would re-measure the database
and re-verify the vector width before answering anything.

The state is passed in rather than reached for, so a test can serve a stub
pipeline without a database, a corpus or an API key.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sqlalchemy import Engine

from rail_rag.core.config import Settings, get_settings
from rail_rag.db.engine import create_db_engine
from rail_rag.rag.pipeline import Answer, AnswerPipeline
from rail_rag.rag.providers.config import load_model_config
from rail_rag.rag.providers.factory import build_embedder, build_generator
from rail_rag.rag.sql.policy import load_retrieval_config
from rail_rag.rag.store.models import KbSchema, build_kb_schema

logger = logging.getLogger(__name__)

DEFAULT_RETRIEVAL_CONFIG = Path("config/retrieval_config.yaml")


class Answerer(Protocol):
    """The only thing the API needs from the pipeline."""

    def answer(self, question: str) -> Answer: ...


@dataclass(frozen=True)
class AppState:
    """Dependencies shared by every request."""

    engine: Engine
    pipeline: Answerer
    kb: KbSchema
    environment: str


def build_state(
    settings: Settings | None = None,
    *,
    profile: str | None = None,
    retrieval_config: Path = DEFAULT_RETRIEVAL_CONFIG,
) -> AppState:
    """Assemble the pipeline and its dependencies.

    Raises:
        ConfigError: if either configuration file is missing or invalid.
        DatabaseError: if the knowledge base is absent or its width differs.
        ProviderError: if the configured provider cannot be constructed.
    """
    resolved = settings or get_settings()
    config = load_model_config(resolved.llm_config_path, profile=profile)
    kb = build_kb_schema(config.embedding.dimension)
    engine = create_db_engine(resolved)
    policy = load_retrieval_config(retrieval_config).sql

    pipeline = AnswerPipeline(
        engine,
        build_generator(config, resolved.llm_api_key),
        build_embedder(config, resolved.llm_api_key),
        kb,
        policy,
    )
    logger.info(
        "Pipeline ready (provider=%s, generation=%s, embedding=%s)",
        config.provider,
        config.generation.model,
        config.embedding.model,
    )
    return AppState(
        engine=engine,
        pipeline=pipeline,
        kb=kb,
        environment=resolved.environment,
    )
