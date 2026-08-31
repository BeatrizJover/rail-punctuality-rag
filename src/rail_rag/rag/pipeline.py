"""End-to-end question answering: routing, retrieval, generation, execution.

The pipeline is built once (at application startup or CLI boot) and reused for
every question.  Construction pays the one-time costs — measuring the database,
verifying the knowledge base, building the context block — so none of that
appears on the hot path.

The ``answer`` method is deliberately synchronous.  The database layer is
synchronous SQLAlchemy Core, and FastAPI already runs ``def`` endpoints in a
worker thread, so colouring this coroutine would buy nothing.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy import Engine

from rail_rag.rag.context import render_context
from rail_rag.rag.context.profile import DataProfile, load_profile
from rail_rag.rag.exceptions import AnswerError, UnsafeQueryError
from rail_rag.rag.prompts import (
    NO_SQL,
    Passage,
    build_answer_prompt,
    build_conceptual_prompt,
    build_repair_prompt,
    build_sql_prompt,
    build_sql_system,
    extract_sql,
)
from rail_rag.rag.providers.base import Embedder, TextGenerator
from rail_rag.rag.router import Route, classify
from rail_rag.rag.sql.executor import QueryResult, execute_safe_query
from rail_rag.rag.sql.guard import SafeQuery, validate_sql
from rail_rag.rag.sql.policy import SqlPolicy
from rail_rag.rag.store.models import KbSchema
from rail_rag.rag.store.retriever import Retriever

logger = logging.getLogger(__name__)

#: Guard attempts: the initial generation plus one retry with the error message.
_MAX_SQL_ATTEMPTS = 2


@dataclass(frozen=True)
class Source:
    """A passage the answer drew on, surfaced so the caller can cite it."""

    doc_id: str
    heading: str | None


@dataclass(frozen=True)
class Answer:
    """Everything the caller needs to render a response."""

    text: str
    route: Route
    sql: str | None = None
    result: QueryResult | None = None
    sources: tuple[Source, ...] = field(default_factory=tuple)


class AnswerPipeline:
    """Wires retrieval, generation, validation and execution into one call."""

    def __init__(
        self,
        engine: Engine,
        generator: TextGenerator,
        embedder: Embedder,
        kb: KbSchema,
        policy: SqlPolicy,
        *,
        top_k: int = 4,
    ) -> None:
        self._engine = engine
        self._generator = generator
        self._policy = policy

        profile = load_profile(engine)
        self._profile = profile
        self._system = build_sql_system(render_context(policy, profile))
        self._retriever = Retriever(engine, kb, embedder, top_k=top_k)

    @property
    def profile(self) -> DataProfile:
        return self._profile

    def answer(self, question: str) -> Answer:
        """Return a complete answer, choosing the path automatically.

        Never raises on a recoverable failure: a misrouted question falls back
        to the other path, and a guard rejection is retried once.  Only an
        unrecoverable failure (empty model reply, two guard rejections, database
        down) surfaces as :class:`AnswerError`.
        """
        if not question.strip():
            raise AnswerError("The question is empty")

        passages = self._retriever.retrieve(question)
        sources = _sources_from(passages)
        route = classify(question, self._generator)

        if route is Route.CONCEPTUAL:
            return self._answer_conceptual(question, passages, sources)

        return self._answer_data(question, passages, sources)

    def _answer_data(
        self,
        question: str,
        passages: Sequence[Passage],
        sources: tuple[Source, ...],
    ) -> Answer:
        """Generate SQL, validate, execute, narrate — or fall back."""
        sql_text = self._generate_sql(question, passages)
        if sql_text is None:
            logger.info("generator declined with %s; falling back to conceptual", NO_SQL)
            return self._answer_conceptual(question, passages, sources)

        safe = self._validate_with_retry(question, sql_text)
        result = execute_safe_query(self._engine, safe, self._policy)

        if result.is_empty and not self._profile.has_data:
            return Answer(
                text="The fact table is empty — no data has been loaded yet.",
                route=Route.DATA,
                sql=safe.sql,
                result=result,
                sources=sources,
            )

        narration = self._narrate(question, result, passages)
        return Answer(
            text=narration,
            route=Route.DATA,
            sql=safe.sql,
            result=result,
            sources=sources,
        )

    def _answer_conceptual(
        self,
        question: str,
        passages: Sequence[Passage],
        sources: tuple[Source, ...],
    ) -> Answer:
        """Answer from documentation only, no SQL."""
        prompt = build_conceptual_prompt(question, passages)
        from rail_rag.rag.prompts import CONCEPTUAL_INSTRUCTIONS

        text = self._generator.generate(system=CONCEPTUAL_INSTRUCTIONS, prompt=prompt)
        return Answer(text=text, route=Route.CONCEPTUAL, sources=sources)

    def _generate_sql(self, question: str, passages: Sequence[Passage]) -> str | None:
        """Ask the model for a query; return ``None`` if it declines."""
        prompt = build_sql_prompt(question, passages)
        reply = self._generator.generate(system=self._system, prompt=prompt)
        return extract_sql(reply)

    def _validate_with_retry(self, question: str, sql_text: str) -> SafeQuery:
        """Validate, and on rejection feed the error back once."""
        try:
            return validate_sql(sql_text, self._policy)
        except UnsafeQueryError as exc:
            rejection = str(exc)
            logger.info("first SQL attempt rejected: %s", rejection)

        repair_prompt = build_repair_prompt(question, sql_text, rejection)
        reply = self._generator.generate(system=self._system, prompt=repair_prompt)
        repaired = extract_sql(reply)
        if repaired is None:
            raise AnswerError("The model declined to write a query on the retry attempt")

        try:
            return validate_sql(repaired, self._policy)
        except UnsafeQueryError as second_error:
            raise AnswerError(f"The query was rejected twice: {second_error}") from second_error

    def _narrate(
        self,
        question: str,
        result: QueryResult,
        passages: Sequence[Passage],
    ) -> str:
        """Turn a result set into prose the user can read."""
        from rail_rag.rag.prompts import ANSWER_INSTRUCTIONS

        prompt = build_answer_prompt(
            question,
            result.columns,
            result.rows,
            truncated=result.truncated,
            passages=passages,
        )
        return self._generator.generate(system=ANSWER_INSTRUCTIONS, prompt=prompt)


def _sources_from(passages: Sequence[Passage]) -> tuple[Source, ...]:
    return tuple(Source(doc_id=p.doc_id, heading=p.heading) for p in passages)
