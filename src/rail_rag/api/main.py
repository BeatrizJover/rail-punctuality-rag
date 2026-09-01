"""HTTP endpoints over the answering pipeline.

Endpoints are ``def``, not ``async def``. The whole stack below them is
synchronous — SQLAlchemy Core, the provider SDK — so a coroutine would block the
event loop for the duration of a model call. FastAPI runs ``def`` endpoints in a
worker thread instead, which is exactly the behaviour wanted here.

``/health`` and ``/ready`` are deliberately different questions. Liveness says
the process is up; readiness says the schema, the knowledge base and the vector
width all agree. An orchestrator that conflates them restarts a healthy
container because a corpus has not been built yet.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status

from rail_rag.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    ReadyResponse,
    ResultOut,
    SourceOut,
    jsonable_row,
)
from rail_rag.api.state import AppState, build_state
from rail_rag.core.exceptions import RailRagError
from rail_rag.db.schema import missing_tables
from rail_rag.rag.exceptions import AnswerError
from rail_rag.rag.pipeline import Answer
from rail_rag.rag.store.schema import kb_schema_exists, stored_dimension

logger = logging.getLogger(__name__)

_STATE_ATTR = "rag_state"
_UNPROCESSABLE = 422


def get_state(request: Request) -> AppState:
    """Return the state assembled at startup."""
    state = getattr(request.app.state, _STATE_ATTR, None)
    if not isinstance(state, AppState):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The service is still starting up.",
        )
    return state


def create_app(state: AppState | None = None) -> FastAPI:
    """Build the application, optionally with a pre-assembled state.

    Passing ``state`` skips startup construction entirely, which is how the
    tests serve a stub pipeline with no database or provider behind it.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Built once here rather than per request: see the module docstring.
        setattr(app.state, _STATE_ATTR, state if state is not None else build_state())
        yield

    app = FastAPI(
        title="Rail Punctuality RAG",
        description="Question answering over the Belgian railway punctuality Gold layer.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health(app_state: Annotated[AppState, Depends(get_state)]) -> HealthResponse:
        """Liveness. Touches nothing external, so it stays true while Postgres is down."""
        return HealthResponse(status="ok", environment=app_state.environment)

    @app.get("/ready", response_model=ReadyResponse)
    def ready(
        response: Response, app_state: Annotated[AppState, Depends(get_state)]
    ) -> ReadyResponse:
        """Readiness. Reports 503 with the reason rather than an opaque failure."""
        report = _readiness(app_state)
        if not report.ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return report

    @app.post("/ask", response_model=AskResponse)
    def ask(payload: AskRequest, app_state: Annotated[AppState, Depends(get_state)]) -> AskResponse:
        """Answer one question, choosing the conceptual or data path automatically."""
        try:
            answer = app_state.pipeline.answer(payload.question)
        except AnswerError as exc:
            # The pipeline ran and could not produce an answer: a client-visible
            # outcome, not a server fault.
            raise HTTPException(status_code=_UNPROCESSABLE, detail=str(exc)) from exc
        except RailRagError as exc:
            logger.exception("Answering failed")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
            ) from exc
        return _to_response(answer)

    return app


def _readiness(app_state: AppState) -> ReadyResponse:
    """Check every dependency, reporting the first problem in plain language."""
    try:
        missing = sorted(missing_tables(app_state.engine))
        kb_present = False if missing else kb_schema_exists(app_state.engine)
        width = stored_dimension(app_state.engine) if kb_present else None
    except RailRagError as exc:
        return ReadyResponse(
            ready=False,
            missing_tables=[],
            knowledge_base=False,
            detail=str(exc),
        )

    detail: str | None = None
    if missing:
        detail = f"Missing tables: {', '.join(missing)}. Run 'db-init'."
    elif not kb_present:
        detail = "Knowledge base is absent. Run 'kb-init' then 'kb-build'."
    elif width != app_state.kb.dimension:
        detail = (
            f"Embedding dimension mismatch: table has {width},"
            f" configuration expects {app_state.kb.dimension}."
        )

    return ReadyResponse(
        ready=detail is None,
        missing_tables=missing,
        knowledge_base=kb_present,
        embedding_dimension=width,
        detail=detail,
    )


def _to_response(answer: Answer) -> AskResponse:
    """Map the pipeline's answer onto the wire contract."""
    result: ResultOut | None = None
    if answer.result is not None:
        result = ResultOut(
            columns=list(answer.result.columns),
            rows=[jsonable_row(row) for row in answer.result.rows],
            truncated=answer.result.truncated,
        )
    return AskResponse(
        text=answer.text,
        route=answer.route.value,
        sql=answer.sql,
        result=result,
        sources=[SourceOut(doc_id=s.doc_id, heading=s.heading) for s in answer.sources],
    )


app = create_app()
