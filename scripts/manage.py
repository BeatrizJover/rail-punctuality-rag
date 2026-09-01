"""Administrative command line entry point.
Usage:
 python scripts/manage.py db-ping
 python scripts/manage.py db-init
 python scripts/manage.py db-drop --yes
 python scripts/manage.py load-dims --source-dir DIR
 python scripts/manage.py load-fact --source-dir DIR [--date YYYY-MM-DD]
 python scripts/manage.py kb-init
 python scripts/manage.py kb-drop --yes
 python scripts/manage.py kb-build [--force]
 python scripts/manage.py kb-search --query TEXT
"""

import argparse
import datetime as dt
import logging
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path

from rail_rag.core.config import get_settings
from rail_rag.core.exceptions import RailRagError
from rail_rag.core.logging import configure_logging
from rail_rag.db.engine import create_db_engine
from rail_rag.db.schema import create_schema, drop_schema, missing_tables, ping
from rail_rag.ingestion.loader import OnViolation, load_dimensions, load_fact
from rail_rag.rag.pipeline import AnswerPipeline
from rail_rag.rag.providers.config import ModelConfig, load_model_config
from rail_rag.rag.providers.factory import build_embedder, build_generator
from rail_rag.rag.sql.policy import load_retrieval_config
from rail_rag.rag.store.builder import build_knowledge_base
from rail_rag.rag.store.models import KbSchema, build_kb_schema
from rail_rag.rag.store.repository import kb_stats
from rail_rag.rag.store.retriever import DEFAULT_TOP_K, Retriever
from rail_rag.rag.store.schema import create_kb_schema, drop_kb_schema, stored_dimension

logger = logging.getLogger("rail_rag.manage")

EXIT_OK = 0
EXIT_ERROR = 1

DEFAULT_CORPUS_DIR = Path("docs/knowledge")

#: Enough of a passage to recognise it in the terminal without flooding it.
_PREVIEW_CHARS = 160


def _cmd_db_ping() -> int:
    """Report connectivity and whether the Gold schema is complete."""
    engine = create_db_engine(get_settings())
    logger.info("Connected: %s", ping(engine))
    missing = missing_tables(engine)
    if missing:
        logger.warning("Missing tables: %s. Run 'db-init'.", ", ".join(sorted(missing)))
    else:
        logger.info("Gold schema is complete.")
    return EXIT_OK


def _cmd_db_init() -> int:
    """Create the Gold schema if it does not already exist."""
    create_schema(create_db_engine(get_settings()))
    logger.info("Done.")
    return EXIT_OK


def _cmd_db_drop(confirmed: bool, include_ops: bool) -> int:
    """Drop the Gold schema, guarded by an explicit flag."""
    if not confirmed:
        logger.error("Refusing to drop the schema without --yes.")
        return EXIT_ERROR
    drop_schema(create_db_engine(get_settings()), include_ops=include_ops)
    logger.info("Done.")
    return EXIT_OK


def _cmd_load_dims(source_dir: Path) -> int:
    """Upsert the three dimensions from a Gold export directory."""
    results = load_dimensions(create_db_engine(get_settings()), source_dir)
    for name, counts in results.items():
        logger.info("%s: %s", name, counts)
    return EXIT_OK


def _cmd_load_fact(source_dir: Path, service_date: dt.date | None, on_violation: str) -> int:
    """Stage, validate and promote the fact export."""
    engine = create_db_engine(get_settings())
    policy: OnViolation = "skip" if on_violation == "skip" else "fail"
    counts = load_fact(engine, source_dir, service_date=service_date, on_violation=policy)
    logger.info("fact_stop_event: %s", counts)
    return EXIT_OK


def _model_setup(profile: str | None) -> tuple[ModelConfig, KbSchema]:
    """Resolve the selected profile and the schema shape it implies."""
    settings = get_settings()
    config = load_model_config(settings.llm_config_path, profile=profile)
    return config, build_kb_schema(config.embedding.dimension)


def _cmd_kb_init(profile: str | None) -> int:
    """Create the knowledge-base schema at the configured embedding dimension."""
    config, kb = _model_setup(profile)
    create_kb_schema(create_db_engine(get_settings()), kb)
    logger.info(
        "Knowledge base ready (model=%s, dimension=%d)", config.embedding.model, kb.dimension
    )
    return EXIT_OK


def _cmd_kb_drop(confirmed: bool) -> int:
    """Drop the knowledge-base schema, guarded by an explicit flag."""
    if not confirmed:
        logger.error("Refusing to drop the knowledge base without --yes.")
        return EXIT_ERROR
    engine = create_db_engine(get_settings())
    width = stored_dimension(engine)
    drop_kb_schema(engine)
    logger.info("Done. Dropped vectors of dimension %s; 'kb-build' will re-embed.", width)
    return EXIT_OK


def _cmd_kb_build(corpus_dir: Path, force: bool, profile: str | None) -> int:
    """Chunk the corpus, upsert it, and embed whatever still needs a vector."""
    settings = get_settings()
    config, kb = _model_setup(profile)
    engine = create_db_engine(settings)
    report = build_knowledge_base(
        engine,
        kb,
        build_embedder(config, settings.llm_api_key),
        corpus_dir,
        max_chars=config.chunking.max_chars,
        min_chars=config.chunking.min_chars,
        batch_size=config.embedding.batch_size,
        force=force,
    )
    logger.info(
        "Corpus: %d inserted, %d updated, %d unchanged, %d deleted",
        report.sync.inserted,
        report.sync.updated,
        report.sync.unchanged,
        report.sync.deleted,
    )
    logger.info(
        "Embedded %d chunks in %d request(s) with %s",
        report.embedded,
        report.batches,
        config.embedding.model,
    )
    stats = kb_stats(engine, kb)
    logger.info("Knowledge base: %d chunks, %d embedded", stats.total, stats.embedded)
    return EXIT_OK


def _cmd_kb_search(query: str, top_k: int, profile: str | None) -> int:
    """Retrieve passages for one question, to eyeball what the model will see."""
    settings = get_settings()
    config, kb = _model_setup(profile)
    retriever = Retriever(
        create_db_engine(settings),
        kb,
        build_embedder(config, settings.llm_api_key),
        top_k=top_k,
    )
    results = retriever.retrieve(query)
    if not results:
        logger.warning("No passages retrieved. Has 'kb-build' run?")
        return EXIT_OK
    for rank, passage in enumerate(results, start=1):
        logger.info(
            "%d. [%.3f] %s / %s", rank, passage.similarity, passage.doc_id, passage.heading or "-"
        )
        logger.info("   %s", textwrap.shorten(passage.content, _PREVIEW_CHARS))
    return EXIT_OK


def _cmd_ask(question: str, show_sql: bool, profile: str | None,) -> int:
    """Answer one question using the full pipeline."""
    settings = get_settings()
    config, kb = _model_setup(profile)
    engine = create_db_engine(settings)
    retrieval = load_retrieval_config(Path("config/retrieval_config.yaml"))

    generator = build_generator(config, settings.llm_api_key)
    embedder = build_embedder(config, settings.llm_api_key)

    pipe = AnswerPipeline(engine, generator, embedder, kb, retrieval.sql)
    answer = pipe.answer(question)

    logger.info("[%s]", answer.route.value)
    if show_sql and answer.sql:
        logger.info("SQL:\n%s", answer.sql)
    if answer.result and not answer.result.is_empty:
        header = " | ".join(answer.result.columns)
        logger.info("%s", header)
        for row in answer.result.rows:
            logger.info("%s", " | ".join("" if v is None else str(v) for v in row))
    logger.info("")
    logger.info("%s", answer.text)
    if answer.sources:
        logger.info(
            "Sources: %s",
            ", ".join(f"{s.doc_id}/{s.heading}" if s.heading else s.doc_id for s in answer.sources),
        )
    return EXIT_OK


def _service_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def _add_profile_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile", default=None, help="model profile to use (default: the active one)"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(prog="manage.py", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("db-ping", help="check the database connection and schema state")
    sub.add_parser("db-init", help="create the Gold schema")
    drop = sub.add_parser("db-drop", help="drop the Gold schema (destructive)")
    drop.add_argument("--yes", action="store_true", help="confirm the destructive operation")
    drop.add_argument(
        "--include-ops",
        action="store_true",
        help="also drop the load history in ops (kept by default)",
    )
    dims = sub.add_parser("load-dims", help="upsert the Gold dimensions")
    dims.add_argument("--source-dir", type=Path, required=True, help="Gold export directory")
    fact = sub.add_parser("load-fact", help="stage, validate and promote the Gold fact")
    fact.add_argument("--source-dir", type=Path, required=True, help="Gold export directory")
    fact.add_argument("--date", type=_service_date, default=None, help="restrict to one date_key")
    fact.add_argument(
        "--on-violation",
        choices=("fail", "skip"),
        default="fail",
        help="abort the load, or exclude and count the offending rows",
    )
    kb_init = sub.add_parser("kb-init", help="create the knowledge-base schema")
    _add_profile_option(kb_init)

    kb_drop = sub.add_parser("kb-drop", help="drop the knowledge base (destructive)")
    kb_drop.add_argument("--yes", action="store_true", help="confirm the destructive operation")
    _add_profile_option(kb_drop)

    kb_build = sub.add_parser("kb-build", help="embed the corpus into the knowledge base")
    kb_build.add_argument(
        "--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR, help="markdown corpus directory"
    )
    kb_build.add_argument(
        "--force", action="store_true", help="re-embed every chunk, ignoring the hashes"
    )
    _add_profile_option(kb_build)

    kb_search = sub.add_parser("kb-search", help="retrieve passages for one question")
    kb_search.add_argument("--query", required=True, help="the question to embed and search with")
    kb_search.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, help="how many passages to return"
    )
    _add_profile_option(kb_search)

    ask = sub.add_parser("ask", help="answer one question using the full pipeline")
    ask.add_argument("--question", required=True, help="the question to answer")
    ask.add_argument("--show-sql", action="store_true", help="print the generated SQL")
    _add_profile_option(ask)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = build_parser().parse_args(argv)
    configure_logging("DEBUG" if args.verbose else "INFO")
    try:
        if args.command == "db-ping":
            return _cmd_db_ping()
        if args.command == "db-init":
            return _cmd_db_init()
        if args.command == "load-dims":
            return _cmd_load_dims(args.source_dir)
        if args.command == "load-fact":
            return _cmd_load_fact(args.source_dir, args.date, args.on_violation)
        if args.command == "kb-init":
            return _cmd_kb_init(args.profile)
        if args.command == "kb-drop":
            return _cmd_kb_drop(confirmed=bool(args.yes))
        if args.command == "kb-build":
            return _cmd_kb_build(args.corpus_dir, bool(args.force), args.profile)
        if args.command == "kb-search":
            return _cmd_kb_search(args.query, int(args.top_k), args.profile)
        if args.command == "ask":
            return _cmd_ask(args.question, bool(args.show_sql), args.profile)
        return _cmd_db_drop(confirmed=bool(args.yes), include_ops=bool(args.include_ops))
    except RailRagError as exc:
        logger.error("%s", exc)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
