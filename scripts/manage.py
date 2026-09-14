"""Administrative command line entry point.
Usage:
 python scripts/manage.py db-ping
 python scripts/manage.py db-init
 python scripts/manage.py db-drop --yes
 python scripts/manage.py load-dims --source-dir DIR
 python scripts/manage.py load-fact --source-dir DIR [--date D | --from A --to B]
 python scripts/manage.py finalize
 python scripts/manage.py docs-fetch --source ID [--bootstrap]
 python scripts/manage.py docs-parse --source ID
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
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from rail_rag.core.config import get_settings
from rail_rag.core.exceptions import RailRagError
from rail_rag.core.logging import configure_logging
from rail_rag.db.engine import create_db_engine
from rail_rag.db.schema import create_schema, drop_schema, missing_tables, ping
from rail_rag.ingestion.derivations import (
    assert_referential_integrity,
    derive_station_activity,
    optimize_fact_for_reads,
)
from rail_rag.ingestion.docs.canonical import (
    DEFAULT_PARSED_DIR,
    PARSER_VERSION,
    ParseManifest,
    artefact_paths,
    write_blocks,
)
from rail_rag.ingestion.docs.exceptions import ParseError
from rail_rag.ingestion.docs.fetcher import (
    DEFAULT_RAW_DIR,
    FetchManifest,
    artefact_path,
    fetch_source,
    manifest_path,
    probe_digest,
    sha256_of,
)
from rail_rag.ingestion.docs.pdf_reader import parse_document
from rail_rag.ingestion.docs.sources import load_source
from rail_rag.ingestion.fact_source import DateRange, count_fact_rows, is_partitioned
from rail_rag.ingestion.loader import OnViolation, load_dimensions, load_fact
from rail_rag.ingestion.manifest import (
    assert_dimension_counts,
    assert_fact_coverage,
    read_manifest,
)
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
DEFAULT_SOURCES_REGISTRY = Path("config/sources.yaml")

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
    manifest = read_manifest(source_dir)
    logger.info("Export manifest: %s", manifest.describe())
    results = load_dimensions(create_db_engine(get_settings()), source_dir)
    for name, counts in results.items():
        logger.info("%s: %s", name, counts)
    assert_dimension_counts(manifest, results)
    return EXIT_OK


def _cmd_load_fact(
    source_dir: Path,
    service_date: dt.date | None,
    range_start: dt.date | None,
    range_end: dt.date | None,
    on_violation: str,
) -> int:
    """Stage, validate and promote the fact export over an optional date scope."""
    if service_date is not None and (range_start is not None or range_end is not None):
        logger.error("Pass --date, or --from/--to, not both.")
        return EXIT_ERROR
    manifest = read_manifest(source_dir)
    logger.info("Export manifest: %s", manifest.describe())
    engine = create_db_engine(get_settings())
    policy: OnViolation = "skip" if on_violation == "skip" else "fail"

    scope = (
        DateRange.for_day(service_date)
        if service_date is not None
        else DateRange(range_start, range_end)
    )
    counts = load_fact(engine, source_dir, date_range=scope, on_violation=policy)
    logger.info("fact_stop_event: %s", counts)

    on_disk = (
        count_fact_rows(source_dir, scope.start, scope.end)
        if is_partitioned(source_dir)
        else counts.rows_read
    )
    assert_fact_coverage(
        manifest,
        loaded_rows=counts.rows_inserted,
        on_disk_rows=on_disk,
        range_start=scope.start,
        range_end=scope.end,
    )
    return EXIT_OK


def _cmd_finalize() -> int:
    """Close the load: derive station activity, assert integrity, optimize for reads."""
    engine = create_db_engine(get_settings())
    updated = derive_station_activity(engine)
    logger.info("Derived activity for %d station(s).", updated)
    assert_referential_integrity(engine)
    logger.info("Referential integrity holds.")
    optimize_fact_for_reads(engine)
    logger.info("Fact optimized for reads (lookup indexes + analyze).")
    return EXIT_OK


def _model_setup(profile: str | None) -> tuple[ModelConfig, KbSchema]:
    """Resolve the selected profile and the schema shape it implies."""
    settings = get_settings()
    config = load_model_config(settings.llm_config_path, profile=profile)
    return config, build_kb_schema(config.embedding.dimension)


def _cmd_docs_fetch(source_id: str, dest_dir: Path, bootstrap: bool) -> int:
    """Download one declared external document, or print the digest to pin it with."""
    source = load_source(DEFAULT_SOURCES_REGISTRY, source_id)
    if bootstrap:
        digest = probe_digest(source)
        logger.info("Pin it in %s under %s:", DEFAULT_SOURCES_REGISTRY, source.source_id)
        logger.info('    sha256: "%s"', digest)
        return EXIT_OK
    outcome = fetch_source(source, dest_dir)
    if outcome.downloaded:
        logger.info("Stored %s (%d bytes).", outcome.path, outcome.manifest.bytes)
    else:
        logger.info("Already present and verified: %s", outcome.path)
    return EXIT_OK


def _cmd_docs_parse(source_id: str, dest_dir: Path) -> int:
    """Parse one fetched document into canonical JSONL, verified against its fetch manifest."""
    source = load_source(DEFAULT_SOURCES_REGISTRY, source_id)
    artefact = artefact_path(source, DEFAULT_RAW_DIR)
    if not artefact.exists():
        raise ParseError(f"{artefact} does not exist; run 'docs-fetch --source {source_id}' first.")

    fetched = FetchManifest.model_validate_json(manifest_path(artefact).read_text(encoding="utf-8"))
    digest = sha256_of(artefact)
    if digest != fetched.sha256:
        raise ParseError(
            f"{artefact} has digest {digest}, but its fetch manifest recorded "
            f"{fetched.sha256}. Something touched the file after it was fetched."
        )

    blocks = parse_document(artefact, source)
    sections: Counter[str] = Counter(block.section for block in blocks)

    manifest = ParseManifest(
        source_id=source.source_id,
        version=source.version,
        source_sha256=digest,
        parser_version=PARSER_VERSION,
        block_count=len(blocks),
        blocks_per_section=dict(sections),
        parsed_at=dt.datetime.now(dt.UTC),
    )

    stream_path, meta_path = artefact_paths(source.source_id, source.version, dest_dir)
    written = write_blocks(stream_path, blocks)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")

    logger.info("Parsed %d block(s) into %s", written, stream_path)
    for heading, count in sections.items():
        logger.info("  %s: %d", heading, count)
    return EXIT_OK


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


def _cmd_ask(
    question: str,
    show_sql: bool,
    profile: str | None,
) -> int:
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
        "--from",
        dest="range_start",
        type=_service_date,
        default=None,
        help="range start, inclusive (with --to)",
    )
    fact.add_argument(
        "--to",
        dest="range_end",
        type=_service_date,
        default=None,
        help="range end, exclusive (with --from)",
    )
    fact.add_argument(
        "--on-violation",
        choices=("fail", "skip"),
        default="fail",
        help="abort the load, or exclude and count the offending rows",
    )
    sub.add_parser("finalize", help="derive station activity and assert referential integrity")
    docs_fetch = sub.add_parser("docs-fetch", help="download a declared external document")
    docs_fetch.add_argument(
        "--source", required=True, help="source_id declared in config/sources.yaml"
    )
    docs_fetch.add_argument(
        "--dest-dir", type=Path, default=DEFAULT_RAW_DIR, help="where the artefact is stored"
    )
    docs_fetch.add_argument(
        "--bootstrap", action="store_true", help="print the digest to pin, store nothing"
    )
    docs_parse = sub.add_parser("docs-parse", help="parse a fetched document into canonical JSONL")
    docs_parse.add_argument(
        "--source", required=True, help="source_id declared in config/sources.yaml"
    )
    docs_parse.add_argument(
        "--dest-dir",
        type=Path,
        default=DEFAULT_PARSED_DIR,
        help="where the parsed JSONL is stored",
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
            return _cmd_load_fact(
                args.source_dir, args.date, args.range_start, args.range_end, args.on_violation
            )
        if args.command == "finalize":
            return _cmd_finalize()
        if args.command == "docs-fetch":
            return _cmd_docs_fetch(args.source, args.dest_dir, bool(args.bootstrap))
        if args.command == "docs-parse":
            return _cmd_docs_parse(args.source, args.dest_dir)
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
