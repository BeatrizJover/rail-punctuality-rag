"""Administrative command line entry point.
Usage:
 python scripts/manage.py db-ping
 python scripts/manage.py db-init
 python scripts/manage.py db-drop --yes
 python scripts/manage.py load-dims --source-dir DIR
 python scripts/manage.py load-fact --source-dir DIR [--date YYYY-MM-DD]
"""

import argparse
import datetime as dt
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from rail_rag.core.config import get_settings
from rail_rag.core.exceptions import RailRagError
from rail_rag.core.logging import configure_logging
from rail_rag.db.engine import create_db_engine
from rail_rag.db.schema import create_schema, drop_schema, missing_tables, ping
from rail_rag.ingestion.loader import OnViolation, load_dimensions, load_fact

logger = logging.getLogger("rail_rag.manage")

EXIT_OK = 0
EXIT_ERROR = 1


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


def _service_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


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
        return _cmd_db_drop(confirmed=bool(args.yes), include_ops=bool(args.include_ops))
    except RailRagError as exc:
        logger.error("%s", exc)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
