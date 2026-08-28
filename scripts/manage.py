"""Administrative command line entry point.
Usage:
 python scripts/manage.py db-ping
 python scripts/manage.py db-init
 python scripts/manage.py db-drop --yes
"""

import argparse
import logging
import sys
from collections.abc import Sequence

from rail_rag.core.config import get_settings
from rail_rag.core.exceptions import RailRagError
from rail_rag.core.logging import configure_logging
from rail_rag.db.engine import create_db_engine
from rail_rag.db.schema import create_schema, drop_schema, missing_tables, ping

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
        return _cmd_db_drop(confirmed=bool(args.yes), include_ops=bool(args.include_ops))
    except RailRagError as exc:
        logger.error("%s", exc)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
