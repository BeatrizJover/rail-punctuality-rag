"""Generate a small, referentially coherent Gold dataset as Parquet.

The real sample CSVs are three independent 100-row cuts: many fact keys point at
dimension rows that were never exported alongside them. That is fine for reading
tests, but it cannot exercise the loader's happy path, where every fact key must
resolve. This generator builds the four tables together, so every ``station_key``
and ``relation_key`` in the fact exists in its dimension.

It deliberately reproduces the shapes the contracts have to tolerate:
  * fact rows with a null ``planned_hour`` / ``delay_arr_s`` / ``punctual_arrivals``
    but a present ``delay_dep_s`` - an arrival that was never measured;
  * a relation row with a null ``relation_direction`` - the upstream ``concat_ws``
    case;
  * the unresolved station ``-1``, exported with no fact rows, as upstream does today;
  * two stop points sharing one station, and a station no stop point uses;
  * a resolved station with no type and no coordinates.

It also writes ``_manifest/coverage.json``, as the lakehouse export does, so the
loader's coverage gate accepts the sample. The generator refuses to write into a
partitioned export, where its files would shadow or contradict the real data.

Determinism: a fixed seed and a fixed ``exported_at``, so the output is byte-stable
across runs and safe to use as a committed fixture. Usage::

    python scripts/generate_sample_gold.py --out-dir data/sample_gold --stations 20 --days 3
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import logging
import random
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from rail_rag.core.logging import configure_logging
from rail_rag.db.models import UNRESOLVED_STATION_KEY
from rail_rag.ingestion.fact_source import FACT_DIRNAME
from rail_rag.ingestion.manifest import (
    CONTRACT_VERSION,
    MANIFEST_PATH,
    CoverageManifest,
    FactCoverage,
    RequestedRange,
)

logger = logging.getLogger("rail_rag.generate_sample_gold")

_SEED = 20260826
_OPERATORS = ("SNCB/NMBS", "SNCF", "DB", "NS")
_RELATION_NAMES = ("IC 01", "IC 09", "S53", "S52", "THAL", "ICE", "P")
_STATION_TYPES = ("Station", "Stop in open track", "Service installation")
_PASSENGER_TYPES = frozenset({"Station", "Stop in open track"})
_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_DAY_NAMES = ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")


def _md5(text: str) -> str:
    """MD5 hex digest, matching the upstream surrogate-key convention."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()  # noqa: S324 - mirrors upstream, not security


def _build_dim_date(days: list[dt.date]) -> pa.Table:
    """Calendar rows for exactly the service dates the fact will reference."""
    rows = []
    for d in days:
        # Spark's dayofweek(): 1 = Sunday .. 7 = Saturday.
        spark_dow = (d.weekday() + 1) % 7 + 1
        rows.append(
            {
                "date_key": d,
                "year": d.year,
                "quarter": (d.month - 1) // 3 + 1,
                "month": d.month,
                "month_name": _MONTHS[d.month - 1],
                "week_of_year": d.isocalendar().week,
                "day_of_week": spark_dow,
                "day_name": _DAY_NAMES[spark_dow - 1],
                "is_weekend": spark_dow in (1, 7),
            }
        )
    return pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("date_key", pa.date32()),
                ("year", pa.int16()),
                ("quarter", pa.int16()),
                ("month", pa.int16()),
                ("month_name", pa.string()),
                ("week_of_year", pa.int16()),
                ("day_of_week", pa.int16()),
                ("day_name", pa.string()),
                ("is_weekend", pa.bool_()),
            ]
        ),
    )


def _build_dim_station(rng: random.Random, count: int, days: list[dt.date]) -> pa.Table:
    """``count`` stations keyed by their PTCAR_NO, plus the unresolved row ``-1``."""
    rows: list[dict[str, Any]] = [
        {
            "station_key": UNRESOLVED_STATION_KEY,
            "station_name": "Unresolved",
            "ptcar_no": None,
            "first_seen": None,
            "last_seen": None,
            "station_type": None,
            "is_passenger": False,
            "latitude": None,
            "longitude": None,
        }
    ]
    for i in range(count):
        station_type = _STATION_TYPES[i % len(_STATION_TYPES)]
        # The second station mirrors the resolved rows upstream has no reference data for.
        located = i != 1
        rows.append(
            {
                "station_key": 100 + i,
                "station_name": f"STATION-{i:03d}",
                "ptcar_no": 100 + i,
                # Alias-derived upstream, so the span is wider than the fact window.
                "first_seen": days[0] - dt.timedelta(days=rng.randint(30, 900)),
                "last_seen": days[-1],
                "station_type": station_type if located else None,
                "is_passenger": station_type in _PASSENGER_TYPES,
                "latitude": round(rng.uniform(49.5, 51.5), 6) if located else None,
                "longitude": round(rng.uniform(2.5, 6.4), 6) if located else None,
            }
        )
    return pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("station_key", pa.int32()),
                ("station_name", pa.string()),
                ("ptcar_no", pa.int32()),
                ("first_seen", pa.date32()),
                ("last_seen", pa.date32()),
                ("station_type", pa.string()),
                ("is_passenger", pa.bool_()),
                ("latitude", pa.float64()),
                ("longitude", pa.float64()),
            ]
        ),
    )


def _stop_points(station_keys: list[int]) -> list[tuple[str, int]]:
    """One stop point per resolved station, except the last, which shares the first's PTCAR."""
    resolved = [key for key in station_keys if key != UNRESOLVED_STATION_KEY]
    stations = resolved[:-1] + resolved[:1] if len(resolved) > 1 else resolved
    return [(_md5(f"STOP POINT {i:03d}"), key) for i, key in enumerate(stations)]


def _build_dim_relation(rng: random.Random) -> pa.Table:
    """One relation per name; the last drops its direction to hit the null case."""
    rows = []
    for i, name in enumerate(_RELATION_NAMES):
        operator = rng.choice(_OPERATORS)
        # Last relation: null direction, so concat_ws hashes name|operator only.
        direction = None if i == len(_RELATION_NAMES) - 1 else f"{name}: A -> B"
        key_input = "|".join(part for part in (name, direction, operator) if part is not None)
        rows.append(
            {
                "relation_key": _md5(key_input),
                "relation": name,
                "relation_direction": direction,
                "operator": operator,
            }
        )
    return pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("relation_key", pa.string()),
                ("relation", pa.string()),
                ("relation_direction", pa.string()),
                ("operator", pa.string()),
            ]
        ),
    )


def _build_fact(
    rng: random.Random,
    days: list[dt.date],
    stop_points: list[tuple[str, int]],
    relation_keys: list[str],
) -> pa.Table:
    """Fact rows that reference only existing dimension keys.

    One row per stop point per day, so the natural key never collides within a
    batch. Roughly 1 in 6 rows models an unmeasured arrival: null
    ``planned_hour`` / ``delay_arr_s`` / ``punctual_arrivals`` with a present
    ``delay_dep_s``.
    """
    rows = []
    for d in days:
        for stop_point_key, station_key in stop_points:
            train_no = rng.randint(1, 9999)
            relation_key = rng.choice(relation_keys)
            unmeasured = rng.random() < 1 / 6
            if unmeasured:
                planned_hour = None
                delay_arr_s = None
                punctual_arrivals = None
            else:
                planned_hour = rng.randint(0, 23)
                delay_arr_s = rng.randint(-120, 3600)
                punctual_arrivals = 1 if delay_arr_s < 360 else 0
            rows.append(
                {
                    "date_key": d,
                    "station_key": station_key,
                    "stop_point_key": stop_point_key,
                    "relation_key": relation_key,
                    "train_no": train_no,
                    "planned_hour": planned_hour,
                    "delay_arr_s": delay_arr_s,
                    "delay_dep_s": rng.randint(-120, 3600),
                    "dwell_delta_s": rng.randint(0, 300),
                    "punctual_arrivals": punctual_arrivals,
                    "stop_events": 1,
                }
            )
    return pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("date_key", pa.date32()),
                ("station_key", pa.int32()),
                ("stop_point_key", pa.string()),
                ("relation_key", pa.string()),
                ("train_no", pa.int32()),
                ("planned_hour", pa.int16()),
                ("delay_arr_s", pa.int32()),
                ("delay_dep_s", pa.int32()),
                ("dwell_delta_s", pa.int32()),
                ("punctual_arrivals", pa.int16()),
                ("stop_events", pa.int16()),
            ]
        ),
    )


def _build_manifest(tables: dict[str, pa.Table], service_days: list[dt.date]) -> CoverageManifest:
    """Coverage manifest derived from the tables actually written."""
    fact = tables["fact_stop_event"]
    dates: list[dt.date] = fact.column("date_key").to_pylist()
    rows_per_year: dict[str, int] = {}
    for day in dates:
        rows_per_year[str(day.year)] = rows_per_year.get(str(day.year), 0) + 1
    return CoverageManifest(
        contract_version=CONTRACT_VERSION,
        exported_at=dt.datetime.combine(service_days[-1], dt.time(), tzinfo=dt.UTC),
        requested_range=RequestedRange(start=service_days[0], end=service_days[-1]),
        fact_stop_event=FactCoverage(
            min_date_key=min(dates),
            max_date_key=max(dates),
            total_rows=fact.num_rows,
            rows_per_year=dict(sorted(rows_per_year.items())),
        ),
        dimensions={
            name: table.num_rows for name, table in tables.items() if name.startswith("dim_")
        },
        layout={name: f"{name}.parquet" for name in tables},
    )


def generate(out_dir: Path, *, stations: int, days: int) -> dict[str, int]:
    """Write the four Gold tables and their coverage manifest to ``out_dir``. Returns row counts."""
    if (out_dir / FACT_DIRNAME).is_dir():
        raise FileExistsError(f"{out_dir} holds a partitioned export; choose another --out-dir")
    rng = random.Random(_SEED)
    end = dt.date(2026, 8, 23)
    service_days = [end - dt.timedelta(days=offset) for offset in reversed(range(days))]

    dim_date = _build_dim_date(service_days)
    dim_station = _build_dim_station(rng, stations, service_days)
    dim_relation = _build_dim_relation(rng)
    fact = _build_fact(
        rng,
        service_days,
        _stop_points(dim_station.column("station_key").to_pylist()),
        dim_relation.column("relation_key").to_pylist(),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "dim_date": dim_date,
        "dim_station": dim_station,
        "dim_relation": dim_relation,
        "fact_stop_event": fact,
    }
    for name, table in tables.items():
        pq.write_table(table, out_dir / f"{name}.parquet")
        logger.info("wrote %s (%d rows)", name, table.num_rows)

    manifest_file = out_dir / MANIFEST_PATH
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        _build_manifest(tables, service_days).model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    logger.info("wrote %s", manifest_file)
    return {name: table.num_rows for name, table in tables.items()}


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(prog="generate_sample_gold.py", description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True, help="output directory")
    parser.add_argument("--stations", type=int, default=20, help="number of stations")
    parser.add_argument("--days", type=int, default=3, help="number of service dates")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the generator from the command line."""
    args = build_parser().parse_args(argv)
    configure_logging("INFO")
    try:
        counts = generate(args.out_dir, stations=args.stations, days=args.days)
    except FileExistsError as exc:
        logger.error("%s", exc)
        return 1
    logger.info("done: %s", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
