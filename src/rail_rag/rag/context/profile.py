"""What the data actually contains, measured rather than assumed.

A text-to-SQL system will happily answer "how did punctuality evolve between 2020
and 2024" with valid SQL and an empty result set, which reads as an answer and is
not one. The calendar dimension spans years the fact table has never seen, so the
gap is invisible from the schema alone.

Measuring the real window once at startup and putting it in the prompt turns a
silent wrong answer into an explicit "there is no data for that period". The same
query also samples column values, because a model that has never seen
"Bruxelles-Midi" will guess "Brussels Central" and match nothing.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

from sqlalchemy import Engine, func, select
from sqlalchemy.exc import SQLAlchemyError

from rail_rag.core.exceptions import DatabaseError
from rail_rag.db.models import dim_date, dim_relation, dim_station, fact_stop_event

logger = logging.getLogger(__name__)

#: Enough values to teach the model the spelling conventions, few enough to stay cheap.
DEFAULT_SAMPLE_SIZE = 12


@dataclass(frozen=True)
class DataProfile:
    """The measured state of the Gold tables at a point in time."""

    fact_rows: int
    fact_min_date: dt.date | None
    fact_max_date: dt.date | None
    fact_distinct_days: int
    fact_distinct_stations: int
    calendar_min_date: dt.date | None
    calendar_max_date: dt.date | None
    sample_stations: tuple[str, ...]
    sample_operators: tuple[str, ...]
    sample_relations: tuple[str, ...]

    @property
    def has_data(self) -> bool:
        """False when the fact table is empty, which no query can work around."""
        return self.fact_rows > 0

    def covers(self, day: dt.date) -> bool:
        """True when ``day`` falls inside the measured window."""
        if self.fact_min_date is None or self.fact_max_date is None:
            return False
        return self.fact_min_date <= day <= self.fact_max_date

    def overlaps(self, start: dt.date, end: dt.date) -> bool:
        """True when a requested range intersects the measured window at all.

        Intersection rather than containment: a question spanning a wider period
        than the data can still be answered, as long as the answer says which
        part of the range it actually covers.
        """
        if self.fact_min_date is None or self.fact_max_date is None:
            return False
        return start <= self.fact_max_date and end >= self.fact_min_date

    def render(self) -> str:
        """Return the coverage block as it appears in the prompt."""
        if not self.has_data:
            return (
                "DATA COVERAGE\n"
                "- fact_stop_event is EMPTY. No question about the data can be answered.\n"
                "  Say so plainly instead of running a query."
            )

        lines = [
            "DATA COVERAGE (measured at startup, not inferred from the calendar)",
            f"- fact_stop_event: {self.fact_rows:,} rows,"
            f" {self.fact_distinct_days} distinct dates,"
            f" {self.fact_distinct_stations} measuring points",
            f"- Data available from {self.fact_min_date} to {self.fact_max_date}",
        ]
        if self.calendar_min_date and self.calendar_max_date:
            lines.append(
                f"- dim_date calendar spans {self.calendar_min_date}"
                f" to {self.calendar_max_date}, which is WIDER than the data"
            )
        lines.append(
            "- A question about any period outside"
            f" {self.fact_min_date}..{self.fact_max_date} has no data behind it."
            " Say that explicitly rather than returning an empty result."
        )

        samples = [
            ("station_name", self.sample_stations),
            ("operator", self.sample_operators),
            ("relation", self.sample_relations),
        ]
        present = [(label, values) for label, values in samples if values]
        if present:
            lines.append("")
            lines.append("KNOWN VALUES (a sample, not the full list)")
            lines.extend(f"- {label}: {', '.join(values)}" for label, values in present)
        return "\n".join(lines)


def load_profile(engine: Engine, *, sample_size: int = DEFAULT_SAMPLE_SIZE) -> DataProfile:
    """Measure the Gold tables.

    Raises:
        DatabaseError: if the database is unreachable or the query fails.
    """
    try:
        with engine.connect() as conn:
            fact_row = conn.execute(
                select(
                    func.count().label("rows"),
                    func.min(fact_stop_event.c.date_key),
                    func.max(fact_stop_event.c.date_key),
                    func.count(func.distinct(fact_stop_event.c.date_key)),
                    func.count(func.distinct(fact_stop_event.c.station_key)),
                )
            ).one()
            calendar_row = conn.execute(
                select(func.min(dim_date.c.date_key), func.max(dim_date.c.date_key))
            ).one()
            stations = conn.execute(
                select(dim_station.c.station_name)
                .where(dim_station.c.station_name.is_not(None))
                .order_by(dim_station.c.observed_stop_events.desc().nulls_last())
                .limit(sample_size)
            ).scalars()
            operators = conn.execute(
                select(dim_relation.c.operator)
                .where(dim_relation.c.operator.is_not(None))
                .distinct()
                .limit(sample_size)
            ).scalars()
            relations = conn.execute(
                select(dim_relation.c.relation)
                .where(dim_relation.c.relation.is_not(None))
                .distinct()
                .limit(sample_size)
            ).scalars()
            profile = DataProfile(
                fact_rows=int(fact_row[0]),
                fact_min_date=fact_row[1],
                fact_max_date=fact_row[2],
                fact_distinct_days=int(fact_row[3]),
                fact_distinct_stations=int(fact_row[4]),
                calendar_min_date=calendar_row[0],
                calendar_max_date=calendar_row[1],
                sample_stations=tuple(stations),
                sample_operators=tuple(operators),
                sample_relations=tuple(relations),
            )
    except SQLAlchemyError as exc:
        raise DatabaseError(f"Could not profile the Gold tables: {type(exc).__name__}") from exc

    logger.info(
        "profiled Gold: %d fact rows covering %s..%s",
        profile.fact_rows,
        profile.fact_min_date,
        profile.fact_max_date,
    )
    return profile
