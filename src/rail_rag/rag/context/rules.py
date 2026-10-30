"""Semantic rules the SQL generator must always see.

These are injected into every prompt rather than retrieved from the knowledge
base. Retrieval is best-effort by nature: a question phrased unusually may not
surface the chunk that defines the punctuality ratio, and the model would then
invent an average of percentages. An invariant that must never be missed cannot
depend on a similarity score.

The knowledge base in the next stage carries the *explanatory* material — why
``ptcar_no`` is missing, how the surrogate keys were derived. That is genuinely
optional context. This is not.
"""

from __future__ import annotations

#: Kept as prose rather than structured data: it is prompt text, not configuration.
SEMANTIC_RULES = """\
MEASURES AND RATIOS
- Punctuality rate is always SUM(punctual_arrivals)::float / NULLIF(SUM(measured_arrivals), 0).
  Never AVG(punctual_arrivals) and never an average of per-row percentages: averaging
  ratios weights a station with 3 trains the same as one with 3000.
- measured_arrivals is 0 when no arrival was observed, so summing it excludes
  unmeasured events from the denominator automatically. Do not filter them by hand.
- stop_events counts observations; measured_arrivals counts observations where an
  arrival time existed. They are not interchangeable.

PUNCTUALITY DEFINITION
- Infrabel considers a train punctual when its arrival delay is under 6 minutes
  (360 seconds). punctual_arrivals already encodes this: 1 punctual, 0 late.
- delay_arr_s and delay_dep_s are in seconds. Negative values are early arrivals,
  which are valid data and must not be filtered out or treated as errors.

GRAIN
- One row of fact_stop_event is one train passing one measuring point on one
  service date, identified by (date_key, stop_point_key, train_no).
- Counting rows is therefore counting stop events, not counting trains. Use
  COUNT(DISTINCT train_no) when the question is about trains.
- A station (station_key) can hold several measuring points (stop_point_key).
  Questions about stations group by station_key, never by stop_point_key.

KEYS AND NAMES
- station_key is Infrabel's PTCAR_NO, an integer; stop_point_key and relation_key
  are MD5 hashes. None of them means anything to a reader: always join to
  dim_station or dim_relation and return the human-readable name.
- station_key -1 is 'Unresolved': measuring points not yet matched to a station.
  Count it in network totals, but exclude it (station_key <> -1) from per-station
  rankings and comparisons.

DAILY ROLLUPS
- agg_station_day (date_key, station_key) and agg_relation_day (date_key, relation_key)
  hold exact daily sums of fact_stop_event: row_count, stop_events, measured_arrivals
  and punctual_arrivals. Network totals are sums over all of agg_station_day,
  station_key -1 included.
- For punctuality rates and for stop_events or measured_arrivals counts, by station,
  relation or the whole network, at day, month or year level, query a rollup. Join
  dim_station or dim_relation only for names and labels, and to filter by name.
- Query fact_stop_event only for what the rollups cannot answer: train-level detail,
  delays and distinct counts such as COUNT(DISTINCT train_no).
- Stop events are SUM(stop_events) on a rollup. row_count is the number of fact rows,
  which COUNT(*) gives on the fact.

PERIODS
- Always filter a period with a literal half-open range on date_key, on the fact and
  on the rollups, for example
  date_key >= DATE '2025-01-01' AND date_key < DATE '2026-01-01'.
- Use dim_date only to output calendar attributes such as year, month or month_name,
  never to choose the period.

THE CALENDAR IS NOT THE DATA
- dim_date is a generated calendar covering many years. fact_stop_event covers
  far less. Never infer the available period from dim_date.
- Do not drive a report from dim_date with a LEFT JOIN to the fact: dates with no
  data return NULL, which reads as zero punctuality rather than as absent data.
  Join from the fact outwards instead.
"""
