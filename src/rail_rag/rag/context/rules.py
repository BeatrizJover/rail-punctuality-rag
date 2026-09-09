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
  service date, identified by (date_key, station_key, train_no).
- Counting rows is therefore counting stop events, not counting trains. Use
  COUNT(DISTINCT train_no) when the question is about trains.

KEYS AND NAMES
- station_key and relation_key are MD5 hashes and are meaningless to a reader.
  Always join to dim_station or dim_relation and return the human-readable name.
- ptcar_no is nullable by design: the daily feed does not carry it. Never filter
  on ptcar_no IS NOT NULL, as that silently drops most of the network.

THE CALENDAR IS NOT THE DATA
- dim_date is a generated calendar covering many years. fact_stop_event covers
  far less. Never infer the available period from dim_date.
- Do not drive a report from dim_date with a LEFT JOIN to the fact: dates with no
  data return NULL, which reads as zero punctuality rather than as absent data.
  Join from the fact outwards instead.
"""
