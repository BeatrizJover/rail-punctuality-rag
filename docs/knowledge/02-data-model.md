# Data model

## Medallion architecture

The upstream lakehouse follows the Bronze / Silver / Gold pattern. Raw data is
landed and rescued in Bronze with types left as strings by design. Silver types,
deduplicates and enriches it, with data quality checks as a first-class pipeline
stage. Gold models it into conformed dimensions and an additive fact table.

Only the Gold layer is mirrored into this service. Bronze and Silver are not
reachable from here and cannot be queried.

## The star schema

Three dimensions surround one fact table.

`dim_date` is a generated calendar. `dim_station` holds measuring point
identity, keyed by the MD5 surrogate described in the data source document.
`dim_relation` holds the commercial relation, its direction and the operator,
keyed by an MD5 of those three fields concatenated.

`fact_stop_event` is the only fact table. Its grain is **one train passing one
measuring point on one service date**, identified by the combination of
`date_key`, `station_key` and `train_no`.

## Why the measures are counts, not percentages

`fact_stop_event` stores `punctual_arrivals`, `measured_arrivals` and
`stop_events` as additive integer counts. It deliberately does not store a
punctuality percentage.

A stored percentage cannot be aggregated. Averaging the punctuality rates of
two stations gives the wrong answer whenever they saw different volumes: a stop
with three trains would count as much as one with three thousand. Storing the
numerator and denominator separately, and dividing only at query time, makes
that class of error structurally impossible rather than merely discouraged.

This is why the punctuality rate is always computed as
`SUM(punctual_arrivals) / SUM(measured_arrivals)` at whatever grain the question
requires.

## Why measured_arrivals exists separately from stop_events

`stop_events` counts observations. `measured_arrivals` counts observations where
an arrival time actually existed. They differ whenever a stop was recorded but
its arrival was not measured.

Using `stop_events` as the denominator would treat an unmeasured arrival as a
late one, understating punctuality. `measured_arrivals` excludes those rows from
the denominator automatically, with no manual filtering.

`measured_arrivals` is a derived measure. It is computed in SQL when rows are
promoted from staging into the fact table, never carried from Python, because a
value transported through application code can drift from the rule that defines
it.
