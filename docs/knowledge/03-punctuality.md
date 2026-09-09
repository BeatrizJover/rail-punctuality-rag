# Punctuality

## The threshold

Infrabel considers a train punctual when its arrival delay is **under 6 minutes**,
that is under 360 seconds. This is Infrabel's own operational definition, not one
chosen by this project, and it is applied in the Silver layer upstream.

The result is already encoded in the fact table: `punctual_arrivals` is 1 for a
punctual arrival and 0 for a late one. Re-deriving punctuality from `delay_arr_s`
in a query is unnecessary and risks disagreeing with the upstream definition.

## Delays are stored in seconds and may be negative

`delay_arr_s`, `delay_dep_s` and `dwell_delta_s` are all in seconds.

A negative delay is an **early** arrival or departure. Early arrivals are valid
observations and are common. Filtering them out, or treating them as data errors,
biases every result. There is deliberately no range constraint forbidding
negative values on these columns.

## What a punctuality rate means at each grain

Because the measures are additive, the same expression is correct at any level of
aggregation: per station, per relation, per hour, per day, or over the whole
table. The denominator changes; the formula does not.

A rate computed over a single stop event is either 0 or 1 and is not meaningful.
Punctuality is a property of a population of trains, not of an individual one.

## Counting trains versus counting stop events

Counting rows in `fact_stop_event` counts stop events, not trains. One train
passing six measuring points contributes six rows.

A question about how many trains ran needs `COUNT(DISTINCT train_no)`. A question
about how many observations exist needs a row count. Confusing the two inflates
train counts by roughly the average number of measuring points per journey.
