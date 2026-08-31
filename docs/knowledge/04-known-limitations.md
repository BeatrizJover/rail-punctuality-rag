# Known limitations

## The calendar is much wider than the data

`dim_date` is a generated calendar spanning many years. It exists so that date
attributes such as weekday, quarter and month name are available for any date.

`fact_stop_event` covers a far shorter period. The presence of a date in
`dim_date` says nothing about whether data exists for it.

A question about a period with no data produces perfectly valid SQL and an empty
result, which reads like an answer and is not one. The actual coverage window is
measured at startup and stated explicitly; questions outside it should be
answered by saying the data is not there.

Driving a report from `dim_date` with a left join to the fact is a specific trap:
dates with no data return NULL, and a NULL punctuality rate is easily read as
zero punctuality rather than as absent data.

## Surrogate keys are hashes and are not human readable

`station_key` and `relation_key` are MD5 digests. They are meaningless to a
reader and must always be resolved to `station_name`, `relation`,
`relation_direction` or `operator` by joining to the corresponding dimension.

## A latent hash collision risk upstream

`relation_key` is derived upstream with Spark's `concat_ws`, which skips NULL
values rather than preserving their position. Two distinct tuples with NULLs in
different positions can therefore produce the same concatenated string, and thus
the same MD5 digest.

This is tracked as a known defect in the upstream Gold layer. It has not been
observed to cause a collision in the current data, but it is a real risk rather
than a theoretical one.

## Station names come from the daily feed

Station identity is derived from the normalized station name. If the upstream
feed changes how it spells a station, the derived key changes with it and the
station appears as a new measuring point. Names should be treated as the values
they are in the data, not as canonical official names.
