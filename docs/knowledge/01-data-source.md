# Data source

## Where the data comes from

The underlying data is Belgian railway punctuality, published as open data by
Infrabel, the infrastructure manager of the Belgian rail network, through its
OpenDataSoft API. It describes train movements past measuring points on the
network, not passenger-facing timetables.

The upstream project `rail-punctuality-lakehouse` ingests this data on
Databricks and models it into a star schema. This service reads a mirror of
that schema and answers questions about it; it performs no transformation of
its own.

## The two exports and why they differ

Infrabel publishes two exports from the same dataset family, and their
structural difference drives several design decisions.

The **D-1 daily export** carries the previous day's data and is overwritten
each morning upstream. It is the only feed that can drive a daily pipeline.
It does **not** contain `PTCAR_NO`.

The **monthly export** is historical, appended rather than overwritten, and
published with a lag. It **does** contain `PTCAR_NO`.

`PTCAR_NO` is Infrabel's official, stable numeric identifier for a point of
observation on the network. It is the natural candidate for station identity
and would have made an ideal compact integer key.

## Why station identity is a hash, not the official ID

Anchoring station identity to `PTCAR_NO` would have made daily ingestion depend
on a monthly publication cycle. A pipeline that cannot run until a monthly file
arrives is not a daily pipeline.

Station identity is therefore a deterministic surrogate: `station_key`, an MD5
hash of the normalized, accent-stripped station name. It is reproducible from
the daily feed alone.

`PTCAR_NO` was demoted from a join dependency to a nullable enrichment
attribute, left-joined in from a monthly-derived crosswalk on the same
normalized name.

## Why ptcar_no is often null

Because the daily feed does not carry it. This is expected behaviour, not a
data quality failure, and the upstream pipeline deliberately tracks it as a
coverage metric rather than as a pass/fail rule — a check on it would produce a
permanently red signal with no action attached.

Any query that filters on `ptcar_no IS NOT NULL` silently discards most of the
network. It should never be used as a join key or a filter.
