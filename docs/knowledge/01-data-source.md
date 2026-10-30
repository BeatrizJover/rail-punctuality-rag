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
observation on the network, and it is what this service uses as station
identity.

## Stations and stop points

The data distinguishes two identities.

A **stop point** is a measuring point as the feeds name it. Its key,
`stop_point_key`, is an MD5 hash of the normalized, accent-stripped name, so it
can be computed from the daily feed alone. It never changes.

A **station** is a `PTCAR_NO`. Its key, `station_key`, is that number, and
`dim_station` holds one row per `PTCAR_NO`. Each stop point is linked to its
station by an alias table maintained upstream. Usually one stop point is one
station; when two stop points share a `PTCAR_NO`, they are one station. Questions
about stations are answered at station level.

`station_name` is Infrabel's official Dutch long name for the `PTCAR_NO` when the
reference data has it, otherwise the most recent name seen in the feeds.

## How a stop point gets its station

The daily feed does not carry `PTCAR_NO`, so the daily pipeline cannot wait for
it. Upstream resolves each stop point's `PTCAR_NO` from the monthly export, or
from Infrabel's operational-points reference when the name matches exactly one
point. A stop point that neither source resolves yet is assigned to the
unresolved station until it does.

When a stop point is resolved late or corrected, upstream re-points its rows to
the right station. A `station_key` can therefore change for rows already loaded;
a `stop_point_key` cannot.

## The unresolved station

`station_key = -1`, named `Unresolved`, collects the stop points that have no
`PTCAR_NO` yet. It is always present in `dim_station`, and it is the only row
where `ptcar_no`, the dates, the type and the coordinates are NULL. It currently
carries no rows of the fact.

Its stop events are real and count in network totals. It is not a place, though,
so it is excluded from per-station rankings and comparisons.
