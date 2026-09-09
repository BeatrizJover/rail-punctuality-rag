# Architecture of this service

## Two planes

The system is split into an analytic plane and a serving plane.

The analytic plane is Databricks with Unity Catalog, where the medallion
pipeline runs on a schedule and produces the Gold star schema. It is optimised
for throughput over large scans.

The serving plane is PostgreSQL, which holds a mirror of the Gold layer and
answers single questions in milliseconds. It is optimised for latency on small
result sets.

The two are connected by a file export rather than a live query. A live JDBC
connection would make every question depend on a remote cluster being awake, and
would put a network round trip on the critical path of a question a local index
can answer.

## Why questions become SQL rather than vector search

The data is a star schema of numeric measures, not a corpus of prose. A question
such as "which station had the worst punctuality in August" has an exact answer
that requires aggregation over thousands of rows. Similarity search over embedded
text cannot compute a sum, and returning the passages most similar to the
question would produce something that reads like an answer without being one.

Questions about the data are therefore translated into SQL, executed, and
answered from the numbers.

## Why there is a knowledge base as well

Not every question is about the data. "Why do some stations have no ptcar_no"
and "how is punctuality defined here" are questions about the *system*, and no
SQL query answers them. These are answered from this documentation, retrieved by
semantic similarity.

The knowledge base also feeds the SQL generator: passages relevant to a question
are included in its context, so it writes queries informed by the modelling
decisions rather than by the column names alone.

## Guardrails on generated SQL

Generated SQL is parsed into a syntax tree and validated before it reaches a
connection. Only a single SELECT statement is permitted, only over an explicit
allow-list of tables, with a mandatory row limit and a blocked list of functions
that can read files or stall the server.

Execution adds defences that do not trust the validator: the transaction is
declared read only, a server-side statement timeout applies, and the transaction
is always rolled back rather than committed.

Staging and operational tables are excluded from the allow-list. For staging the
reason is correctness rather than security: it is an unvalidated landing zone,
and answering from it would report numbers the fact table rejected.
