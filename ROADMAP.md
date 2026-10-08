# Roadmap

The v0.1 scope is closed: the system runs end to end locally on three years of real data. This file tracks what comes next. The evaluation and monitoring layer (tracing, SQL quality signals, golden set, runner and report) is built; what remains is publishing a baseline and the performance work it will measure.

**Status:** ✅ Done · 🔧 In progress · ⏳ Pending · 📋 Planned · 💡 Idea

## P0 — Portfolio readiness

| # | Item | Status |
| --- | --- | :---: |
| P0-1 | Synthetic quickstart writes its coverage manifest, with a test that loads it end to end | ✅ |
| P0-2 | CI on GitHub Actions: `ruff`, `mypy --strict`, `pytest` against pgvector Postgres; a missing database fails the job | ✅ |
| P0-3 | Dependency cleanup and a complete CLI usage docstring | ✅ |
| P0-4 | README with real example outputs: a data question with `--show-sql`, a conceptual question with glossary sources, and a `curl /ask` | ⏳ |

## P1 — Quick wins

| # | Item | Status |
| --- | --- | :---: |
| P1-1 | Pin the versions of the gate tools (`ruff`, `mypy`) so local and CI results match | ⏳ |
| P1-2 | General application logging when the API starts. The trace sink is already configured at API startup; the API's own `INFO` logs are still not shown, and `RAIL_RAG_LOG_LEVEL` is declared in the settings but read by nothing | ⏳ |
| P1-3 | **Slow startup** (5–7 min on 61M rows). Measured: the startup trace puts about 99.98% of pipeline construction in the `profile` span (402 s in one run). The cause is `load_profile` running `count(*)` and two `count(DISTINCT …)` aggregates over the whole fact table. Fix: take each figure from a cheap source (index `min`/`max` for the coverage window, catalogue statistics for approximate counts, the load audit for row counts) and keep full-table distincts off the startup path | ⏳ |
| P1-4 | **Slow aggregations.** Measured: a filter through `dim_date` gives the planner no predicate on the partition key, so a one-month question plans 36 of 36 monthly partitions against 1 of 36 for the same month as a literal `f.date_key` range (estimated cost about 1.50M against 46.8K). The `fact_without_partition_filter` lint flags it. Pruning alone is not enough for year-level questions: a year still plans 12 of 36 partitions and is only about 2.3× cheaper by estimated cost, so daily rollups by station and by relation are needed | ⏳ |
| P1-5 | Minimum-volume rule for ratio rankings, next to the punctuality formula in the semantic rules | ⏳ |
| P1-6 | Detect truncated model responses (`finish_reason`) instead of treating them as format errors | ⏳ |
| P1-7 | Log the provider's real error message, with a regression test | ⏳ |
| P1-8 | Disable the Gemini SDK's automatic function calling on tool-less calls | ⏳ |
| P1-9 | Measure the similarity floor against out-of-domain questions | ⏳ |
| P1-10 | Update the SQL and conceptual instructions now that the corpus includes an external source | ⏳ |
| P1-11 | Router: the data marker `count` sends conceptual questions down the data path (for example "What delay threshold makes a train count as punctual?"). Fix after the baseline, so the change is measured | ⏳ |
| P1-12 | Chore: remove `config/logging_config.yaml`, an unused stub (logging is configured in code) | ⏳ |

## P2 — Visible improvements

| # | Item | Status |
| --- | --- | :---: |
| P2-1 | **Minimal web frontend** over `/ask`: question, answer, route, generated SQL, result table and sources (Streamlit or a static page; decision open) | 📋 |
| P2-2 | **Evaluation set:** questions with expected answers and a per-route, per-family report, run as a regression check. The golden set (24 cases), the comparison rules, the budgeted runner and the report are done; publishing the baseline is pending (E1-6) | 🔧 |
| P2-3 | Offline demo mode, so the quickstart gives meaningful answers without an API key | 📋 |
| P2-4 | Application `Dockerfile`; Compose currently runs only PostgreSQL | 📋 |
| P2-5 | Hosted demo on free tiers (managed Postgres + API) over a reduced data window | 📋 |
| P2-6 | Batch embedding requests | 📋 |
| P2-7 | `kb-search --scope`, and similarity scores in per-scope measurements | 📋 |
| P2-8 | Diagnostic endpoint separate from `/ready` | 💡 |
| P2-9 | Move the API tests off the deprecated `TestClient` transport | ⏳ |

## P3 — Evolution

| # | Item | Status |
| --- | --- | :---: |
| P3-1 | User document uploads, after an attack-surface review | 💡 |
| P3-2 | Pluggable document parsers (geometric, layout-model based) | 💡 |
| P3-3 | Evaluate managed document-AI services | 💡 |
| P3-4 | Better segmentation of short glossary entries such as abbreviations | 💡 |
| P3-5 | More external sources: the full Network Statement and other Belgian mobility open data | 💡 |
| P3-6 | Per-source weighting on the conceptual path; an HNSW index once the corpus reaches tens of thousands of chunks | 💡 |
| P3-7 | On-demand loading of older partitions into the serving mirror | 💡 |

## Evaluation & Monitoring

The goal is to make quality and performance measurable, then improve them against a baseline instead of by feel. The free tier (20 generation requests per day) shapes the design: a full run needs at least two days.

### Stage 1 — Measure

| # | Item | Status |
| --- | --- | :---: |
| E1-1 | Per-request tracing: spans, provider calls with tokens and attempts, the executed SQL, errors, a startup trace and a rotating `logs/traces.jsonl` sink | ✅ |
| E1-2 | SQL quality signals: the `fact_without_partition_filter` lint and the `EXPLAIN` estimate (partitions scanned out of total, cost) | ✅ |
| E1-3 | Golden set (24 cases, reviewed ground truth) and result comparison: columns matched by value, precision-aware numeric matching | ✅ |
| E1-4 | Budgeted runner: rate limit, persisted daily budget, reply cache, resumable results, `narrate=False` | ✅ |
| E1-5 | `manage.py eval` (`--layer`, `--tag`, `--dry-run`, `--resume`, `--no-cache`) and the markdown/JSON report in `reports/eval/` | ✅ |
| E1-6 | **Baseline report:** run the golden set (at least two days of budget), commit the report, and record it in the README | ⏳ |

### Stage 2 — Performance, measured against the baseline

| # | Item | Status |
| --- | --- | :---: |
| E2-1 | Cheap startup profile (see P1-3) | 📋 |
| E2-2 | Postgres settings: `jit`, `work_mem`, `shared_buffers` | 📋 |
| E2-3 | Daily rollups by station and by relation (see P1-4) | 📋 |
| E2-4 | SQL rules: prefer the rollups, and always bound `f.date_key` with literal ranges | 📋 |
| E2-5 | A configurable `thinking_level` for generation, decided by evaluation (thinking tokens dominate call time) | 📋 |
| E2-6 | Lower `statement_timeout` once queries are fast | 📋 |
| E2-7 | Post-fix report, compared with the baseline | 📋 |

### Stage 3 — Deeper quality and operations

| # | Item | Status |
| --- | --- | :---: |
| E3-1 | Numeric grounding of the narration: every number in the prose must come from the result | 💡 |
| E3-2 | LLM-as-judge for conceptual answers | 💡 |
| E3-3 | Persist traces to Postgres | 💡 |
| E3-4 | Dashboard over the persisted traces | 💡 |
| E3-5 | CI regression gate on the evaluation | 💡 |
| E3-6 | Drift comparison across reports | 💡 |
| E3-7 | Multi-process safety for trace writing: the rotating file sink assumes one writer. Covered by E3-3 | 💡 |

## Upstream (lakehouse)

Tracked here because they change the contract this service consumes.

| # | Item | Status |
| --- | --- | :---: |
| U-1 | Automate the Gold export as a Databricks job task writing to a Unity Catalog volume, plus a local download runner | 📋 |
| U-2 | Range-parameterised backfill, decoupled from Auto Loader | 📋 |
| U-3 | Fix a latent `relation_key` collision: `concat_ws` skips NULLs | ⏳ |
