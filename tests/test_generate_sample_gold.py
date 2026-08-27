"""Tests for the synthetic Gold generator.

The generator's reason to exist is referential coherence, so that is what is pinned
hardest: every fact key must resolve against a dimension. Determinism and the
deliberately-included edge shapes are checked too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq

from rail_rag.ingestion.gold_source import (
    read_dim_relation,
    read_dim_station,
    read_fact_stop_event,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from generate_sample_gold import generate  # noqa: E402


def test_generated_fact_keys_all_resolve(tmp_path: Path) -> None:
    """Every station_key and relation_key in the fact must exist in its dimension."""
    generate(tmp_path, stations=10, days=2)
    station_keys = {r.station_key for r in read_dim_station(tmp_path / "dim_station.parquet")}
    relation_keys = {r.relation_key for r in read_dim_relation(tmp_path / "dim_relation.parquet")}
    fact = list(read_fact_stop_event(tmp_path / "fact_stop_event.parquet"))

    assert fact, "generator produced no fact rows"
    assert all(r.station_key in station_keys for r in fact)
    assert all(r.relation_key in relation_keys for r in fact)


def test_generated_natural_key_is_unique(tmp_path: Path) -> None:
    """(date_key, station_key, train_no) must not collide, matching the fact's index."""
    generate(tmp_path, stations=10, days=3)
    fact = list(read_fact_stop_event(tmp_path / "fact_stop_event.parquet"))
    keys = [(r.date_key, r.station_key, r.train_no) for r in fact]
    assert len(keys) == len(set(keys))


def test_generation_is_deterministic(tmp_path: Path) -> None:
    """A fixed seed makes the output byte-stable, so it is safe as a committed fixture."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    generate(a, stations=8, days=2)
    generate(b, stations=8, days=2)
    for name in ("dim_date", "dim_station", "dim_relation", "fact_stop_event"):
        assert (a / f"{name}.parquet").read_bytes() == (b / f"{name}.parquet").read_bytes()


def test_generated_data_includes_null_ptcar_no(tmp_path: Path) -> None:
    generate(tmp_path, stations=10, days=1)
    stations = list(read_dim_station(tmp_path / "dim_station.parquet"))
    assert any(s.ptcar_no is None for s in stations)


def test_generated_data_includes_null_relation_direction(tmp_path: Path) -> None:
    generate(tmp_path, stations=5, days=1)
    relations = list(read_dim_relation(tmp_path / "dim_relation.parquet"))
    assert any(r.relation_direction is None for r in relations)


def test_generated_data_includes_unmeasured_arrival(tmp_path: Path) -> None:
    """Some fact rows must carry a null punctual_arrivals with a present delay_dep_s."""
    generate(tmp_path, stations=30, days=3)
    fact = list(read_fact_stop_event(tmp_path / "fact_stop_event.parquet"))
    assert any(r.punctual_arrivals is None and r.delay_dep_s is not None for r in fact)


def test_generated_files_are_valid_parquet(tmp_path: Path) -> None:
    counts = generate(tmp_path, stations=6, days=2)
    for name, expected in counts.items():
        assert pq.read_table(tmp_path / f"{name}.parquet").num_rows == expected
