"""Tests for the Gold export coverage manifest gate."""

from __future__ import annotations

import datetime as dt
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from rail_rag.db.run_log import LoadCounts
from rail_rag.ingestion.gold_source import IngestionError
from rail_rag.ingestion.manifest import (
    CoverageManifest,
    assert_dimension_counts,
    read_manifest,
)

# Copy of a real export manifest; gold_export/ is gitignored.
_REAL_PAYLOAD: dict[str, Any] = {
    "exported_at": "2026-09-09T05:50:19.018546+00:00",
    "requested_range": {"start": "2024-01-01", "end": "2026-12-31"},
    "fact_stop_event": {
        "min_date_key": "2024-01-01",
        "max_date_key": "2026-09-07",
        "total_rows": 60950122,
        "rows_per_year": {"2024": 23114440, "2025": 22426978, "2026": 15408704},
    },
    "dimensions": {"dim_date": 5113, "dim_station": 701, "dim_relation": 566},
    "layout": {
        "fact_stop_event": "fact_stop_event/date_key=YYYY-MM-DD/*.parquet",
        "dim_date": "dim_date/*.parquet",
        "dim_station": "dim_station/*.parquet",
        "dim_relation": "dim_relation/*.parquet",
    },
}


def _write_manifest(source_dir: Path, payload: Any, *, raw: str | None = None) -> Path:
    """Write a manifest into an export directory and return the export root."""
    manifest_dir = source_dir / "_manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    text = raw if raw is not None else json.dumps(payload)
    (manifest_dir / "coverage.json").write_text(text, encoding="utf-8")
    return source_dir


def _payload_without(key: str, *, section: str | None = None) -> dict[str, Any]:
    """Return the real payload with one key removed."""
    payload = deepcopy(_REAL_PAYLOAD)
    target = payload if section is None else payload[section]
    del target[key]
    return payload


def test_a_real_manifest_parses_into_typed_values(tmp_path: Path) -> None:
    manifest = read_manifest(_write_manifest(tmp_path / "gold_export", _REAL_PAYLOAD))

    assert manifest.fact_stop_event.min_date_key == dt.date(2024, 1, 1)
    assert manifest.fact_stop_event.max_date_key == dt.date(2026, 9, 7)
    assert manifest.fact_stop_event.total_rows == 60950122
    assert manifest.dimensions == {"dim_date": 5113, "dim_station": 701, "dim_relation": 566}
    assert manifest.requested_range.end == dt.date(2026, 12, 31)


def test_the_declared_span_is_narrower_than_the_requested_one(tmp_path: Path) -> None:
    """Declared span stops before the requested end."""
    manifest = read_manifest(_write_manifest(tmp_path / "gold_export", _REAL_PAYLOAD))

    assert manifest.fact_stop_event.max_date_key < manifest.requested_range.end


def test_an_unknown_field_is_ignored(tmp_path: Path) -> None:
    """A new producer key must not break loading."""
    payload = deepcopy(_REAL_PAYLOAD)
    payload["exported_by"] = "databricks-job-42"

    manifest = read_manifest(_write_manifest(tmp_path / "gold_export", payload))

    assert manifest.fact_stop_event.total_rows == 60950122


def test_the_description_names_the_span_and_the_volume(tmp_path: Path) -> None:
    manifest = read_manifest(_write_manifest(tmp_path / "gold_export", _REAL_PAYLOAD))

    description = manifest.describe()

    assert "2024-01-01..2026-09-07" in description
    assert "60,950,122" in description
    assert "dim_station 701" in description


def test_a_missing_manifest_is_a_loud_error(tmp_path: Path) -> None:
    source_dir = tmp_path / "gold_export"
    source_dir.mkdir()

    with pytest.raises(IngestionError, match="Missing coverage manifest"):
        read_manifest(source_dir)


def test_a_missing_source_directory_is_a_loud_error(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="Missing coverage manifest"):
        read_manifest(tmp_path / "does_not_exist")


def test_corrupt_json_is_a_loud_error(tmp_path: Path) -> None:
    source_dir = _write_manifest(tmp_path / "gold_export", None, raw="{not json,")

    with pytest.raises(IngestionError, match="Could not read coverage manifest"):
        read_manifest(source_dir)


@pytest.mark.parametrize(
    ("key", "section"),
    [
        ("total_rows", "fact_stop_event"),
        ("max_date_key", "fact_stop_event"),
        ("dimensions", None),
        ("exported_at", None),
    ],
)
def test_a_missing_required_field_is_a_loud_error(
    tmp_path: Path, key: str, section: str | None
) -> None:
    source_dir = _write_manifest(tmp_path / "gold_export", _payload_without(key, section=section))

    with pytest.raises(IngestionError, match="is invalid"):
        read_manifest(source_dir)


def test_rows_per_year_must_sum_to_total_rows(tmp_path: Path) -> None:
    payload = deepcopy(_REAL_PAYLOAD)
    payload["fact_stop_event"]["rows_per_year"]["2026"] = 1

    source_dir = _write_manifest(tmp_path / "gold_export", payload)

    with pytest.raises(IngestionError, match="contradicts total_rows"):
        read_manifest(source_dir)


def test_an_inverted_date_span_is_a_loud_error(tmp_path: Path) -> None:
    payload = deepcopy(_REAL_PAYLOAD)
    payload["fact_stop_event"]["max_date_key"] = "2023-12-31"

    source_dir = _write_manifest(tmp_path / "gold_export", payload)

    with pytest.raises(IngestionError, match="precedes min_date_key"):
        read_manifest(source_dir)


def test_a_manifest_is_immutable(tmp_path: Path) -> None:
    manifest = read_manifest(_write_manifest(tmp_path / "gold_export", _REAL_PAYLOAD))

    with pytest.raises(ValueError, match="frozen"):
        manifest.dimensions = {}  # type: ignore[misc]


def _manifest() -> CoverageManifest:
    return CoverageManifest.model_validate(_REAL_PAYLOAD)


def test_dimension_counts_matching_the_manifest_pass() -> None:
    results = {
        "dim_date": LoadCounts(rows_read=5113, rows_inserted=5113),
        "dim_station": LoadCounts(rows_read=701, rows_inserted=701),
        "dim_relation": LoadCounts(rows_read=566, rows_inserted=566),
    }

    assert_dimension_counts(_manifest(), results)


def test_a_rerun_that_updates_instead_of_inserting_still_passes() -> None:
    """Only rows_read is compared, not inserted vs updated."""
    results = {"dim_station": LoadCounts(rows_read=701, rows_updated=701)}

    assert_dimension_counts(_manifest(), results)


def test_a_short_dimension_load_is_a_loud_error() -> None:
    results = {"dim_station": LoadCounts(rows_read=700, rows_inserted=700)}

    with pytest.raises(IngestionError, match="read 700 row\\(s\\), manifest declares 701"):
        assert_dimension_counts(_manifest(), results)


def test_a_dimension_absent_from_the_manifest_is_a_loud_error() -> None:
    results = {"dim_platform": LoadCounts(rows_read=12, rows_inserted=12)}

    with pytest.raises(IngestionError, match="not declared in the manifest"):
        assert_dimension_counts(_manifest(), results)


def test_every_mismatching_dimension_is_reported() -> None:
    results = {
        "dim_date": LoadCounts(rows_read=5113),
        "dim_station": LoadCounts(rows_read=1),
        "dim_relation": LoadCounts(rows_read=2),
    }

    with pytest.raises(IngestionError) as error:
        assert_dimension_counts(_manifest(), results)

    message = str(error.value)
    assert "dim_station" in message
    assert "dim_relation" in message
    assert "dim_date" not in message
