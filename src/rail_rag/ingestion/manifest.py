"""Gold export coverage manifest reader and load-integrity gate.

Validates `_manifest/coverage.json` before ingestion to prevent loading
truncated exports into the database.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rail_rag.db.run_log import LoadCounts
from rail_rag.ingestion.gold_source import IngestionError

#: Relative path to the coverage manifest.
MANIFEST_PATH = Path("_manifest") / "coverage.json"


class _ManifestModel(BaseModel):
    """Immutable base model ignoring extra fields for forward compatibility."""

    model_config = ConfigDict(frozen=True)


class RequestedRange(_ManifestModel):
    """Informational target date span requested for the export."""

    start: dt.date
    end: dt.date


class FactCoverage(_ManifestModel):
    """Declared coverage and row counts for `fact_stop_event`."""

    min_date_key: dt.date
    max_date_key: dt.date
    total_rows: int = Field(ge=0)
    rows_per_year: dict[str, int]

    @model_validator(mode="after")
    def _check_internal_consistency(self) -> Self:
        """Validates date boundaries and internal row count alignment."""
        if self.max_date_key < self.min_date_key:
            raise ValueError(
                f"max_date_key {self.max_date_key} precedes min_date_key {self.min_date_key}"
            )
        counted = sum(self.rows_per_year.values())
        if counted != self.total_rows:
            raise ValueError(
                f"rows_per_year sums to {counted}, which contradicts total_rows {self.total_rows}"
            )
        return self


class CoverageManifest(_ManifestModel):
    """Schema representing complete export coverage manifest."""

    exported_at: dt.datetime
    requested_range: RequestedRange
    fact_stop_event: FactCoverage
    dimensions: dict[str, int]
    layout: dict[str, str] = Field(default_factory=dict)

    def describe(self) -> str:
        """Returns single-line summary of manifest metrics for logging."""
        fact = self.fact_stop_event
        return (
            f"exported {self.exported_at.isoformat()}, "
            f"fact {fact.min_date_key}..{fact.max_date_key} "
            f"({fact.total_rows:,} rows), "
            + ", ".join(f"{name} {count:,}" for name, count in sorted(self.dimensions.items()))
        )


def read_manifest(source_dir: Path) -> CoverageManifest:
    """Reads and validates `_manifest/coverage.json` under `source_dir`.

    Raises IngestionError if the file is missing, unreadable, or invalid.
    """
    path = source_dir / MANIFEST_PATH
    if not path.is_file():
        raise IngestionError(
            f"Missing coverage manifest: {path}. The export is incomplete or the "
            f"source directory is wrong."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IngestionError(
            f"Could not read coverage manifest {path}: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        return CoverageManifest.model_validate(payload)
    except ValidationError as exc:
        raise IngestionError(
            f"Coverage manifest {path} is invalid: {exc.error_count()} error(s); {exc}"
        ) from exc


def assert_dimension_counts(
    manifest: CoverageManifest,
    results: Mapping[str, LoadCounts],
) -> None:
    """Validates loaded dimension row counts against declared manifest totals.

    Raises IngestionError on missing dimensions or row count mismatches.
    """
    mismatches: list[str] = []
    for name, counts in results.items():
        if name not in manifest.dimensions:
            mismatches.append(
                f"{name}: read {counts.rows_read} row(s), not declared in the manifest"
            )
            continue
        declared = manifest.dimensions[name]
        if counts.rows_read != declared:
            mismatches.append(
                f"{name}: read {counts.rows_read} row(s), manifest declares {declared}"
            )
    if mismatches:
        raise IngestionError("Dimension load does not match the manifest: " + "; ".join(mismatches))
