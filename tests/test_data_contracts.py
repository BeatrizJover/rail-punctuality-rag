"""Tests for the Gold data contracts.

These pin the nullability the real export exhibits and the invariants that protect
the punctuality ratio, using minimal in-memory dicts rather than a file.
"""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from rail_rag.ingestion.data_contracts import (
    DimRelationRow,
    DimStationRow,
    FactStopEventRow,
)

_HASH = "0" * 32


def _fact(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "date_key": dt.date(2026, 8, 23),
        "station_key": 1211,
        "stop_point_key": _HASH,
        "relation_key": _HASH,
        "train_no": 42,
        "planned_hour": 8,
        "delay_arr_s": 120,
        "delay_dep_s": 130,
        "dwell_delta_s": 10,
        "punctual_arrivals": 1,
        "stop_events": 1,
    }
    base.update(overrides)
    return base


def test_fact_accepts_a_measured_arrival() -> None:
    row = FactStopEventRow.model_validate(_fact())
    assert row.punctual_arrivals == 1


def test_fact_preserves_null_punctual_arrivals() -> None:
    """A null arrival stays null: it must never be coerced to 0 (see the ratio)."""
    row = FactStopEventRow.model_validate(
        _fact(planned_hour=None, delay_arr_s=None, punctual_arrivals=None, delay_dep_s=23)
    )
    assert row.punctual_arrivals is None


def test_fact_accepts_negative_delay() -> None:
    """Negative delays are early arrivals: valid, not an error."""
    row = FactStopEventRow.model_validate(_fact(delay_arr_s=-97))
    assert row.delay_arr_s == -97


def test_fact_rejects_punctual_arrivals_out_of_range() -> None:
    with pytest.raises(ValidationError):
        FactStopEventRow.model_validate(_fact(punctual_arrivals=2))


def test_fact_rejects_planned_hour_out_of_range() -> None:
    with pytest.raises(ValidationError):
        FactStopEventRow.model_validate(_fact(planned_hour=24))


def test_fact_rejects_missing_required_key() -> None:
    payload = _fact()
    del payload["station_key"]
    with pytest.raises(ValidationError):
        FactStopEventRow.model_validate(payload)


def test_fact_rejects_short_hash_key() -> None:
    with pytest.raises(ValidationError):
        FactStopEventRow.model_validate(_fact(stop_point_key="tooshort"))


def test_fact_rejects_missing_stop_point_key() -> None:
    """stop_point_key is the fact's grain, so a row without it cannot be placed."""
    payload = _fact()
    del payload["stop_point_key"]
    with pytest.raises(ValidationError):
        FactStopEventRow.model_validate(payload)


def test_fact_accepts_the_unresolved_station() -> None:
    row = FactStopEventRow.model_validate(_fact(station_key=-1))
    assert row.station_key == -1


def test_fact_rejects_a_station_key_below_the_sentinel() -> None:
    with pytest.raises(ValidationError):
        FactStopEventRow.model_validate(_fact(station_key=-2))


def test_fact_rejects_unexpected_column() -> None:
    """extra='forbid' is the ingestion-side mirror of upstream schema-drift detection."""
    with pytest.raises(ValidationError):
        FactStopEventRow.model_validate(_fact(surprise_column="x"))


def test_measured_arrivals_is_not_part_of_the_contract() -> None:
    """It is derived in SQL on promotion, so it must not be a transportable field."""
    assert "measured_arrivals" not in FactStopEventRow.model_fields


def test_station_accepts_the_unresolved_row() -> None:
    """The sentinel carries only its key, its name and is_passenger."""
    row = DimStationRow.model_validate(
        {"station_key": -1, "station_name": "Unresolved", "is_passenger": False}
    )
    assert row.ptcar_no is None
    assert row.latitude is None


def test_station_accepts_the_v2_attributes() -> None:
    row = DimStationRow.model_validate(
        {
            "station_key": 455,
            "station_name": "GENT-SINT-PIETERS",
            "ptcar_no": 455,
            "first_seen": dt.date(2014, 1, 1),
            "last_seen": dt.date(2026, 9, 7),
            "station_type": "Station",
            "is_passenger": True,
            "latitude": 51.036,
            "longitude": 3.710,
        }
    )
    assert row.station_key == 455
    assert row.is_passenger is True


@pytest.mark.parametrize("missing", ["station_name", "is_passenger"])
def test_station_requires_its_non_null_attributes(missing: str) -> None:
    payload: dict[str, object] = {
        "station_key": 455,
        "station_name": "GENT-SINT-PIETERS",
        "is_passenger": True,
    }
    del payload[missing]
    with pytest.raises(ValidationError):
        DimStationRow.model_validate(payload)


def test_station_rejects_a_hash_key() -> None:
    """A v1 MD5 station_key must fail, not be coerced."""
    with pytest.raises(ValidationError):
        DimStationRow.model_validate(
            {"station_key": _HASH, "station_name": "MOENSBERG", "is_passenger": True}
        )


def test_relation_allows_null_direction() -> None:
    """The upstream concat_ws case: a null direction is legitimate data."""
    row = DimRelationRow.model_validate(
        {"relation_key": _HASH, "relation": "THAL", "relation_direction": None, "operator": "X"}
    )
    assert row.relation_direction is None
