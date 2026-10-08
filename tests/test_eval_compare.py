"""Tests for result comparison and the retrieval and latency metrics."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from rail_rag.eval.cases import CompareSpec
from rail_rag.eval.compare import Rows, compare_results, hit_at_k, percentile, reciprocal_rank

ORDERED = CompareSpec(mode="ordered")
SET = CompareSpec(mode="set")
SCALAR = CompareSpec(mode="scalar")

_REFERENCE = [("Gent", 0.9), ("Leuven", 0.8), ("Mons", 0.7)]


def _passed(reference: Rows, actual: Rows, spec: CompareSpec) -> bool:
    return compare_results(reference, actual, spec)[0]


# --- columns are matched by value ---------------------------------------------------


def test_identical_results_match() -> None:
    assert compare_results(_REFERENCE, list(_REFERENCE), ORDERED) == (True, "match")


def test_an_extra_actual_column_is_allowed() -> None:
    actual = [("Gent", 0.9, 1200), ("Leuven", 0.8, 900), ("Mons", 0.7, 400)]
    assert _passed(_REFERENCE, actual, ORDERED)


def test_swapped_column_order_is_allowed() -> None:
    actual = [(0.9, "Gent"), (0.8, "Leuven"), (0.7, "Mons")]
    assert _passed(_REFERENCE, actual, ORDERED)
    assert _passed(_REFERENCE, actual, SET)


def test_a_missing_column_fails_with_a_reason() -> None:
    passed, reason = compare_results(_REFERENCE, [("Gent",), ("Leuven",), ("Mons",)], ORDERED)
    assert not passed
    assert "fewer than 2" in reason


def test_a_column_with_different_values_fails() -> None:
    actual = [("Gent", 0.9), ("Leuven", 0.8), ("Mons", 0.1)]
    passed, reason = compare_results(_REFERENCE, actual, ORDERED)
    assert not passed
    assert "reference column 2" in reason


def test_two_identical_reference_columns_need_two_actual_columns() -> None:
    assert _passed([(1, 1)], [(1, 1)], ORDERED)
    assert not _passed([(1, 1)], [(1, 2)], ORDERED)


# --- row order -----------------------------------------------------------------------


def test_wrong_row_order_fails_when_ordered_and_passes_as_a_set() -> None:
    actual = [("Mons", 0.7), ("Gent", 0.9), ("Leuven", 0.8)]
    assert not _passed(_REFERENCE, actual, ORDERED)
    assert _passed(_REFERENCE, actual, SET)


def test_set_mode_compares_rows_as_multisets() -> None:
    assert _passed([("a", 1), ("a", 1), ("b", 2)], [("b", 2), ("a", 1), ("a", 1)], SET)
    assert not _passed([("a", 1), ("a", 1), ("b", 2)], [("a", 1), ("b", 2), ("b", 2)], SET)


def test_set_mode_rejects_columns_that_match_only_separately() -> None:
    reference = [("a", 1), ("b", 2)]
    actual = [("a", 2), ("b", 1)]
    passed, reason = compare_results(reference, actual, SET)
    assert not passed
    assert "not together" in reason


def test_a_different_row_count_fails() -> None:
    assert not _passed(_REFERENCE, _REFERENCE[:2], ORDERED)


def test_top_k_compares_only_the_leading_rows() -> None:
    spec = CompareSpec(mode="ordered", top_k=2)
    actual = [("Gent", 0.9), ("Leuven", 0.8), ("Unrelated", 0.0), ("Rows", 0.0)]
    assert _passed(_REFERENCE, actual, spec)


def test_top_k_still_fails_inside_the_window() -> None:
    spec = CompareSpec(mode="ordered", top_k=2)
    assert not _passed(_REFERENCE, [("Leuven", 0.8), ("Gent", 0.9), ("Mons", 0.7)], spec)


def test_top_k_fails_when_the_actual_result_is_shorter_than_k() -> None:
    assert not _passed(_REFERENCE, _REFERENCE[:1], CompareSpec(mode="ordered", top_k=2))


# --- normalisation -------------------------------------------------------------------


def test_floats_are_compared_after_rounding() -> None:
    assert _passed([(0.123449,)], [(0.123401,)], SCALAR)
    assert not _passed([(0.1234,)], [(0.1236,)], SCALAR)


def test_the_number_of_decimals_is_configurable() -> None:
    loose = CompareSpec(mode="scalar", decimals=1)
    assert _passed([(0.12,)], [(0.14,)], loose)


def test_ints_floats_and_decimals_are_interchangeable() -> None:
    assert _passed([(5,)], [(Decimal("5.0"),)], SCALAR)
    assert _passed([(2.5,)], [(Decimal("2.50"),)], SCALAR)


def test_strings_are_trimmed_and_casefolded() -> None:
    assert _passed([("  BRUSSEL-ZUID",)], [("Brussel-Zuid  ",)], SCALAR)


def test_dates_compare_as_iso_strings() -> None:
    assert _passed([("2025-03-01",)], [(dt.date(2025, 3, 1),)], SCALAR)
    assert _passed([("2025-03-01T10:00:00",)], [(dt.datetime(2025, 3, 1, 10),)], SCALAR)


def test_none_matches_only_none() -> None:
    assert _passed([(None,)], [(None,)], SCALAR)
    assert not _passed([(None,)], [(0,)], SCALAR)


# --- percentages ---------------------------------------------------------------------


def test_a_percentage_matches_only_when_accepted() -> None:
    assert not _passed([(0.952,)], [(95.2,)], SCALAR)
    assert _passed([(0.952,)], [(95.2,)], CompareSpec(mode="scalar", accept_percent=True))


def test_a_percentage_rounded_to_two_decimals_still_matches() -> None:
    spec = CompareSpec(mode="ordered", accept_percent=True)
    assert _passed([("a", 0.95231), ("b", 0.8)], [("a", 95.23), ("b", 80.0)], spec)


def test_accepting_percentages_does_not_break_exact_ratios() -> None:
    assert _passed([(0.952,)], [(0.952,)], CompareSpec(mode="scalar", accept_percent=True))


# --- precision-aware matching ---------------------------------------------------------

_PRECISE = CompareSpec(mode="scalar", accept_percent=True)


@pytest.mark.parametrize(
    ("actual", "expected_to_pass"),
    [
        pytest.param(90.4, True, id="percent-one-decimal-is-the-minimum"),
        pytest.param(90.45, True, id="percent-two-decimals"),
        pytest.param(90, False, id="percent-integer-is-too-coarse"),
        pytest.param(0.904, True, id="ratio-three-decimals-is-the-minimum"),
        pytest.param(0.9, False, id="ratio-one-decimal-is-too-coarse"),
        pytest.param(0.90449123, True, id="unrounded-ratio"),
        pytest.param(90.449123, True, id="unrounded-percent"),
        pytest.param(Decimal("90.45"), True, id="decimal-percent"),
        pytest.param(0.905, False, id="three-decimals-but-a-different-value"),
        pytest.param(90.5, False, id="percent-one-decimal-but-a-different-value"),
    ],
)
def test_a_coarsely_rounded_rate_is_judged_at_its_own_precision(
    actual: float | Decimal, expected_to_pass: bool
) -> None:
    assert _passed([(0.90449,)], [(actual,)], _PRECISE) is expected_to_pass


def test_unrounded_floats_keep_the_existing_tolerance() -> None:
    assert _passed([(0.904491,)], [(0.904523,)], _PRECISE)
    assert not _passed([(0.9045,)], [(0.9047,)], _PRECISE)


def test_the_minimum_precision_applies_to_each_row_of_a_series() -> None:
    spec = CompareSpec(mode="ordered", accept_percent=True)
    reference = [(0.91234,), (0.88765,)]
    assert _passed(reference, [(91.2,), (88.8,)], spec)
    assert not _passed(reference, [(91.0,), (88.8,)], spec)


def test_coarse_values_are_judged_the_same_way_in_set_mode() -> None:
    spec = CompareSpec(mode="set", accept_percent=True)
    reference = [("a", 0.91234), ("b", 0.88765)]
    assert _passed(reference, [("b", 88.8), ("a", 91.2)], spec)
    assert not _passed(reference, [("b", 89.0), ("a", 91.2)], spec)


def test_counts_are_not_loosened_by_the_precision_rule() -> None:
    assert _passed([(1965000,)], [(1965000,)], SCALAR)
    assert not _passed([(1965000,)], [(1965001,)], SCALAR)
    assert _passed([(5,)], [(5.0,)], SCALAR)


def test_a_reference_that_is_short_itself_needs_no_more_precision() -> None:
    assert _passed([(0.5,)], [(0.5,)], CompareSpec(mode="scalar"))
    assert _passed([(0.5,)], [(50.0,)], _PRECISE)
    assert _passed([(1.0,)], [(100,)], _PRECISE)


# --- series: the period column is optional -------------------------------------------

_SERIES = [(0.9161,), (0.8968,), (0.9129,)]
_PERIODS = {
    "month-number": [1, 2, 3],
    "month-name": ["January", "February", "March"],
    "iso-string": ["2025-01-01", "2025-02-01", "2025-03-01"],
    "date": [dt.date(2025, 1, 1), dt.date(2025, 2, 1), dt.date(2025, 3, 1)],
    "datetime": [dt.datetime(2025, m, 1, tzinfo=dt.UTC) for m in (1, 2, 3)],
}


@pytest.mark.parametrize("period", _PERIODS.values(), ids=_PERIODS.keys())
@pytest.mark.parametrize("first", [True, False], ids=["period-first", "period-last"])
def test_a_series_passes_with_the_period_in_any_representation(
    period: list[object], first: bool
) -> None:
    spec = CompareSpec(mode="ordered", accept_percent=True)
    actual = [
        (label, rate) if first else (rate, label)
        for label, (rate,) in zip(period, _SERIES, strict=True)
    ]
    assert _passed(_SERIES, actual, spec)


def test_a_series_with_the_right_period_but_wrong_values_still_fails() -> None:
    spec = CompareSpec(mode="ordered", accept_percent=True)
    assert not _passed(_SERIES, [(1, 0.5), (2, 0.8968), (3, 0.9129)], spec)


# --- empty results -------------------------------------------------------------------


def test_empty_versus_non_empty_fails_in_both_directions() -> None:
    assert not _passed(_REFERENCE, [], ORDERED)
    assert not _passed([], _REFERENCE, ORDERED)


def test_an_empty_reference_matches_only_an_empty_actual() -> None:
    assert compare_results([], [], SET) == (True, "both results are empty")


# --- scalar --------------------------------------------------------------------------


def test_scalar_passes_when_any_cell_of_the_first_row_matches() -> None:
    assert _passed([(0.95,)], [("Gent", 0.95, 12)], SCALAR)


def test_scalar_ignores_rows_after_the_first() -> None:
    assert not _passed([(0.95,)], [("Gent", 0.1), ("Leuven", 0.95)], SCALAR)


def test_scalar_fails_when_no_cell_matches() -> None:
    passed, reason = compare_results([(0.95,)], [(0.5, 0.6)], SCALAR)
    assert not passed
    assert "0.95" in reason


def test_a_scalar_reference_must_be_one_by_one() -> None:
    passed, reason = compare_results([(1, 2)], [(1, 2)], SCALAR)
    assert not passed
    assert "single" in reason


# --- retrieval metrics ---------------------------------------------------------------

_A, _B, _C, _D = ("doc-a", "A"), ("doc-b", None), ("doc-c", "C"), ("doc-d", "D")


def test_a_hit_at_rank_one() -> None:
    returned = [_A, _B, _C]
    assert hit_at_k({_A}, returned, 1)
    assert reciprocal_rank({_A}, returned) == 1.0


def test_a_hit_at_rank_three() -> None:
    returned = [_B, _D, _C]
    assert not hit_at_k({_C}, returned, 2)
    assert hit_at_k({_C}, returned, 3)
    assert reciprocal_rank({_C}, returned) == pytest.approx(1 / 3)


def test_a_miss() -> None:
    assert not hit_at_k({_A}, [_B, _C, _D], 3)
    assert reciprocal_rank({_A}, [_B, _C, _D]) == 0.0
    assert reciprocal_rank({_A}, []) == 0.0


def test_the_first_of_several_expected_passages_counts() -> None:
    assert reciprocal_rank({_A, _C}, [_B, _C, _A]) == 0.5


def test_a_heading_must_match_as_well_as_the_document() -> None:
    assert not hit_at_k({("doc-a", "A")}, [("doc-a", "other")], 1)


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        hit_at_k({_A}, [_A], 0)


# --- percentile ----------------------------------------------------------------------


def test_percentiles_interpolate_between_ranks() -> None:
    values = [4.0, 1.0, 3.0, 2.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 50) == 2.5
    assert percentile(values, 100) == 4.0
    assert percentile(values, 95) == pytest.approx(3.85)


def test_the_percentile_of_one_value_is_that_value() -> None:
    assert percentile([7.0], 95) == 7.0


def test_the_percentile_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="no values"):
        percentile([], 50)
    with pytest.raises(ValueError, match="within"):
        percentile([1.0], 101)
