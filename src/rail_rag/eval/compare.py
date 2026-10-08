"""Pure functions that hold an actual result and a retrieval ranking against the ground truth."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable, Collection, Sequence
from decimal import Decimal
from typing import Any

from rail_rag.eval.cases import CompareSpec

Rows = Sequence[Sequence[Any]]
SourceKey = tuple[str, str | None]

#: An actual column index, with the divisor that brings its values onto the reference scale.
_Candidate = tuple[int, float]

_PERCENT = 100.0
#: Fewest decimal places, on the ratio scale, that a coarsely rounded value must still carry.
_MIN_RATIO_PLACES = 3


def compare_results(reference: Rows, actual: Rows, spec: CompareSpec) -> tuple[bool, str]:
    """Return whether ``actual`` matches ``reference`` under ``spec``, and why not if it doesn't."""
    if not reference:
        return (
            (True, "both results are empty")
            if not actual
            else (
                False,
                f"the reference is empty but the actual result has {len(actual)} rows",
            )
        )
    if not actual:
        return False, f"the actual result is empty but the reference has {len(reference)} rows"

    if spec.mode == "scalar":
        return _compare_scalar(reference, actual, spec)

    expected_rows, got_rows = list(reference), list(actual)
    if spec.mode == "ordered" and spec.top_k is not None:
        expected_rows, got_rows = expected_rows[: spec.top_k], got_rows[: spec.top_k]
    if len(expected_rows) != len(got_rows):
        return False, f"row count differs: reference {len(expected_rows)}, actual {len(got_rows)}"

    width = len(expected_rows[0])
    if len(got_rows[0]) < width:
        return False, f"the actual result has {len(got_rows[0])} columns, fewer than {width}"

    ordered = spec.mode == "ordered"
    candidates = [
        _candidate_columns(i, expected_rows, got_rows, spec, ordered) for i in range(width)
    ]
    for index, options in enumerate(candidates):
        if not options:
            return False, f"no actual column matches reference column {index + 1} by value"

    if _assign(candidates, expected_rows, got_rows, spec, ordered):
        return True, "match"
    return False, "the columns match individually but not together as rows"


def hit_at_k(expected: Collection[SourceKey], returned: Sequence[SourceKey], k: int) -> bool:
    """True when any expected passage is among the first ``k`` returned."""
    if k <= 0:
        raise ValueError("k must be positive")
    return any(item in expected for item in returned[:k])


def reciprocal_rank(expected: Collection[SourceKey], returned: Sequence[SourceKey]) -> float:
    """One over the 1-based rank of the first expected passage, or zero if none was returned."""
    for rank, item in enumerate(returned, start=1):
        if item in expected:
            return 1.0 / rank
    return 0.0


def percentile(values: Sequence[float], q: float) -> float:
    """The ``q``-th percentile (0 to 100) by linear interpolation between closest ranks."""
    if not values:
        raise ValueError("percentile of no values")
    if not 0 <= q <= 100:
        raise ValueError("q must be within [0, 100]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _compare_scalar(reference: Rows, actual: Rows, spec: CompareSpec) -> tuple[bool, str]:
    if len(reference) != 1 or len(reference[0]) != 1:
        return False, "a scalar reference must be a single row with a single column"
    wanted = reference[0][0]
    for cell in actual[0]:
        if any(_values_match(wanted, cell, spec, scale) for scale in _scales(spec)):
            return True, "match"
    return False, f"no cell of the first actual row equals {_normalise(wanted, spec.decimals)!r}"


def _scales(spec: CompareSpec) -> tuple[float, ...]:
    return (1.0, _PERCENT) if spec.accept_percent else (1.0,)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float | Decimal) and not isinstance(value, bool)


def _normalise(value: Any, decimals: int) -> Any:
    """Reduce a non-numeric cell to something comparable; numbers are rounded."""
    if value is None or isinstance(value, bool):
        return value
    if _is_number(value):
        return round(float(value), decimals)
    if isinstance(value, dt.date):
        value = value.isoformat()
    if isinstance(value, str):
        return value.strip().casefold()
    return value


def _places(value: Any, *, significant: bool = False) -> int:
    """Decimal places the value is written with, so 90.4 has one and 90 has none."""
    digits = Decimal(str(value))
    exponent = (digits.normalize() if significant else digits).as_tuple().exponent
    return -exponent if isinstance(exponent, int) and exponent < 0 else 0


def _values_match(wanted: Any, got: Any, spec: CompareSpec, divisor: float = 1.0) -> bool:
    """Compare two cells; a number written to fewer places is judged at its own precision."""
    if not (_is_number(wanted) and _is_number(got)):
        return bool(_normalise(wanted, spec.decimals) == _normalise(got, spec.decimals))

    target = round(float(wanted), spec.decimals)
    places = _places(got) + round(math.log10(divisor))
    if places >= spec.decimals:
        return round(float(got) / divisor, spec.decimals) == target

    needed = min(_MIN_RATIO_PLACES, spec.decimals, _places(target, significant=True))
    if places < needed:
        return False
    return round(float(got) / divisor, places) == round(float(wanted), places)


def _matchable(size: int, pair: Callable[[int, int], bool]) -> bool:
    """True when every reference item can be paired with a distinct actual item."""
    owner = [-1] * size

    def place(item: int, seen: set[int]) -> bool:
        for slot in range(size):
            if slot in seen or not pair(item, slot):
                continue
            seen.add(slot)
            if owner[slot] == -1 or place(owner[slot], seen):
                owner[slot] = item
                return True
        return False

    return all(place(item, set()) for item in range(size))


def _candidate_columns(
    index: int,
    expected_rows: Rows,
    got_rows: Rows,
    spec: CompareSpec,
    ordered: bool,
) -> list[_Candidate]:
    """Actual columns, with the scale they need, whose values match reference column ``index``."""
    options: list[_Candidate] = []
    for position in range(len(got_rows[0])):
        for scale in _scales(spec):

            def same(row: int, other: int, position: int = position, scale: float = scale) -> bool:
                return _values_match(
                    expected_rows[row][index], got_rows[other][position], spec, scale
                )

            count = len(expected_rows)
            if all(same(row, row) for row in range(count)) if ordered else _matchable(count, same):
                options.append((position, scale))
    return options


def _assign(
    candidates: list[list[_Candidate]],
    expected_rows: Rows,
    got_rows: Rows,
    spec: CompareSpec,
    ordered: bool,
) -> bool:
    """Find distinct actual columns for every reference column such that the rows agree."""
    chosen: list[_Candidate] = []

    def rows_agree() -> bool:
        if ordered:
            return True  # column-wise equality already fixes every row

        def same(row: int, other: int) -> bool:
            return all(
                _values_match(expected_rows[row][column], got_rows[other][position], spec, scale)
                for column, (position, scale) in enumerate(chosen)
            )

        return _matchable(len(expected_rows), same)

    def search(depth: int) -> bool:
        if depth == len(candidates):
            return rows_agree()
        for option in candidates[depth]:
            if any(option[0] == taken[0] for taken in chosen):
                continue
            chosen.append(option)
            if search(depth + 1):
                return True
            chosen.pop()
        return False

    return search(0)
