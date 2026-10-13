"""``SeriesGroupBy.agg`` never hands the pivot back a Series.

Instruction 288. ``_aggregate`` carried a branch converting a Series
result into a frame, marked ``# pragma: no cover``. Its reason was right
and is pinned here, because the deletion rests on three separate facts
and any one of them could change without anyone noticing.

1. ``PivotSpec.__post_init__`` starts its aggregation list with ``n`` and
   appends the rest, so ``n`` survives ``aggs=()`` and ``with_aggs(())``.
2. ``n`` maps to a real pandas function name, so it is never filtered out
   of ``wanted``.
3. pandas returns a DataFrame for a LIST of function names, however short.

Checked exhaustively when the branch was removed: all 256 subsets of the
eight aggregations, none of which produced a Series.
"""
from __future__ import annotations

import itertools

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import pivot_spec as P


def _frame():
    return pd.DataFrame({"g": ["a", "a", "b", "b"],
                         "v": [1.0, 2.0, 3.0, 4.0]})


def test_n_survives_an_empty_aggregation_list():
    """PREMISE 1. Without this the list could be empty and pandas would
    be asked to aggregate by nothing."""
    spec = P.PivotSpec(rows=("g",), values=("v",), aggs=())
    assert P.N in spec.aggs
    assert spec.with_aggs(()).aggs == (P.N,)


def test_n_maps_to_a_real_pandas_name():
    """PREMISE 2. `wanted` filters to names pandas knows, so an `n` that
    was not in that table would be dropped and could empty the list."""
    assert P.N in P._PANDAS_NAMES
    assert P._PANDAS_NAMES[P.N] == "count"


@pytest.mark.parametrize("count", [1, 2, 3])
def test_pandas_returns_a_frame_for_a_list_however_short(count):
    """PREMISE 3, against the real pandas in this environment.

    A one-element LIST is the interesting case: `agg("count")` returns a
    Series, `agg(["count"])` returns a DataFrame, and the pivot always
    passes a list.
    """
    grouped = _frame().groupby("g")["v"]
    names = ["count", "mean", "median"][:count]
    assert isinstance(grouped.agg(names), pd.DataFrame)


def test_a_bare_name_would_have_returned_a_series():
    """WHY premise 3 is about the list and not about pandas generally.

    If the pivot ever stopped wrapping its names in a list, the deleted
    branch would be needed again. This is the mutation that would break
    it, asserted directly.
    """
    grouped = _frame().groupby("g")["v"]
    assert isinstance(grouped.agg("count"), pd.Series)


def test_no_combination_of_aggregations_produces_a_series():
    """All 256 subsets, the search that settled it."""
    grouped = _frame().groupby("g")["v"]
    every = list(P._PANDAS_NAMES) + [P.QUANTILE]
    checked = 0
    for size in range(len(every) + 1):
        for combo in itertools.combinations(every, size):
            spec = P.PivotSpec(rows=("g",), values=("v",), aggs=combo)
            wanted = [P._PANDAS_NAMES[a] for a in spec.aggs
                      if a in P._PANDAS_NAMES]
            assert wanted, f"{combo} produced an empty aggregation list"
            assert isinstance(grouped.agg(wanted), pd.DataFrame)
            checked += 1
    assert checked == 256, f"expected 256 subsets, checked {checked}"
